"""Figures for sample-shared and within-sample sensor-output coupling."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from nasp_compendium.display import humanize_module_name  # type: ignore[import]

from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


class SampleCouplingPlotter(_PlotterBase):
    """Export within/between-sample slopes and variance partitions.

    Both views read the tables written by `sample_coupling` and save PNG,
    vector PDF and the displayed rows as CSV. Non-estimable and excluded
    results are labelled rather than drawn as zero.

    Example Usage:
      >>> plotter = SampleCouplingPlotter("path/to/output")
      >>> paths = plotter.plot_coupling(coupling, row_height=0.11)
      >>> paths += plotter.plot_variance_partition(variance_partition)
    """

    def plot_coupling(
        self,
        coupling: pd.DataFrame,
        *,
        responses: Sequence[str] | None = None,
        order_by_response: str | None = None,
        alpha: float = 0.05,
        x_quantile: float = 0.95,
        filename: str = "sensor_output_sample_coupling",
        figsize: tuple[float, float] | None = None,
        panel_width: float = 1.35,
        row_height: float = 0.105,
        panel_spacing: float = 0.03,
        marker_size: float = 7.0,
        title: str | None = None,
        title_fontsize: float | None = None,
        tick_fontsize: float | None = None,
        within_color: str = "#0072B2",
        between_color: str = "#D55E00",
        connector_color: str = "#999999",
        x_label: str = "Score change per sensor log1p unit",
    ) -> list[Path]:
        """Plot within- and between-sample slopes with 95% CIs per sensor.

        Each panel is one output module and keeps its own x limits, because
        module scores have module-specific units. Within a panel, both slopes
        share one axis. Filled markers have BH-adjusted p-values below
        `alpha` within their estimand family; open markers do not. A gray
        connector marks a contextual (between minus within) slope with
        adjusted p below `alpha`. Sensors that belong to the output module
        are labelled "in module"; unfitted pairs are labelled "n.e.".

        Args:
          coupling: `sensor_output_sample_coupling.csv` rows.
          responses: Output modules to show, in panel order. Defaults to the
            order of first appearance.
          order_by_response: Module whose within-sample slope orders the
            sensors (descending). Defaults to the first panel.
          alpha: Adjusted p-value threshold for filled markers (0 to 1).
          x_quantile: Quantile of CI endpoints that sets each panel's x
            limits (0.5 to 1; 1 shows every interval in full). Intervals
            beyond the limits end in an edge arrow; the CSV keeps values.
          filename: Output stem for PNG, PDF and displayed CSV.
          figsize: Overall inches; overrides the panel and row geometry.
          panel_width: Inches per panel when `figsize` is None.
          row_height: Inches per sensor row when `figsize` is None.
          panel_spacing: Horizontal space between panels, as a fraction of
            the figure width (constrained layout).
          marker_size: Marker area in points squared.
          title: Figure title; an empty string hides it.
          title_fontsize: Title size in points; None keeps the shared style.
          tick_fontsize: Tick-label size in points; None keeps the style.
          within_color: Within-sample marker and CI color.
          between_color: Between-sample marker and CI color.
          connector_color: Color of the contextual-difference connector.
          x_label: Axis label for every panel, for example when plotting
            arm differences of slopes rather than slopes.

        Returns:
          Paths of the saved PNG, PDF and displayed CSV, or an empty list
          when no pair is estimable.

        Example Usage:
          >>> paths = plotter.plot_coupling(
          ...     coupling, responses=["IFN_I_OUTPUT", "ISR"],
          ...     panel_width=1.5, row_height=0.12, alpha=0.05,
          ... )
        """
        _validate_positive(
            panel_width=panel_width,
            row_height=row_height,
            marker_size=marker_size,
        )
        _validate_fontsizes(title_fontsize, tick_fontsize)
        _validate_figsize(figsize)

        if not 0 < alpha < 1:
            raise ValueError("alpha must lie between 0 and 1")
        if not 0.5 <= x_quantile <= 1:
            raise ValueError("x_quantile must lie between 0.5 and 1")
        if not np.isfinite(panel_spacing) or panel_spacing < 0:
            raise ValueError("panel_spacing must be finite and nonnegative")

        _require_columns(
            coupling,
            (
                "predictor",
                "response",
                "estimand",
                "estimate",
                "ci_low",
                "ci_high",
                "pvalue_fdr",
                "estimable",
                "status",
            ),
            table="coupling",
        )

        panels = list(
            responses
            if responses is not None
            else dict.fromkeys(coupling["response"].astype(str))
        )
        if missing := set(panels).difference(coupling["response"]):
            raise ValueError(f"Responses absent from coupling: {missing}")

        displayed = _coupling_display(
            coupling.loc[coupling["response"].isin(panels)],
            order_by_response=order_by_response or panels[0],
            alpha=alpha,
        )
        if not displayed["estimable"].any():
            logger.info("No estimable sensor-output coupling to plot")
            return []

        n_rows = displayed["predictor"].nunique()
        dimensions = figsize or (
            0.8 + panel_width * len(panels),
            0.75 + row_height * n_rows,
        )

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axes = plt.subplots(
                1,
                len(panels),
                figsize=dimensions,
                sharey=True,
                squeeze=False,
                layout="constrained",
            )
            figure.get_layout_engine().set(wspace=panel_spacing)  # type: ignore[call-arg]
            try:
                for axis, response in zip(axes[0], panels, strict=True):
                    _draw_coupling_panel(
                        axis,
                        displayed.loc[displayed["response"].eq(response)],
                        response=response,
                        x_quantile=x_quantile,
                        marker_size=marker_size,
                        within_color=within_color,
                        between_color=between_color,
                        connector_color=connector_color,
                        x_label=x_label,
                    )
                    if tick_fontsize is not None:
                        axis.tick_params(labelsize=tick_fontsize)

                support = displayed.loc[
                    displayed["estimable"], "n_independent_units"
                ]
                figure.suptitle(
                    _coupling_title(support) if title is None else title,
                    x=0.01,
                    ha="left",
                    fontsize=title_fontsize,
                )

                figure.legend(
                    handles=_coupling_legend(
                        within_color=within_color,
                        between_color=between_color,
                        connector_color=connector_color,
                        alpha=alpha,
                    ),
                    loc="outside lower center",
                    ncol=6,
                    frameon=False,
                    handletextpad=0.4,
                    columnspacing=1.0,
                )

                return self._save_with_data(figure, displayed, filename)
            finally:
                plt.close(figure)

    def plot_variance_partition(
        self,
        variance_partition: pd.DataFrame,
        *,
        filename: str = "sample_variance_partition",
        figsize: tuple[float, float] | None = None,
        width: float = 3.5,
        row_height: float = 0.075,
        panel_spacing: float = 0.04,
        title: str | None = None,
        title_fontsize: float | None = None,
        tick_fontsize: float | None = None,
        donor_color: str = "#33608C",
        sample_color: str = "#8FB9D9",
        residual_color: str = "#DDDDDD",
    ) -> list[Path]:
        """Plot donor, donor-tissue and residual variance fractions.

        Module scores and sensor genes are separate panels, each ordered by
        the sample-shared fraction (donor plus donor-tissue). Saved fractions
        are not renormalized. A hatched segment shows variance that was not
        allocated to a fitted component, for example a component that the
        design could not support. Non-estimable features are labelled.

        Args:
          variance_partition: `sample_variance_partition.csv` rows.
          filename: Output stem for PNG, PDF and displayed CSV.
          figsize: Overall inches; overrides `width` and `row_height`.
          width: Figure width in inches when `figsize` is None.
          row_height: Inches per feature row when `figsize` is None.
          panel_spacing: Vertical space between panels, as a fraction of
            the figure height (constrained layout).
          title: Figure title; an empty string hides it.
          title_fontsize: Title size in points; None keeps the shared style.
          tick_fontsize: Tick-label size in points; None keeps the style.
          donor_color: Independent-unit (donor) component color.
          sample_color: Donor-tissue sample component color.
          residual_color: Residual component color.

        Returns:
          Paths of the saved PNG, PDF and displayed CSV, or an empty list
          when the table is empty.

        Example Usage:
          >>> paths = plotter.plot_variance_partition(
          ...     variance_partition, width=3.2, row_height=0.08,
          ... )
        """
        _validate_positive(width=width, row_height=row_height)
        _validate_fontsizes(title_fontsize, tick_fontsize)
        _validate_figsize(figsize)

        if not np.isfinite(panel_spacing) or panel_spacing < 0:
            raise ValueError("panel_spacing must be finite and nonnegative")

        _require_columns(
            variance_partition,
            (
                "feature_type",
                "feature_label",
                "component_type",
                "variance_fraction",
                "estimable",
                "status",
            ),
            table="variance_partition",
        )

        displayed = _partition_display(variance_partition)
        if displayed.empty:
            logger.info("No variance partition rows to plot")
            return []

        feature_types = [
            feature_type
            for feature_type in ("module_score", "gene_expression")
            if feature_type in set(displayed["feature_type"])
        ]
        counts = [
            int(displayed["feature_type"].eq(feature_type).sum())
            for feature_type in feature_types
        ]
        dimensions = figsize or (
            width,
            0.7 + row_height * sum(counts) + 0.25 * len(counts),
        )

        colors = {
            "donor_fraction": donor_color,
            "sample_fraction": sample_color,
            "residual_fraction": residual_color,
        }

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axes = plt.subplots(
                len(feature_types),
                1,
                figsize=dimensions,
                squeeze=False,
                gridspec_kw={"height_ratios": counts},
                layout="constrained",
            )
            figure.get_layout_engine().set(hspace=panel_spacing)  # type: ignore[call-arg]
            try:
                for axis, feature_type in zip(
                    axes[:, 0], feature_types, strict=True
                ):
                    _draw_partition_panel(
                        axis,
                        displayed.loc[
                            displayed["feature_type"].eq(feature_type)
                        ],
                        feature_type=feature_type,
                        colors=colors,
                    )
                    if tick_fontsize is not None:
                        axis.tick_params(labelsize=tick_fontsize)

                axes[-1, 0].set_xlabel(
                    "Fraction of context-centered variance (model-conditional)",
                    labelpad=3,
                )

                figure.suptitle(
                    (
                        "Variance shared within donors and donor-tissue samples"
                        if title is None
                        else title
                    ),
                    x=0.01,
                    ha="left",
                    fontsize=title_fontsize,
                )

                figure.legend(
                    handles=_partition_legend(colors),
                    loc="outside lower center",
                    ncol=4,
                    frameon=False,
                    handletextpad=0.4,
                    columnspacing=1.0,
                )

                return self._save_with_data(figure, displayed, filename)
            finally:
                plt.close(figure)


def _coupling_display(
    coupling: pd.DataFrame,
    *,
    order_by_response: str,
    alpha: float,
) -> pd.DataFrame:
    """Order sensors by within-sample slope and flag displayed evidence."""
    frame = coupling.copy()
    frame["estimable"] = frame["estimable"].astype(bool)
    frame["significant"] = frame["estimable"] & frame["pvalue_fdr"].lt(alpha)

    within = frame.loc[
        frame["response"].eq(order_by_response)
        & frame["estimand"].eq("within_sample_slope")
    ].set_index("predictor")["estimate"]

    predictors = list(dict.fromkeys(frame["predictor"].astype(str)))
    ordered = sorted(
        predictors,
        key=lambda predictor: (
            not np.isfinite(within.get(predictor, np.nan)),
            -within.get(predictor, 0.0)
            if np.isfinite(within.get(predictor, np.nan))
            else 0.0,
            predictor,
        ),
    )

    frame["display_row"] = frame["predictor"].map(
        {predictor: row for row, predictor in enumerate(ordered)}
    )

    return frame.sort_values(
        ["display_row", "response", "estimand"], kind="stable"
    ).reset_index(drop=True)


def _draw_coupling_panel(
    axis: Axes,
    panel: pd.DataFrame,
    *,
    response: str,
    x_quantile: float,
    marker_size: float,
    within_color: str,
    between_color: str,
    connector_color: str,
    x_label: str,
) -> None:
    """Draw one output module's slopes, CIs and unavailable labels."""
    rows = panel["display_row"].max() + 1
    estimates, significant = _coupling_panel_tables(panel)
    status = panel.groupby("display_row")["status"].first()

    for estimand, offset, marker, color in (
        ("within_sample_slope", -0.17, "o", within_color),
        ("between_sample_slope", 0.17, "D", between_color),
    ):
        if ("estimate", estimand) not in estimates:
            continue

        rows_drawn = estimates[("estimate", estimand)].dropna().index
        y = rows_drawn + offset
        axis.hlines(
            y,
            estimates.loc[rows_drawn, ("ci_low", estimand)],
            estimates.loc[rows_drawn, ("ci_high", estimand)],
            color=color,
            linewidth=0.5,
            zorder=2,
        )

        filled = significant.loc[rows_drawn, estimand].astype(bool)
        axis.scatter(
            estimates.loc[rows_drawn, ("estimate", estimand)],
            y,
            s=marker_size * (0.8 if marker == "D" else 1.0),
            marker=marker,
            facecolors=np.where(filled, color, "white"),
            edgecolors=color,
            linewidths=0.5,
            zorder=3,
        )

    if "sample_contextual_slope" in significant:
        _draw_contextual_connectors(
            axis, estimates, significant, color=connector_color
        )

    drawn = estimates[("estimate", "within_sample_slope")].dropna().index
    low, high = _panel_limits(panel, x_quantile=x_quantile)
    pad = 0.06 * (high - low or 1.0)
    axis.set_xlim(low - pad, high + pad)
    _mark_clipped(axis, panel, low=low - pad, high=high + pad)
    _label_unfitted_rows(
        axis, status.loc[~status.index.isin(drawn)], x=(low + high) / 2
    )

    labels = panel.groupby("display_row")["predictor"].first()
    axis.set_yticks(labels.index, labels.tolist())
    axis.set_ylim(rows - 0.4, -0.6)
    axis.axvline(0, color="#AAAAAA", linewidth=0.4, zorder=0)
    axis.set_title(humanize_module_name(response), pad=3)
    axis.set_xlabel(x_label, labelpad=2)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.tick_params(axis="x", length=2, pad=1.5)
    axis.spines[["top", "right", "left"]].set_visible(False)


def _coupling_panel_tables(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot a panel's slopes and significance to display rows by estimand.

    Reindexing keeps every row and estimand column, so a panel whose pairs
    are all non-estimable does not lose them to pivot's NaN dropping.
    """
    estimands = [
        "within_sample_slope",
        "between_sample_slope",
        "sample_contextual_slope",
    ]

    rows_index = sorted(panel["display_row"].unique())
    estimates = panel.pivot_table(
        index="display_row",
        columns="estimand",
        values=["estimate", "ci_low", "ci_high"],
        aggfunc="first",
    ).reindex(
        index=rows_index,
        columns=pd.MultiIndex.from_product(
            [["estimate", "ci_low", "ci_high"], estimands]
        ),
    )

    significant = (
        panel.pivot_table(
            index="display_row",
            columns="estimand",
            values="significant",
            aggfunc="first",
        )
        .reindex(index=rows_index, columns=estimands)
        .fillna(False)
        .astype(bool)
    )
    return estimates, significant


def _draw_contextual_connectors(
    axis: Axes,
    estimates: pd.DataFrame,
    significant: pd.DataFrame,
    *,
    color: str,
) -> None:
    """Join within and between slopes whose difference is significant."""
    contextual = significant["sample_contextual_slope"].fillna(False)
    connected = estimates.loc[
        contextual.index[contextual.astype(bool)],
        [
            ("estimate", "within_sample_slope"),
            ("estimate", "between_sample_slope"),
        ],
    ]

    for row, slopes in zip(
        connected.index.to_numpy(dtype=float),
        connected.to_numpy(dtype=float),
        strict=True,
    ):
        axis.plot(
            slopes,
            [row - 0.17, row + 0.17],
            color=color,
            linewidth=0.5,
            zorder=1,
        )


def _label_unfitted_rows(axis: Axes, status: pd.Series, *, x: float) -> None:
    """Mark rows without a slope as "in module" (excluded) or "n.e."."""
    for row, value in zip(
        status.index.to_numpy(dtype=float),
        status.astype(str),
        strict=True,
    ):
        axis.text(
            x,
            row,
            "in module" if value == "excluded" else "n.e.",
            ha="center",
            va="center",
            color="#777777",
            fontstyle="italic",
        )


def _panel_limits(
    panel: pd.DataFrame, *, x_quantile: float
) -> tuple[float, float]:
    """Choose x limits that include zero from central CI endpoints.

    A quantile rather than the extremes keeps one very sparse sensor's wide
    interval from compressing every other sensor in the panel.
    """
    estimable = panel.loc[panel["estimable"]]
    lows = estimable["ci_low"].to_numpy(dtype=float)
    highs = estimable["ci_high"].to_numpy(dtype=float)
    lows, highs = lows[np.isfinite(lows)], highs[np.isfinite(highs)]
    if not lows.size or not highs.size:
        return -1.0, 1.0

    low = float(np.quantile(lows, 1.0 - x_quantile))
    high = float(np.quantile(highs, x_quantile))
    return min(low, 0.0), max(high, 0.0)


def _mark_clipped(
    axis: Axes,
    panel: pd.DataFrame,
    *,
    low: float,
    high: float,
) -> None:
    """Draw edge arrows where a CI or estimate extends beyond the axis."""
    offsets = {"within_sample_slope": -0.17, "between_sample_slope": 0.17}
    shown = panel.loc[
        panel["estimable"] & panel["estimand"].isin(list(offsets))
    ]
    y = shown["display_row"].to_numpy(dtype=float) + shown["estimand"].map(
        offsets
    ).to_numpy(dtype=float)

    for edge, marker, beyond in (
        (low, "<", shown["ci_low"].to_numpy(dtype=float) < low),
        (high, ">", shown["ci_high"].to_numpy(dtype=float) > high),
    ):
        if beyond.any():
            axis.scatter(
                np.full(int(beyond.sum()), edge),
                y[beyond],
                marker=marker,
                s=6,
                color="#555555",
                zorder=4,
            )


def _coupling_title(support: pd.Series) -> str:
    """State the estimand, unit and independent support of the figure."""
    donors = support.dropna()
    if donors.empty:
        span = "no estimable support"
    elif donors.min() == donors.max():
        span = f"{int(donors.min())} donors"
    else:
        span = f"{int(donors.min())}-{int(donors.max())} donors per pair"

    return (
        "Sensor-output coupling within and between donor-tissue samples "
        f"(mixed-model slopes, 95% CI; {span})"
    )


def _coupling_legend(
    *,
    within_color: str,
    between_color: str,
    connector_color: str,
    alpha: float,
) -> list[Line2D]:
    """Explain marker shape, fill and the contextual connector."""
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            markerfacecolor=within_color,
            markeredgecolor=within_color,
            label="Within sample (cell types)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markersize=2.6,
            markerfacecolor=between_color,
            markeredgecolor=between_color,
            label="Between samples",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            markerfacecolor="black",
            markeredgecolor="black",
            label=f"Filled: BH-FDR < {alpha:g}",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            markerfacecolor="white",
            markeredgecolor="black",
            label=f"Open: BH-FDR >= {alpha:g}",
        ),
        Line2D(
            [],
            [],
            color=connector_color,
            linewidth=0.6,
            label=f"Between differs from within (BH-FDR < {alpha:g})",
        ),
        Line2D(
            [],
            [],
            marker=">",
            linestyle="",
            markersize=2.5,
            color="#555555",
            label="CI beyond axis (see CSV)",
        ),
    ]


def _partition_legend(colors: dict[str, str]) -> list[Patch]:
    """Legend patches for the variance components and unallocated share."""
    return [
        Patch(color=colors["donor_fraction"], label="Donor"),
        Patch(color=colors["sample_fraction"], label="Donor-tissue sample"),
        Patch(color=colors["residual_fraction"], label="Residual"),
        Patch(
            facecolor="white",
            edgecolor="#777777",
            hatch="////",
            linewidth=0.3,
            label="Unallocated / not estimable",
        ),
    ]


def _partition_display(variance_partition: pd.DataFrame) -> pd.DataFrame:
    """Return one row per feature with component fractions and ordering."""
    components = {
        "group_random_intercept": "donor_fraction",
        "variance_component": "sample_fraction",
        "residual": "residual_fraction",
    }

    frame = variance_partition.loc[
        variance_partition["component_type"].isin(components)
    ].assign(component=lambda table: table["component_type"].map(components))
    if frame.empty:
        return pd.DataFrame()

    keys = ["feature_type", "feature_label"]
    estimable = frame.groupby(keys)["estimable"].all()
    status = frame.groupby(keys)["status"].agg(
        lambda values: ";".join(sorted(set(values.astype(str))))
    )

    wide = (
        frame.loc[frame["estimable"].astype(bool)]
        .pivot_table(
            index=keys,
            columns="component",
            values="variance_fraction",
            aggfunc="first",
        )
        .reindex(index=estimable.index, columns=list(components.values()))
    )

    wide["estimable"] = estimable
    wide["status"] = status
    wide["shared_fraction"] = wide[["donor_fraction", "sample_fraction"]].sum(
        axis=1, min_count=1
    )
    wide["unallocated_fraction"] = np.where(
        wide["estimable"],
        (1.0 - wide[list(components.values())].sum(axis=1)).clip(lower=0.0),
        np.nan,
    )

    return (
        wide.reset_index()
        .sort_values(
            ["feature_type", "estimable", "shared_fraction", "feature_label"],
            ascending=[True, False, False, True],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )


def _draw_partition_panel(
    axis: Axes,
    panel: pd.DataFrame,
    *,
    feature_type: str,
    colors: dict[str, str],
) -> None:
    """Draw stacked fractions, keeping missing and unallocated parts visible."""
    positions = np.arange(len(panel))
    left = np.zeros(len(panel))
    for column, color in colors.items():
        values = panel[column].fillna(0.0).to_numpy()
        axis.barh(
            positions, values, left=left, height=0.75, color=color, linewidth=0
        )
        left += values

    hatched = np.where(
        panel["estimable"], panel["unallocated_fraction"].fillna(0.0), 1.0
    )
    axis.barh(
        positions,
        hatched,
        left=np.where(panel["estimable"], left, 0.0),
        height=0.75,
        facecolor="white",
        edgecolor="#777777",
        hatch="////",
        linewidth=0.3,
    )

    unavailable = ~panel["estimable"].to_numpy(dtype=bool)
    for position, status in zip(
        positions[unavailable].astype(float),
        panel.loc[unavailable, "status"].astype(str),
        strict=True,
    ):
        axis.text(
            0.5,
            position,
            f"not estimable ({status})",
            ha="center",
            va="center",
            color="#555555",
            fontstyle="italic",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )

    labels = [
        humanize_module_name(label)
        if feature_type == "module_score"
        else str(label)
        for label in panel["feature_label"]
    ]
    axis.set_yticks(positions, labels)
    axis.set_ylim(len(panel) - 0.4, -0.6)
    axis.set_xlim(0, 1)
    axis.set_title(
        "Module scores" if feature_type == "module_score" else "Sensor genes",
        loc="left",
        pad=3,
    )
    axis.tick_params(axis="y", length=0, pad=2)
    axis.tick_params(axis="x", length=2, pad=1.5)
    axis.spines[["top", "right", "left"]].set_visible(False)


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    table: str,
) -> None:
    """Fail when a plotted table lacks a required column."""
    if missing := [column for column in columns if column not in frame]:
        raise KeyError(f"{table} table lacks columns: {missing}")


def _validate_positive(**values: float) -> None:
    """Reject nonpositive or non-finite geometry."""
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")


def _validate_fontsizes(*sizes: float | None) -> None:
    """Reject nonpositive or non-finite font sizes."""
    for size in sizes:
        if size is not None and (not np.isfinite(size) or size <= 0):
            raise ValueError("Font sizes must be positive and finite")


def _validate_figsize(figsize: tuple[float, float] | None) -> None:
    """Reject a figure size that is not two positive finite inches."""
    if figsize is not None and (
        len(figsize) != 2
        or any(not np.isfinite(value) or value <= 0 for value in figsize)
    ):
        raise ValueError("figsize must contain two positive finite inches")
