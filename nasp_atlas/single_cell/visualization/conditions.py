"""Figures for disease-versus-reference (condition cohort) analyses."""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text  # type: ignore[import]
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from nasp_compendium.display import humanize_module_name  # type: ignore[import]

from nasp_atlas.single_cell.utils import dedupe_stem
from nasp_atlas.single_cell.utils import snake_case
from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


class ConditionPlotter(_PlotterBase):
    """Export design, descriptive and inferential condition-cohort figures.

    Every figure is saved as PNG, vector PDF and the displayed rows as CSV.
    Unsupported or non-estimable cells are drawn as missing, never as zero.

    Example Usage:
      >>> plotter = ConditionPlotter("path/to/output")
      >>> paths = plotter.plot_arm_support(arm_support, minimum_donors=3)
      >>> paths += plotter.plot_condition_effects(
      ...     contrasts, estimand="ckd_vs_reference",
      ...     feature_type="gene_expression",
      ... )
    """

    def plot_arm_support(
        self,
        arm_support: pd.DataFrame,
        *,
        minimum_donors: int = 3,
        filename: str = "arm_support",
        figsize: tuple[float, float] | None = None,
        row_height: float = 0.12,
        title: str | None = None,
    ) -> list[Path]:
        """Plot supported donors per cell type and arm.

        Bars count donors with at least one supported unit; the dashed line
        marks `minimum_donors`, below which a contrast is not estimable.

        Args:
          arm_support: `design/arm_support.csv` rows.
          minimum_donors: Donors needed per arm, drawn as a reference line.
          filename: Output stem for PNG, PDF and displayed CSV.
          figsize: Overall inches; overrides `row_height`.
          row_height: Inches per cell type when `figsize` is None.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files.

        Example Usage:
          >>> paths = plotter.plot_arm_support(
          ...     arm_support, minimum_donors=3, row_height=0.14,
          ... )
        """
        _validate_geometry(figsize=figsize, row_height=row_height)

        donors = arm_support.pivot(
            index="cell_type", columns="condition", values="n_donors"
        ).fillna(0)
        donors = donors.loc[donors.max(axis=1).sort_values().index]
        arms = list(donors.columns)
        dimensions = figsize or (3.5, 0.8 + row_height * len(donors))

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axis = plt.subplots(
                figsize=dimensions, layout="constrained"
            )
            try:
                height = 0.8 / len(arms)
                positions = np.arange(len(donors))
                for index, arm in enumerate(arms):
                    axis.barh(
                        positions + (index - (len(arms) - 1) / 2) * height,
                        donors[arm],
                        height=height,
                        label=arm,
                        color=plt.get_cmap("tab10")(index),
                    )

                axis.axvline(
                    minimum_donors, color="#555555", linestyle="--", lw=0.5
                )
                axis.set_yticks(positions, donors.index.tolist())
                axis.set_xlabel("Donors with a supported unit")
                axis.set_title(
                    "Donor support per cell type and arm"
                    if title is None
                    else title,
                    loc="left",
                )
                axis.legend(frameon=False, loc="lower right")
                axis.spines[["top", "right"]].set_visible(False)
                axis.tick_params(axis="y", length=0)

                return self._save_with_data(
                    figure, donors.reset_index(), filename
                )
            finally:
                plt.close(figure)

    def plot_condition_dotplot(
        self,
        features: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        arms: Sequence[str],
        minimum_donors: int = 3,
        filename: str = "sensor_condition_dotplot",
        figsize: tuple[float, float] | None = None,
        column_width: float = 0.11,
        row_height: float = 0.1,
        max_dot_size: float = 18.0,
        color_cap: float | None = None,
        title: str | None = None,
    ) -> list[Path]:
        """Plot sensor level and detection by cell type, arms side by side.

        Colour is the donor median of unit mean log1p CP10k; dot area is the
        donor median of the unit detected fraction. Medians use supported
        units, so each donor counts once. Groups with fewer than
        `minimum_donors` donors are blank. Descriptive only.

        Args:
          features: `units/unit_features.csv.gz` rows.
          metadata: `units/unit_metadata.csv` rows.
          arms: Arms to show, adjacent within each cell type.
          minimum_donors: Donors needed to show a group.
          filename: Output stem for PNG, PDF and displayed CSV.
          figsize: Overall inches; overrides the cell geometry.
          column_width: Inches per cell-type-by-arm column.
          row_height: Inches per sensor row.
          max_dot_size: Area in points squared at detected fraction 1.
          color_cap: Upper colour limit in log1p units; None uses the 95th
            percentile of displayed groups.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files, or an empty list without sensors.

        Example Usage:
          >>> paths = plotter.plot_condition_dotplot(
          ...     features, metadata, arms=["reference", "ckd"],
          ...     column_width=0.12, max_dot_size=20.0,
          ... )
        """
        _validate_geometry(
            figsize=figsize, row_height=row_height, column_width=column_width
        )
        summary = _donor_median_sensors(
            features, metadata, arms=arms, minimum_donors=minimum_donors
        )
        if summary.empty:
            logger.info("No supported sensor groups to plot")
            return []

        genes, columns, shown = _dotplot_layout(summary, arms)
        cap = color_cap or float(np.nanquantile(shown["mean_expression"], 0.95))
        dimensions = figsize or (
            1.2 + column_width * len(columns),
            1.1 + row_height * len(genes),
        )

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axis = plt.subplots(
                figsize=dimensions, layout="constrained"
            )
            try:
                points = axis.scatter(
                    shown["x"],
                    shown["y"],
                    s=shown["fraction_detected"].clip(0, 1) * max_dot_size,
                    c=shown["mean_expression"].clip(upper=cap),
                    cmap="Reds",
                    vmin=0,
                    vmax=cap,
                    edgecolors="#555555",
                    linewidths=0.2,
                )

                arms_shown = columns["condition"].tolist()
                axis.set_xticks(
                    range(len(columns)),
                    [
                        f"{cell} · {arm}"
                        for cell, arm in zip(
                            columns["cell_type"], arms_shown, strict=True
                        )
                    ],
                    rotation=90,
                )
                axis.set_yticks(range(len(genes)), genes)
                for label in axis.get_yticklabels():
                    label.set_fontstyle("italic")
                boundaries = np.flatnonzero(
                    columns["cell_type"].to_numpy()[1:]
                    != columns["cell_type"].to_numpy()[:-1]
                )
                for boundary in boundaries:
                    axis.axvline(boundary + 0.5, color="#dddddd", lw=0.4)

                axis.set_xlim(-0.6, len(columns) - 0.4)
                axis.set_ylim(-0.6, len(genes) - 0.4)
                axis.tick_params(length=0, pad=1.5)
                axis.spines[:].set_visible(False)
                colorbar = figure.colorbar(
                    points, ax=axis, shrink=0.3, aspect=10, pad=0.01
                )
                colorbar.set_label("Donor-median mean log1p CP10k")

                axis.set_title(
                    "Sensor expression by cell type and arm "
                    f"(descriptive; groups with >= {minimum_donors} donors)"
                    if title is None
                    else title,
                    loc="left",
                )

                return self._save_with_data(figure, shown, filename)
            finally:
                plt.close(figure)

    def plot_donor_contrasts(
        self,
        donors: pd.DataFrame,
        sensor_mapping: pd.DataFrame,
        *,
        comparisons: Sequence[tuple[str, str, str | None]],
        scorer: str,
        minimum_donors: int = 3,
        value_name: str = "donor level",
        filename_prefix: str = "donor_contrast",
        column_width: float = 0.11,
        row_height: float = 0.1,
        max_dot_size: float = 30.0,
        color_quantile: float = 0.98,
        color_limits: tuple[float, float] | None = None,
        colorbar_length: float = 0.9,
    ) -> list[Path]:
        """Plot every sensor and module by cell type, donors as replicates.

        Draws one figure per comparison. Rows are every sensor in
        `sensor_mapping`, grouped as DNA, DNA + RNA and RNA sensors, then
        every module; columns are cell types. Colour is the mean donor value
        of `target` minus that of `reference`, in log1p CP10k for sensors and
        `scorer` score units for modules. Dot area is how consistently the
        donors follow that difference: Cliff's delta, the share of
        target-reference donor pairs ordered as the means are minus the share
        ordered against them (ties count as neither). A comparison whose
        reference is None reads `donors` values as within-donor changes and
        compares them with zero: colour is the mean change and dot area the
        share of donors changing in its direction minus the share changing
        against it. Separation against the mean's direction draws only a
        pinpoint (the CSV keeps the signed value), so a mean carried by a few
        donors shows colour without area; every measured cell keeps at least
        that pinpoint, so an empty cell means no value. Descriptive only: no
        covariates, no test.

        Cell types need `minimum_donors` donors in each compared group;
        others are named in the title and kept in the CSV. Sensors without a
        usable source gene are grey and labelled, and a cell whose donors
        are all zero gets a cross, never a no-difference dot. Each panel's
        colour limit is the `color_quantile` of absolute differences across
        every comparison of the call, so comparisons and cell types are
        directly comparable; values beyond it are clipped in colour only.

        Args:
          donors: One row per `cell_type`, `feature_type` ("gene_expression"
            or "module_score"), `feature_label`, `group` and `donor_id`,
            with the donor's `value`.
          sensor_mapping: `units/sensor_mapping.csv` rows.
          comparisons: (name, target, reference) per figure; a None
            reference compares within-donor changes in `target` with zero.
          scorer: Scorer of the module rows, for labels.
          minimum_donors: Donors needed per compared group in a cell type.
          value_name: What one donor value is, for the colour bar.
          filename_prefix: Output stems are `<prefix>_<name>`.
          column_width: Inches per cell type.
          row_height: Inches per feature row.
          max_dot_size: Area in points squared at separation 1.
          color_quantile: Quantile of absolute differences used as each
            panel's colour limit (0.5 to 1).
          color_limits: Fixed (sensor, module) colour limits; overrides
            `color_quantile`.
          colorbar_length: Inches of each colour bar, at most 90% of its
            panel's height.

        Returns:
          Paths of the saved files, or an empty list without a drawable
          comparison.

        Example Usage:
          >>> paths = plotter.plot_donor_contrasts(
          ...     donors, sensor_mapping,
          ...     comparisons=[("sle_vs_control", "sle", "control")],
          ...     scorer="scanpy", column_width=0.12, max_dot_size=18.0,
          ... )
        """
        _validate_geometry(
            column_width=column_width,
            row_height=row_height,
            max_dot_size=max_dot_size,
            colorbar_length=colorbar_length,
        )
        if not 0.5 <= color_quantile <= 1:
            raise ValueError("color_quantile must be in [0.5, 1]")
        if color_limits is not None:
            _validate_geometry(
                sensor_limit=color_limits[0], module_limit=color_limits[1]
            )

        rows = _feature_rows(
            sensor_mapping,
            modules=donors.loc[
                donors["feature_type"].eq("module_score"), "feature_label"
            ].unique(),
            scorer=scorer,
        )
        unknown = (
            donors[["feature_type", "feature_label"]]
            .drop_duplicates()
            .merge(rows, how="left", indicator=True)
        )
        unknown = unknown.loc[unknown["_merge"].eq("left_only")]
        if not unknown.empty:
            raise ValueError(
                "Donor features absent from the sensor mapping or of another "
                f"type: {unknown['feature_label'].tolist()[:5]}"
            )

        summaries = [
            _donor_contrast(
                donors,
                target=target,
                reference=reference,
                minimum_donors=minimum_donors,
            )
            for _, target, reference in comparisons
        ]
        measured = pd.concat(summaries, ignore_index=True)
        measured = measured.loc[measured["status"].eq("measured")]
        if measured.empty:
            logger.info(
                "No cell type has %d donors per compared group", minimum_donors
            )
            return []

        limits = color_limits or tuple(
            _color_limit(
                measured.loc[
                    measured["feature_type"].eq(feature_type), "difference"
                ].to_numpy(dtype=float),
                color_quantile,
            )
            for feature_type in ("gene_expression", "module_score")
        )

        paths: list[Path] = []
        stems: set[str] = set()
        for (name, target, reference), summary in zip(
            comparisons, summaries, strict=True
        ):
            if not summary["status"].ne("insufficient_donors").any():
                logger.info(
                    "No %s figure: no cell type has %d donors per group",
                    name,
                    minimum_donors,
                )
                continue

            paths += self._plot_donor_contrast(
                _contrast_table(
                    summary,
                    rows,
                    comparison=(name, target, reference),
                    limits=(float(limits[0]), float(limits[1])),
                ),
                rows,
                comparison=(name, target, reference),
                scorer=scorer,
                limits=(float(limits[0]), float(limits[1])),
                value_name=value_name,
                minimum_donors=minimum_donors,
                column_width=column_width,
                row_height=row_height,
                max_dot_size=max_dot_size,
                colorbar_length=colorbar_length,
                filename=dedupe_stem(
                    f"{filename_prefix}_{snake_case(name)}", stems
                ),
            )

        return paths

    def _plot_donor_contrast(
        self,
        displayed: pd.DataFrame,
        rows: pd.DataFrame,
        *,
        comparison: tuple[str, str, str | None],
        scorer: str,
        limits: tuple[float, float],
        value_name: str,
        minimum_donors: int,
        column_width: float,
        row_height: float,
        max_dot_size: float,
        colorbar_length: float,
        filename: str,
    ) -> list[Path]:
        """Draw one comparison's sensor and module panels on shared columns."""
        name, target, reference = comparison
        paired = reference is None
        drawn = displayed.loc[
            displayed["status"].isin(["measured", "not_detected"])
        ]
        support = drawn.groupby("cell_type")[["n_target", "n_reference"]].max()
        cell_types = sorted(map(str, support.index))
        skipped = sorted(
            set(displayed["cell_type"].dropna().astype(str)).difference(
                cell_types
            )
        )

        panels = [
            _contrast_panel(
                drawn, rows, cell_types=cell_types, feature_type=feature_type
            )
            for feature_type in ("gene_expression", "module_score")
        ]
        heights = [len(labels) for _, labels in panels]
        versus = f" in {target}" if paired else f", {target} - {reference}"

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axes = plt.subplots(
                2,
                1,
                figsize=(
                    2.6 + column_width * len(cell_types),
                    1.7 + row_height * sum(heights),
                ),
                height_ratios=heights,
                sharex=True,
                layout="constrained",
            )
            try:
                for axis, (points, labels), limit, unit in zip(
                    axes,
                    panels,
                    limits,
                    ("log1p CP10k", f"{scorer} score"),
                    strict=True,
                ):
                    _draw_contrast_panel(
                        figure,
                        axis,
                        points,
                        labels,
                        n_columns=len(cell_types),
                        limit=limit,
                        max_dot_size=max_dot_size,
                        color_label=f"Mean {value_name}{versus}\n({unit})",
                        colorbar_shrink=min(
                            0.9, colorbar_length / (row_height * len(labels))
                        ),
                    )

                counts = support.loc[cell_types].fillna(0).astype(int)
                axes[1].set_xticks(
                    range(len(cell_types)),
                    [
                        f"{cell} ({n_target})"
                        if paired
                        else f"{cell} ({n_target}/{n_reference})"
                        for cell, n_target, n_reference in zip(
                            cell_types,
                            counts["n_target"],
                            counts["n_reference"],
                            strict=True,
                        )
                    ],
                    rotation=90,
                )
                axes[1].set_xlabel(
                    f"Cell type ({target} donors)"
                    if paired
                    else f"Cell type ({target}/{reference} donors)"
                )
                _separation_legend(
                    figure, paired=paired, max_dot_size=max_dot_size
                )

                what = (
                    f"{value_name} in {target}"
                    if paired
                    else f"{target} vs {reference} donors"
                )
                lines = [
                    f"{name}: {what}",
                    "Every sensor and module by cell type; donors are the "
                    "replicates. Descriptive: no covariates, no test.",
                ]
                if skipped:
                    lines += textwrap.wrap(
                        f"Not drawn (< {minimum_donors} donors in a compared "
                        "group): " + ", ".join(skipped),
                        width=90,
                    )
                heading = figure.suptitle(
                    "\n".join(lines),
                    x=0.02,
                    ha="left",
                    fontsize=plt.rcParams["axes.titlesize"],
                )
                needed = heading.get_window_extent().width / figure.dpi + 0.15
                if needed > figure.get_figwidth():
                    figure.set_figwidth(needed)

                return self._save_with_data(figure, displayed, filename)
            finally:
                plt.close(figure)

    def plot_condition_effects(
        self,
        contrasts: pd.DataFrame,
        *,
        estimand: str,
        feature_type: str,
        scorer: str | None = None,
        alpha: float = 0.05,
        filename: str | None = None,
        figsize: tuple[float, float] | None = None,
        cell_size: float = 0.13,
        color_quantile: float = 0.95,
        title: str | None = None,
    ) -> list[Path]:
        """Plot one estimand's effects as features by cell types.

        Colour is the estimate (an arm difference, a slope or a state
        difference, as named by `effect_scale`) on a diverging scale centred
        on zero, limited by the `color_quantile` of absolute estimates so
        one extreme cell cannot wash out the rest (clipped cells keep their
        value in the CSV). A black dot marks BH-FDR below `alpha` within the
        estimand and feature family. Non-estimable cells are grey with a
        cross, never zero.

        Args:
          contrasts: Rows of a condition-cohort effect table, such as
            `condition_effects/condition_contrasts.csv`, with `estimand`,
            `feature_type`, `feature_label`, `cell_type`, `estimate`,
            `pvalue_fdr`, `estimable`, `status` and `effect_scale`.
          estimand: Estimand to plot.
          feature_type: "module_score", "gene_expression" or
            "gene_detection".
          scorer: Scorer for module scores; None for sensor features.
          alpha: Adjusted p-value threshold for the significance dot.
          filename: Output stem; defaults to the estimand and family.
          figsize: Overall inches; overrides `cell_size`. Cells stay
            square, and the figure widens when its title needs more room.
            Without it, the figure also widens to fit long row labels.
          cell_size: Inches per heatmap cell.
          color_quantile: Quantile of absolute estimates used as the colour
            limit (0.5 to 1).
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files, or an empty list without rows.

        Example Usage:
          >>> paths = plotter.plot_condition_effects(
          ...     contrasts, estimand="sle_vs_control",
          ...     feature_type="gene_detection", cell_size=0.15,
          ... )
        """
        _validate_geometry(figsize=figsize, cell_size=cell_size)
        if not 0.5 <= color_quantile <= 1 or not 0 < alpha < 1:
            raise ValueError("color_quantile in [0.5, 1] and alpha in (0, 1)")

        selected = contrasts.loc[
            contrasts["estimand"].eq(estimand)
            & contrasts["feature_type"].eq(feature_type)
            & contrasts["cell_type"].notna()
        ]
        if scorer is not None:
            selected = selected.loc[selected["scorer"].eq(scorer)]
        if selected.empty:
            logger.info("No %s rows for %s", feature_type, estimand)
            return []

        estimate, fdr = _effect_matrix(selected)
        limit = _color_limit(estimate.to_numpy(dtype=float), color_quantile)
        dimensions = figsize or (
            1.4 + cell_size * estimate.shape[1],
            1.0 + cell_size * estimate.shape[0],
        )

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axis = plt.subplots(
                figsize=dimensions, layout="constrained"
            )
            try:
                _draw_effect_matrix(
                    figure,
                    axis,
                    estimate,
                    fdr,
                    alpha=alpha,
                    limit=limit,
                    feature_type=feature_type,
                    color_label=_effect_label(
                        selected["effect_scale"].dropna()
                    ),
                )
                if figsize is None:
                    # Long row labels (module names) must not be clipped
                    # when a matrix has few cell types.
                    _fit_row_labels(
                        figure,
                        axis,
                        matrix_width=cell_size * estimate.shape[1],
                    )
                _significance_legend(figure, alpha=alpha)

                family = feature_type + (f" ({scorer})" if scorer else "")
                heading = figure.suptitle(
                    f"{estimand}:\n{family} within cell type\n"
                    "(donor means; OLS with HC3 errors)"
                    if title is None
                    else title,
                    x=0.02,
                    ha="left",
                    fontsize=plt.rcParams["axes.titlesize"],
                )
                needed = heading.get_window_extent().width / figure.dpi + 0.15
                if needed > figure.get_figwidth():
                    figure.set_figwidth(needed)

                shown = selected.assign(color_limit=limit)
                stem = filename or "_".join(
                    part
                    for part in (estimand, feature_type, scorer or "")
                    if part
                )
                return self._save_with_data(figure, shown, stem)
            finally:
                plt.close(figure)

    def plot_atlas_calibration(
        self,
        profiles: pd.DataFrame,
        agreement: pd.DataFrame,
        *,
        atlas_id: str,
        feature_type: str = "gene_expression",
        n_labels: int = 4,
        filename: str | None = None,
        panel_size: float = 1.7,
        ncols: int = 4,
        marker_size: float = 7.0,
        title: str | None = None,
    ) -> list[Path]:
        """Plot reference-arm against atlas sensor profiles per cell type.

        Each panel is one matched cell type; each point is a sensor's donor
        median in the cohort reference arm (y) and in the atlas (x), on the
        same log1p CP10k or detected-fraction scale. The dashed line is
        equality and the panel title gives Spearman's rho across sensors.
        The `n_labels` sensors furthest from equality are named. Cohort and
        atlas differ in donors, chemistry and processing: descriptive.

        Args:
          profiles: `atlas_calibration/calibration_profiles.csv` rows.
          agreement: `atlas_calibration/calibration_agreement.csv` rows.
          atlas_id: Atlas name for axis labels.
          feature_type: "gene_expression" or "gene_detection".
          n_labels: Sensors labelled per panel.
          filename: Output stem; defaults to "atlas_calibration_<measure>".
          panel_size: Inches per square panel.
          ncols: Panels per row.
          marker_size: Marker area in points squared.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files, or an empty list without profiles.

        Example Usage:
          >>> paths = plotter.plot_atlas_calibration(
          ...     profiles, agreement, atlas_id="tabula_sapiens",
          ...     panel_size=1.8, ncols=4,
          ... )
        """
        _validate_geometry(panel_size=panel_size, marker_size=marker_size)
        shown = profiles.loc[profiles["feature_type"].eq(feature_type)]
        if shown.empty:
            return []

        cell_types = sorted(shown["cell_type"].unique())
        ncols = min(ncols, len(cell_types))
        nrows = int(np.ceil(len(cell_types) / ncols))
        rho = agreement.loc[
            agreement["feature_type"].eq(feature_type)
        ].set_index("cell_type")

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(panel_size * ncols, panel_size * nrows + 0.5),
                layout="constrained",
                squeeze=False,
            )
            try:
                for axis, cell_type in zip(
                    axes.ravel(), cell_types, strict=False
                ):
                    _draw_calibration_panel(
                        axis,
                        shown.loc[shown["cell_type"].eq(cell_type)],
                        rho.loc[cell_type] if cell_type in rho.index else None,
                        cell_type=cell_type,
                        marker_size=marker_size,
                        n_labels=n_labels,
                    )
                for axis in axes.ravel()[len(cell_types) :]:
                    axis.set_visible(False)

                scale = (
                    "donor-median log1p CP10k"
                    if feature_type == "gene_expression"
                    else "donor-median detected fraction"
                )
                figure.supxlabel(f"{atlas_id}: {scale}")
                figure.supylabel(f"Cohort reference arm: {scale}")
                figure.suptitle(
                    f"Reference arm versus {atlas_id} sensor profiles "
                    "(descriptive)"
                    if title is None
                    else title,
                    x=0.02,
                    ha="left",
                )

                stem = filename or f"atlas_calibration_{feature_type}"
                return self._save_with_data(figure, shown, stem)
            finally:
                plt.close(figure)

    def plot_covariate_balance(
        self,
        balance: pd.DataFrame,
        *,
        filename: str = "covariate_balance",
        width: float = 4.6,
        panel_height: float = 0.32,
        title: str | None = None,
    ) -> list[Path]:
        """Plot donor-level covariate balance between arms, one row each.

        Numeric covariates show each arm's donor mean with a one-SD bar and
        the standardized mean difference from the reference arm. Categorical
        covariates show each arm's donor shares per level as stacked bars;
        levels found in only one arm are hatched, because adjusting for them
        removes those donors' information about the arm. Descriptive.

        Args:
          balance: `design/covariate_balance.csv` rows.
          filename: Output stem for PNG, PDF and displayed CSV.
          width: Figure width in inches.
          panel_height: Inches per arm within each covariate panel.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files, or an empty list without covariates.

        Example Usage:
          >>> paths = plotter.plot_covariate_balance(
          ...     balance, width=5.0, panel_height=0.3,
          ... )
        """
        _validate_geometry(width=width, panel_height=panel_height)
        if balance.empty:
            return []
        covariates = list(dict.fromkeys(balance["covariate"]))
        arms = list(dict.fromkeys(balance["condition"].astype(str)))
        heights = [panel_height * len(arms) + 0.35 for _ in covariates]

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axes = plt.subplots(
                len(covariates),
                1,
                figsize=(width, sum(heights) + 0.5),
                height_ratios=heights,
                layout="constrained",
                squeeze=False,
            )
            try:
                for axis, covariate in zip(axes[:, 0], covariates, strict=True):
                    rows = balance.loc[balance["covariate"].eq(covariate)]
                    if rows["kind"].iloc[0] == "categorical":
                        _draw_categorical_balance(axis, rows, arms)
                    else:
                        _draw_numeric_balance(axis, rows, arms)
                    axis.set_title(str(covariate), loc="left", pad=2)

                figure.suptitle(
                    "Covariate balance between arms (donor level; descriptive)"
                    if title is None
                    else title,
                    x=0.02,
                    ha="left",
                )

                return self._save_with_data(figure, balance, filename)
            finally:
                plt.close(figure)

    def plot_split_umap(
        self,
        embedding: pd.DataFrame,
        *,
        arms: Sequence[str],
        value_label: str,
        filename: str,
        max_cells_per_arm: int = 40_000,
        random_state: int = 42,
        color_quantiles: tuple[float, float] = (0.01, 0.99),
        panel_size: float = 2.3,
        point_size: float = 0.4,
        cmap: str = "viridis",
        title: str | None = None,
    ) -> list[Path]:
        """Plot one per-cell value on the UMAP, one panel per arm.

        Every panel shares one colour scale, set by `color_quantiles` of the
        displayed values, so arms are directly comparable. Each arm shows at
        most `max_cells_per_arm` cells drawn with `random_state`, so a larger
        arm does not look denser; other arms' displayed cells are light grey
        underneath. Descriptive: cells are not independent replicates.

        Args:
          embedding: One row per cell with `x`, `y`, `condition` and `value`.
          arms: Arms in panel order.
          value_label: Colour-bar label, such as "IFN-I output (Scanpy)".
          filename: Output stem for PNG, PDF and displayed CSV.
          max_cells_per_arm: Cells drawn per arm (downsampled when larger).
          random_state: Seed for downsampling and drawing order.
          color_quantiles: Lower and upper quantiles of the shared scale.
          panel_size: Inches per square panel.
          point_size: Marker area in points squared.
          cmap: Matplotlib colour map name.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files.

        Example Usage:
          >>> paths = plotter.plot_split_umap(
          ...     embedding, arms=["control", "sle"],
          ...     value_label="IFN-I output (Scanpy)",
          ...     filename="umap_ifn_i_output_by_arm", panel_size=2.5,
          ... )
        """
        _validate_geometry(panel_size=panel_size, point_size=point_size)
        low_q, high_q = color_quantiles
        if not 0 <= low_q < high_q <= 1 or max_cells_per_arm < 1:
            raise ValueError("Need 0 <= low < high <= 1 and positive cells")

        shown = _equal_arm_sample(
            embedding,
            arms,
            max_cells=max_cells_per_arm,
            random_state=random_state,
        )
        low, high = np.nanquantile(shown["value"], [low_q, high_q])
        norm = Normalize(vmin=float(low), vmax=float(high))

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axes = plt.subplots(
                1,
                len(arms),
                figsize=(panel_size * len(arms) + 0.7, panel_size + 0.45),
                layout="constrained",
                squeeze=False,
            )
            try:
                for axis, arm in zip(axes[0], arms, strict=True):
                    own = shown["condition"].astype(str).eq(arm)
                    axis.scatter(
                        shown.loc[~own, "x"],
                        shown.loc[~own, "y"],
                        s=point_size,
                        c="#e3e3e3",
                        linewidths=0,
                        rasterized=True,
                    )
                    axis.scatter(
                        shown.loc[own, "x"],
                        shown.loc[own, "y"],
                        s=point_size,
                        c=shown.loc[own, "value"].clip(low, high),
                        cmap=cmap,
                        norm=norm,
                        linewidths=0,
                        rasterized=True,
                    )

                    axis.set_title(f"{arm} (n = {int(own.sum()):,} cells)")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    axis.set_aspect("equal", adjustable="datalim")
                    axis.spines[:].set_visible(False)

                colorbar = figure.colorbar(
                    ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes[0, -1],
                    shrink=0.6,
                    aspect=12,
                    pad=0.02,
                )
                colorbar.set_label(value_label)

                figure.suptitle(
                    f"{value_label} by arm (shared scale; descriptive)"
                    if title is None
                    else title,
                    x=0.02,
                    ha="left",
                )

                return self._save_with_data(
                    figure,
                    shown.assign(color_low=low, color_high=high),
                    filename,
                )
            finally:
                plt.close(figure)

    def plot_ifn_adjustment(
        self,
        unadjusted: pd.DataFrame,
        adjusted: pd.DataFrame,
        *,
        estimand: str,
        feature_type: str = "gene_expression",
        alpha: float = 0.05,
        n_labels: int = 12,
        filename: str | None = None,
        figsize: tuple[float, float] = (4.2, 3.8),
        marker_size: float = 9.0,
        title: str | None = None,
    ) -> list[Path]:
        """Compare each sensor's arm effect before and after IFN adjustment.

        One point per sensor and cell type: x is the AB1 estimate, y the
        estimate at a fixed IFN-I output level (AB3), each with its 95%
        interval. Shifts with BH-FDR below `alpha` are coloured "persists"
        when the adjusted interval excludes zero with the same sign, and
        "attenuated" otherwise; the rest are grey. Points on the diagonal
        are unaffected by adjustment; points pulled to zero track IFN.

        Args:
          unadjusted: `condition_effects/condition_contrasts.csv` rows.
          adjusted: `ifn_adjusted_effects/ifn_adjusted_contrasts.csv` rows.
          estimand: Planned contrast to plot.
          feature_type: "gene_expression" or "gene_detection".
          alpha: Adjusted p-value threshold for a modulated shift.
          n_labels: Persisting points labelled, largest adjusted first.
          filename: Output stem; defaults to "<estimand>_ifn_adjustment_..".
          figsize: Figure inches.
          marker_size: Marker area in points squared.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files, or an empty list without paired rows.

        Example Usage:
          >>> paths = plotter.plot_ifn_adjustment(
          ...     unadjusted, adjusted, estimand="sle_vs_control",
          ...     n_labels=10, figsize=(4.5, 4.0),
          ... )
        """
        _validate_geometry(figsize=figsize, marker_size=marker_size)
        paired = _ifn_adjustment_pairs(
            unadjusted,
            adjusted,
            estimand=estimand,
            feature_type=feature_type,
            alpha=alpha,
        )
        if paired.empty:
            return []

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axis = plt.subplots(figsize=figsize, layout="constrained")
            try:
                _draw_ifn_adjustment_points(
                    axis, paired, marker_size=marker_size
                )
                _draw_identity_axes(axis, paired)
                _label_persisting(axis, paired, n_labels=n_labels)

                scale = _effect_label(
                    unadjusted.loc[
                        unadjusted["feature_type"].eq(feature_type),
                        "effect_scale",
                    ].dropna()
                )
                axis.set_xlabel(f"Unadjusted: {scale}")
                axis.set_ylabel("At fixed IFN-I output")
                axis.legend(frameon=False, loc="upper left", fontsize=5)
                axis.spines[["top", "right"]].set_visible(False)
                axis.set_title(
                    f"{estimand}: {feature_type} shifts before and after "
                    "IFN-I adjustment\n(one point per sensor and cell type; "
                    "95% CIs)"
                    if title is None
                    else title,
                    loc="left",
                )

                stem = filename or f"{estimand}_ifn_adjustment_{feature_type}"
                return self._save_with_data(figure, paired, stem)
            finally:
                plt.close(figure)

    def plot_ifn_strata(
        self,
        donors: pd.DataFrame,
        *,
        filename: str = "ifn_strata_donors",
        figsize: tuple[float, float] = (3.2, 2.8),
        marker_size: float = 7.0,
        random_state: int = 42,
        title: str | None = None,
    ) -> list[Path]:
        """Plot donor IFN-I output by stratum with the reference threshold.

        One point per donor: the mean IFN_I_OUTPUT of its units in the
        stratifying cell type. The dashed line is the threshold fixed from
        reference donors alone. Donors without that cell type are
        "unassigned" and omitted, with their count in the legend title.

        Args:
          donors: `ifn_strata/ifn_strata_donors.csv` rows.
          filename: Output stem for PNG, PDF and displayed CSV.
          figsize: Figure inches.
          marker_size: Marker area in points squared.
          random_state: Seed for horizontal jitter.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files.

        Example Usage:
          >>> paths = plotter.plot_ifn_strata(donors, figsize=(3.5, 3.0))
        """
        _validate_geometry(figsize=figsize, marker_size=marker_size)
        assigned = donors.loc[donors["ifn_output"].notna()]
        # Reference first, then disease strata from low to high.
        strata = sorted(
            dict.fromkeys(assigned["stratum"].astype(str)),
            key=lambda name: (
                name.endswith(("_ifn_low", "_ifn_high")),
                name.endswith("_ifn_high"),
                name,
            ),
        )
        rng = np.random.default_rng(random_state)
        unassigned = int(donors["ifn_output"].isna().sum())

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axis = plt.subplots(figsize=figsize, layout="constrained")
            try:
                for index, stratum in enumerate(strata):
                    values = assigned.loc[
                        assigned["stratum"].astype(str).eq(stratum),
                        "ifn_output",
                    ]
                    axis.scatter(
                        index + rng.uniform(-0.18, 0.18, len(values)),
                        values,
                        s=marker_size,
                        color=plt.get_cmap("tab10")(index),
                        edgecolors="white",
                        linewidths=0.2,
                    )
                    axis.text(
                        index,
                        1.0,
                        f"n = {len(values)}",
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=5,
                    )

                axis.axhline(
                    float(donors["threshold"].iloc[0]),
                    ls="--",
                    lw=0.6,
                    c="#555555",
                    label="Reference-donor threshold",
                )
                axis.set_xticks(
                    range(len(strata)), strata, rotation=30, ha="right"
                )
                axis.set_ylabel("Donor IFN-I output (unit mean score)")
                figure.legend(
                    frameon=False,
                    loc="outside lower center",
                    fontsize=5,
                    title=f"Unassigned donors: {unassigned}",
                    title_fontsize=5,
                )
                axis.spines[["top", "right"]].set_visible(False)
                axis.set_title(
                    "IFN-I strata of disease donors (AB2)"
                    if title is None
                    else title,
                    loc="left",
                    pad=10,
                )

                return self._save_with_data(figure, donors, filename)
            finally:
                plt.close(figure)


def _draw_numeric_balance(
    axis: Axes,
    rows: pd.DataFrame,
    arms: Sequence[str],
) -> None:
    """Draw donor mean and one SD per arm, with the SMD as text."""
    for position, arm in enumerate(arms):
        row = rows.loc[rows["condition"].astype(str).eq(arm)]
        if row.empty or not np.isfinite(row["mean"].iloc[0]):
            continue

        mean, spread = float(row["mean"].iloc[0]), float(row["std"].iloc[0])
        axis.errorbar(
            mean,
            position,
            xerr=spread if np.isfinite(spread) else None,
            fmt="o",
            ms=3,
            color=plt.get_cmap("tab10")(position),
            elinewidth=0.6,
        )

        smd = row["smd_vs_reference"].iloc[0]
        label = f"n={int(row['n_donors'].iloc[0])}" + (
            f", SMD {smd:+.2f}" if pd.notna(smd) else ""
        )
        axis.annotate(
            label,
            (1.0, position),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=5,
        )

    axis.set_yticks(range(len(arms)), arms)
    axis.set_ylim(-0.6, len(arms) - 0.4)
    axis.invert_yaxis()
    axis.spines[["top", "right"]].set_visible(False)


def _draw_categorical_balance(
    axis: Axes,
    rows: pd.DataFrame,
    arms: Sequence[str],
) -> None:
    """Draw each arm's donor shares per level as stacked bars."""
    shares = rows.pivot_table(
        index="condition", columns="level", values="mean", aggfunc="first"
    ).reindex(list(arms))
    single = rows.groupby("level")["level_in_single_arm"].first()
    left = np.zeros(len(arms))

    for index, level in enumerate(shares.columns):
        values = shares[level].fillna(0).to_numpy()
        axis.barh(
            range(len(arms)),
            values,
            left=left,
            color=plt.get_cmap("tab20")(index % 20),
            hatch="////" if bool(single.get(level, False)) else None,
            edgecolor="white",
            linewidth=0.3,
            label=str(level),
        )
        left += values

    axis.set_yticks(range(len(arms)), arms)
    axis.set_xlim(0, 1)
    axis.invert_yaxis()
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=5,
        ncol=1 + len(shares.columns) // 6,
    )
    axis.spines[["top", "right"]].set_visible(False)


def _donor_median_sensors(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    arms: Sequence[str],
    minimum_donors: int,
) -> pd.DataFrame:
    """Summarize supported sensor units to donor medians per group."""
    keys = ["donor_id", "sample_id", "cell_type"]
    units = metadata.loc[
        metadata["supported"] & metadata["condition"].isin(arms),
        [*keys, "condition"],
    ]
    genes = features.loc[
        features["feature_type"].isin(["gene_expression", "gene_detection"])
        & features["status"].isin(["present", "undetected"])
    ].merge(units, on=keys, how="inner", validate="many_to_one")

    donors = (
        genes.groupby(
            [
                "feature_type",
                "feature_label",
                "cell_type",
                "condition",
                "donor_id",
            ],
            observed=True,
        )["feature_value"]
        .mean()
        .reset_index()
    )

    summary = (
        donors.pivot_table(
            index=["feature_label", "cell_type", "condition"],
            columns="feature_type",
            values="feature_value",
            aggfunc="median",
        )
        .rename(
            columns={
                "gene_expression": "mean_expression",
                "gene_detection": "fraction_detected",
            }
        )
        .reset_index()
    )

    support = (
        donors.loc[donors["feature_type"].eq("gene_expression")]
        .groupby(["feature_label", "cell_type", "condition"], observed=True)[
            "donor_id"
        ]
        .nunique()
        .rename("n_donors")
        .reset_index()
    )
    summary = summary.merge(
        support, on=["feature_label", "cell_type", "condition"]
    )

    return summary.loc[summary["n_donors"].ge(minimum_donors)]


def _feature_rows(
    sensor_mapping: pd.DataFrame,
    *,
    modules: Iterable[str],
    scorer: str,
) -> pd.DataFrame:
    """Order sensors by class, then symbol, followed by modules by identifier.

    Every compendium sensor keeps a row, including those without a usable
    source gene, so their absence stays visible.
    """
    classes = {
        ("dna_sensor",): "DNA sensor",
        ("dna_sensor", "rna_sensor"): "DNA + RNA sensor",
        ("rna_sensor",): "RNA sensor",
    }
    tags = [
        tuple(sorted(str(text).split(";")))
        for text in sensor_mapping["sensor_class"]
    ]
    unknown = set(tags).difference(classes)
    if unknown:
        raise ValueError(
            f"Unknown sensor classes {sorted(unknown)}; expected dna_sensor "
            "and/or rna_sensor"
        )

    order = list(classes)
    sensors = (
        pd.DataFrame(
            {
                "feature_type": "gene_expression",
                "panel_group": [classes[tag] for tag in tags],
                "feature_label": sensor_mapping["gene"].astype(str).to_numpy(),
                "mapping_status": sensor_mapping["mapping_status"]
                .astype(str)
                .to_numpy(),
                "order": [order.index(tag) for tag in tags],
            }
        )
        .sort_values(["order", "feature_label"])
        .drop(columns="order")
    )
    sensors["display_label"] = sensors["feature_label"]

    module_ids = sorted(map(str, modules))
    module_rows = pd.DataFrame(
        {
            "feature_type": "module_score",
            "panel_group": f"Modules ({scorer})",
            "feature_label": module_ids,
            "mapping_status": "present",
            "display_label": [
                humanize_module_name(name) for name in module_ids
            ],
        }
    )
    return pd.concat([sensors, module_rows], ignore_index=True)


def _donor_contrast(
    donors: pd.DataFrame,
    *,
    target: str,
    reference: str | None,
    minimum_donors: int,
) -> pd.DataFrame:
    """Summarize one comparison per cell type and feature from donor values.

    With a reference group, separation is Cliff's delta from midranks
    pooled over both groups, so ties count as neither order. Without one,
    values are within-donor changes and separation is their mean sign.
    Non-finite donor values are left out of both.

    Returns:
      One row per cell type and feature: donors and mean per group, the
      difference, the signed separation and its measure, and a status
      ("measured", "not_detected" or "insufficient_donors").
    """
    keys = ["cell_type", "feature_type", "feature_label"]
    groups = [target] if reference is None else [target, reference]
    values = donors.loc[
        donors["group"].isin(groups)
        & np.isfinite(donors["value"].to_numpy(dtype=float))
    ]
    values = values.assign(
        in_target=values["group"].eq(target),
        detected=values["value"].ne(0),
        rank=values.groupby(keys, observed=True)["value"].rank(),
        sign=np.sign(values["value"]),
    )

    per_group = pd.DataFrame(
        values.groupby([*keys, "in_target"], observed=True)
        .agg(
            n=("value", "size"),
            mean=("value", "mean"),
            rank_sum=("rank", "sum"),
            sign=("sign", "mean"),
        )
        .unstack("in_target")
    ).reindex(
        columns=pd.MultiIndex.from_product(
            [["n", "mean", "rank_sum", "sign"], [True, False]]
        )
    )
    n_target = per_group[("n", True)].fillna(0)
    n_reference = per_group[("n", False)].fillna(0)
    detected = (
        values.groupby(keys, observed=True)["detected"]
        .any()
        .reindex(per_group.index)
    )

    if reference is None:
        summary = pd.DataFrame(
            {
                "n_target": n_target,
                "n_reference": np.nan,
                "target_mean": per_group[("mean", True)],
                "reference_mean": np.nan,
                "difference": per_group[("mean", True)],
                "separation": per_group[("sign", True)],
                "separation_measure": "sign_balance",
            }
        )
        enough = n_target.ge(minimum_donors)
    else:
        u_target = per_group[("rank_sum", True)] - n_target * (n_target + 1) / 2
        summary = pd.DataFrame(
            {
                "n_target": n_target,
                "n_reference": n_reference,
                "target_mean": per_group[("mean", True)],
                "reference_mean": per_group[("mean", False)],
                "difference": per_group[("mean", True)]
                - per_group[("mean", False)],
                "separation": 2 * u_target / (n_target * n_reference) - 1,
                "separation_measure": "cliffs_delta",
            }
        )
        enough = n_target.ge(minimum_donors) & n_reference.ge(minimum_donors)

    is_sensor = summary.index.get_level_values("feature_type").to_numpy() == (
        "gene_expression"
    )
    summary["status"] = np.select(
        [~enough.to_numpy(), is_sensor & ~detected.to_numpy(dtype=bool)],
        ["insufficient_donors", "not_detected"],
        default="measured",
    )
    return summary.reset_index()


def _contrast_table(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    comparison: tuple[str, str, str | None],
    limits: tuple[float, float],
) -> pd.DataFrame:
    """Return the displayed rows: every feature row, with its cell types.

    Features without a usable source gene keep one row without cells.
    `dot_separation` is the separation toward the mean difference, floored
    at zero, as drawn.
    """
    name, target, reference = comparison
    table = rows.merge(
        summary.assign(cell_type=summary["cell_type"].astype(str)),
        on=["feature_type", "feature_label"],
        how="left",
        validate="one_to_many",
    )
    toward = table["separation"] * np.sign(table["difference"])

    return table.assign(
        comparison=name,
        target=target,
        reference=reference,
        dot_separation=toward.clip(0, 1).where(table["status"].eq("measured")),
        color_limit=np.where(
            table["feature_type"].eq("gene_expression"), limits[0], limits[1]
        ),
    )


def _contrast_panel(
    drawn: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    cell_types: Sequence[str],
    feature_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lay out one panel: a heading per row group, then its features.

    Returns:
      The cells to draw with `x` and `y` positions, and one label record
      per panel row with `kind`, `text`, `feature_type` and `status`.
    """
    panel_rows = rows.loc[rows["feature_type"].eq(feature_type)]
    cells = drawn.loc[drawn["feature_type"].eq(feature_type)]
    with_cells = set(cells["feature_label"])

    labels, positions = [], {}
    for heading, block in panel_rows.groupby("panel_group", sort=False):
        labels.append({"kind": "heading", "text": heading, "status": ""})

        for row in block.itertuples(index=False):
            if row.mapping_status == "absent":
                status = "Not mapped"
            elif row.mapping_status != "present":
                status = str(row.mapping_status).replace("_", " ").capitalize()
            elif row.feature_label not in with_cells:
                status = "No values"
            else:
                status = "measured"
                positions[row.feature_label] = len(labels)

            labels.append(
                {"kind": "feature", "text": row.display_label, "status": status}
            )

    columns = {cell: index for index, cell in enumerate(cell_types)}
    points = cells.assign(
        x=cells["cell_type"].map(columns),
        y=cells["feature_label"].map(positions),
    ).dropna(subset=["x", "y"])
    return points, pd.DataFrame(labels).assign(feature_type=feature_type)


def _draw_contrast_panel(
    figure: Figure,
    axis: Axes,
    points: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_columns: int,
    limit: float,
    max_dot_size: float,
    color_label: str,
    colorbar_shrink: float,
) -> None:
    """Draw one panel's dots, crosses, headings, row notes and colour bar.

    Colour is the clipped difference and area the separation toward it,
    with a pinpoint floor so measured cells never vanish. Cells whose donors
    are all zero get a grey cross; rows without a usable source gene carry
    their status as grey text.
    """
    colormap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    features = np.flatnonzero(labels["kind"].eq("feature"))
    axis.hlines(features, -0.6, n_columns - 0.4, color="#eeeeee", lw=0.3)
    axis.vlines(
        range(n_columns), -0.5, len(labels) - 0.5, color="#eeeeee", lw=0.3
    )

    measured = points.loc[points["status"].eq("measured")]
    axis.scatter(
        measured["x"],
        measured["y"],
        s=np.maximum(measured["dot_separation"] * max_dot_size, 1.0),
        c=measured["difference"].clip(-limit, limit),
        cmap=colormap,
        norm=norm,
        edgecolors="#555555",
        linewidths=0.2,
        zorder=2,
    )
    zeros = points.loc[points["status"].eq("not_detected")]
    axis.scatter(
        zeros["x"], zeros["y"], marker="x", s=4, c="#9a9a9a", lw=0.4, zorder=2
    )

    note = {"facecolor": "white", "edgecolor": "none", "pad": 0.4}
    for position, label in enumerate(labels.itertuples(index=False)):
        if label.kind == "heading":
            axis.text(
                0.0,
                position,
                str(label.text),
                transform=axis.get_yaxis_transform(),
                va="center",
                fontweight="bold",
                bbox=note,
                zorder=3,
            )
        elif label.status != "measured":
            axis.text(
                (n_columns - 1) / 2,
                position,
                str(label.status),
                ha="center",
                va="center",
                color="#626262",
                fontsize=4,
                bbox=note,
                zorder=3,
            )

    axis.set_yticks(
        range(len(labels)),
        [
            text if kind == "feature" else ""
            for kind, text in zip(labels["kind"], labels["text"], strict=True)
        ],
    )
    if labels["feature_type"].eq("gene_expression").all():
        for tick in axis.get_yticklabels():
            tick.set_fontstyle("italic")
    axis.set_xlim(-0.6, n_columns - 0.4)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.tick_params(length=0, pad=1.5)
    axis.spines[:].set_visible(False)

    differences = measured["difference"].to_numpy(dtype=float)
    clipped = (
        bool((differences < -limit).any()),
        bool((differences > limit).any()),
    )
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=colormap),
        ax=axis,
        shrink=colorbar_shrink,
        aspect=12,
        pad=0.02,
        extend={
            (False, False): "neither",
            (True, False): "min",
            (False, True): "max",
            (True, True): "both",
        }[clipped],
        extendrect=True,
    )
    colorbar.set_label(color_label)


def _separation_legend(
    figure: Figure,
    *,
    paired: bool,
    max_dot_size: float,
) -> None:
    """Explain dot area and the all-zero cross below the panels."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=float(np.sqrt(value * max_dot_size)),
            markerfacecolor="#bdbdbd",
            markeredgecolor="#555555",
            markeredgewidth=0.2,
            label=f"{value:g}",
        )
        for value in (0.25, 0.5, 1.0)
    ]
    handles.append(
        Line2D(
            [], [], marker="x", ls="", ms=2, color="#9a9a9a", label="All zero"
        )
    )
    figure.legend(
        handles=handles,
        title=(
            "Dot area: donors changing with the mean minus against it (share)"
            if paired
            else "Dot area: Cliff's delta toward the mean difference\n"
            "(donor pairs ordered with it minus against it, as a share)"
        ),
        title_fontsize=plt.rcParams["font.size"],
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
    )


def _effect_matrix(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot effects to features by cell types, blanking non-estimable cells.

    Rows are ordered by their largest absolute estimate; the BH-FDR matrix
    follows the same order.
    """
    estimate = selected.pivot_table(
        index="feature_label",
        columns="cell_type",
        values="estimate",
        aggfunc="first",
        dropna=False,
    )
    estimable = selected.assign(
        ok=selected["estimable"].astype(bool) & selected["status"].eq("ok")
    ).pivot_table(
        index="feature_label",
        columns="cell_type",
        values="ok",
        aggfunc="first",
        dropna=False,
    )
    fdr = selected.pivot_table(
        index="feature_label",
        columns="cell_type",
        values="pvalue_fdr",
        aggfunc="first",
        dropna=False,
    )

    estimate = estimate.where(estimable.fillna(False).astype(bool))
    order = estimate.abs().max(axis=1).sort_values(ascending=False).index
    return estimate.loc[order], fdr.reindex_like(estimate.loc[order])


def _color_limit(values: np.ndarray, quantile: float) -> float:
    """Symmetric colour limit: a quantile of absolute finite values, or 1."""
    finite = np.abs(values[np.isfinite(values)])
    limit = float(np.quantile(finite, quantile)) if finite.size else 1.0
    return limit or 1.0


def _draw_effect_matrix(
    figure: Figure,
    axis: Axes,
    estimate: pd.DataFrame,
    fdr: pd.DataFrame,
    *,
    alpha: float,
    limit: float,
    feature_type: str,
    color_label: str,
) -> None:
    """Draw the clipped diverging matrix, its marks, labels and colour bar.

    Non-estimable cells are grey with a cross; a black dot marks BH-FDR
    below `alpha`.
    """
    values = estimate.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    image = axis.imshow(
        np.clip(masked, -limit, limit),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
        aspect="auto",
    )
    # Square cells: a figure widened for its title must not stretch a matrix
    # of few cell types.
    axis.set_box_aspect(estimate.shape[0] / estimate.shape[1])
    image.cmap.set_bad("#e6e6e6")

    missing_y, missing_x = np.nonzero(~np.isfinite(values))
    axis.scatter(missing_x, missing_y, marker="x", s=6, c="#999999", lw=0.4)
    significant = (fdr.to_numpy(dtype=float) < alpha) & np.isfinite(values)
    sig_y, sig_x = np.nonzero(significant)
    axis.scatter(sig_x, sig_y, s=3, c="black")

    labels = [
        humanize_module_name(label)
        if feature_type == "module_score"
        else str(label)
        for label in estimate.index
    ]
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(
        range(estimate.shape[1]),
        estimate.columns.astype(str).tolist(),
        rotation=90,
    )
    axis.tick_params(length=0, pad=1.5)
    axis.spines[:].set_visible(False)

    colorbar = figure.colorbar(image, ax=axis, shrink=0.35, aspect=10, pad=0.02)
    colorbar.set_label(color_label)


def _fit_row_labels(
    figure: Figure,
    axis: Axes,
    *,
    matrix_width: float,
) -> None:
    """Widen the figure so the longest row label fits beside the matrix."""
    label_width = max(
        text.get_window_extent().width for text in axis.get_yticklabels()
    )
    content = label_width / figure.dpi + matrix_width + 1.2
    if content > figure.get_figwidth():
        figure.set_figwidth(content)


def _significance_legend(figure: Figure, *, alpha: float) -> None:
    """Explain the significance dot and the not-estimable cross."""
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                ms=2,
                color="black",
                label=f"BH-FDR < {alpha:g}",
            ),
            Line2D(
                [],
                [],
                marker="x",
                ls="",
                ms=3,
                color="#999999",
                label="Not estimable",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )


def _ifn_adjustment_pairs(
    unadjusted: pd.DataFrame,
    adjusted: pd.DataFrame,
    *,
    estimand: str,
    feature_type: str,
    alpha: float,
) -> pd.DataFrame:
    """Pair fitted AB1 and AB3 rows and classify each modulated shift.

    A shift with BH-FDR below `alpha` "persists" when the adjusted
    estimate keeps its sign with an interval excluding zero, and is
    "attenuated" otherwise; other rows are "not_modulated".
    """
    keys = ["feature_label", "cell_type"]
    columns = [*keys, "estimate", "ci_low", "ci_high", "pvalue_fdr"]

    def rows(table: pd.DataFrame) -> pd.DataFrame:
        return table.loc[
            table["estimand"].eq(estimand)
            & table["feature_type"].eq(feature_type)
            & table["status"].eq("ok"),
            columns,
        ]

    paired = rows(unadjusted).merge(
        rows(adjusted), on=keys, suffixes=("", "_adjusted")
    )
    if paired.empty:
        return paired

    modulated = paired["pvalue_fdr"].lt(alpha)
    persists = (
        modulated
        & (np.sign(paired["estimate"]) == np.sign(paired["estimate_adjusted"]))
        & ((paired["ci_low_adjusted"] > 0) | (paired["ci_high_adjusted"] < 0))
    )
    paired["category"] = np.select(
        [persists, modulated], ["persists", "attenuated"], "not_modulated"
    )
    return paired


def _draw_ifn_adjustment_points(
    axis: Axes,
    paired: pd.DataFrame,
    *,
    marker_size: float,
) -> None:
    """Draw each category's points with their unadjusted and adjusted CIs."""
    styles = {
        "not_modulated": ("#c8c8c8", "Not modulated"),
        "attenuated": ("#E69F00", "Modulated, IFN-attenuated"),
        "persists": ("#0072B2", "Modulated, persists after adjustment"),
    }

    for category, (color, label) in styles.items():
        part = paired.loc[paired["category"].eq(category)]
        axis.errorbar(
            part["estimate"],
            part["estimate_adjusted"],
            xerr=[
                part["estimate"] - part["ci_low"],
                part["ci_high"] - part["estimate"],
            ],
            yerr=[
                part["estimate_adjusted"] - part["ci_low_adjusted"],
                part["ci_high_adjusted"] - part["estimate_adjusted"],
            ],
            fmt="none",
            ecolor=color,
            elinewidth=0.3,
            alpha=0.5,
        )
        axis.scatter(
            part["estimate"],
            part["estimate_adjusted"],
            s=marker_size,
            c=color,
            edgecolors="white",
            linewidths=0.2,
            label=f"{label} ({len(part)})",
            zorder=3,
        )


def _draw_identity_axes(axis: Axes, paired: pd.DataFrame) -> None:
    """Set symmetric limits from the estimates and draw the reference lines.

    Wide intervals clip at the edge and keep their values in the CSV.
    """
    limit = 1.15 * float(
        np.nanmax(np.abs(paired[["estimate", "estimate_adjusted"]].to_numpy()))
    )

    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.plot([-limit, limit], [-limit, limit], ls="--", lw=0.5, c="#777777")
    axis.axhline(0, lw=0.4, c="#777777")
    axis.axvline(0, lw=0.4, c="#777777")


def _label_persisting(
    axis: Axes,
    paired: pd.DataFrame,
    *,
    n_labels: int,
) -> None:
    """Label the persisting shifts with the largest adjusted estimates."""
    persisting = paired.loc[paired["category"].eq("persists")]
    labelled = persisting.assign(
        magnitude=persisting["estimate_adjusted"].abs()
    ).nlargest(n_labels, "magnitude")

    texts = [
        axis.text(x, y, f"{gene} · {cell_type}", fontsize=5)
        for x, y, gene, cell_type in zip(
            labelled["estimate"].to_numpy(dtype=float),
            labelled["estimate_adjusted"].to_numpy(dtype=float),
            labelled["feature_label"],
            labelled["cell_type"],
            strict=True,
        )
    ]

    if texts:
        adjust_text(
            texts,
            ax=axis,
            arrowprops={"arrowstyle": "-", "lw": 0.3, "color": "#555555"},
        )


def _draw_calibration_panel(
    axis: Axes,
    rows: pd.DataFrame,
    stats_row: pd.Series | None,
    *,
    cell_type: str,
    marker_size: float,
    n_labels: int,
) -> None:
    """Scatter one cell type's donor medians against the atlas, with rho.

    The `n_labels` sensors farthest from the identity line are named.
    """
    axis.scatter(
        rows["atlas_median"],
        rows["cohort_median"],
        s=marker_size,
        color="#0072B2",
        edgecolors="white",
        linewidths=0.2,
    )
    upper = float(np.nanmax(rows[["atlas_median", "cohort_median"]].to_numpy()))
    axis.plot([0, upper], [0, upper], ls="--", lw=0.5, c="#777777")

    deviation = (rows["cohort_median"] - rows["atlas_median"]).abs()
    labelled = rows.loc[deviation.nlargest(n_labels).index]
    texts = [
        axis.text(x, y, str(gene), fontsize=4.5, fontstyle="italic")
        for x, y, gene in zip(
            labelled["atlas_median"].to_numpy(dtype=float),
            labelled["cohort_median"].to_numpy(dtype=float),
            labelled["gene"],
            strict=True,
        )
    ]
    if texts:
        adjust_text(
            texts,
            ax=axis,
            arrowprops={"arrowstyle": "-", "lw": 0.3, "color": "#555555"},
        )

    axis.set_title(
        f"{cell_type}\nrho {stats_row['spearman_rho']:.2f} "
        f"({int(stats_row['cohort_donors'])} vs "
        f"{int(stats_row['atlas_donors'])} donors)"
        if stats_row is not None
        else str(cell_type),
        fontsize=6,
    )
    axis.spines[["top", "right"]].set_visible(False)


def _equal_arm_sample(
    embedding: pd.DataFrame,
    arms: Sequence[str],
    *,
    max_cells: int,
    random_state: int,
) -> pd.DataFrame:
    """Draw up to `max_cells` cells per arm reproducibly, arms in order."""
    rng = np.random.default_rng(random_state)
    parts = []
    for arm in arms:
        cells = embedding.loc[embedding["condition"].astype(str).eq(arm)]
        if cells.empty:
            continue
        parts.append(cells.iloc[rng.permutation(len(cells))[:max_cells]])

    return pd.concat(parts, ignore_index=True)


def _dotplot_layout(
    summary: pd.DataFrame,
    arms: Sequence[str],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Place sensors on rows and cell-type-by-arm groups on columns.

    Columns keep each cell type's arms adjacent, in `arms` order.

    Returns:
      Sensor row labels, the ordered columns, and `summary` with `x` and `y`.
    """
    genes = sorted(summary["feature_label"].unique(), reverse=True)
    columns = (
        summary[["cell_type", "condition"]]
        .drop_duplicates()
        .assign(
            order=lambda frame: (
                frame["condition"]
                .astype(str)
                .map({arm: index for index, arm in enumerate(arms)})
            )
        )
        .sort_values(["cell_type", "order"])
    )

    column_index = {
        (row.cell_type, row.condition): index
        for index, row in enumerate(columns.itertuples(index=False))
    }
    shown = summary.assign(
        x=[
            column_index[(cell, arm)]
            for cell, arm in zip(
                summary["cell_type"], summary["condition"], strict=True
            )
        ],
        y=summary["feature_label"].map(
            {gene: index for index, gene in enumerate(genes)}
        ),
    )

    return genes, columns, shown


def _effect_label(effect_scales: pd.Series) -> str:
    """Describe a table's effect scale for a colour bar."""
    if effect_scales.empty:
        return "Estimate"

    scale = str(effect_scales.iloc[0])
    known = {
        "score_difference": "Score difference (target - reference)",
        "log1p_cp10k_difference": "log1p CP10k difference",
        "detected_fraction_difference": "Detected-fraction difference",
    }
    return known.get(scale, scale.replace("_", " "))


def _validate_geometry(
    *,
    figsize: tuple[float, float] | None = None,
    **dimensions: float,
) -> None:
    """Reject non-finite or nonpositive figure geometry."""
    if figsize is not None and (
        len(figsize) != 2
        or any(not np.isfinite(value) or value <= 0 for value in figsize)
    ):
        raise ValueError("figsize must contain two positive finite inches")

    for name, value in dimensions.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
