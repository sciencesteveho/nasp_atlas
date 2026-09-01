"""Regression and association plotting methods."""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.typing import ColorType

from nasp_atlas.cellxgene.filter import humanize_label
from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import darken_color
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


def _numeric_scalar(value: object) -> float:
    """Return a numeric scalar or NaN for an unparseable value."""
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return np.nan


def _predictor_display_name(predictor_key: str) -> str:
    """Return a publication label for an association predictor."""
    return humanize_label(
        predictor_key,
        {
            "age_years": "Age (years)",
            "tissue_specific_eqtls": (
                "Tissue-specific\nsignificant eQTL pairs"
            ),
            "total_eqtls": "Total significant eQTL pairs",
        },
    )


def _group_display_name(group_key: str) -> str:
    """Return a publication label for a grouping variable."""
    return humanize_label(
        group_key,
        {
            "cell_type": "Cell type",
            "sex": "Sex",
            "tissue_in_publication": "Tissue",
        },
    )


def _statistical_unit_display_name(statistical_unit: str) -> str:
    """Return a plural publication label for a statistical unit."""
    return {
        "cell": "Cells",
        "metacell": "Metacells",
        "donor": "Donors",
        "donor_tissue": "Donor-tissue units",
        "donor_tissue_cell_type": "Donor-tissue-cell type units",
        "donor_tissue_sex": "Donor-tissue-sex units",
    }.get(statistical_unit, humanize_label(statistical_unit))


class AssociationPlotter(_PlotterBase):
    """Render regression and grouped-association figures.

    Example Usage:
      >>> plotter = AssociationPlotter(output_dir="path/to/output")
      >>> plotter.plot_feature_regression(
      ...     unit_frame,
      ...     feature_id="CGAS",
      ...     predictor_key="age_years",
      ...     filename="cgas_by_age",
      ... )
    """

    def plot_feature_regression(
        self,
        unit_frame: pd.DataFrame,
        *,
        feature_id: str,
        predictor_key: str,
        filename: str,
        result_row: pd.Series | None = None,
        color_key: str | None = None,
        value_column: str = "feature_value",
        feature_label: str | None = None,
        point_color: str | None = "black",
        category_colors: Mapping[str, ColorType] | None = None,
        legend_handletextpad: float = 0.1,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Scatter one feature against a continuous predictor with a fit line.

        The frame is expected to be aggregated to a statistical unit (one point
        per donor/donor_tissue/etc.), so the plot shows donor-level rather than
        cell-level structure. The title identifies the biological stratum when
        available, and the correlation annotation records its p-value.

        Args:
          unit_frame: Aggregated unit frame filtered or filterable to the
            feature, carrying the predictor and value columns.
          feature_id: Feature id to select from `unit_frame`.
          predictor_key: Continuous predictor column for the x axis.
          filename: Output filename stem under `output_dir`.
          result_row: Optional regression result row supplying r, p-value, slope
            and intercept for the annotation and fit line.
          color_key: Optional column used to color points by category when
            `point_color` is None.
          value_column: Column holding the feature value for the y axis.
          feature_label: Optional display label; defaults to `feature_id`.
          point_color: Fixed point color. Set to None to use categorical colors
            from `color_key`.
          category_colors: Optional mapping from `color_key` categories to
            point colors when `point_color` is None.
          legend_handletextpad: Horizontal gap in font-size units between a
            legend marker and its label.
          figsize: Optional figure size in inches.

        Example Usage:
          >>> plotter.plot_feature_regression(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     predictor_key="age_years",
          ...     filename="il6_age_regression",
          ...     color_key="sex",
          ...     point_color=None,
          ...     category_colors={
          ...         "male": "#d2e7ef",
          ...         "female": "#f9bebc",
          ...     },
          ...     legend_handletextpad=0.1,
          ... )
        """
        set_matplotlib_publication_parameters()
        if not np.isfinite(legend_handletextpad) or legend_handletextpad < 0.0:
            raise ValueError(
                "legend_handletextpad must be a nonnegative finite value"
            )
        block = unit_frame[unit_frame["feature_id"].astype(str) == feature_id]
        plot_df = pd.DataFrame(
            {
                "x": pd.to_numeric(block[predictor_key], errors="coerce"),
                "y": pd.to_numeric(block[value_column], errors="coerce"),
            }
        )
        if color_key is not None and color_key in block.columns:
            plot_df["color"] = block[color_key].astype(str).to_numpy()
        plot_df = plot_df.dropna(subset=["x", "y"])
        if plot_df.empty:
            logger.warning(
                "[plot] No valid points for regression %s. Skipping.", filename
            )
            return

        fig, ax = plt.subplots(figsize=figsize or (1.2, 1.2))
        if "color" in plot_df.columns and point_color is None:
            default_colors = plt.get_cmap("tab10")
            for index, category in enumerate(sorted(plot_df["color"].unique())):
                sub = plot_df[plot_df["color"] == category]
                color = (
                    category_colors[category]
                    if category_colors is not None
                    and category in category_colors
                    else default_colors(index % 10)
                )
                ax.scatter(
                    sub["x"],
                    sub["y"],
                    s=2.5,
                    color=color,
                    label=str(category),
                )
            ax.legend(
                frameon=False,
                loc="best",
                handletextpad=legend_handletextpad,
            )
        else:
            ax.scatter(
                plot_df["x"],
                plot_df["y"],
                s=2.5,
                color=point_color or "black",
            )

        unit = "" if block.empty else str(block["statistical_unit"].iloc[0])
        self._draw_regression_line(ax=ax, result_row=result_row)
        ax.set_xlabel(_predictor_display_name(predictor_key))
        y_label = ax.set_ylabel(
            self._regression_y_label(
                feature_id=feature_id,
                feature_label=feature_label or feature_id,
                result_row=result_row,
            )
        )
        if self._is_gene_expression(feature_id, result_row):
            y_label.set_fontstyle("italic")
        title = self._regression_title(
            statistical_unit=unit,
            result_row=result_row,
        )
        if annotation := self._regression_annotation(result_row):
            title += f"\n{annotation}"
        ax.set_title(title)
        self._save_figure_and_log(
            fig, self.output_dir / filename, "[plot] feature regression -> %s"
        )

    def plot_feature_regression_grid(
        self,
        unit_frame: pd.DataFrame,
        *,
        feature_id: str,
        predictor_key: str,
        filename: str,
        results: pd.DataFrame,
        stratify_key: str,
        value_column: str = "feature_value",
        feature_label: str | None = None,
        stratum_order: Sequence[str] | None = None,
        max_columns: int = 10,
        panel_width: float = 0.9,
        panel_height: float = 0.9,
        point_size: float = 3.0,
        label_wrap_width: int = 20,
        column_gap: float = 0.45,
        row_gap: float = 0.85,
        y_label_x: float = 0.09,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Combine one feature's stratum-specific regressions in one figure.

        Every saved stratum receives a panel with shared predictor and response
        limits. Estimable panels show their saved OLS line and correlation;
        non-estimable panels retain their observed points and are labeled as
        such. Panel dimensions are in inches and `point_size` is marker area in
        points squared. `label_wrap_width` is the target character count for
        long stratum labels; and `column_gap`/`row_gap` are Matplotlib subplot
        spacing fractions that expand the adaptive total figure size rather
        than shrinking its panels. `y_label_x` positions the shared response
        label in figure coordinates. Text uses the shared 5-point publication
        style.

        Example Usage:
          >>> plotter.plot_feature_regression_grid(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     predictor_key="age_years",
          ...     filename="il6_age_by_tissue",
          ...     results=regression_results,
          ...     stratify_key="tissue",
          ...     max_columns=6,
          ...     column_gap=0.45,
          ...     row_gap=0.85,
          ...     y_label_x=0.09,
          ... )
        """
        set_matplotlib_publication_parameters()
        required_frame = {
            "feature_id",
            predictor_key,
            stratify_key,
            value_column,
        }
        if missing := sorted(required_frame.difference(unit_frame.columns)):
            raise KeyError(
                f"unit frame missing regression-grid columns: {missing}"
            )
        required_results = {"feature_id", "stratum", "skipped"}
        if missing := sorted(required_results.difference(results.columns)):
            raise KeyError(
                f"results missing regression-grid columns: {missing}"
            )
        if max_columns < 1:
            raise ValueError("max_columns must be at least 1")
        for parameter_name, value in (
            ("panel_width", panel_width),
            ("panel_height", panel_height),
            ("point_size", point_size),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{parameter_name} must be a positive finite value"
                )
        if label_wrap_width < 1:
            raise ValueError("label_wrap_width must be at least 1")
        for parameter_name, value in (
            ("column_gap", column_gap),
            ("row_gap", row_gap),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{parameter_name} must be a nonnegative finite value"
                )
        if not np.isfinite(y_label_x) or not 0.0 <= y_label_x <= 1.0:
            raise ValueError("y_label_x must be finite and between 0 and 1")
        if figsize is not None and (
            len(figsize) != 2
            or any(
                not np.isfinite(dimension) or dimension <= 0.0
                for dimension in figsize
            )
        ):
            raise ValueError("figsize dimensions must be positive and finite")

        selected_results = results.loc[
            results["feature_id"].astype(str) == feature_id
        ].copy()
        if "stratify_key" in selected_results:
            selected_results = selected_results.loc[
                selected_results["stratify_key"].astype(str) == stratify_key
            ]
        selected_results = selected_results.loc[
            selected_results["stratum"].notna()
            & selected_results["stratum"].astype(str).ne("")
        ]
        if selected_results.empty:
            logger.warning(
                "[plot] No saved strata for regression grid %s. Skipping.",
                filename,
            )
            return
        if selected_results["stratum"].astype(str).duplicated().any():
            raise ValueError(
                f"results contain duplicate {stratify_key} strata for "
                f"feature_id={feature_id}"
            )

        rows_by_stratum = {
            str(row["stratum"]): row for _, row in selected_results.iterrows()
        }
        strata = (
            sorted(rows_by_stratum)
            if stratum_order is None
            else [str(stratum) for stratum in stratum_order]
        )
        if missing := sorted(set(strata).difference(rows_by_stratum)):
            raise KeyError(
                f"requested regression-grid strata are unavailable: {missing}"
            )

        n_columns = min(max_columns, len(strata))
        n_rows = (len(strata) + n_columns - 1) // n_columns
        resolved_figsize = figsize or (
            panel_width * (n_columns + max(0, n_columns - 1) * column_gap),
            panel_height * (n_rows + max(0, n_rows - 1) * row_gap),
        )
        fig, axes_grid = plt.subplots(
            n_rows,
            n_columns,
            squeeze=False,
            sharex=True,
            sharey=True,
            figsize=resolved_figsize,
        )
        axes = axes_grid.ravel().tolist()
        feature_block = unit_frame.loc[
            unit_frame["feature_id"].astype(str) == feature_id
        ]
        feature_strata = feature_block[stratify_key].astype(str)
        plotted: list[tuple[Axes, pd.Series]] = []
        for ax, stratum in zip(axes, strata, strict=False):
            result_row = rows_by_stratum[stratum]
            block = feature_block.loc[feature_strata == stratum]
            plot_df = pd.DataFrame(
                {
                    "x": pd.to_numeric(
                        block[predictor_key],
                        errors="coerce",
                    ),
                    "y": pd.to_numeric(
                        block[value_column],
                        errors="coerce",
                    ),
                }
            ).replace([np.inf, -np.inf], np.nan)
            plot_df.dropna(inplace=True)
            ax.scatter(
                plot_df["x"],
                plot_df["y"],
                s=point_size,
                color="black",
            )

            title = self._regression_facet_label(
                stratum,
                wrap_width=label_wrap_width,
            )
            if self._result_is_skipped(result_row):
                title += "\nNot estimable"
                ax.set_facecolor("#f3f3f3")
            elif annotation := self._regression_annotation(result_row):
                title += f"\n{annotation}"
            ax.set_title(title)
            plotted.append((ax, result_row))

        for ax, result_row in plotted:
            if not self._result_is_skipped(result_row):
                self._draw_regression_line(ax=ax, result_row=result_row)
        for ax in axes[len(strata) :]:
            ax.set_visible(False)
        fig.subplots_adjust(wspace=column_gap, hspace=row_gap)

        representative = selected_results.iloc[0]
        resolved_feature_label = feature_label or (
            str(feature_block["feature_label"].iloc[0])
            if "feature_label" in feature_block and not feature_block.empty
            else feature_id
        )
        fig.supxlabel(
            _predictor_display_name(predictor_key),
            fontsize=5,
            y=-0.025,
        )
        y_label = fig.supylabel(
            self._regression_y_label(
                feature_id=feature_id,
                feature_label=resolved_feature_label,
                result_row=representative,
            ),
            fontsize=5,
            x=y_label_x,
        )
        if self._is_gene_expression(feature_id, representative):
            y_label.set_fontstyle("italic")
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] feature regression grid -> %s",
        )

    def plot_feature_group_boxplot(
        self,
        unit_frame: pd.DataFrame,
        *,
        feature_id: str,
        group_key: str,
        filename: str,
        result_row: pd.Series | None = None,
        value_column: str = "feature_value",
        feature_label: str | None = None,
        central: Literal["mean", "median"] = "mean",
        errorbar: Literal["sem", "sd", "none"] | None = None,
        stratify_key: str | None = None,
        stratify_colors: Mapping[str, str] | None = None,
        box_width: float | None = None,
        point_size: float = 1.8,
        x_margin: float = 0.01,
        y_margin: float = 0.01,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot unit-level distributions as boxplots across groups.

        Boxes show the median, interquartile range, and 1.5-IQR whiskers at the
        supplied statistical unit. Individual unit values are overlaid as
        deterministic jittered points. An optional stratifying variable draws
        adjacent boxes within every x-axis group.

        Args:
          unit_frame: Aggregated unit frame carrying the value and group
            columns for the feature.
          feature_id: Feature id to select from `unit_frame`.
          group_key: Categorical column defining the x-axis groups.
          filename: Output filename stem under `output_dir`.
          result_row: Optional group-test result row for the title annotation.
          value_column: Column holding the feature value.
          feature_label: Optional display label; defaults to `feature_id`.
          central: Central tendency used to order x-axis groups.
          errorbar: Deprecated compatibility argument from the former barplot;
            ignored because boxplot whiskers show the 1.5-IQR range.
          stratify_key: Optional categorical column drawn as adjacent boxes.
          stratify_colors: Optional mapping from strata to box fill colors.
          box_width: Box width in categorical-axis units. Defaults to an
            adaptive width based on the number of strata.
          point_size: Individual unit marker area in points squared.
          x_margin: Fractional whitespace at each end of the x axis.
          y_margin: Fractional whitespace at each end of the y axis.
          figsize: Optional figure size in inches.

        Example Usage:
          >>> plotter.plot_feature_group_boxplot(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     group_key="disease_status",
          ...     filename="il6_by_disease",
          ...     box_width=0.25,
          ...     point_size=1.8,
          ... )
        """
        set_matplotlib_publication_parameters()
        if errorbar is not None:
            logger.warning(
                "[plot] errorbar=%s is ignored for true boxplots", errorbar
            )
        if box_width is not None and (
            not np.isfinite(box_width) or box_width <= 0.0
        ):
            raise ValueError("box_width must be a positive finite value")
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("point_size must be a positive finite value")

        block, plot_df = self._prepare_feature_group_plot(
            unit_frame,
            feature_id=feature_id,
            group_key=group_key,
            value_column=value_column,
            stratify_key=stratify_key,
            filename=filename,
        )
        if plot_df.empty:
            logger.warning(
                "[plot] No valid points for boxplot %s. Skipping.", filename
            )
            return

        group_order, stratum_order, color_map = self._group_plot_layout(
            plot_df,
            central=central,
            stratify_key=stratify_key,
            stratify_colors=stratify_colors,
        )

        fig, ax = plt.subplots(
            figsize=figsize or self._feature_group_figure_size(group_order)
        )
        centers = np.arange(len(group_order), dtype=float)
        slot_width = 0.72 / len(stratum_order)
        resolved_box_width = (
            slot_width * 0.72 if box_width is None else box_width
        )
        offsets = (
            np.arange(len(stratum_order), dtype=float)
            - (len(stratum_order) - 1) / 2.0
        ) * slot_width
        rng = np.random.default_rng(0)

        for stratum, offset in zip(stratum_order, offsets, strict=True):
            values_by_group: list[np.ndarray] = []
            positions: list[float] = []

            for group_index, group_name in enumerate(group_order):
                selected = (plot_df["group"] == group_name) & (
                    plot_df["stratum"] == stratum
                )
                values = plot_df.loc[selected, ["value"]].to_numpy(dtype=float)[
                    :, 0
                ]
                if values.size == 0:
                    continue

                position = float(centers[group_index] + offset)
                values_by_group.append(values)
                positions.append(position)

            fill_color = color_map[stratum]
            detail_color = darken_color(fill_color, factor=0.70)
            ax.boxplot(
                values_by_group,
                positions=positions,
                widths=resolved_box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": fill_color,
                    "edgecolor": detail_color,
                    "linewidth": 0.6,
                },
                medianprops={"color": detail_color, "linewidth": 0.7},
                whiskerprops={"color": detail_color, "linewidth": 0.5},
                capprops={"color": detail_color, "linewidth": 0.5},
            )

            for position, values in zip(
                positions, values_by_group, strict=True
            ):
                jitter = (
                    (rng.random(values.size) - 0.5) * resolved_box_width * 0.55
                )
                ax.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=point_size,
                    color=detail_color,
                    alpha=0.7,
                    linewidths=0,
                    zorder=3,
                )

        ax.margins(x=x_margin, y=y_margin)
        self._style_feature_group_axis(
            ax=ax,
            centers=centers,
            group_order=group_order,
            group_key=group_key,
            feature_id=feature_id,
            feature_label=feature_label,
            block=block,
            result_row=result_row,
            stratify_key=stratify_key,
            stratum_order=stratum_order,
            color_map=color_map,
        )

        self._save_figure_and_log(
            fig, self.output_dir / filename, "[plot] feature boxplot -> %s"
        )

    def plot_feature_group_barplot(
        self,
        unit_frame: pd.DataFrame,
        *,
        feature_id: str,
        group_key: str,
        filename: str,
        result_row: pd.Series | None = None,
        value_column: str = "feature_value",
        feature_label: str | None = None,
        central: Literal["mean", "median"] = "mean",
        errorbar: Literal["sem", "sd", "none"] = "sem",
        stratify_key: str | None = None,
        stratify_colors: Mapping[str, str] | None = None,
        bar_width: float | None = None,
        point_size: float = 1.8,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot grouped central values, uncertainty, and unit-level points.

        Args:
          unit_frame: Aggregated unit frame carrying feature and group columns.
          feature_id: Feature id to select from `unit_frame`.
          group_key: Categorical column defining the x-axis groups.
          filename: Output filename stem under `output_dir`.
          result_row: Optional group-test result row for the title annotation.
          value_column: Column holding the feature value.
          feature_label: Optional display label; defaults to `feature_id`.
          central: Central value represented by each bar.
          errorbar: Standard error, standard deviation, or no error bar.
          stratify_key: Optional categorical column drawn as adjacent bars.
          stratify_colors: Optional mapping from strata to bar fill colors.
          bar_width: Bar width in categorical-axis units. Defaults to an
            adaptive width that leaves space between adjacent strata.
          point_size: Individual unit marker area in points squared.
          figsize: Optional figure size in inches.

        Example Usage:
          >>> plotter.plot_feature_group_barplot(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     group_key="tissue",
          ...     filename="il6_by_tissue",
          ...     bar_width=0.25,
          ...     point_size=1.8,
          ... )
        """
        set_matplotlib_publication_parameters()
        if bar_width is not None and (
            not np.isfinite(bar_width) or bar_width <= 0.0
        ):
            raise ValueError("bar_width must be a positive finite value")
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("point_size must be a positive finite value")

        block, plot_df = self._prepare_feature_group_plot(
            unit_frame,
            feature_id=feature_id,
            group_key=group_key,
            value_column=value_column,
            stratify_key=stratify_key,
            filename=filename,
        )
        if plot_df.empty:
            logger.warning(
                "[plot] No valid points for barplot %s. Skipping.", filename
            )
            return

        group_order, stratum_order, color_map = self._group_plot_layout(
            plot_df,
            central=central,
            stratify_key=stratify_key,
            stratify_colors=stratify_colors,
        )

        fig, ax = plt.subplots(
            figsize=figsize or self._feature_group_figure_size(group_order)
        )
        centers = np.arange(len(group_order), dtype=float)
        slot_width = 0.72 / len(stratum_order)
        resolved_bar_width = (
            slot_width * 0.72 if bar_width is None else bar_width
        )
        offsets = (
            np.arange(len(stratum_order), dtype=float)
            - (len(stratum_order) - 1) / 2.0
        ) * slot_width
        rng = np.random.default_rng(0)

        for stratum, offset in zip(stratum_order, offsets, strict=True):
            stratum_frame = plot_df.loc[plot_df["stratum"] == stratum]
            grouped = stratum_frame.groupby("group", observed=True)["value"]
            summary = grouped.agg(["mean", "median", "std", "count"]).reindex(
                group_order
            )
            summary["sem"] = summary["std"] / np.sqrt(summary["count"])

            positions = centers + offset
            valid = summary[central].notna().to_numpy(dtype=bool)
            yerr = None
            if errorbar in ("sem", "sd"):
                error_column = "sem" if errorbar == "sem" else "std"
                yerr = (
                    summary.loc[valid, error_column]
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )

            fill_color = color_map[stratum]
            detail_color = darken_color(fill_color, factor=0.85)
            ax.bar(
                positions[valid],
                summary.loc[valid, central].to_numpy(dtype=float),
                yerr=yerr,
                width=resolved_bar_width,
                color=fill_color,
                edgecolor=detail_color,
                linewidth=0.6,
                error_kw={
                    "ecolor": detail_color,
                    "elinewidth": 0.5,
                    "capsize": 1.5,
                    "capthick": 0.5,
                },
            )

            for group_index, group_name in enumerate(group_order):
                selected = stratum_frame["group"] == group_name
                values = stratum_frame.loc[selected, ["value"]].to_numpy(
                    dtype=float
                )[:, 0]
                if values.size == 0:
                    continue

                position = float(positions[group_index])
                jitter = (
                    (rng.random(values.size) - 0.5) * resolved_bar_width * 0.55
                )
                ax.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=point_size,
                    color=detail_color,
                    alpha=0.7,
                    linewidths=0,
                    zorder=3,
                )

        self._style_feature_group_axis(
            ax=ax,
            centers=centers,
            group_order=group_order,
            group_key=group_key,
            feature_id=feature_id,
            feature_label=feature_label,
            block=block,
            result_row=result_row,
            stratify_key=stratify_key,
            stratum_order=stratum_order,
            color_map=color_map,
        )

        self._save_figure_and_log(
            fig, self.output_dir / filename, "[plot] feature barplot -> %s"
        )

    @staticmethod
    def _prepare_feature_group_plot(
        unit_frame: pd.DataFrame,
        *,
        feature_id: str,
        group_key: str,
        value_column: str,
        stratify_key: str | None,
        filename: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select and clean one feature for a grouped association plot."""
        required = {"feature_id", group_key, value_column}
        if stratify_key is not None:
            required.add(stratify_key)
        if missing := sorted(required.difference(unit_frame.columns)):
            raise KeyError(
                f"unit frame missing grouped plot columns: {missing}"
            )

        block = unit_frame.loc[
            unit_frame["feature_id"].astype(str) == feature_id
        ]
        plot_df = pd.DataFrame(
            {
                "group": block[group_key],
                "value": pd.to_numeric(
                    block[value_column], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan),
            }
        )
        plot_df["stratum"] = (
            block[stratify_key] if stratify_key is not None else ""
        )

        invalid = plot_df[["group", "value", "stratum"]].isna().any(axis=1)
        if invalid.any():
            logger.warning(
                "[plot] Excluding %d rows with missing or non-finite grouped "
                "plot values from %s",
                int(invalid.sum()),
                filename,
            )

        plot_df = plot_df.loc[~invalid].copy()
        plot_df["group"] = plot_df["group"].astype(str)
        plot_df["stratum"] = plot_df["stratum"].astype(str)

        return block, plot_df

    @staticmethod
    def _group_plot_layout(
        plot_df: pd.DataFrame,
        *,
        central: Literal["mean", "median"],
        stratify_key: str | None,
        stratify_colors: Mapping[str, str] | None,
    ) -> tuple[list[str], list[str], dict[str, ColorType]]:
        """Resolve group order, stratum order, and stratum colors."""
        group_order = (
            plot_df.groupby("group", observed=True)["value"]
            .agg(central)
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        if stratify_key is None:
            return group_order, [""], {"": "#4c78a8"}

        available_strata = set(plot_df["stratum"].astype(str))
        requested_order = (
            [] if stratify_colors is None else list(stratify_colors)
        )
        stratum_order = [
            stratum
            for stratum in requested_order
            if stratum in available_strata
        ]
        stratum_order.extend(sorted(available_strata - set(stratum_order)))

        default_colors = plt.get_cmap("tab10")
        color_map: dict[str, ColorType] = {
            stratum: (
                stratify_colors[stratum]
                if stratify_colors is not None and stratum in stratify_colors
                else default_colors(index % 10)
            )
            for index, stratum in enumerate(stratum_order)
        }
        return group_order, stratum_order, color_map

    def _style_feature_group_axis(
        self,
        *,
        ax: Axes,
        centers: np.ndarray,
        group_order: list[str],
        group_key: str,
        feature_id: str,
        feature_label: str | None,
        block: pd.DataFrame,
        result_row: pd.Series | None,
        stratify_key: str | None,
        stratum_order: list[str],
        color_map: Mapping[str, ColorType],
    ) -> None:
        """Apply shared labels, title, ticks, and legend to a grouped plot."""
        ax.set_xticks(centers)
        ax.set_xticklabels(
            [humanize_label(group) for group in group_order], rotation=90
        )
        ax.tick_params(axis="both", which="major", pad=1.0)
        ax.set_xlabel(_group_display_name(group_key))

        resolved_feature_label = feature_label or (
            str(block["feature_label"].iloc[0])
            if "feature_label" in block.columns and not block.empty
            else feature_id
        )
        y_label = ax.set_ylabel(
            self._regression_y_label(
                feature_id=feature_id,
                feature_label=resolved_feature_label,
                result_row=result_row,
            )
        )
        if self._is_gene_expression(feature_id, result_row):
            y_label.set_fontstyle("italic")

        statistical_unit = (
            "" if block.empty else str(block["statistical_unit"].iloc[0])
        )
        ax.set_title(
            self._group_title(
                statistical_unit=statistical_unit,
                group_key=group_key,
                result_row=result_row,
            )
        )

        if stratify_key is None:
            return

        handles = [
            Patch(
                facecolor=color_map[stratum],
                edgecolor=darken_color(color_map[stratum], factor=0.70),
                linewidth=0.6,
                label=humanize_label(stratum),
            )
            for stratum in stratum_order
        ]
        ax.legend(handles=handles, frameon=False)

    @staticmethod
    def _feature_group_figure_size(
        group_order: list[str],
    ) -> tuple[float, float]:
        """Scale categorical width while retaining the tuned panel height."""
        return max(1.8, len(group_order) * 0.145), 1.5

    @staticmethod
    def _draw_regression_line(
        ax: Axes,
        result_row: pd.Series | None,
    ) -> None:
        """Overlay an OLS fit line on a regression scatter.

        Args:
          ax: Axes to draw on.
          result_row: Optional result row with `slope`/`intercept`; when
            absent or non-finite the line is omitted.
        """
        if result_row is None:
            return
        slope = float(result_row.get("slope", np.nan))
        intercept = float(result_row.get("intercept", np.nan))
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return
        x_limits = ax.get_xlim()
        x_line = np.linspace(*x_limits, 100)
        ax.plot(x_line, slope * x_line + intercept, color="#e45756", lw=0.8)
        ax.set_xlim(x_limits)

    @staticmethod
    def _regression_facet_label(
        stratum: str,
        *,
        wrap_width: int,
    ) -> str:
        """Return a compact tissue or tissue-cell-type panel label."""
        if "::" not in stratum:
            return textwrap.fill(
                humanize_label(stratum),
                width=wrap_width,
            )
        tissue, cell_type = stratum.split("::", 1)
        cell_type_label = textwrap.fill(
            humanize_label(cell_type),
            width=wrap_width,
        )
        return f"{humanize_label(tissue)}\n{cell_type_label}"

    @staticmethod
    def _result_is_skipped(result_row: pd.Series) -> bool:
        """Return whether a serialized result row is non-estimable."""
        return str(result_row.get("skipped", "false")).lower() in {
            "true",
            "1",
            "yes",
        }

    @staticmethod
    def _regression_title(
        *,
        statistical_unit: str,
        result_row: pd.Series | None,
    ) -> str:
        """Build a contextual regression plot title.

        Args:
          statistical_unit: Statistical unit of the plotted points.
          result_row: Optional result row supplying analysis context.

        Returns:
          A compact biological-context title.
        """
        if result_row is not None:
            stratum = result_row.get("stratum")
            if pd.notna(stratum) and str(stratum):
                stratum_text = str(stratum)
                stratify_key = str(result_row.get("stratify_key", ""))
                analysis_scope = str(result_row.get("analysis_scope", ""))
                if "cell_type" in stratify_key or "cell_type" in analysis_scope:
                    return humanize_label(stratum_text.rsplit("::", 1)[-1])
                if "sex" in stratify_key or "sex" in analysis_scope:
                    return f"{humanize_label(stratum_text)} donors"
                return humanize_label(stratum_text)
            if str(result_row.get("analysis_scope", "")) == "global_donor":
                return "All donors"
        return humanize_label(statistical_unit)

    @staticmethod
    def _regression_annotation(result_row: pd.Series | None) -> str:
        """Return a compact correlation annotation when values are finite."""
        if result_row is None:
            return ""
        pearson = _numeric_scalar(result_row.get("pearson_r"))
        pvalue = _numeric_scalar(result_row.get("pearson_pvalue"))
        if (
            pd.notna(pearson)
            and pd.notna(pvalue)
            and np.isfinite(float(pearson))
            and np.isfinite(float(pvalue))
        ):
            return f"r={float(pearson):.2f}, p={float(pvalue):.1e}"
        return ""

    @staticmethod
    def _regression_y_label(
        *,
        feature_id: str,
        feature_label: str,
        result_row: pd.Series | None,
    ) -> str:
        """Return a readable feature label with module-scoring provenance."""
        if AssociationPlotter._is_gene_expression(feature_id, result_row):
            return feature_label.strip().upper()

        label = feature_label.replace("_", " ").strip()
        if label.isupper():
            label = label.lower()
        replacements = {
            "nasp": "NASP",
            "dna": "DNA",
            "rna": "RNA",
            "ifn": "IFN",
            "nfkb": "NF-κB",
            "aucell": "AUCell",
        }
        label = " ".join(
            replacements.get(token.lower(), token) for token in label.split()
        )
        feature_type = (
            ""
            if result_row is None
            else str(result_row.get("feature_type", ""))
        )
        is_module = feature_type == "module_score" or feature_id.endswith(
            ("_score", "_auc")
        )
        if not is_module:
            return label
        if not label.lower().endswith(" score"):
            label += " score"
        if feature_id.endswith("_auc"):
            return f"{label}\n(AUCell)"
        return f"{label}\n(Scanpy)" if feature_id.endswith("_score") else label

    @staticmethod
    def _is_gene_expression(
        feature_id: str,
        result_row: pd.Series | None,
    ) -> bool:
        """Return whether a plotted feature represents a human gene."""
        feature_type = (
            ""
            if result_row is None
            else str(result_row.get("feature_type", ""))
        )
        return (
            feature_type == "gene_expression"
            or feature_id.upper().startswith("ENSG")
        )

    @staticmethod
    def _group_title(
        *,
        statistical_unit: str,
        group_key: str,
        result_row: pd.Series | None,
    ) -> str:
        """Build a grouped plot title.

        Args:
          statistical_unit: Statistical unit of the plotted points.
          group_key: Categorical comparison represented on the x axis.
          result_row: Optional group-test result row.

        Returns:
          A compact title identifying the unit and optional raw test p-value.
        """
        title = _statistical_unit_display_name(statistical_unit)
        if result_row is None:
            return title
        test = str(result_row.get("parametric_test", ""))
        pvalue = _numeric_scalar(result_row.get("pvalue"))
        test_labels = {
            "anova": "ANOVA",
            "welch_t": "Welch's t-test",
        }
        if test and pd.notna(pvalue) and np.isfinite(float(pvalue)):
            comparison = _group_display_name(group_key).lower()
            title += (
                f"\n{test_labels.get(test, humanize_label(test))} across "
                f"{comparison}, raw p={float(pvalue):.1e}"
            )
        return title
