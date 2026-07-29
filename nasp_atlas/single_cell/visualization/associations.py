"""Regression and association plotting methods."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.typing import ColorType

from nasp_atlas.cellxgene.filter import humanize_label
from nasp_atlas.single_cell.visualization.style import _VisualizationStyleMixin
from nasp_atlas.visualization import darken_color


logger = logging.getLogger(__name__)

_PREDICTOR_DISPLAY_NAMES = {
    "age_years": "Age (years)",
    "tissue_specific_eqtls": "Tissue-specific significant eQTL pairs",
    "total_eqtls": "Total significant eQTL pairs",
}
_GROUP_DISPLAY_NAMES = {
    "cell_type": "Cell type",
    "sex": "Sex",
    "tissue_in_publication": "Tissue",
}
_STATISTICAL_UNIT_DISPLAY_NAMES = {
    "cell": "Cells",
    "metacell": "Metacells",
    "donor": "Donors",
    "donor_tissue": "Donor-tissue units",
    "donor_tissue_cell_type": "Donor-tissue-cell type units",
    "donor_tissue_sex": "Donor-tissue-sex units",
}


def _numeric_scalar(value: object) -> float:
    """Return a numeric scalar or NaN for an unparseable value."""
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return np.nan


class _AssociationPlotMixin(_VisualizationStyleMixin):
    """Regression and association plotting methods."""

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
          figsize: Optional figure size in inches.

        Example Usage:
          >>> viz.plot_feature_regression(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     predictor_key="age_years",
          ...     filename="il6_age_regression",
          ... )
        """
        self._set_matplotlib_publication_parameters()
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
            for category in sorted(plot_df["color"].unique()):
                sub = plot_df[plot_df["color"] == category]
                ax.scatter(sub["x"], sub["y"], s=6, label=str(category))
            ax.legend(frameon=False, loc="best")
        else:
            ax.scatter(
                plot_df["x"],
                plot_df["y"],
                s=3,
                color=point_color or "black",
            )

        unit = "" if block.empty else str(block["statistical_unit"].iloc[0])
        self._draw_regression_line(ax=ax, result_row=result_row)
        ax.set_xlabel(humanize_label(predictor_key, _PREDICTOR_DISPLAY_NAMES))
        ax.set_ylabel(
            self._regression_y_label(
                feature_id=feature_id,
                feature_label=feature_label or feature_id,
                result_row=result_row,
            )
        )
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
          x_margin: Fractional whitespace at each end of the x axis.
          y_margin: Fractional whitespace at each end of the y axis.
          figsize: Optional figure size in inches.

        Example Usage:
          >>> viz.plot_feature_group_boxplot(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     group_key="disease_status",
          ...     filename="il6_by_disease",
          ...     box_width=0.25,
          ... )
        """
        self._set_matplotlib_publication_parameters()
        if errorbar is not None:
            logger.warning(
                "[plot] errorbar=%s is ignored for true boxplots", errorbar
            )
        if box_width is not None and box_width <= 0:
            raise ValueError("box_width must be greater than zero")

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
            slot_width * 1.25 if box_width is None else box_width
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
                    s=3,
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
          figsize: Optional figure size in inches.

        Example Usage:
          >>> viz.plot_feature_group_barplot(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     group_key="tissue",
          ...     filename="il6_by_tissue",
          ... )
        """
        self._set_matplotlib_publication_parameters()

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
        bar_width = slot_width * 0.86
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
                width=bar_width,
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
                jitter = (rng.random(values.size) - 0.5) * bar_width * 0.55
                ax.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=3,
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
        ax.set_xlabel(humanize_label(group_key, _GROUP_DISPLAY_NAMES))

        resolved_feature_label = feature_label or (
            str(block["feature_label"].iloc[0])
            if "feature_label" in block.columns and not block.empty
            else feature_id
        )
        ax.set_ylabel(
            self._regression_y_label(
                feature_id=feature_id,
                feature_label=resolved_feature_label,
                result_row=result_row,
            )
        )

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
        title = _STATISTICAL_UNIT_DISPLAY_NAMES.get(
            statistical_unit, humanize_label(statistical_unit)
        )
        if result_row is None:
            return title
        test = str(result_row.get("parametric_test", ""))
        pvalue = _numeric_scalar(result_row.get("pvalue"))
        test_labels = {
            "anova": "ANOVA",
            "welch_t": "Welch's t-test",
        }
        if test and pd.notna(pvalue) and np.isfinite(float(pvalue)):
            comparison = humanize_label(group_key, _GROUP_DISPLAY_NAMES).lower()
            title += (
                f"\n{test_labels.get(test, humanize_label(test))} across "
                f"{comparison}, raw p={float(pvalue):.1e}"
            )
        return title
