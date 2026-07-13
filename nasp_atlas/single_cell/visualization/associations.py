"""Regression and association plotting methods."""

from __future__ import annotations

import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from nasp_atlas.single_cell.visualization.style import _VisualizationStyleMixin


logger = logging.getLogger(__name__)


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
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Scatter one feature against a continuous predictor with a fit line.

        The frame is expected to be aggregated to a statistical unit (one point
        per donor/donor_tissue/etc.), so the plot shows donor-level rather than
        cell-level structure. The title records the statistical unit and, when a
        result row is supplied, the correlation and its p-value.

        Args:
          unit_frame: Aggregated unit frame filtered or filterable to the
            feature, carrying the predictor and value columns.
          feature_id: Feature id to select from `unit_frame`.
          predictor_key: Continuous predictor column for the x axis.
          filename: Output filename stem under `output_dir`.
          result_row: Optional regression result row supplying r, p-value, slope
            and intercept for the annotation and fit line.
          color_key: Optional column used to color points by category.
          value_column: Column holding the feature value for the y axis.
          feature_label: Optional display label; defaults to `feature_id`.
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

        fig, ax = plt.subplots(figsize=figsize or (2.2, 2.0))
        if "color" in plot_df.columns:
            for category in sorted(plot_df["color"].unique()):
                sub = plot_df[plot_df["color"] == category]
                ax.scatter(sub["x"], sub["y"], s=6, label=str(category))
            ax.legend(frameon=False, loc="best")
        else:
            ax.scatter(plot_df["x"], plot_df["y"], s=6, color="#4c78a8")

        unit = "" if block.empty else str(block["statistical_unit"].iloc[0])
        self._draw_regression_line(
            ax=ax, plot_df=plot_df, result_row=result_row
        )
        ax.set_xlabel(predictor_key)
        ax.set_ylabel(feature_label or feature_id)
        ax.set_title(
            self._regression_title(
                feature_label=feature_label or feature_id,
                statistical_unit=unit,
                result_row=result_row,
            )
        )
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
        errorbar: Literal["sem", "sd", "none"] = "sem",
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Bar the central value per group with error bars and unit points.

        Bars show the per-group central value at the supplied statistical unit
        (donor-collapsed when the frame is donor-level), individual unit values
        are overlaid as points, and the title records the unit and, when a
        result row is supplied, the omnibus test and its p-value.

        Args:
          unit_frame: Aggregated unit frame carrying the value and group
            columns for the feature.
          feature_id: Feature id to select from `unit_frame`.
          group_key: Categorical column defining the x-axis groups.
          filename: Output filename stem under `output_dir`.
          result_row: Optional group-test result row for the title annotation.
          value_column: Column holding the feature value.
          feature_label: Optional display label; defaults to `feature_id`.
          central: Central tendency drawn as the bar height.
          errorbar: Error bar type: standard error, standard deviation or none.
          figsize: Optional figure size in inches.

        Example Usage:
          >>> viz.plot_feature_group_boxplot(
          ...     unit_frame,
          ...     feature_id="IL6",
          ...     group_key="disease_status",
          ...     filename="il6_by_disease",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        block = unit_frame[unit_frame["feature_id"].astype(str) == feature_id]
        plot_df = pd.DataFrame(
            {
                "group": block[group_key].astype(str),
                "value": pd.to_numeric(block[value_column], errors="coerce"),
            }
        ).dropna()
        if plot_df.empty:
            logger.warning(
                "[plot] No valid points for boxplot %s. Skipping.", filename
            )
            return

        grouped = plot_df.groupby("group", observed=True)["value"]
        table = grouped.agg(["mean", "median", "std", "count"])
        table["sem"] = table["std"] / np.sqrt(table["count"])
        table[["std", "sem"]] = table[["std", "sem"]].fillna(0.0)
        table = table.sort_values(central, ascending=False)

        x = np.arange(table.shape[0])
        yerr = None
        if errorbar == "sem":
            yerr = table["sem"].to_numpy(dtype=float)
        elif errorbar == "sd":
            yerr = table["std"].to_numpy(dtype=float)

        fig, ax = plt.subplots(
            figsize=figsize or (max(1.8, table.shape[0] * 0.3), 1.9)
        )
        ax.bar(
            x,
            table[central].to_numpy(dtype=float),
            yerr=yerr,
            width=0.72,
            color="#4c78a8",
            error_kw={"elinewidth": 0.5},
        )
        for position, group_name in enumerate(table.index.tolist()):
            values = grouped.get_group(group_name).to_numpy(dtype=float)
            jitter = (np.random.default_rng(0).random(values.size) - 0.5) * 0.3
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=3,
                color="#333333",
                alpha=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(table.index.tolist(), rotation=90)
        ax.set_ylabel(feature_label or feature_id)
        unit = "" if block.empty else str(block["statistical_unit"].iloc[0])
        ax.set_title(
            self._group_title(
                feature_label=feature_label or feature_id,
                statistical_unit=unit,
                result_row=result_row,
            )
        )
        self._save_figure_and_log(
            fig, self.output_dir / filename, "[plot] feature boxplot -> %s"
        )

    @staticmethod
    def _draw_regression_line(
        ax: Axes,
        plot_df: pd.DataFrame,
        result_row: pd.Series | None,
    ) -> None:
        """Overlay an OLS fit line on a regression scatter.

        Args:
          ax: Axes to draw on.
          plot_df: Frame with `x` and `y` columns for the fit range.
          result_row: Optional result row with `slope`/`intercept`; when
            absent or non-finite the line is omitted.
        """
        if result_row is None:
            return
        slope = float(result_row.get("slope", np.nan))
        intercept = float(result_row.get("intercept", np.nan))
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return
        x_line = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#e45756", lw=0.8)

    @staticmethod
    def _regression_title(
        *,
        feature_label: str,
        statistical_unit: str,
        result_row: pd.Series | None,
    ) -> str:
        """Build a regression plot title.

        Args:
          feature_label: Feature display label.
          statistical_unit: Statistical unit of the plotted points.
          result_row: Optional result row supplying r and p-value.

        Returns:
          A compact one-line title.
        """
        title = f"{feature_label} (unit={statistical_unit})"
        if result_row is None:
            return title
        pearson = result_row.get("pearson_r", np.nan)
        pvalue = result_row.get("pearson_pvalue", np.nan)
        if np.isfinite(float(pearson)) and np.isfinite(float(pvalue)):
            title += f"\nr={float(pearson):.2f}, p={float(pvalue):.1e}"
        return title

    @staticmethod
    def _group_title(
        *,
        feature_label: str,
        statistical_unit: str,
        result_row: pd.Series | None,
    ) -> str:
        """Build a grouped plot title.

        Args:
          feature_label: Feature display label.
          statistical_unit: Statistical unit of the plotted points.
          result_row: Optional group-test result row.

        Returns:
          A compact one-line title.
        """
        title = f"{feature_label} (unit={statistical_unit})"
        if result_row is None:
            return title
        test = str(result_row.get("parametric_test", ""))
        pvalue = result_row.get("pvalue", np.nan)
        if test and np.isfinite(float(pvalue)):
            title += f"\n{test} p={float(pvalue):.1e}"
        return title
