"""Grouped score summaries and annotation distributions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore[import]
from scipy import stats  # type: ignore[import]

from nasp_atlas.single_cell.visualization.style import _VisualizationStyleMixin


logger = logging.getLogger(__name__)


class _SummaryPlotMixin(_VisualizationStyleMixin):
    """Grouped score bars and annotation distributions."""

    def plot_grouped_obs_score_barplot(
        self,
        adata: Any,
        *,
        score_key: str,
        groupby: str,
        filename: str,
        stat_test: Literal["anova", "kruskal"] = "anova",
        color: str = "#4c78a8",
        figsize: tuple[float, float] | None = None,
        bar_width: float = 0.75,
        errorbar: Literal["sem", "sd", "none"] = "sem",
        min_group_size: int = 2,
    ) -> pd.DataFrame:
        """Plot one obs score as ordered bars across obs groups.

        Args:
          adata: AnnData object containing the score and grouping columns in
            obs.
          score_key: Numeric obs score column to summarize on the y axis.
          groupby: Obs column used for x-axis groups, such as cell type or
            tissue.
          filename: Output filename under output_dir.
          stat_test: Omnibus test for group differences. "anova" uses Welch's
            t-test for two groups and one-way ANOVA for three or more;
            "kruskal" uses Kruskal-Wallis for three or more groups.
          color: Bar face color.
          figsize: Optional figure size in inches.
          bar_width: Width of each bar.
          errorbar: Error bar type: standard error, standard deviation, or none.
          min_group_size: Minimum non-null observations required per group for
            inclusion in the omnibus test.

        Returns:
          Per-group summary table in plotted order.

        Example Usage:
          >>> summary = viz.plot_grouped_obs_score_barplot(
          ...     adata,
          ...     score_key="senescence_score",
          ...     groupby="cell_type",
          ...     filename="senescence_by_cell_type",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        if groupby not in adata.obs.columns:
            raise KeyError(f"obs column not found for barplot: {groupby}")
        if score_key not in adata.obs.columns:
            raise KeyError(f"score column not found for barplot: {score_key}")

        summary, groups = self._group_obs_score_bar_stats(
            adata=adata,
            score_key=score_key,
            groupby=groupby,
        )
        if summary.empty:
            logger.warning(
                "[plot] No valid groups found for score barplot %s. Skipping.",
                filename,
            )
            return summary

        p_value = self._score_group_p_value(
            groups=groups,
            stat_test=stat_test,
            min_group_size=min_group_size,
        )
        x = np.arange(summary.shape[0])
        yerr = None
        if errorbar == "sem":
            yerr = summary["sem"].to_numpy(dtype=float)
        elif errorbar == "sd":
            yerr = summary["std"].to_numpy(dtype=float)

        if figsize is None:
            figsize = (max(1.8, summary.shape[0] * 0.28), 1.8)
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(
            x,
            summary["mean"].to_numpy(dtype=float),
            yerr=yerr,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            error_kw={"elinewidth": 0.4, "capsize": 1.5, "capthick": 0.4},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            summary.index.tolist(),
            rotation=90,
            ha="center",
            va="top",
            color="black",
        )
        ax.set_xlabel("")
        ax.set_ylabel(score_key)
        if p_value is not None:
            ax.set_title(f"{stat_test} p={p_value:.2e}", pad=2)

        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", length=2, pad=1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.25)

        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] grouped score barplot -> %s",
        )
        return summary

    def plot_annotation_violins(
        self,
        score: Any,
        *,
        leiden_key: str,
        score_keys: list[str],
        filename: str,
        n_rows: int = 2,
        n_cols: int = 3,
        ref_n_groups: int = 15,
        ref_ax_w: float = 2.0,
        ref_ax_h: float = 0.8,
    ) -> None:
        """Save violin plots of per-cluster enrichment scores.

        Args:
          score: Per-cell score AnnData with Leiden cluster labels in obs.
          leiden_key: Obs key for Leiden clustering.
          score_keys: Cell-type score column names to plot.
          filename: Output filename under output_dir.
          n_rows: Number of subplot rows.
          n_cols: Number of subplot columns.
          ref_n_groups: Reference cluster count for axis-width scaling.
          ref_ax_w: Reference axis width in inches at ref_n_groups clusters.
          ref_ax_h: Height per subplot row in inches.

        Example Usage:
          >>> viz.plot_annotation_violins(
          ...     score_adata,
          ...     leiden_key="leiden",
          ...     score_keys=["T_cell_score", "B_cell_score"],
          ...     filename="annotation_violins",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename

        n_groups = score.obs[leiden_key].nunique()
        ax_w = ref_ax_w * (n_groups / ref_n_groups)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(ax_w * n_cols, ref_ax_h * n_rows),
            sharey=False,
        )

        for ax, key in zip(axes.flat, score_keys, strict=False):
            sc.pl.violin(
                adata=score,
                keys=key,
                groupby=leiden_key,
                rotation=90,
                show=False,
                size=0.2,  # type: ignore[arg-type]  # Scanpy stub requires int
                linewidth=0.25,
                ax=ax,
            )

        for ax in axes.flat[len(score_keys) :]:
            ax.axis("off")

        fig.tight_layout()
        self._save_figure_and_log(fig, out, "[plot] annotation violins -> %s")

    @staticmethod
    def _group_obs_score_bar_stats(
        *,
        adata: Any,
        score_key: str,
        groupby: str,
    ) -> tuple[pd.DataFrame, list[pd.Series]]:
        """Return ordered score summaries and per-group score vectors."""
        obs = cast(pd.DataFrame, adata.obs)
        plot_df = pd.DataFrame(
            {
                groupby: obs[groupby].astype(str),
                score_key: pd.to_numeric(obs[score_key], errors="coerce"),
            }
        ).dropna()
        if plot_df.empty:
            return pd.DataFrame(), []

        grouped = plot_df.groupby(groupby, observed=True)[score_key]
        summary = grouped.agg(["mean", "std", "count"])
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])
        summary = summary.sort_values("mean", ascending=False)
        summary[["std", "sem"]] = summary[["std", "sem"]].fillna(0.0)
        groups = [
            grouped.get_group(group).astype(float)
            for group in summary.index.tolist()
        ]
        return summary, groups

    @staticmethod
    def _score_group_p_value(
        *,
        groups: Sequence[pd.Series],
        stat_test: Literal["anova", "kruskal"],
        min_group_size: int,
    ) -> float | None:
        """Return an omnibus p-value for grouped score distributions."""
        arrays = [
            group.dropna().to_numpy(dtype=float)
            for group in groups
            if group.dropna().shape[0] >= min_group_size
        ]
        if len(arrays) < 2:
            return None
        if len(arrays) == 2:
            ttest = cast(
                tuple[float, float],
                stats.ttest_ind(
                    arrays[0],
                    arrays[1],
                    equal_var=False,
                    nan_policy="omit",
                ),
            )
            return ttest[1]
        if stat_test == "kruskal":
            kruskal = cast(
                tuple[float, float],
                stats.kruskal(*arrays, nan_policy="omit"),
            )
            return kruskal[1]
        anova = cast(tuple[float, float], stats.f_oneway(*arrays))
        return anova[1]
