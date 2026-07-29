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

from nasp_atlas.single_cell.visualization.style import ColorbarStyle
from nasp_atlas.single_cell.visualization.style import _VisualizationStyleMixin


logger = logging.getLogger(__name__)


class _SummaryPlotMixin(_VisualizationStyleMixin):
    """Grouped score bars and annotation distributions."""

    def plot_scorer_concordance_heatmap(
        self,
        concordance: pd.DataFrame,
        *,
        filename: str,
        statistic: str = "spearman_r",
        module_order: Sequence[str] | None = None,
        cell_size: float = 0.0975,
        min_width: float = 1.5,
        min_height: float = 1.5,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        cbar_pad: float | None = None,
        annotate: bool = False,
    ) -> None:
        """Plot all Scanpy-module/AUCell-module score correlations.

        Scanpy modules are columns and AUCell modules are rows. The same module
        order is used on both axes, placing matched modules on the diagonal.
        Off-diagonal values show cross-module score correlation, which may
        reflect shared genes or biological covariation rather than scorer
        disagreement.

        Args:
          concordance: Output from `cross_scorer_module_correlations`, with one
            row per Scanpy-module/AUCell-module pair.
          filename: Output filename stem under `output_dir`.
          statistic: Correlation statistic to display.
          module_order: Optional shared module order for both axes.
          cell_size: Width and height of each heatmap cell in inches.
          min_width: Minimum heatmap panel width in inches.
          min_height: Minimum heatmap panel height in inches.
          colorbar_style: Explicit inset colorbar geometry.
          cbar_height: Optional colorbar-height override.
          cbar_width: Optional colorbar-width override.
          cbar_pad: Optional heatmap-to-colorbar padding override.
          annotate: Whether to print correlation values in heatmap cells.

        Example Usage:
          >>> viz.plot_scorer_concordance_heatmap(
          ...     concordance,
          ...     filename="scorer_concordance",
          ... )
        """
        self._set_matplotlib_publication_parameters()

        required = {"scanpy_module_id", "aucell_module_id", statistic}
        if missing := sorted(required.difference(concordance.columns)):
            raise KeyError(f"concordance missing required columns: {missing}")

        table = concordance.loc[
            :, ["scanpy_module_id", "aucell_module_id", statistic]
        ].copy()
        table["scanpy_module_id"] = table["scanpy_module_id"].astype(str)
        table["aucell_module_id"] = table["aucell_module_id"].astype(str)
        table[statistic] = pd.to_numeric(table[statistic], errors="coerce")
        if table.empty or table[statistic].notna().sum() == 0:
            logger.warning(
                "[plot] No finite scorer concordance for %s. Skipping.",
                filename,
            )
            return

        pair_columns = ["scanpy_module_id", "aucell_module_id"]
        if table.duplicated(pair_columns).any():
            raise ValueError(
                "concordance contains duplicate Scanpy/AUCell module pairs"
            )

        available_modules = set(table["scanpy_module_id"].astype(str)) | set(
            table["aucell_module_id"].astype(str)
        )
        if module_order is None:
            order = sorted(available_modules)
        else:
            order = [
                str(module_id)
                for module_id in dict.fromkeys(module_order)
                if str(module_id) in available_modules
            ]

        if not order:
            raise ValueError("module_order contains no available modules")

        matrix = table.pivot(
            index="aucell_module_id",
            columns="scanpy_module_id",
            values=statistic,
        ).reindex(index=order, columns=order)
        values = matrix.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)

        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        panel_width = max(min_width, len(order) * cell_size)
        panel_height = max(min_height, len(order) * cell_size)
        fig, ax = plt.subplots(figsize=(panel_width, panel_height))
        image = ax.imshow(
            masked,
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_box_aspect(1.0)

        labels = [self._scorer_concordance_label(value) for value in order]
        self._style_score_heatmap_axis(
            ax=ax,
            score_labels=labels,
            groups=labels,
        )
        ax.set_xlabel("Scanpy", labelpad=6.5)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("AUCell", rotation=270, labelpad=9.5)
        ax.yaxis.set_label_position("right")

        if annotate:
            for row_index, column_index in np.ndindex(values.shape):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    text_color = "white" if abs(value) >= 0.55 else "black"
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                    )

        resolved_colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.4, width=0.08, pad=0.12)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )

        self._add_embedding_colorbar(
            fig,
            ax,
            resolved_colorbar_style,
            mappable=image,
            title="Cell-level Spearman correlation",
        )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] scorer concordance heatmap -> %s",
        )

    @staticmethod
    def _scorer_concordance_label(value: object) -> str:
        """Return a compact readable module label."""
        label = str(value)
        known_labels = {
            "CGAMP_TRANSPORT": "cGAMP transport",
            "IFN_I_OUTPUT": "IFN-I output",
            "NFKB_CYTOKINE_OUTPUT": "NF-κB cytokine output",
        }
        if label in known_labels:
            return known_labels[label]
        label = label.replace("_", " ").lower()
        replacements = {
            "nasp": "NASP",
            "dna": "DNA",
            "rna": "RNA",
            "ifn": "IFN",
            "nfkb": "NF-κB",
            "isr": "ISR",
            "sasp": "SASP",
            "jak": "JAK",
            "stat": "STAT",
            "tbk1": "TBK1",
            "irf": "IRF",
            "tlr": "TLR",
            "na": "NA",
        }
        return " ".join(
            replacements.get(token, token) for token in label.split()
        )

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
