"""Heatmap plotting methods."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore[import]
from matplotlib.colors import Colormap

from nasp_atlas.single_cell.visualization.gene_resolution import (
    _VisualizationGeneMixin,
)
from nasp_atlas.single_cell.visualization.style import ColorbarStyle
from nasp_atlas.single_cell.visualization.style import _VisualizationStyleMixin


logger = logging.getLogger(__name__)


class _HeatmapMixin(_VisualizationGeneMixin, _VisualizationStyleMixin):
    """Grouped gene-expression and observation-score heatmaps."""

    def plot_multi_gene_expression_heatmap(
        self,
        adata: Any,
        genes: list[str],
        *,
        groupby: str,
        filename: str,
        cmap: Colormap | None = None,
        expression_layer: str | None = None,
        use_raw: bool = False,
        gene_symbol_column: str | None = None,
        obs_order: Sequence[str] | None = None,
        cell_size: float = 0.0975,
        min_width: float = 1.5,
        min_height: float = 1.5,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        cbar_pad: float | None = None,
        cbar_title: str | None = None,
        vmin: float | None = 0,
        vmax: float | None = None,
    ) -> None:
        """Save mean gene expression as a grouped square-cell heatmap.

        Args:
          adata: AnnData object containing expression values.
          genes: Gene names to plot.
          groupby: Observation column used for heatmap rows.
          filename: Output filename under output_dir.
          cmap: Colormap for expression values.
          expression_layer: Layer to use for expression values.
          use_raw: Whether to use adata.raw when expression_layer is None.
          gene_symbol_column: Optional adata.var column used to resolve symbols.
          obs_order: Optional ordered subset of group labels for heatmap rows.
          cell_size: Width and height of each heatmap cell in inches.
          min_width: Minimum heatmap panel width in inches.
          min_height: Minimum heatmap panel height in inches.
          colorbar_style: Base heatmap colorbar styling. Defaults to the
            heatmap-tuned inset colorbar style.
          cbar_height: Optional override for colorbar height.
          cbar_width: Optional override for colorbar width.
          cbar_pad: Optional override for heatmap-to-colorbar padding.
          cbar_title: Optional title drawn above the heatmap colorbar.
          vmin: Lower color limit. Defaults to 0 so zero maps to gray when
            using `umap_expression_cmap`.
          vmax: Upper color limit.

        Example Usage:
          >>> viz.plot_multi_gene_expression_heatmap(
          ...     adata,
          ...     genes=["CD3D", "MS4A1"],
          ...     groupby="cell_type",
          ...     filename="marker_heatmap",
          ...     gene_symbol_column="gene_symbol",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename
        colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.36, width=0.07, pad=0.02)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )
        cmap = self.zero_gray_cmap(
            cmap if cmap is not None else self.expression_cmap,
            zero_position=self._zero_cmap_position(vmin=vmin, vmax=vmax),
        )
        if expression_layer is not None and use_raw:
            raise ValueError(
                "use_raw=True cannot be combined with expression_layer"
            )
        if groupby not in adata.obs.columns:
            raise KeyError(f"obs column not found for heatmap: {groupby}")

        resolved = self._resolve_genes(
            adata,
            genes,
            gene_symbol_column=gene_symbol_column,
        )
        if not resolved.var_names:
            logger.warning(
                "[plot] No valid genes found for %s. Skipping.", filename
            )
            return

        expr_kwargs: dict[str, Any] = {"layer": expression_layer}
        if expression_layer is None:
            expr_kwargs["use_raw"] = use_raw
        expr_df = sc.get.obs_df(
            adata,
            keys=[*resolved.var_names, groupby],
            **expr_kwargs,
        )
        grouped_expression = self._group_gene_expression_by_obs(
            adata=adata,
            expr_df=expr_df,
            var_names=resolved.var_names,
            labels=resolved.labels,
            groupby=groupby,
            obs_order=obs_order,
        )
        if grouped_expression.empty:
            logger.warning(
                "[plot] No valid groups found for heatmap %s. Skipping.",
                filename,
            )
            return

        n_genes, n_groups = grouped_expression.shape
        heatmap_values = grouped_expression.T.to_numpy(dtype=float)
        panel_w = max(min_width, n_genes * cell_size)
        panel_h = max(min_height, n_groups * cell_size)
        fig, ax = plt.subplots(figsize=(panel_w, panel_h))
        image = ax.imshow(
            heatmap_values,
            aspect="equal",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_box_aspect(n_groups / n_genes)
        self._style_gene_expression_heatmap_axis(
            ax=ax,
            genes=grouped_expression.index.tolist(),
            groups=grouped_expression.columns.tolist(),
        )
        self._add_embedding_colorbar(
            fig=fig,
            ax=ax,
            mappable=image,
            colorbar_style=colorbar_style,
            title=cbar_title,
        )

        self._save_figure_and_log(
            fig, out, "[plot] multi-gene expression heatmap -> %s"
        )

    def plot_grouped_obs_score_heatmap(
        self,
        adata: Any,
        score_keys: Sequence[str],
        *,
        groupby: str,
        filename: str,
        score_labels: Sequence[str] | None = None,
        cmap: Colormap | str = "RdBu_r",
        obs_order: Sequence[str] | None = None,
        cell_size: float = 0.0975,
        min_width: float = 1.5,
        min_height: float = 1.5,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        cbar_pad: float | None = None,
        cbar_title: str | None = "Mean score",
        vmin: float | None = None,
        vmax: float | None = None,
        center_zero: bool = True,
    ) -> None:
        """Save mean observation scores as a grouped heatmap.

        Args:
          adata: AnnData object containing score columns in obs.
          score_keys: Numeric obs score columns to aggregate.
          groupby: Observation column used for heatmap rows.
          filename: Output filename under output_dir.
          score_labels: Optional labels for score columns. Defaults to
            `score_keys`.
          cmap: Colormap for score values.
          obs_order: Optional ordered subset of group labels for heatmap rows.
          cell_size: Width and height of each heatmap cell in inches.
          min_width: Minimum heatmap panel width in inches.
          min_height: Minimum heatmap panel height in inches.
          colorbar_style: Base heatmap colorbar styling. Defaults to the
            heatmap-tuned inset colorbar style.
          cbar_height: Optional override for colorbar height.
          cbar_width: Optional override for colorbar width.
          cbar_pad: Optional override for heatmap-to-colorbar padding.
          cbar_title: Optional vertical colorbar label.
          vmin: Lower color limit. When omitted with center_zero=True, the
            lower limit is the negative maximum absolute grouped score.
          vmax: Upper color limit. When omitted with center_zero=True, the
            upper limit is the maximum absolute grouped score.
          center_zero: Whether to derive symmetric color limits around zero
            when explicit limits are not supplied.

        Example Usage:
          >>> viz.plot_grouped_obs_score_heatmap(
          ...     adata,
          ...     score_keys=["senescence_score", "immune_score"],
          ...     groupby="cell_type",
          ...     filename="score_heatmap",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename
        colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.36, width=0.07, pad=0.02)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )
        if groupby not in adata.obs.columns:
            raise KeyError(f"obs column not found for score heatmap: {groupby}")

        valid_score_keys = [key for key in score_keys if key in adata.obs]
        if not valid_score_keys:
            logger.warning(
                "[plot] No valid score keys found for heatmap %s. Skipping.",
                filename,
            )
            return

        labels = (
            list(score_labels)
            if score_labels is not None
            else list(valid_score_keys)
        )
        if len(labels) != len(valid_score_keys):
            raise ValueError("score_labels must match score_keys length")

        grouped_scores = self._group_obs_scores_by_obs(
            adata=adata,
            score_keys=valid_score_keys,
            labels=labels,
            groupby=groupby,
            obs_order=obs_order,
        )
        if grouped_scores.empty:
            logger.warning(
                "[plot] No valid groups found for score heatmap %s. Skipping.",
                filename,
            )
            return

        values = grouped_scores.to_numpy(dtype=float)
        if center_zero and (vmin is None or vmax is None):
            finite = np.abs(values[np.isfinite(values)])
            limit = float(finite.max()) if finite.size else 1.0
            if limit <= 0.0:
                limit = 1.0
            if vmin is None:
                vmin = -limit
            if vmax is None:
                vmax = limit
        cmap = self.zero_gray_cmap(
            cmap,
            zero_position=self._zero_cmap_position(vmin=vmin, vmax=vmax),
        )

        n_groups, n_scores = grouped_scores.shape
        panel_w = max(min_width, n_scores * cell_size)
        panel_h = max(min_height, n_groups * cell_size)
        fig, ax = plt.subplots(figsize=(panel_w, panel_h))
        image = ax.imshow(
            values,
            aspect="equal",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_box_aspect(n_groups / n_scores)
        self._style_score_heatmap_axis(
            ax=ax,
            score_labels=grouped_scores.columns.tolist(),
            groups=grouped_scores.index.tolist(),
        )
        self._add_embedding_colorbar(
            fig=fig,
            ax=ax,
            mappable=image,
            colorbar_style=colorbar_style,
            title=cbar_title,
        )

        self._save_figure_and_log(fig, out, "[plot] score heatmap -> %s")

    @staticmethod
    def _group_gene_expression_by_obs(
        *,
        adata: Any,
        expr_df: pd.DataFrame,
        var_names: list[str],
        labels: list[str],
        groupby: str,
        obs_order: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return mean expression with genes on rows and obs groups."""
        expr_df = expr_df.copy()
        expr_df[groupby] = expr_df[groupby].astype(str)

        if obs_order is not None:
            categories = [str(category) for category in obs_order]
        elif hasattr(adata.obs[groupby], "cat"):
            categories = [
                str(category) for category in adata.obs[groupby].cat.categories
            ]
        else:
            categories = sorted(
                str(category)
                for category in pd.Series(expr_df[groupby]).dropna().unique()
            )

        mean_exp = expr_df.groupby(groupby, observed=True)[var_names].mean()
        categories = [
            category for category in categories if category in mean_exp.index
        ]
        if not categories:
            return pd.DataFrame(index=labels)

        gene_means = expr_df[var_names].mean(axis=0)
        ordered_var_names = gene_means.sort_values(ascending=False).index
        label_lookup = dict(zip(var_names, labels, strict=True))
        ordered_labels = [
            label_lookup[var_name] for var_name in ordered_var_names
        ]

        grouped = mean_exp.loc[categories, ordered_var_names].T
        grouped.index = ordered_labels
        return grouped

    @staticmethod
    def _group_obs_scores_by_obs(
        *,
        adata: Any,
        score_keys: list[str],
        labels: list[str],
        groupby: str,
        obs_order: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return mean obs scores with groups on rows and scores on columns."""
        obs = cast(pd.DataFrame, adata.obs)
        score_df = obs.loc[:, [*score_keys, groupby]].copy()
        score_df[groupby] = score_df[groupby].astype(str)

        if obs_order is not None:
            categories = [str(category) for category in obs_order]
        elif hasattr(obs[groupby], "cat"):
            categories = [
                str(category) for category in obs[groupby].cat.categories
            ]
        else:
            categories = sorted(
                str(category)
                for category in pd.Series(score_df[groupby]).dropna().unique()
            )

        mean_scores = score_df.groupby(groupby, observed=True)[
            score_keys
        ].mean()
        categories = [
            category for category in categories if category in mean_scores.index
        ]
        if not categories:
            return pd.DataFrame(columns=labels)

        grouped = mean_scores.loc[categories, score_keys]
        grouped.columns = labels
        return grouped
