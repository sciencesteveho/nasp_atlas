"""Heatmap plotting methods."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]
from matplotlib.colors import Colormap

from nasp_atlas.single_cell.utils import expression_matrix
from nasp_atlas.single_cell.visualization._expression import (
    _expression_column_means,
)
from nasp_atlas.single_cell.visualization.gene_resolution import (
    _VisualizationGeneMixin,
)
from nasp_atlas.single_cell.visualization.style import ColorbarStyle
from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupedGeneExpression:
    """Reusable full-cell gene-expression means grouped by one obs column.

    Attributes:
      groupby: Observation column defining the groups.
      means: Mean expression with var names on rows and groups on columns.
      overall_means: Full-cell mean expression indexed by var name.
      requested_var_names: Requested gene token to resolved var-name mapping.
      requested_labels: Requested gene token to display-label mapping.
      expression_layer: AnnData layer used to prepare the summary.
      use_raw: Whether the summary was prepared from adata.raw.
    """

    groupby: str
    means: pd.DataFrame
    overall_means: pd.Series
    requested_var_names: Mapping[str, str]
    requested_labels: Mapping[str, str]
    expression_layer: str | None
    use_raw: bool


class HeatmapPlotter(_VisualizationGeneMixin, _PlotterBase):
    """Render grouped expression and observation-score heatmaps.

    Example Usage:
      >>> plotter = HeatmapPlotter(output_dir="path/to/output")
      >>> plotter.plot_multi_gene_expression_heatmap(
      ...     adata,
      ...     genes=["CGAS", "STING1"],
      ...     groupby="cell_type",
      ...     filename="sensor_expression",
      ... )
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dpi: int = 450,
        expression_cmap: Colormap | None = None,
    ) -> None:
        """Initialize heatmap rendering dependencies.

        Args:
          output_dir: Directory where figures are written.
          dpi: Saved PNG resolution.
          expression_cmap: Optional expression colormap override.
        """
        super().__init__(output_dir, dpi=dpi)
        if expression_cmap is None:
            expression_cmap = self._pastelize_cmap("YlGnBu", blend=0.20)
            expression_cmap = self._zero_gray_cmap(expression_cmap)
        self.expression_cmap = expression_cmap

    def summarize_gene_expression_by_obs(
        self,
        adata: Any,
        genes: Sequence[str],
        *,
        groupby: str,
        expression_layer: str | None = None,
        use_raw: bool = False,
        gene_symbol_column: str | None = None,
        obs_order: Sequence[str] | None = None,
    ) -> GroupedGeneExpression:
        """Return reusable sparse-aware grouped means for requested genes.

        The result retains full-cell weighted means while bounding dense
        materialization to `n_groups x n_genes`.

        Args:
          adata: AnnData object containing expression values.
          genes: Gene names to aggregate.
          groupby: Observation column defining output groups.
          expression_layer: Layer to use for expression values.
          use_raw: Whether to use adata.raw when expression_layer is None.
          gene_symbol_column: Optional var column used to resolve symbols.
          obs_order: Optional ordered subset of group labels.

        Returns:
          Grouped expression values and gene-resolution metadata.
        """
        if expression_layer is not None and use_raw:
            raise ValueError(
                "use_raw=True cannot be combined with expression_layer"
            )
        if groupby not in adata.obs.columns:
            raise KeyError(f"obs column not found for heatmap: {groupby}")

        gene_list = list(genes)
        resolved = self._resolve_genes(
            adata,
            gene_list,
            gene_symbol_column=gene_symbol_column,
        )
        requested_var_names = dict(resolved.requested_var_names)
        requested_labels = dict(resolved.requested_labels)
        if not resolved.var_names:
            return GroupedGeneExpression(
                groupby=groupby,
                means=pd.DataFrame(),
                overall_means=pd.Series(dtype=float),
                requested_var_names=requested_var_names,
                requested_labels=requested_labels,
                expression_layer=expression_layer,
                use_raw=use_raw,
            )

        source_var_names, source_matrix = expression_matrix(
            adata,
            resolved.var_names,
            expression_layer=expression_layer,
            use_raw=use_raw,
        )
        if source_matrix is None or source_var_names != resolved.var_names:
            missing = [
                var_name
                for var_name in resolved.var_names
                if var_name not in source_var_names
            ]
            raise KeyError(
                "resolved genes are absent from the requested expression "
                f"source: {missing}"
            )

        group_values = adata.obs[groupby].astype(str)
        categories = self._ordered_obs_groups(
            adata=adata,
            groupby=groupby,
            group_values=group_values,
            obs_order=obs_order,
        )
        if not categories:
            return GroupedGeneExpression(
                groupby=groupby,
                means=pd.DataFrame(index=source_var_names),
                overall_means=pd.Series(
                    _expression_column_means(source_matrix),
                    index=source_var_names,
                ),
                requested_var_names=requested_var_names,
                requested_labels=requested_labels,
                expression_layer=expression_layer,
                use_raw=use_raw,
            )

        group_codes = pd.Categorical(
            group_values,
            categories=categories,
        ).codes
        grouped_values, overall_means = self._group_expression_means(
            source_matrix,
            group_codes=group_codes,
            n_groups=len(categories),
        )
        means = pd.DataFrame(
            grouped_values.T,
            index=source_var_names,
            columns=categories,
        )
        return GroupedGeneExpression(
            groupby=groupby,
            means=means,
            overall_means=pd.Series(
                overall_means,
                index=source_var_names,
            ),
            requested_var_names=requested_var_names,
            requested_labels=requested_labels,
            expression_layer=expression_layer,
            use_raw=use_raw,
        )

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
        grouped_expression: GroupedGeneExpression | None = None,
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
          vmin: Lower color limit. Defaults to 0 so zero maps to gray with the
            default expression colormap.
          vmax: Upper color limit.
          grouped_expression: Optional reusable grouped means prepared by
            `summarize_gene_expression_by_obs`. When supplied, its recorded
            expression source is used instead of reading adata again.

        Example Usage:
          >>> plotter.plot_multi_gene_expression_heatmap(
          ...     adata,
          ...     genes=["CD3D", "MS4A1"],
          ...     groupby="cell_type",
          ...     filename="marker_heatmap",
          ...     gene_symbol_column="gene_symbol",
          ... )
        """
        set_matplotlib_publication_parameters()
        out = self.output_dir / filename
        colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.36, width=0.07, pad=0.02)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )
        cmap = self._zero_gray_cmap(
            cmap if cmap is not None else self.expression_cmap,
            zero_position=self._zero_cmap_position(vmin=vmin, vmax=vmax),
        )
        if grouped_expression is None:
            grouped_expression = self.summarize_gene_expression_by_obs(
                adata,
                genes,
                groupby=groupby,
                expression_layer=expression_layer,
                use_raw=use_raw,
                gene_symbol_column=gene_symbol_column,
                obs_order=obs_order,
            )
        elif grouped_expression.groupby != groupby:
            raise ValueError(
                "grouped_expression was prepared for "
                f"{grouped_expression.groupby!r}, not {groupby!r}"
            )
        elif (
            grouped_expression.expression_layer != expression_layer
            or grouped_expression.use_raw != use_raw
        ):
            raise ValueError(
                "grouped_expression was prepared from a different "
                "expression source"
            )

        var_names = []
        display_label_by_var_name = {}
        seen = set()
        for gene in genes:
            var_name = grouped_expression.requested_var_names.get(gene)
            if var_name is None or var_name in seen:
                continue
            seen.add(var_name)
            var_names.append(var_name)
            display_label_by_var_name[var_name] = (
                grouped_expression.requested_labels[gene]
            )
        if not var_names:
            logger.warning(
                "[plot] No valid genes found for %s. Skipping.", filename
            )
            return

        ordered_var_names = (
            grouped_expression.overall_means.loc[var_names]
            .sort_values(ascending=False)
            .index.tolist()
        )
        heatmap_expression = grouped_expression.means.loc[ordered_var_names]
        display_labels = [
            display_label_by_var_name[var_name]
            for var_name in ordered_var_names
        ]
        if heatmap_expression.empty or heatmap_expression.shape[1] == 0:
            logger.warning(
                "[plot] No valid groups found for heatmap %s. Skipping.",
                filename,
            )
            return

        n_genes, n_groups = heatmap_expression.shape
        heatmap_values = heatmap_expression.T.to_numpy(dtype=float)
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
            genes=display_labels,
            groups=heatmap_expression.columns.tolist(),
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
          >>> plotter.plot_grouped_obs_score_heatmap(
          ...     adata,
          ...     score_keys=["senescence_score", "immune_score"],
          ...     groupby="cell_type",
          ...     filename="score_heatmap",
          ... )
        """
        set_matplotlib_publication_parameters()
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
        cmap = self._zero_gray_cmap(
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
    def _ordered_obs_groups(
        *,
        adata: Any,
        groupby: str,
        group_values: pd.Series,
        obs_order: Sequence[str] | None = None,
    ) -> list[str]:
        """Return requested observed groups in deterministic display order."""
        if obs_order is not None:
            categories = [str(category) for category in obs_order]
        elif hasattr(adata.obs[groupby], "cat"):
            categories = [
                str(category) for category in adata.obs[groupby].cat.categories
            ]
        else:
            categories = sorted(
                str(category) for category in group_values.dropna().unique()
            )

        observed = set(group_values)
        return [
            category
            for category in dict.fromkeys(categories)
            if category in observed
        ]

    @classmethod
    def _group_expression_means(
        cls,
        matrix: Any,
        *,
        group_codes: np.ndarray,
        n_groups: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return grouped and overall means without full dense conversion."""
        if sp.issparse(matrix):
            return cls._sparse_group_expression_means(
                matrix,
                group_codes=group_codes,
                n_groups=n_groups,
            )
        return cls._dense_group_expression_means(
            np.asarray(matrix),
            group_codes=group_codes,
            n_groups=n_groups,
        )

    @staticmethod
    def _sparse_group_expression_means(
        matrix: Any,
        *,
        group_codes: np.ndarray,
        n_groups: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse-source grouped and overall expression means."""
        n_obs, n_vars = matrix.shape
        valid_rows = np.flatnonzero(group_codes >= 0)
        indicator = sp.csr_matrix(
            (
                np.ones(len(valid_rows), dtype=float),
                (group_codes[valid_rows], valid_rows),
            ),
            shape=(n_groups, n_obs),
        )

        sparse_matrix = matrix
        coo = sparse_matrix.tocoo(copy=False)
        nan_mask = np.isnan(coo.data)
        if bool(nan_mask.any()):
            sparse_matrix = sparse_matrix.copy()
            sparse_matrix.data[np.isnan(sparse_matrix.data)] = 0
            nan_matrix = sp.csr_matrix(
                (
                    np.ones(int(nan_mask.sum()), dtype=float),
                    (coo.row[nan_mask], coo.col[nan_mask]),
                ),
                shape=matrix.shape,
            )
            grouped_nan_product: Any = indicator @ nan_matrix
            grouped_nan_counts = grouped_nan_product.toarray()
            overall_nan_counts = np.asarray(nan_matrix.sum(axis=0)).reshape(-1)
        else:
            grouped_nan_counts = np.zeros((n_groups, n_vars), dtype=float)
            overall_nan_counts = np.zeros(n_vars, dtype=float)

        grouped_product: Any = indicator @ sparse_matrix
        grouped_sums = grouped_product.toarray()
        grouped_counts = np.bincount(
            group_codes[valid_rows],
            minlength=n_groups,
        ).astype(float)[:, np.newaxis]
        grouped_counts = grouped_counts - grouped_nan_counts
        grouped_means = np.divide(
            grouped_sums,
            grouped_counts,
            out=np.full((n_groups, n_vars), np.nan, dtype=float),
            where=grouped_counts > 0,
        )

        overall_sums = np.asarray(sparse_matrix.sum(axis=0)).reshape(-1)
        overall_counts = n_obs - overall_nan_counts
        overall_means = np.divide(
            overall_sums,
            overall_counts,
            out=np.full(n_vars, np.nan, dtype=float),
            where=overall_counts > 0,
        )
        return grouped_means, overall_means

    @staticmethod
    def _dense_group_expression_means(
        matrix: np.ndarray,
        *,
        group_codes: np.ndarray,
        n_groups: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return dense-source grouped and overall expression means."""
        n_vars = matrix.shape[1]
        grouped_means = np.full((n_groups, n_vars), np.nan, dtype=float)
        for group_index in range(n_groups):
            block = matrix[group_codes == group_index]
            if block.shape[0] == 0:
                continue
            nan_mask = np.isnan(block)
            sums = np.nansum(block, axis=0)
            counts = block.shape[0] - nan_mask.sum(axis=0)
            grouped_means[group_index] = np.divide(
                sums,
                counts,
                out=np.full(n_vars, np.nan, dtype=float),
                where=counts > 0,
            )

        overall_nan_mask = np.isnan(matrix)
        overall_sums = np.nansum(matrix, axis=0)
        overall_counts = matrix.shape[0] - overall_nan_mask.sum(axis=0)
        overall_means = np.divide(
            overall_sums,
            overall_counts,
            out=np.full(n_vars, np.nan, dtype=float),
            where=overall_counts > 0,
        )
        return grouped_means, overall_means

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
