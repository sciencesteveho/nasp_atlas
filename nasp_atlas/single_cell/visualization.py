"""Visualization utilities for single-cell datasets."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, Required, TypedDict, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.typing import ColorType
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # type: ignore
from scipy import stats  # type: ignore
from scipy.cluster.hierarchy import (  # type: ignore
    dendrogram as scipy_dendrogram,
)


logger = logging.getLogger(__name__)


@dataclass
class _ResolvedGenes:
    """Resolved plotting genes and display labels."""

    var_names: list[str]
    labels: list[str]


@dataclass
class _DotplotGenes:
    """Resolved gene list and optional marker group annotations."""

    var_names: list[str]
    labels: list[str]
    group_labels: list[str] | None
    group_positions: list[tuple[int, int]] | None


@dataclass
class _DotplotStats:
    """Per-group mean expression and fraction-expressing cells."""

    mean_exp: pd.DataFrame
    frac_exp: pd.DataFrame
    categories: list[str]


class UmapPanelSpec(TypedDict, total=False):
    """User-facing specification for one observation-colored UMAP panel."""

    obs_key: Required[str]
    title: NotRequired[str]
    kind: NotRequired[Literal["categorical", "numeric"]]
    color_map: NotRequired[dict[str, str]]
    cmap: NotRequired[Colormap | str]
    legend_loc: NotRequired[Literal["right", "bottom"]]
    legend_ncol: NotRequired[int]
    vmin: NotRequired[float | None]
    vmax: NotRequired[float | None]
    cbar_ticks: NotRequired[Sequence[float] | None]


@dataclass(frozen=True, slots=True)
class _UmapPanel:
    """Resolved UMAP panel configuration."""

    obs_key: str
    title: str
    kind: Literal["categorical", "numeric"]
    color_map: dict[str, str] | None = None
    cmap: Colormap | str = "viridis"
    legend_loc: Literal["right", "bottom"] = "right"
    legend_ncol: int = 1
    vmin: float | None = None
    vmax: float | None = None
    cbar_ticks: Sequence[float] | None = None


class SCVisualizer:
    """Visualization utilities for single-cell data.

    Example usage (standalone):
      >>> from nasp_atlas.single_cell import SCVisualizer
      >>> viz = SCVisualizer(output_dir="results/")
      >>> viz.plot_embedding(
      ...     adata,
      ...     color="condition",
      ...     filename="umap_condition",
      ... )
      >>> viz.plot_multi_gene_umap_panel(
      ...     adata,
      ...     genes=["AIM2", "CGAS", "ZBP1"],
      ...     filename="umap_marker_panel",
      ...     gene_symbol_column="feature_name",
      ... )
      >>> viz.plot_marker_dotplot(
      ...     adata,
      ...     groupby="leiden_0.5",
      ...     marker_groups={
      ...         "DNA sensing": ["AIM2", "CGAS"],
      ...         "RNA sensing": ["DDX58", "IFIH1"],
      ...     },
      ...     filename="dotplot_leiden_0.5",
      ...     gene_symbol_column="feature_name",
      ... )

    Attributes:
      output_dir: Directory where all figures are written.
    """

    dpi: int = 450
    legend_w: float = 0.52
    size_legend_h: float = 0.40
    cbar_w: float = 0.38
    cbar_h: float = 0.065
    legend_inner_gap: float = 0.08
    bar_h: float = 0.035
    bar_gap: float = 0.020
    left_margin: float = 0.18
    bottom_margin: float = 0.18
    annotation_height: float = 0.34
    annotation_gap: float = 0.02

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.expression_cmap = self.umap_expression_cmap(
            "RdYlBu_r",
            blue_blend=0.35,
        )
        self.dotplot_cmap = self.pastelize_cmap("Blues", blend=0.35)

        logging.getLogger("matplotlib.category").setLevel(logging.WARNING + 1)

    def plot_embedding(
        self,
        adata: Any,
        *,
        color: str | list[str],
        basis: str = "X_umap",
        filename: str | None = None,
        figsize: tuple[float, float] = (2, 2),
        color_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Save an embedding plot colored by any obs key.

        Args:
          adata: AnnData object with embedding computed.
          color: obs column(s) or gene symbol(s) to color by.
          basis: obsm key for the embedding
          filename: Output filename under output_dir.
          figsize: Figure size in inches.
          color_map: Optional mapping from category name -> hex color for
            categorical obs columns.
          **kwargs: Forwarded to sc.pl.embedding.
        """
        self._set_matplotlib_publication_parameters()

        if filename is None:
            label = color if isinstance(color, str) else "_".join(color)
            filename = f"embedding_{label}"
        out = self.output_dir / filename

        colors = [color] if isinstance(color, str) else color
        use_custom_legend = len(colors) == 1 and self._is_categorical_obs(
            adata, colors[0]
        )

        if use_custom_legend:
            fig, (ax, ax_legend) = plt.subplots(
                1,
                2,
                figsize=figsize,
                gridspec_kw={"width_ratios": [9, 1], "wspace": 0},
            )
            axes = [ax]
        else:
            fig, axes_raw = plt.subplots(1, len(colors), figsize=figsize)
            axes = [axes_raw] if len(colors) == 1 else list(axes_raw)
            ax_legend = None

        for ax, col in zip(axes, colors, strict=True):
            if color_map is not None and self._is_categorical_obs(adata, col):
                self._apply_color_map(adata, color_key=col, color_map=color_map)

            sc.pl.embedding(
                adata,
                basis=basis,
                color=col,
                ax=ax,
                title=None,
                size=kwargs.pop("size", 2.0),
                show=False,
                frameon=True,
                legend_loc=(
                    None
                    if self._is_categorical_obs(adata, col)
                    else "right margin"
                ),
                **kwargs,
            )
            self._style_embedding_axes(ax)

        if use_custom_legend and ax_legend is not None:
            self._add_square_legend(
                adata=adata,
                ax_legend=ax_legend,
                color_key=colors[0],
                color_map=color_map,
            )

        fig.savefig(f"{out}.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("[plot] embedding (%s) -> %s", color, out)

    def plot_umap_panel(
        self,
        adata: Any,
        panels: Sequence[str | UmapPanelSpec],
        *,
        filename: str,
        basis: str = "X_umap",
        ncols: int | None = None,
        panel_w: float = 3.25,
        panel_h: float = 3.25,
        size: float = 2.0,
        row_hspace: float = 0.22,
        col_wspace: float = 0.275,
        cbar_height: str = "23.375%",
        cbar_width: str = "4%",
        cbar_pad: float = 0.02,
    ) -> None:
        """Save ordered obs-colored UMAP panels with publication styling.

        Args:
          adata: AnnData object with embedding coordinates.
          panels: Ordered obs keys or panel specifications. Plain obs-key
            strings infer categorical or numeric styling from adata.obs.
          filename: Output filename under output_dir, without extension.
          basis: obsm key for the embedding.
          ncols: Number of columns. Defaults to one row.
          panel_w: Width of each panel in inches.
          panel_h: Height of each panel in inches.
          size: Scatter point size.
          row_hspace: Vertical spacing between panel rows.
          col_wspace: Horizontal spacing between panel columns.
          cbar_height: Inset numeric colorbar height as a percentage.
          cbar_width: Inset numeric colorbar width as a percentage.
          cbar_pad: Padding between panel and numeric colorbar.
        """
        self._set_matplotlib_publication_parameters()
        if not panels:
            logger.warning("[plot] no obs UMAP panels requested")
            return

        resolved_panels = self._resolve_umap_panel_specs(adata, panels)
        ncols = ncols if ncols is not None else len(resolved_panels)
        nrows = math.ceil(len(resolved_panels) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_w * ncols, panel_h * nrows),
            squeeze=False,
        )

        for ax, panel in zip(axes.flat, resolved_panels, strict=False):
            self._plot_umap_panel_axis(
                adata=adata,
                ax=ax,
                panel=panel,
                basis=basis,
                size=size,
                cbar_height=cbar_height,
                cbar_width=cbar_width,
                cbar_pad=cbar_pad,
            )

        for ax in axes.flat[len(resolved_panels) :]:
            ax.axis("off")

        fig.subplots_adjust(hspace=row_hspace, wspace=col_wspace)
        out = self.output_dir / filename
        self._save_figure_and_log(fig, out, "[plot] UMAP panel -> %s")

    def plot_obs_umap_panel(
        self,
        adata: Any,
        panels: Sequence[str | UmapPanelSpec],
        *,
        filename: str,
        basis: str = "X_umap",
        ncols: int | None = None,
        panel_w: float = 3.25,
        panel_h: float = 3.25,
        size: float = 2.0,
        row_hspace: float = 0.22,
        col_wspace: float = 0.275,
        cbar_height: str = "23.375%",
        cbar_width: str = "4%",
        cbar_pad: float = 0.02,
    ) -> None:
        """Compatibility wrapper for obs-colored UMAP panels."""
        self.plot_umap_panel(
            adata,
            panels=panels,
            filename=filename,
            basis=basis,
            ncols=ncols,
            panel_w=panel_w,
            panel_h=panel_h,
            size=size,
            row_hspace=row_hspace,
            col_wspace=col_wspace,
            cbar_height=cbar_height,
            cbar_width=cbar_width,
            cbar_pad=cbar_pad,
        )

    def plot_multi_gene_umap_panel(
        self,
        adata: Any,
        genes: list[str],
        *,
        filename: str,
        basis: str = "X_umap",
        ncols: int = 4,
        panel_w: float = 1.75,
        panel_h: float = 1.75,
        cmap: Colormap | None = None,
        expression_layer: str | None = None,
        use_raw: bool = False,
        gene_symbol_column: str | None = None,
        row_hspace: float = 0.22,
        cbar_height: str = "27.5%",
        cbar_width: str = "4%",
        cbar_pad: float = 0.02,
        size: float = 8.0,
    ) -> None:
        """Save a multi-panel UMAP figure.

        Args:
          adata: AnnData object with UMAP embedding computed.
          genes: Gene names to plot.
          filename: Output filename under output_dir.
          basis: obsm key for the embedding.
          ncols: Number of columns in the panel grid.
          panel_w: Width of each panel in inches.
          panel_h: Height of each panel in inches.
          cmap: Colormap for expression values.
          expression_layer: Layer to use for expression values.
          use_raw: Whether to use adata.raw when expression_layer is None.
          gene_symbol_column: Optional adata.var column used to resolve symbols.
          row_hspace: Vertical spacing between rows.
          cbar_height: Inset colorbar height as a percentage.
          cbar_width: Inset colorbar width as a percentage.
          cbar_pad: Padding between panel and colorbar.
          size: Point size forwarded to Scanpy.
        """
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename
        cmap = (
            cmap if cmap is not None else self.umap_expression_cmap("RdYlBu_r")
        )
        if expression_layer is not None and use_raw:
            raise ValueError(
                "use_raw=True cannot be combined with expression_layer"
            )

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
        expr_df = sc.get.obs_df(adata, keys=resolved.var_names, **expr_kwargs)
        gene_means = expr_df.mean(axis=0)
        order = gene_means.sort_values(ascending=False).index.tolist()
        label_lookup = dict(
            zip(resolved.var_names, resolved.labels, strict=True)
        )
        var_names = order
        labels = [label_lookup[var_name] for var_name in var_names]

        nrows = math.ceil(len(var_names) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_w * ncols, panel_h * nrows),
            squeeze=False,
        )

        for ax, var_name, label in zip(
            axes.flat, var_names, labels, strict=False
        ):
            plot_kwargs: dict[str, Any] = {"layer": expression_layer}
            if expression_layer is None:
                plot_kwargs["use_raw"] = use_raw

            sc.pl.embedding(
                adata,
                basis=basis,
                color=var_name,
                ax=ax,
                show=False,
                frameon=False,
                size=size,
                color_map=cmap,
                vmin=0,
                colorbar_loc=None,
                **plot_kwargs,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(label, fontstyle="italic", pad=0.0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)

            for spine in ax.spines.values():
                spine.set_visible(False)

            cax = inset_axes(
                ax,
                width=cbar_width,
                height=cbar_height,
                loc="center left",
                bbox_to_anchor=(1.02 + cbar_pad, 0.0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )

            cb = fig.colorbar(ax.collections[0], cax=cax)
            cb.ax.tick_params(length=1.5, pad=0.5)

        for ax in axes.flat[len(var_names) :]:
            ax.axis("off")

        fig.subplots_adjust(hspace=row_hspace, wspace=0.35)
        self._save_figure_and_log(
            fig, out, "[plot] multi-gene UMAP panel -> %s"
        )

    def plot_multi_obs_umap_panel(
        self,
        adata: Any,
        obs_keys: list[str],
        *,
        filename: str,
        basis: str = "X_umap",
        ncols: int = 4,
        panel_w: float = 1.75,
        panel_h: float = 1.75,
        cmap: Colormap | str | None = None,
        row_hspace: float = 0.22,
        cbar_height: str = "27.5%",
        cbar_width: str = "4%",
        cbar_pad: float = 0.02,
        size: float = 8.0,
        vmin: float | None = 0,
        vmax: float | None = None,
        center_zero: bool = False,
    ) -> None:
        """Save a multi-panel UMAP figure for numeric obs columns.

        This is a convenience wrapper around `plot_umap_panel` for applying one
        numeric style to several observation columns.

        Args:
          adata: AnnData object with UMAP embedding computed.
          obs_keys: Numeric obs columns to plot.
          filename: Output filename under output_dir.
          basis: obsm key for the embedding.
          ncols: Number of columns in the panel grid.
          panel_w: Width of each panel in inches.
          panel_h: Height of each panel in inches.
          cmap: Colormap for values.
          row_hspace: Vertical spacing between rows.
          cbar_height: Inset colorbar height as a percentage.
          cbar_width: Inset colorbar width as a percentage.
          cbar_pad: Padding between panel and colorbar.
          size: Point size forwarded to Scanpy.
          vmin: Lower color limit. Defaults to 0 so zero maps to gray when
            using `umap_expression_cmap`.
          vmax: Upper color limit.
          center_zero: Whether to derive per-panel symmetric color limits
            around zero when explicit limits are not supplied.
        """
        valid_keys = [key for key in obs_keys if key in adata.obs.columns]
        if not valid_keys:
            logger.warning(
                "[plot] No valid obs keys found for %s. Skipping.", filename
            )
            return

        numeric_cmap = (
            cmap if cmap is not None else self.umap_expression_cmap("viridis")
        )
        panels: list[UmapPanelSpec] = []
        for key in valid_keys:
            panel_vmin = vmin
            panel_vmax = vmax
            if center_zero and (panel_vmin is None or panel_vmax is None):
                limit = self._symmetric_obs_limit(adata, key)
                if panel_vmin is None:
                    panel_vmin = -limit
                if panel_vmax is None:
                    panel_vmax = limit
            panels.append(
                {
                    "obs_key": key,
                    "kind": "numeric",
                    "cmap": numeric_cmap,
                    "vmin": panel_vmin,
                    "vmax": panel_vmax,
                }
            )
        self.plot_umap_panel(
            adata,
            panels=panels,
            filename=filename,
            basis=basis,
            ncols=ncols,
            panel_w=panel_w,
            panel_h=panel_h,
            size=size,
            row_hspace=row_hspace,
            col_wspace=0.35,
            cbar_height=cbar_height,
            cbar_width=cbar_width,
            cbar_pad=cbar_pad,
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
        cell_size: float = 0.1,
        min_width: float = 1.5,
        min_height: float = 1.5,
        cbar_height: str | float = 0.36,
        cbar_width: str | float = 0.07,
        cbar_pad: float = 0.02,
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
          cbar_height: Inset colorbar height. Floats are inches; strings such
            as "17.5%" are relative to the heatmap axis.
          cbar_width: Inset colorbar width. Floats are inches; strings such as
            "3%" are relative to the heatmap axis.
          cbar_pad: Padding between heatmap and colorbar.
          cbar_title: Optional title drawn above the heatmap colorbar.
          vmin: Lower color limit. Defaults to 0 so zero maps to gray when
            using `umap_expression_cmap`.
          vmax: Upper color limit.
        """
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename
        cmap = cmap if cmap is not None else self.expression_cmap
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
            cbar_height=cbar_height,
            cbar_width=cbar_width,
            cbar_pad=cbar_pad,
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
        cell_size: float = 0.18,
        min_width: float = 1.5,
        min_height: float = 1.5,
        cbar_height: str | float = 0.36,
        cbar_width: str | float = 0.07,
        cbar_pad: float = 0.02,
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
          cbar_height: Inset colorbar height. Floats are inches; strings are
            relative to the heatmap axis.
          cbar_width: Inset colorbar width. Floats are inches; strings are
            relative to the heatmap axis.
          cbar_pad: Padding between heatmap and colorbar.
          cbar_title: Optional vertical colorbar label.
          vmin: Lower color limit. When omitted with center_zero=True, the
            lower limit is the negative maximum absolute grouped score.
          vmax: Upper color limit. When omitted with center_zero=True, the
            upper limit is the maximum absolute grouped score.
          center_zero: Whether to derive symmetric color limits around zero
            when explicit limits are not supplied.
        """
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename
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
            cbar_height=cbar_height,
            cbar_width=cbar_width,
            cbar_pad=cbar_pad,
            title=cbar_title,
        )

        self._save_figure_and_log(fig, out, "[plot] score heatmap -> %s")

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

    def plot_marker_dotplot(
        self,
        adata: Any,
        groupby: str,
        *,
        filename: str,
        genes: list[str] | None = None,
        marker_groups: dict[str, list[str]] | None = None,
        standard_scale: str = "var",
        expression_layer: str | None = "log1p",
        gene_symbol_column: str | None = None,
        cell_w: float = 0.0675,
        cell_h: float = 0.0825,
        largest_dot: float = 4.5,
        size_exponent: float = 1.0,
        dot_edge_color: str = "0.5",
        dot_edge_lw: float = 0.1,
        cmap: Colormap | str | None = None,
        group_cmap: str = "tab20c",
    ) -> None:
        """Save a marker gene dot plot grouped by a categorical obs key.

        Provide either a `genes` list or a `marker_groups` dict. When
        `marker_groups` is provided it takes precedence and genes are drawn in
        group order.
        """
        self._set_matplotlib_publication_parameters()
        cmap = cmap if cmap is not None else self.dotplot_cmap
        out = self.output_dir / filename

        resolved = self._resolve_dotplot_genes(
            adata=adata,
            genes=genes,
            marker_groups=marker_groups,
            gene_symbol_column=gene_symbol_column,
        )
        if not resolved.var_names:
            logger.warning(
                "[plot] No valid genes for dotplot %s. Skipping.", filename
            )
            return

        dendro_key = f"dendrogram_{groupby}"
        if dendro_key not in adata.uns:
            sc.tl.dendrogram(adata, groupby=groupby)

        categories_order = list(adata.uns[dendro_key]["categories_ordered"])
        categories_order = [str(c) for c in categories_order]

        stats = self._compute_dotplot_stats(
            adata=adata,
            groupby=groupby,
            var_names=resolved.var_names,
            standard_scale=standard_scale,
            expression_layer=expression_layer,
            categories_order=categories_order,
        )

        group_colors = self._get_group_colors(
            group_labels=resolved.group_labels,
            group_cmap=group_cmap,
        )

        plot_width = len(resolved.var_names) * cell_w
        plot_height = len(stats.categories) * cell_h
        gap_plot_to_legend = 0.10
        right_margin = gap_plot_to_legend + self.legend_w + 0.12
        top_annotation_block = 0.0
        bottom_annotation_block = 0.0

        if (
            resolved.group_labels is not None
            and resolved.group_positions is not None
            and group_colors is not None
        ):
            top_annotation_block = (
                self.bar_h
                + self.bar_gap
                + self.annotation_gap
                + self.annotation_height
            )
            bottom_annotation_block = self.bar_h + self.bar_gap

        fig_width = plot_width + self.left_margin + right_margin
        fig_height = (
            plot_height
            + 0.02
            + self.bottom_margin
            + top_annotation_block
            + bottom_annotation_block
        )
        fig = plt.figure(figsize=(fig_width, fig_height))

        scatter_y = self.bottom_margin + bottom_annotation_block
        scatter_ax = self._add_axes(
            fig=fig,
            fig_w=fig_width,
            fig_h=fig_height,
            x=self.left_margin,
            y=scatter_y,
            w=plot_width,
            h=plot_height,
        )
        scatter = self._draw_dotplot_scatter(
            ax=scatter_ax,
            stats=stats,
            labels=resolved.labels,
            cmap=cmap,
            largest_dot=largest_dot,
            size_exponent=size_exponent,
            dot_edge_color=dot_edge_color,
            dot_edge_lw=dot_edge_lw,
        )

        if (
            resolved.group_labels is not None
            and resolved.group_positions is not None
            and group_colors is not None
        ):
            self._draw_marker_group_annotations(
                fig=fig,
                fig_width=fig_width,
                fig_height=fig_height,
                scatter_y=scatter_y,
                plot_width=plot_width,
                plot_height=plot_height,
                group_positions=resolved.group_positions,
                group_labels=resolved.group_labels,
                group_colors=group_colors,
                n_genes=len(resolved.var_names),
            )

        self._draw_dotplot_legends(
            fig=fig,
            scat=scatter,
            fig_w=fig_width,
            fig_h=fig_height,
            scatter_y=scatter_y,
            plot_h=plot_height,
            legend_left=self.left_margin + plot_width + gap_plot_to_legend,
            legend_w=self.legend_w,
            size_legend_h=self.size_legend_h,
            cbar_w=self.cbar_w,
            cbar_h=self.cbar_h,
            legend_inner_gap=self.legend_inner_gap,
            largest_dot=largest_dot,
            size_exponent=size_exponent,
            dot_edge_color=dot_edge_color,
            dot_edge_lw=dot_edge_lw,
        )

        plt.savefig(
            f"{out}.png", dpi=self.dpi, bbox_inches="tight", pad_inches=0.02
        )
        plt.close(fig)
        logger.info("[plot] dotplot -> %s", out)

    def plot_rank_genes_dotplot(
        self,
        adata: Any,
        *,
        groupby: str,
        rank_key: str,
        filename: str,
        n_genes_per_group: int = 5,
        expression_layer: str | None = "log1p",
        gene_symbol_column: str | None = None,
        group_cmap: str = "tab20c",
        cmap: Colormap | str = "Reds",
        cell_w: float = 0.0675,
        cell_h: float = 0.0825,
        largest_dot: float = 4.5,
        size_exponent: float = 1.0,
        dot_edge_color: str = "black",
        dot_edge_lw: float = 0.1,
        dendro_frac: float = 0.125,
        header_rotation: int = 0,
    ) -> None:
        """Save a scaled mean expression ranked-genes dot plot."""
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename

        dendro_key = f"dendrogram_{groupby}"
        if dendro_key not in adata.uns:
            sc.tl.dendrogram(adata, groupby=groupby)

        categories_order = list(adata.uns[dendro_key]["categories_ordered"])
        categories_order = [str(c) for c in categories_order]

        gene_slots, group_labels, group_positions = (
            self._extract_rank_genes_grouped(
                adata=adata,
                rank_key=rank_key,
                groupby=groupby,
                n_genes_per_group=n_genes_per_group,
                categories_order=categories_order,
            )
        )
        resolved = self._resolve_genes(
            adata,
            gene_slots,
            gene_symbol_column=gene_symbol_column,
        )
        mean_mat, frac_mat = self._compute_rank_dotplot_stats(
            adata=adata,
            groupby=groupby,
            categories_order=categories_order,
            var_names=resolved.var_names,
            expression_layer=expression_layer,
        )

        n_groups = len(categories_order)
        n_slots = len(resolved.var_names)
        tab = plt.get_cmap(group_cmap)
        group_colors = [tab(i % 20) for i in range(len(group_labels))]

        plot_w = n_slots * cell_w
        plot_h = n_groups * cell_h
        dendro_w = plot_w * dendro_frac
        legend_gap = 0.15
        right_margin = legend_gap + self.legend_w + 0.12

        fig_w = self.left_margin + plot_w + dendro_w + right_margin
        fig_h = (
            self.bottom_margin
            + self.bar_h
            + self.bar_gap
            + plot_h
            + self.bar_gap
            + self.bar_h
            + self.annotation_gap
            + self.annotation_height
            + 0.02
        )

        fig = plt.figure(figsize=(fig_w, fig_h))
        scatter_y = self.bottom_margin + self.bar_h + self.bar_gap
        xlim: tuple[float, float] = (-0.5, n_slots - 0.5)

        scatter_ax = self._add_axes(
            fig, fig_w, fig_h, self.left_margin, scatter_y, plot_w, plot_h
        )

        stats = _DotplotStats(
            mean_exp=pd.DataFrame(
                mean_mat,
                columns=resolved.var_names,
                index=categories_order,
            ),
            frac_exp=pd.DataFrame(
                frac_mat,
                columns=resolved.var_names,
                index=categories_order,
            ),
            categories=categories_order,
        )

        scat = self._draw_dotplot_scatter(
            ax=scatter_ax,
            stats=stats,
            labels=resolved.labels,
            cmap=cmap,
            largest_dot=largest_dot,
            size_exponent=size_exponent,
            dot_edge_color=dot_edge_color,
            dot_edge_lw=dot_edge_lw,
        )

        bottom_bar_ax = self._add_axes(
            fig,
            fig_w,
            fig_h,
            self.left_margin,
            self.bottom_margin,
            plot_w,
            self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=bottom_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=xlim,
        )

        top_bar_ax = self._add_axes(
            fig,
            fig_w,
            fig_h,
            self.left_margin,
            scatter_y + plot_h + self.bar_gap,
            plot_w,
            self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=top_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=xlim,
        )

        header_ax = self._add_axes(
            fig,
            fig_w,
            fig_h,
            self.left_margin,
            scatter_y
            + plot_h
            + self.bar_gap
            + self.bar_h
            + self.annotation_gap,
            plot_w,
            self.annotation_height,
        )
        self._draw_dotplot_group_headers(
            hax=header_ax,
            group_labels=group_labels,
            group_positions=group_positions,
            n_genes=n_slots,
            rotation=header_rotation,
        )

        dendro_ax = self._add_axes(
            fig,
            fig_w,
            fig_h,
            self.left_margin + plot_w,
            scatter_y,
            dendro_w,
            plot_h,
        )
        self._draw_rank_dendrogram(
            adata=adata, groupby=groupby, ax=dendro_ax, n_groups=n_groups
        )

        self._draw_dotplot_legends(
            fig=fig,
            scat=scat,
            fig_w=fig_w,
            fig_h=fig_h,
            scatter_y=scatter_y,
            plot_h=plot_h,
            legend_left=self.left_margin + plot_w + dendro_w + legend_gap,
            legend_w=self.legend_w,
            size_legend_h=self.size_legend_h,
            cbar_w=self.cbar_w,
            cbar_h=self.cbar_h,
            legend_inner_gap=self.legend_inner_gap,
            largest_dot=largest_dot,
            size_exponent=size_exponent,
            dot_edge_color=dot_edge_color,
            dot_edge_lw=dot_edge_lw,
        )

        fig.savefig(
            f"{out}.png", dpi=self.dpi, bbox_inches="tight", pad_inches=0.02
        )
        plt.close(fig)
        logger.info("[plot] rank-genes dotplot -> %s", out)

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
                size=0.2,  # type: ignore
                linewidth=0.25,
                ax=ax,
            )

        for ax in axes.flat[len(score_keys) :]:
            ax.axis("off")

        fig.tight_layout()
        self._save_figure_and_log(fig, out, "[plot] annotation violins -> %s")

    def _save_figure_and_log(
        self,
        fig: Figure,
        out: Path,
        log_message: str,
    ) -> None:
        """Save a figure as PNG, close it, and log the output path."""
        fig.savefig(f"{out}.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(log_message, out)

    @staticmethod
    def _resolve_genes(
        adata: Any,
        genes: list[str],
        *,
        gene_symbol_column: str | None = None,
    ) -> _ResolvedGenes:
        """Resolve user-facing gene names to adata.var_names."""
        var_names = pd.Index(adata.var_names).astype(str)
        lookup: dict[str, tuple[str, str]] = {
            str(var_name): (str(var_name), str(var_name))
            for var_name in var_names
        }

        if gene_symbol_column is not None:
            if gene_symbol_column not in adata.var.columns:
                raise ValueError(
                    f"gene_symbol_column not found in adata.var: "
                    f"{gene_symbol_column}"
                )
            for var_name, symbol in zip(
                var_names,
                adata.var[gene_symbol_column].astype(str),
                strict=True,
            ):
                if not symbol or symbol == "nan":
                    continue
                lookup.setdefault(symbol, (str(var_name), symbol))

        resolved_var_names = []
        labels = []
        seen = set()
        for gene in genes:
            if gene not in lookup:
                continue
            var_name, label = lookup[gene]
            if var_name in seen:
                continue
            seen.add(var_name)
            resolved_var_names.append(var_name)
            labels.append(label)

        return _ResolvedGenes(var_names=resolved_var_names, labels=labels)

    @staticmethod
    def _resolve_dotplot_genes(
        adata: Any,
        genes: list[str] | None,
        marker_groups: dict[str, list[str]] | None,
        gene_symbol_column: str | None,
    ) -> _DotplotGenes:
        """Resolve the gene list and optional group annotations from inputs."""
        if marker_groups is None:
            resolved = SCVisualizer._resolve_genes(
                adata,
                genes or [],
                gene_symbol_column=gene_symbol_column,
            )
            return _DotplotGenes(
                var_names=resolved.var_names,
                labels=resolved.labels,
                group_labels=None,
                group_positions=None,
            )

        group_labels: list[str] = []
        group_positions: list[tuple[int, int]] = []
        var_names: list[str] = []
        labels: list[str] = []
        idx = 0

        for label, group_genes in marker_groups.items():
            resolved = SCVisualizer._resolve_genes(
                adata,
                group_genes,
                gene_symbol_column=gene_symbol_column,
            )
            if not resolved.var_names:
                continue
            group_labels.append(label)
            group_positions.append((idx, idx + len(resolved.var_names) - 1))
            var_names.extend(resolved.var_names)
            labels.extend(resolved.labels)
            idx += len(resolved.var_names)

        return _DotplotGenes(var_names, labels, group_labels, group_positions)

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
        """Return mean obs scores with obs groups on rows and scores on
        columns.
        """
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

    @staticmethod
    def _symmetric_obs_limit(adata: Any, obs_key: str) -> float:
        """Return a non-zero symmetric color limit for one numeric obs
        column.
        """
        obs = cast(pd.DataFrame, adata.obs)
        values = pd.to_numeric(obs[obs_key], errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        finite = np.abs(values[np.isfinite(values)])
        if finite.size == 0:
            return 1.0
        limit = float(finite.max())
        return limit if limit > 0.0 else 1.0

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

    @staticmethod
    def _compute_dotplot_stats(
        adata: Any,
        groupby: str,
        var_names: list[str],
        standard_scale: str,
        expression_layer: str | None = "log1p",
        categories_order: list[str] | None = None,
    ) -> _DotplotStats:
        """Compute per-group mean expression.

        Also computes the fraction of cells expressing each gene.
        """
        unique_genes = list(dict.fromkeys(var_names))

        if categories_order is not None:
            categories = [str(c) for c in categories_order]
        else:
            categories = list(
                adata.obs[groupby].cat.categories
                if hasattr(adata.obs[groupby], "cat")
                else sorted(adata.obs[groupby].unique())
            )
            categories = [str(c) for c in categories]

        exp_df = sc.get.obs_df(
            adata,
            keys=[*unique_genes, groupby],
            layer=expression_layer,
        )

        mean_exp = (
            exp_df.groupby(groupby, observed=True)[unique_genes]
            .mean()
            .loc[categories]
            .reindex(columns=var_names)
        )

        if standard_scale == "var":
            mean_exp = (mean_exp - mean_exp.min()) / (
                mean_exp.max() - mean_exp.min() + 1e-9
            )

        frac_exp = (
            (exp_df[unique_genes] > 0)
            .groupby(exp_df[groupby], observed=True)
            .mean()
            .loc[categories]
            .reindex(columns=var_names)
        )

        return _DotplotStats(
            mean_exp=mean_exp,
            frac_exp=frac_exp,
            categories=categories,
        )

    @staticmethod
    def _get_group_colors(
        group_labels: list[str] | None,
        group_cmap: str,
    ) -> list[ColorType] | None:
        """Return a list of colors for each marker group, or None."""
        if group_labels is None:
            return None
        cmaper = plt.get_cmap(group_cmap)
        return [cmaper(i % 10) for i in range(len(group_labels))]

    @staticmethod
    def _draw_dotplot_scatter(
        ax: Axes,
        stats: _DotplotStats,
        labels: list[str],
        cmap: Colormap | str,
        largest_dot: float,
        size_exponent: float,
        dot_edge_color: str,
        dot_edge_lw: float,
    ) -> PathCollection:
        """Draw the dot scatter and style axes without an enclosing box."""
        n_groups = len(stats.categories)
        n_genes = len(labels)
        x_coords, y_coords = np.meshgrid(
            np.arange(n_genes), np.arange(n_groups)
        )
        s = (stats.frac_exp.values.flatten() ** size_exponent) * (
            largest_dot**2
        )

        scatter = ax.scatter(
            x_coords.flatten(),
            y_coords.flatten(),
            s=s,
            c=stats.mean_exp.values.flatten(),
            cmap=cmap,
            edgecolors=dot_edge_color,
            linewidths=dot_edge_lw,
            clip_on=False,
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(
            labels, rotation=90, ha="center", va="top", color="black"
        )
        for label in ax.get_xticklabels():
            label.set_fontstyle("italic")
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(stats.categories, color="black")
        ax.set_xlim(-0.5, n_genes - 0.5)
        ax.set_ylim(n_groups - 0.5, -0.5)
        ax.tick_params(
            axis="x", which="both", length=0, labelbottom=True, pad=7
        )
        ax.tick_params(axis="y", which="both", length=0, labelleft=True)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_visible(True)
            lbl.set_clip_on(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return scatter

    def _draw_marker_group_annotations(
        self,
        *,
        fig: Figure,
        fig_width: float,
        fig_height: float,
        scatter_y: float,
        plot_width: float,
        plot_height: float,
        group_positions: list[tuple[int, int]],
        group_labels: list[str],
        group_colors: list[ColorType],
        n_genes: int,
    ) -> None:
        """Draw dotplot group bars and headers."""
        bottom_bar_ax = self._add_axes(
            fig,
            fig_width,
            fig_height,
            self.left_margin,
            self.bottom_margin,
            plot_width,
            self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=bottom_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=(-0.5, n_genes - 0.5),
            bar_height=0.5,
        )

        top_bar_ax = self._add_axes(
            fig,
            fig_width,
            fig_height,
            self.left_margin,
            scatter_y + plot_height + self.bar_gap,
            plot_width,
            self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=top_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=(-0.5, n_genes - 0.5),
            bar_height=0.5,
        )

        header_ax = self._add_axes(
            fig,
            fig_width,
            fig_height,
            self.left_margin,
            scatter_y
            + plot_height
            + self.bar_gap
            + self.bar_h
            + self.annotation_gap,
            plot_width,
            self.annotation_height,
        )
        self._draw_dotplot_group_headers(
            hax=header_ax,
            group_labels=group_labels,
            group_positions=group_positions,
            n_genes=n_genes,
        )

    @staticmethod
    def _draw_dotplot_group_headers(
        hax: Axes,
        group_labels: list[str],
        group_positions: list[tuple[int, int]],
        n_genes: int,
        rotation: int = 90,
    ) -> None:
        """Draw plain black group labels above the dot scatter."""
        hax.set_xlim(-0.5, n_genes - 0.5)
        hax.set_ylim(0, 1)
        hax.axis("off")

        for (start, end), label in zip(
            group_positions, group_labels, strict=True
        ):
            x_center = (start + end) / 2.0
            hax.text(
                x_center,
                0.0,
                label,
                rotation=rotation,
                ha="center",
                va="bottom",
                clip_on=False,
                color="black",
            )

    @staticmethod
    def _draw_dotplot_group_bars(
        ax: Axes,
        group_positions: list[tuple[int, int]],
        group_colors: list[Any],
        xlim: tuple[float, float],
        bar_height: float = 1.0,
    ) -> None:
        """Draw full-width rectangular group bars edge-to-edge."""
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for color, (start, end) in zip(
            group_colors, group_positions, strict=True
        ):
            ax.add_patch(
                Rectangle(
                    (start - 0.5, 0.0),
                    end - start + 1.0,
                    bar_height,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0,
                )
            )

    @staticmethod
    def _draw_dotplot_size_legend(
        sax: Axes,
        largest_dot: float,
        size_exponent: float,
        dot_edge_color: str,
        dot_edge_lw: float,
    ) -> None:
        """Draw the fraction / dot-size legend."""
        sax.set_axis_off()
        sax.text(
            0.5,
            1.1,
            "Fraction of cells\nin group (%)",
            ha="center",
            va="bottom",
            transform=sax.transAxes,
        )

        ref_fracs = [0.2, 0.5, 0.8, 1.0]
        xs = np.linspace(0.15, 0.85, len(ref_fracs))

        for x, f in zip(xs, ref_fracs, strict=True):
            sax.scatter(
                [x],
                [1.0],
                s=(f**size_exponent) * (largest_dot**2),
                c="gray",
                edgecolors=dot_edge_color,
                linewidths=dot_edge_lw,
                clip_on=False,
                zorder=3,
            )
            sax.plot(
                [x, x],
                [0.75, 0.90],
                color="black",
                linewidth=0.5,
                solid_capstyle="butt",
                zorder=2,
            )
            sax.text(x, 0.45, f"{int(f * 100)}", ha="center", va="bottom")

        sax.set_xlim(0, 1)
        sax.set_ylim(-0.06, 1.05)

    @staticmethod
    def _draw_dotplot_colorbar(
        fig: Figure, scat: PathCollection, cax: Axes
    ) -> None:
        """Draw the mean expression colorbar."""
        cbar = plt.colorbar(
            scat,
            cax=cax,
            orientation="horizontal",
            ticks=[0, 0.5, 1],
        )
        cbar.ax.set_title("Mean expression\nin group", pad=2)
        cbar.ax.tick_params(length=3, pad=1)

    @staticmethod
    def _draw_dotplot_legends(
        fig: Figure,
        scat: PathCollection,
        *,
        fig_w: float,
        fig_h: float,
        scatter_y: float,
        plot_h: float,
        legend_left: float,
        legend_w: float,
        size_legend_h: float,
        cbar_w: float,
        cbar_h: float,
        legend_inner_gap: float,
        largest_dot: float,
        size_exponent: float,
        dot_edge_color: str,
        dot_edge_lw: float,
    ) -> None:
        """Draw the size legend and colorbar for dotplots."""
        legend_block_h = size_legend_h + legend_inner_gap + cbar_h
        legend_bottom = scatter_y + max(0.0, (plot_h - legend_block_h) / 2.0)
        cbar_left = legend_left + (legend_w - cbar_w) / 2.0
        cbar_bottom = legend_bottom - 0.05
        size_bottom = cbar_bottom + cbar_h + legend_inner_gap

        size_ax = SCVisualizer._add_axes(
            fig, fig_w, fig_h, legend_left, size_bottom, legend_w, size_legend_h
        )
        SCVisualizer._draw_dotplot_size_legend(
            sax=size_ax,
            largest_dot=largest_dot,
            size_exponent=size_exponent,
            dot_edge_color=dot_edge_color,
            dot_edge_lw=dot_edge_lw,
        )

        cbar_ax = SCVisualizer._add_axes(
            fig, fig_w, fig_h, cbar_left, cbar_bottom, cbar_w, cbar_h
        )
        SCVisualizer._draw_dotplot_colorbar(fig=fig, scat=scat, cax=cbar_ax)

    @staticmethod
    def _extract_rank_genes_grouped(
        adata: Any,
        rank_key: str,
        groupby: str,
        n_genes_per_group: int,
        categories_order: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[tuple[int, int]]]:
        """Extract ranked gene slots and cluster block positions.

        Returns:
          gene_slots: Flat list of gene names in cluster block order
          group_labels: Cluster labels in the same order
          group_positions: Inclusive (start, end) x-index for each block
        """
        rg = adata.uns[rank_key]
        ranked_names = rg["names"]

        if categories_order is None:
            categories_order = list(adata.obs[groupby].cat.categories)

        gene_slots: list[str] = []
        group_labels: list[str] = []
        group_positions: list[tuple[int, int]] = []
        idx = 0

        for group in categories_order:
            genes = list(ranked_names[group][:n_genes_per_group])
            if not genes:
                continue
            start = idx
            gene_slots.extend(genes)
            idx += len(genes)
            group_labels.append(str(group))
            group_positions.append((start, idx - 1))

        return gene_slots, group_labels, group_positions

    @staticmethod
    def _compute_rank_dotplot_stats(
        adata: Any,
        groupby: str,
        categories_order: list[str],
        var_names: list[str],
        expression_layer: str | None = "log1p",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-cluster mean expression.

        Also computes fraction-expressing arrays.
        """
        n_groups = len(categories_order)
        n_slots = len(var_names)
        mean_mat = np.zeros((n_groups, n_slots), dtype=float)
        frac_mat = np.zeros((n_groups, n_slots), dtype=float)

        for j, var_name in enumerate(var_names):
            if var_name not in adata.var_names:
                continue
            tmp = sc.get.obs_df(
                adata,
                keys=[var_name, groupby],
                layer=expression_layer,
            )
            grp = tmp.groupby(groupby, observed=True)[var_name]
            mean_s = grp.mean()
            frac_s = grp.apply(lambda x: (x > 0).mean())
            mean_mat[:, j] = [mean_s.get(cat, 0.0) for cat in categories_order]
            frac_mat[:, j] = [frac_s.get(cat, 0.0) for cat in categories_order]

        col_min = mean_mat.min(axis=0, keepdims=True)
        col_max = mean_mat.max(axis=0, keepdims=True)
        mean_mat = (mean_mat - col_min) / (col_max - col_min + 1e-9)
        return mean_mat, frac_mat

    @staticmethod
    def _draw_rank_dendrogram(
        adata: Any,
        groupby: str,
        ax: Axes,
        n_groups: int,
    ) -> None:
        """Draw a dendrogram to the right of the rank-genes dot scatter."""
        dendro_key = f"dendrogram_{groupby}"
        linkage = np.asarray(adata.uns[dendro_key]["linkage"])

        result = scipy_dendrogram(linkage, orientation="right", no_plot=True)

        segments = []
        for icoord, dcoord in zip(
            result["icoord"], result["dcoord"], strict=True
        ):
            y = (np.asarray(icoord, dtype=float) - 5.0) / 10.0
            x = np.asarray(dcoord, dtype=float)
            segments.extend(
                [(x[k], y[k]), (x[k + 1], y[k + 1])] for k in range(3)
            )
        lc = LineCollection(
            segments,
            colors="0.45",
            linewidths=0.25,
            capstyle="butt",
            joinstyle="miter",
        )
        ax.add_collection(lc)
        ax.autoscale_view()

        ax.set_ylim(n_groups - 0.5, -0.5)
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            labelleft=False,
            labelbottom=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _plot_umap_panel_axis(
        self,
        *,
        adata: Any,
        ax: Axes,
        panel: _UmapPanel,
        basis: str,
        size: float,
        cbar_height: str,
        cbar_width: str,
        cbar_pad: float,
    ) -> None:
        """Plot one obs-colored UMAP panel onto an existing axis."""
        obs_key = panel.obs_key
        kind = panel.kind
        xy = self._embedding_xy(adata, basis=basis)

        if kind == "categorical":
            color_map = self._resolve_obs_color_map(
                adata=adata,
                obs_key=obs_key,
                color_map=panel.color_map,
            )
            obs_values = adata.obs[obs_key].astype("category")
            for category in obs_values.cat.categories:
                mask = obs_values == category
                ax.scatter(
                    xy[mask.to_numpy(), 0],
                    xy[mask.to_numpy(), 1],
                    c=color_map.get(str(category), "#999999"),
                    s=size,
                    linewidths=0,
                    rasterized=True,
                )
            if obs_values.isna().any():
                mask = obs_values.isna()
                ax.scatter(
                    xy[mask.to_numpy(), 0],
                    xy[mask.to_numpy(), 1],
                    c="#d9d9d9",
                    s=size,
                    linewidths=0,
                    rasterized=True,
                )
                color_map = {**color_map, "NA": "#d9d9d9"}
                adata.uns[f"{obs_key}_colors"] = [
                    *adata.uns.get(f"{obs_key}_colors", []),
                    "#d9d9d9",
                ]

            self._style_umap_panel_axis(ax, panel)
            self._add_point_legend(
                ax=ax,
                color_map=color_map,
                marker_size=4,
                loc=panel.legend_loc,
                ncol=panel.legend_ncol,
            )
            return

        if kind == "numeric":
            values = pd.to_numeric(adata.obs[obs_key], errors="coerce")
            collection = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=values.to_numpy(),
                s=size,
                cmap=panel.cmap,
                vmin=panel.vmin,
                vmax=panel.vmax,
                linewidths=0,
                rasterized=True,
            )
            self._style_umap_panel_axis(ax, panel)
            self._add_embedding_colorbar(
                fig=ax.figure,
                ax=ax,
                mappable=collection,
                cbar_height=cbar_height,
                cbar_width=cbar_width,
                cbar_pad=cbar_pad,
                ticks=panel.cbar_ticks,
            )
            return

        raise ValueError(f"Unsupported obs UMAP panel kind: {kind!r}")

    def _style_umap_panel_axis(self, ax: Axes, panel: _UmapPanel) -> None:
        """Apply shared styling to one UMAP panel axis."""
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        self._pad_embedding_axes(ax)
        ax.set_title(panel.title, pad=2)

    @staticmethod
    def _style_gene_expression_heatmap_axis(
        *,
        ax: Axes,
        genes: list[str],
        groups: list[str],
    ) -> None:
        """Apply shared publication styling to a gene-expression heatmap."""
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(
            genes, rotation=90, ha="center", va="top", color="black"
        )
        for label in ax.get_xticklabels():
            label.set_fontstyle("italic")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, color="black")
        ax.tick_params(axis="both", which="major", length=0, pad=2.1)

        ax.set_xticks(np.arange(-0.5, len(genes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.25)
        ax.tick_params(axis="both", which="minor", length=0)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_visible(True)
            lbl.set_clip_on(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _style_score_heatmap_axis(
        *,
        ax: Axes,
        score_labels: list[str],
        groups: list[str],
    ) -> None:
        """Apply shared publication styling to an obs-score heatmap."""
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(score_labels)))
        ax.set_xticklabels(
            score_labels, rotation=90, ha="center", va="top", color="black"
        )
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, color="black")
        ax.tick_params(axis="both", which="major", length=0, pad=2.1)

        ax.set_xticks(np.arange(-0.5, len(score_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.25)
        ax.tick_params(axis="both", which="minor", length=0)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_visible(True)
            lbl.set_clip_on(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _resolve_umap_panel_specs(
        adata: Any,
        panels: Sequence[str | UmapPanelSpec],
    ) -> list[_UmapPanel]:
        """Normalize user panel inputs while preserving requested order."""
        resolved: list[_UmapPanel] = []
        for panel in panels:
            if isinstance(panel, str):
                spec: UmapPanelSpec = {"obs_key": panel}
            else:
                spec = cast(UmapPanelSpec, dict(panel))

            obs_key = spec["obs_key"]
            if obs_key not in adata.obs.columns:
                raise KeyError(
                    f"obs column not found for UMAP panel: {obs_key}"
                )

            title = spec.get("title", obs_key)
            kind = spec.get("kind")
            if kind is None:
                if pd.api.types.is_numeric_dtype(adata.obs[obs_key]):
                    kind = "numeric"
                else:
                    kind = "categorical"

            resolved.append(
                _UmapPanel(
                    obs_key=obs_key,
                    title=title,
                    kind=kind,
                    color_map=spec.get("color_map"),
                    cmap=spec.get("cmap", "viridis"),
                    legend_loc=spec.get("legend_loc", "right"),
                    legend_ncol=spec.get("legend_ncol", 1),
                    vmin=spec.get("vmin"),
                    vmax=spec.get("vmax"),
                    cbar_ticks=spec.get("cbar_ticks"),
                )
            )

        return resolved

    @staticmethod
    def _embedding_xy(adata: Any, *, basis: str) -> np.ndarray:
        """Return first two embedding coordinates as a dense numpy array."""
        if basis not in adata.obsm:
            raise KeyError(f"embedding basis not found: {basis}")

        xy = np.asarray(adata.obsm[basis])
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(
                f"embedding basis {basis!r} must have at least two columns"
            )

        return xy[:, :2]

    @staticmethod
    def _pad_embedding_axes(
        ax: Axes,
        *,
        x_padding: float = 0.025,
        y_padding: float = 0.025,
    ) -> None:
        """Tighten embedding limits with light padding."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(
            x_min + x_padding * (x_max - x_min),
            x_max - x_padding * (x_max - x_min),
        )
        ax.set_ylim(
            y_min + y_padding * (y_max - y_min),
            y_max - y_padding * (y_max - y_min),
        )
        ax.autoscale(False)

    @staticmethod
    def _resolve_obs_color_map(
        *,
        adata: Any,
        obs_key: str,
        color_map: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return and apply a categorical obs color map."""
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")
        categories = [str(value) for value in adata.obs[obs_key].cat.categories]
        if color_map is None:
            if f"{obs_key}_colors" in adata.uns:
                colors = list(adata.uns[f"{obs_key}_colors"])
                if len(categories) == len(colors):
                    return dict(zip(categories, colors, strict=True))

            tab = plt.get_cmap("tab20")
            color_map = {
                category: mcolors.to_hex(tab(index % tab.N))
                for index, category in enumerate(categories)
            }

        adata.uns[f"{obs_key}_colors"] = [
            color_map.get(category, "#999999") for category in categories
        ]
        return color_map

    @staticmethod
    def _add_point_legend(
        *,
        ax: Axes,
        color_map: dict[str, str],
        marker_size: float = 4,
        loc: Literal["right", "bottom"] = "right",
        ncol: int = 1,
    ) -> None:
        """Add a point-marker categorical legend."""
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=marker_size,
                label=category,
            )
            for category, color in color_map.items()
        ]

        legend_kwargs = (
            {
                "loc": "upper center",
                "bbox_to_anchor": (0.5, -0.04),
                "columnspacing": 0.8,
            }
            if loc == "bottom"
            else {
                "loc": "center left",
                "bbox_to_anchor": (1.02, 0.5),
            }
        )
        ax.legend(
            handles=handles,
            frameon=False,
            handletextpad=0.4,
            borderaxespad=0.0,
            labelspacing=0.35,
            ncol=ncol,
            **legend_kwargs,
        )

    @staticmethod
    def _add_embedding_colorbar(
        fig: Any,
        ax: Axes,
        cbar_height: str | float,
        cbar_width: str | float,
        cbar_pad: float,
        mappable: Any | None = None,
        ticks: Sequence[float] | None = None,
        title: str | None = None,
    ) -> None:
        """Add an inset colorbar beside an embedding panel."""
        cax = inset_axes(
            ax,
            width=cbar_width,
            height=cbar_height,
            loc="center left",
            bbox_to_anchor=(1.02 + cbar_pad, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(
            mappable if mappable is not None else ax.collections[0],
            cax=cax,
            ticks=ticks,
        )
        cbar.ax.tick_params(length=1.5, pad=0.5)
        if title is not None:
            cbar.ax.set_ylabel(title, rotation=270, labelpad=4, va="bottom")

    @staticmethod
    def _add_axes(
        fig: Figure,
        fig_w: float,
        fig_h: float,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> Axes:
        """Add an axes to fig using inch coordinates."""
        rectangle = (x / fig_w, y / fig_h, w / fig_w, h / fig_h)
        return fig.add_axes(rectangle)

    @staticmethod
    def _set_matplotlib_publication_parameters() -> None:
        """Set matplotlib parameters for publication-quality figures."""
        plt.rcParams.update(
            {
                "font.size": 5,
                "axes.titlesize": 5,
                "axes.labelsize": 5,
                "xtick.labelsize": 5,
                "ytick.labelsize": 5,
                "legend.fontsize": 5,
                "figure.titlesize": 5,
                "figure.dpi": 450,
                "font.sans-serif": ["Arial", "Nimbus Sans"],
                "axes.linewidth": 0.25,
                "xtick.major.width": 0.25,
                "ytick.major.width": 0.25,
                "xtick.minor.width": 0.25,
                "ytick.minor.width": 0.25,
            }
        )

    @staticmethod
    def _is_categorical_obs(adata: Any, key: str) -> bool:
        """Return True if key is a categorical obs column."""
        return (
            hasattr(adata.obs[key], "cat")
            if key in adata.obs.columns
            else False
        )

    @staticmethod
    def pastelize_cmap(
        cmap_name: str = "Blues",
        blend: float = 0.35,
    ) -> LinearSegmentedColormap:
        """Returns a more pastel version of a matplotlib cmap.

        Args:
          cmap_name: Base colormap name.
          blend: 0.0 = original cmap, 1.0 = fully white.
        """
        base = plt.colormaps[cmap_name].resampled(256)
        colors = base(np.linspace(0, 1, 256))
        colors[:, :3] = colors[:, :3] * (1 - blend) + blend
        return LinearSegmentedColormap.from_list(f"{cmap_name}_pastel", colors)

    @staticmethod
    def umap_expression_cmap(
        cmap_name: str = "RdYlBu_r",
        *,
        blue_blend: float = 0.0,
    ) -> ListedColormap:
        """Returns a matplotlib cmap with light gray at zero values.

        Args:
          cmap_name: Base colormap name.
          blue_blend: Blend the lower, blue side toward white. 0.0 leaves the
            colormap unchanged; 1.0 makes the blue side white.
        """
        base = plt.colormaps[cmap_name].resampled(256)
        colors = base(np.linspace(0, 1, 256))
        if blue_blend:
            midpoint = len(colors) // 2
            colors[:midpoint, :3] = (
                colors[:midpoint, :3] * (1 - blue_blend) + blue_blend
            )
        colors[0] = mcolors.to_rgba("#eeeeee")
        return mcolors.ListedColormap(colors)

    @staticmethod
    def _apply_color_map(
        adata: Any,
        color_key: str,
        color_map: dict[str, str],
    ) -> None:
        """Apply a fixed categorical color map to adata.uns."""
        if color_key not in adata.obs.columns:
            return
        if not hasattr(adata.obs[color_key], "cat"):
            return
        categories = list(adata.obs[color_key].cat.categories)
        colors = [color_map.get(cat, "#999999") for cat in categories]
        adata.uns[f"{color_key}_colors"] = colors

    @staticmethod
    def _add_square_legend(
        *,
        adata: Any,
        ax_legend: Axes,
        color_key: str,
        color_map: dict[str, str] | None = None,
    ) -> None:
        """Add a square-patch legend in a dedicated legend axis."""
        ax_legend.axis("off")

        if color_key not in adata.obs.columns:
            return
        if not hasattr(adata.obs[color_key], "cat"):
            return

        categories = list(adata.obs[color_key].cat.categories)
        uns_key = f"{color_key}_colors"

        if uns_key in adata.uns:
            colors = list(adata.uns[uns_key])
        elif color_map is not None:
            colors = [color_map.get(cat, "#999999") for cat in categories]
        else:
            return

        handles = [Patch(facecolor=c, edgecolor=c) for c in colors]
        ax_legend.legend(
            handles,
            categories,
            loc="center left",
            frameon=False,
            handlelength=1.15,
            handleheight=1.15,
            labelspacing=0.65,
        )

    @staticmethod
    def _style_embedding_axes(
        ax: Axes,
        x_padding: float = 0.025,
        y_padding: float = 0.025,
    ) -> None:
        """Apply publication-style axes formatting to UMAP plot."""
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(
            x_min + x_padding * (x_max - x_min),
            x_max - x_padding * (x_max - x_min),
        )
        ax.set_ylim(
            y_min + y_padding * (y_max - y_min),
            y_max - y_padding * (y_max - y_min),
        )
        ax.autoscale(False)
