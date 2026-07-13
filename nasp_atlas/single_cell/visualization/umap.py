"""UMAP and embedding panel plotting methods."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore[import]
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nasp_atlas.single_cell.umap import UmapPanel
from nasp_atlas.single_cell.umap import UmapPanelSpec
from nasp_atlas.single_cell.umap import embedding_xy
from nasp_atlas.single_cell.umap import is_categorical_obs
from nasp_atlas.single_cell.umap import resolve_umap_panel_specs
from nasp_atlas.single_cell.visualization.gene_resolution import (
    _VisualizationGeneMixin,
)
from nasp_atlas.single_cell.visualization.style import ColorbarStyle
from nasp_atlas.single_cell.visualization.style import _VisualizationStyleMixin


logger = logging.getLogger(__name__)


class _UmapPlotMixin(_VisualizationGeneMixin, _VisualizationStyleMixin):
    """UMAP and embedding panel plotting methods."""

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

        Example Usage:
          >>> viz.plot_embedding(
          ...     adata,
          ...     color="cell_type",
          ...     filename="cell_type_umap",
          ... )
        """
        self._set_matplotlib_publication_parameters()

        if filename is None:
            label = color if isinstance(color, str) else "_".join(color)
            filename = f"embedding_{label}"
        out = self.output_dir / filename

        colors = [color] if isinstance(color, str) else color
        use_custom_legend = len(colors) == 1 and is_categorical_obs(
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
            self._plot_embedding_axis(
                adata,
                ax,
                basis=basis,
                color=col,
                color_map=color_map,
                plot_kwargs=kwargs,
            )

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
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        cbar_pad: float | None = None,
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
          colorbar_style: Base numeric colorbar styling. Defaults to the
            panel-tuned UMAP colorbar style.
          cbar_height: Optional override for numeric colorbar height.
          cbar_width: Optional override for numeric colorbar width.
          cbar_pad: Optional override for panel-to-colorbar padding.

        Example Usage:
          >>> viz.plot_umap_panel(
          ...     adata,
          ...     panels=["cell_type", "module_score"],
          ...     filename="metadata_panel",
          ...     ncols=2,
          ... )
        """
        self._set_matplotlib_publication_parameters()
        if not panels:
            logger.warning("[plot] no obs UMAP panels requested")
            return

        colorbar_style = (
            colorbar_style
            or ColorbarStyle(height="23.375%", width="4%", pad=0.02)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )
        resolved_panels = resolve_umap_panel_specs(adata, panels)
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
                colorbar_style=colorbar_style,
            )

        for ax in axes.flat[len(resolved_panels) :]:
            ax.axis("off")

        fig.subplots_adjust(hspace=row_hspace, wspace=col_wspace)
        out = self.output_dir / filename
        self._save_figure_and_log(fig, out, "[plot] UMAP panel -> %s")

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
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        cbar_pad: float | None = None,
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
          colorbar_style: Base expression colorbar styling. Defaults to the
            compact multi-panel UMAP colorbar style.
          cbar_height: Optional override for colorbar height.
          cbar_width: Optional override for colorbar width.
          cbar_pad: Optional override for panel-to-colorbar padding.
          size: Point size forwarded to Scanpy.

        Example Usage:
          >>> viz.plot_multi_gene_umap_panel(
          ...     adata,
          ...     genes=["CD3D", "MS4A1"],
          ...     filename="marker_gene_umaps",
          ...     gene_symbol_column="gene_symbol",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        out = self.output_dir / filename
        colorbar_style = (
            colorbar_style
            or ColorbarStyle(height="27.5%", width="4%", pad=0.02)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )
        cmap = self.zero_gray_cmap(
            cmap if cmap is not None else self.expression_cmap
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

            self._add_embedding_colorbar(
                fig=fig,
                ax=ax,
                mappable=ax.collections[0],
                colorbar_style=colorbar_style,
            )

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
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        cbar_pad: float | None = None,
        size: float = 8.0,
        vmin: float | None = 0,
        vmax: float | None = None,
        center_zero: bool = False,
    ) -> None:
        """Save a multi-panel UMAP figure for numeric obs columns.

        Applies one numeric panel style to several observation columns while
        resolving color limits independently for each column.

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
          colorbar_style: Base numeric colorbar styling. Defaults to the
            compact multi-panel UMAP colorbar style.
          cbar_height: Optional override for colorbar height.
          cbar_width: Optional override for colorbar width.
          cbar_pad: Optional override for panel-to-colorbar padding.
          size: Point size forwarded to Scanpy.
          vmin: Lower color limit. Defaults to 0 so zero maps to gray when
            using `umap_expression_cmap`.
          vmax: Upper color limit.
          center_zero: Whether to derive per-panel symmetric color limits
            around zero when explicit limits are not supplied.

        Example Usage:
          >>> viz.plot_multi_obs_umap_panel(
          ...     adata,
          ...     obs_keys=["module_score", "age_accel"],
          ...     filename="score_umaps",
          ...     cbar_height="25%",
          ... )
        """
        valid_keys = [key for key in obs_keys if key in adata.obs.columns]
        if not valid_keys:
            logger.warning(
                "[plot] No valid obs keys found for %s. Skipping.", filename
            )
            return

        colorbar_style = (
            colorbar_style
            or ColorbarStyle(height="27.5%", width="4%", pad=0.02)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )
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
                    "cmap": self.zero_gray_cmap(
                        numeric_cmap,
                        zero_position=self._zero_cmap_position(
                            vmin=panel_vmin,
                            vmax=panel_vmax,
                        ),
                    ),
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
            colorbar_style=colorbar_style,
        )

    @staticmethod
    def _symmetric_obs_limit(adata: Any, obs_key: str) -> float:
        """Return a non-zero symmetric limit for one numeric obs column."""
        values = pd.to_numeric(
            adata.obs[obs_key],
            errors="coerce",
        ).to_numpy(dtype=float, copy=False)
        finite = np.abs(values[np.isfinite(values)])
        if finite.size == 0:
            return 1.0

        limit = float(finite.max())
        return limit if limit > 0.0 else 1.0

    def _plot_embedding_axis(
        self,
        adata: Any,
        ax: Axes,
        *,
        basis: str,
        color: str,
        color_map: dict[str, str] | None,
        plot_kwargs: dict[str, Any],
    ) -> None:
        """Render one categorical or numeric embedding axis."""
        categorical = is_categorical_obs(adata, color)
        if color_map is not None and categorical:
            self._apply_color_map(
                adata,
                color_key=color,
                color_map=color_map,
            )

        resolved_kwargs = dict(plot_kwargs)
        size = resolved_kwargs.pop("size", 2.0)
        if not categorical:
            cmap_key = next(
                (
                    key
                    for key in ("color_map", "cmap")
                    if key in resolved_kwargs
                ),
                None,
            )
            if cmap_key is None:
                resolved_kwargs["color_map"] = self.umap_expression_cmap(
                    "viridis"
                )
            else:
                resolved_kwargs[cmap_key] = self.zero_gray_cmap(
                    resolved_kwargs[cmap_key]
                )

        sc.pl.embedding(
            adata,
            basis=basis,
            color=color,
            ax=ax,
            title=None,
            size=size,
            show=False,
            frameon=True,
            legend_loc=None if categorical else "right margin",
            **resolved_kwargs,
        )
        self._style_embedding_axes(ax=ax)

    def _plot_umap_panel_axis(
        self,
        *,
        adata: Any,
        ax: Axes,
        panel: UmapPanel,
        basis: str,
        size: float,
        colorbar_style: ColorbarStyle,
    ) -> None:
        """Plot one obs-colored UMAP panel onto an existing axis."""
        obs_key = panel.obs_key
        kind = panel.kind
        xy = embedding_xy(adata, basis=basis)

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
            cmap = self.zero_gray_cmap(
                panel.cmap,
                zero_position=self._zero_cmap_position(
                    vmin=panel.vmin,
                    vmax=panel.vmax,
                    values=values.to_numpy(),
                ),
            )
            collection = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=values.to_numpy(),
                s=size,
                cmap=cmap,
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
                colorbar_style=colorbar_style,
                ticks=panel.cbar_ticks,
            )
            return

        raise ValueError(f"Unsupported obs UMAP panel kind: {kind!r}")

    def _style_umap_panel_axis(self, ax: Axes, panel: UmapPanel) -> None:
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
