"""UMAP and embedding panel plotting methods."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore[import]
import scipy.sparse as sp  # type: ignore[import]
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nasp_atlas.single_cell.umap import UmapPanel
from nasp_atlas.single_cell.umap import UmapPanelSpec
from nasp_atlas.single_cell.umap import embedding_xy
from nasp_atlas.single_cell.umap import is_categorical_obs
from nasp_atlas.single_cell.umap import resolve_umap_panel_specs
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


class UmapPlotter(_VisualizationGeneMixin, _PlotterBase):
    """Render UMAP and embedding panels.

    Example Usage:
      >>> plotter = UmapPlotter(output_dir="path/to/output")
      >>> plotter.plot_umap_panel(
      ...     adata,
      ...     panels=["cell_type"],
      ...     filename="cell_type_umap",
      ... )
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dpi: int = 450,
        expression_cmap: Colormap | None = None,
    ) -> None:
        """Initialize UMAP rendering dependencies.

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

        Example Usage:
          >>> plotter.plot_embedding(
          ...     adata,
          ...     color="cell_type",
          ...     filename="cell_type_umap",
          ... )
        """
        set_matplotlib_publication_parameters()

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
          >>> plotter.plot_umap_panel(
          ...     adata,
          ...     panels=["cell_type", "module_score"],
          ...     filename="metadata_panel",
          ...     ncols=2,
          ... )
        """
        set_matplotlib_publication_parameters()
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
        self._render_umap_panel_grid(
            adata=adata,
            panels=resolved_panels,
            filename=filename,
            basis=basis,
            ncols=ncols,
            panel_w=panel_w,
            panel_h=panel_h,
            size=size,
            row_hspace=row_hspace,
            col_wspace=col_wspace,
            colorbar_style=colorbar_style,
        )

    def _render_umap_panel_grid(
        self,
        *,
        adata: Any,
        panels: Sequence[UmapPanel],
        filename: str,
        basis: str,
        ncols: int | None,
        panel_w: float,
        panel_h: float,
        size: float,
        row_hspace: float,
        col_wspace: float,
        colorbar_style: ColorbarStyle,
        numeric_values: dict[str, np.ndarray] | None = None,
        shared_colorbar: bool = False,
        cbar_title: str | None = None,
        cbar_extend: Literal["neither", "both", "min", "max"] = "neither",
    ) -> None:
        """Render resolved UMAP panels with optional shared numeric scaling."""
        if not panels:
            return

        ncols = ncols if ncols is not None else len(panels)
        nrows = math.ceil(len(panels) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_w * ncols, panel_h * nrows),
            squeeze=False,
        )

        numeric_panels: list[tuple[Axes, PathCollection, UmapPanel]] = []
        for ax, panel in zip(axes.flat, panels, strict=False):
            collection = self._plot_umap_panel_axis(
                adata=adata,
                ax=ax,
                panel=panel,
                basis=basis,
                size=size,
                colorbar_style=colorbar_style,
                values=(
                    numeric_values.get(panel.obs_key)
                    if numeric_values is not None
                    else None
                ),
                add_colorbar=not shared_colorbar,
                cbar_title=cbar_title,
                cbar_extend=cbar_extend,
            )
            if collection is not None:
                numeric_panels.append((ax, collection, panel))

        for ax in axes.flat[len(panels) :]:
            ax.axis("off")

        if shared_colorbar and numeric_panels:
            ax, collection, panel = numeric_panels[-1]
            self._add_embedding_colorbar(
                fig=fig,
                ax=ax,
                mappable=collection,
                colorbar_style=colorbar_style,
                ticks=panel.cbar_ticks,
                title=cbar_title,
                extend=cbar_extend,
            )

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
        shared_colorbar: bool = False,
        cbar_title: str | None = None,
        size: float = 8.0,
        max_rows_per_batch: int | None = 4,
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
          shared_colorbar: Whether every gene uses one colorbar with a common
            range from zero to the highest finite expression value shown.
            At least one requested gene must have positive finite expression.
          cbar_title: Optional colorbar label.
          size: Scatter point size.
          max_rows_per_batch: Maximum live scatter rows rendered at once.
            Raster batches are recomposed into the original output filename.
            Use None to render every row in one batch.

        Example Usage:
          >>> plotter.plot_multi_gene_umap_panel(
          ...     adata,
          ...     genes=["CD3D", "MS4A1"],
          ...     filename="marker_gene_umaps",
          ...     gene_symbol_column="gene_symbol",
          ...     shared_colorbar=True,
          ... )
        """
        set_matplotlib_publication_parameters()
        out = self.output_dir / filename
        if ncols <= 0:
            raise ValueError(f"ncols must be positive; got {ncols}.")
        if max_rows_per_batch is not None and max_rows_per_batch <= 0:
            raise ValueError(
                "max_rows_per_batch must be positive or None; "
                f"got {max_rows_per_batch}."
            )
        colorbar_style = (
            colorbar_style
            or ColorbarStyle(height="27.5%", width="4%", pad=0.02)
        ).with_overrides(
            height=cbar_height,
            width=cbar_width,
            pad=cbar_pad,
        )
        cmap = self._zero_gray_cmap(
            cmap if cmap is not None else self.expression_cmap
        ).with_extremes(bad="lightgray")
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

        # Only selected plotting genes are converted; repeated column reads
        # otherwise scan CSR row pointers once per gene on atlas-scale inputs.
        if isinstance(source_matrix, (sp.csr_matrix, sp.csr_array)):
            source_matrix = source_matrix.tocsc()
        gene_means = pd.Series(
            _expression_column_means(source_matrix),
            index=source_var_names,
        )
        ordered_var_names = gene_means.sort_values(
            ascending=False,
        ).index.tolist()
        label_lookup = dict(
            zip(resolved.var_names, resolved.labels, strict=True)
        )
        labels = [label_lookup[var_name] for var_name in ordered_var_names]
        source_positions = {
            var_name: position
            for position, var_name in enumerate(source_var_names)
        }
        shared_vmax = (
            self._highest_gene_expression_value(
                source_matrix,
                positions=list(source_positions.values()),
            )
            if shared_colorbar
            else None
        )
        xy = embedding_xy(adata, basis=basis)
        batch_size = (
            len(ordered_var_names)
            if max_rows_per_batch is None
            else ncols * max_rows_per_batch
        )
        n_batches = math.ceil(len(ordered_var_names) / batch_size)
        rendered_batches: list[np.ndarray] = []
        reference_panel_size_inches: tuple[float, float] | None = None
        for batch_index, start in enumerate(
            range(0, len(ordered_var_names), batch_size),
            start=1,
        ):
            stop = min(start + batch_size, len(ordered_var_names))
            if n_batches > 1:
                logger.info(
                    "[plot] rendering %s batch %d/%d",
                    filename,
                    batch_index,
                    n_batches,
                )
            batch_rgba, panel_size_inches = self._render_gene_umap_batch(
                xy=xy,
                source_matrix=source_matrix,
                source_positions=source_positions,
                var_names=ordered_var_names[start:stop],
                labels=labels[start:stop],
                ncols=ncols,
                panel_w=panel_w,
                panel_h=panel_h,
                row_hspace=row_hspace,
                cmap=cmap,
                colorbar_style=colorbar_style,
                vmax=shared_vmax,
                show_colorbars=not shared_colorbar,
                cbar_title=cbar_title,
                size=size,
            )
            rendered_batches.append(batch_rgba)
            if reference_panel_size_inches is None:
                reference_panel_size_inches = panel_size_inches

        batch_gap = round(row_hspace * panel_h * self.dpi)
        composite = self._stack_rgba_batches(
            rendered_batches,
            gap_pixels=batch_gap,
            pad_pixels=round(0.1 * self.dpi),
        )
        if (
            shared_colorbar
            and shared_vmax is not None
            and reference_panel_size_inches is not None
        ):
            self._save_gene_umap_with_shared_colorbar(
                composite,
                out=out,
                cmap=cmap,
                vmax=shared_vmax,
                colorbar_style=colorbar_style,
                cbar_title=cbar_title,
                panel_size_inches=reference_panel_size_inches,
            )
        else:
            plt.imsave(f"{out}.png", composite, dpi=self.dpi)
            logger.info("[plot] multi-gene UMAP panel -> %s", out)

    def _render_gene_umap_batch(
        self,
        *,
        xy: np.ndarray,
        source_matrix: Any,
        source_positions: dict[str, int],
        var_names: list[str],
        labels: list[str],
        ncols: int,
        panel_w: float,
        panel_h: float,
        row_hspace: float,
        cmap: Colormap,
        colorbar_style: ColorbarStyle,
        vmax: float | None,
        show_colorbars: bool,
        cbar_title: str | None,
        size: float,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Render a gene UMAP batch and report one panel's physical size."""
        nrows = math.ceil(len(var_names) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_w * ncols, panel_h * nrows),
            squeeze=False,
        )
        try:
            for ax, var_name, label in zip(
                axes.flat,
                var_names,
                labels,
                strict=False,
            ):
                values = self._dense_expression_column(
                    source_matrix,
                    source_positions[var_name],
                )
                order = np.argsort(-values, kind="stable")[::-1]
                collection = ax.scatter(
                    xy[order, 0],
                    xy[order, 1],
                    c=values[order],
                    s=size,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=vmax,
                    linewidths=0,
                    edgecolors="none",
                    plotnonfinite=True,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(label, fontstyle="italic", pad=0.0)
                ax.set_aspect("equal", adjustable="box")
                ax.set_box_aspect(1)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if show_colorbars:
                    self._add_embedding_colorbar(
                        fig=fig,
                        ax=ax,
                        mappable=collection,
                        colorbar_style=colorbar_style,
                        title=cbar_title,
                    )

            for ax in axes.flat[len(var_names) :]:
                ax.axis("off")

            fig.subplots_adjust(hspace=row_hspace, wspace=0.35)
            rgba = self._figure_rgba(fig)
            panel_bounds = axes.flat[0].get_window_extent()
            panel_size_inches = (
                panel_bounds.width / self.dpi,
                panel_bounds.height / self.dpi,
            )
            return rgba, panel_size_inches
        finally:
            plt.close(fig)

    def _figure_rgba(self, fig: Figure) -> np.ndarray:
        """Return the tight rendered extent of `fig` as uint8 RGBA pixels."""
        fig.set_dpi(self.dpi)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        renderer = canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer)
        if tight_bbox is None:
            return rgba.copy()

        pixel_bbox = tight_bbox.transformed(fig.dpi_scale_trans)
        height, width = rgba.shape[:2]
        x_start = max(0, math.floor(pixel_bbox.x0))
        x_stop = min(width, math.ceil(pixel_bbox.x1))
        y_start = max(0, height - math.ceil(pixel_bbox.y1))
        y_stop = min(height, height - math.floor(pixel_bbox.y0))
        if x_start >= x_stop or y_start >= y_stop:
            return rgba.copy()
        return rgba[y_start:y_stop, x_start:x_stop].copy()

    @staticmethod
    def _stack_rgba_batches(
        batches: Sequence[np.ndarray],
        *,
        gap_pixels: int,
        pad_pixels: int,
    ) -> np.ndarray:
        """Stack rendered batches vertically on one opaque white canvas."""
        if not batches:
            raise ValueError("at least one rendered UMAP batch is required")

        width = max(batch.shape[1] for batch in batches)
        content_height = sum(batch.shape[0] for batch in batches)
        gaps_height = max(0, len(batches) - 1) * max(0, gap_pixels)
        composite = np.full(
            (
                content_height + gaps_height + 2 * pad_pixels,
                width + 2 * pad_pixels,
                4,
            ),
            255,
            dtype=np.uint8,
        )
        y_offset = pad_pixels
        for batch in batches:
            height, batch_width = batch.shape[:2]
            x_offset = pad_pixels + (width - batch_width) // 2
            composite[
                y_offset : y_offset + height,
                x_offset : x_offset + batch_width,
            ] = batch
            y_offset += height + max(0, gap_pixels)

        return composite

    @staticmethod
    def _dense_expression_column(matrix: Any, index: int) -> np.ndarray:
        """Return one dense expression column from a bounded source matrix."""
        column = matrix[:, index]
        if sp.issparse(column):
            return column.toarray().reshape(-1)
        return np.asarray(column).reshape(-1)

    def _highest_gene_expression_value(
        self,
        matrix: Any,
        *,
        positions: Sequence[int],
    ) -> float:
        """Return the highest finite expression value across selected genes."""
        highest: float | None = None
        for position in positions:
            values = self._dense_expression_column(matrix, position)
            finite = values[np.isfinite(values)]
            if finite.size:
                candidate = float(finite.max())
                highest = (
                    candidate if highest is None else max(highest, candidate)
                )

        if highest is None:
            raise ValueError(
                "shared gene-expression colorbar requires at least one "
                "finite expression value"
            )
        if highest == 0.0:
            raise ValueError(
                "shared gene-expression colorbar requires at least one "
                "positive finite expression value; the requested 0-to-0 "
                "range is undefined"
            )
        if highest < 0.0:
            raise ValueError(
                "shared gene-expression colorbar requires a nonnegative "
                f"maximum because vmin is fixed at 0.0; received {highest}"
            )
        return highest

    def _save_gene_umap_with_shared_colorbar(
        self,
        composite: np.ndarray,
        *,
        out: Path,
        cmap: Colormap,
        vmax: float,
        colorbar_style: ColorbarStyle,
        cbar_title: str | None,
        panel_size_inches: tuple[float, float],
    ) -> None:
        """Save a stacked gene-panel raster beside one shared colorbar."""
        height, width = composite.shape[:2]
        fig, ax = plt.subplots(
            figsize=(width / self.dpi, height / self.dpi),
        )
        try:
            ax.imshow(composite)
            ax.axis("off")
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
            mappable = ScalarMappable(
                norm=mcolors.Normalize(vmin=0.0, vmax=vmax),
                cmap=cmap,
            )
            mappable.set_array([])
            panel_width, panel_height = panel_size_inches
            anchor_width = panel_width / fig.get_figwidth()
            anchor_height = panel_height / fig.get_figheight()
            colorbar_anchor = fig.add_axes(
                (
                    1.0 - anchor_width,
                    (1.0 - anchor_height) / 2.0,
                    anchor_width,
                    anchor_height,
                ),
                frameon=False,
            )
            colorbar_anchor.set_axis_off()
            self._add_embedding_colorbar(
                fig=fig,
                ax=colorbar_anchor,
                mappable=mappable,
                colorbar_style=colorbar_style,
                title=cbar_title,
            )
            self._save_figure_and_log(
                fig,
                out,
                "[plot] multi-gene UMAP panel -> %s",
            )
        finally:
            plt.close(fig)

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
        shared_colorbar: bool = False,
        standardization: Literal["none", "zscore"] = "none",
        cbar_title: str | None = None,
        cbar_extend: Literal["neither", "both", "min", "max"] = "neither",
        size: float = 8.0,
        vmin: float | None = 0,
        vmax: float | None = None,
        center_zero: bool = False,
    ) -> None:
        """Save a multi-panel UMAP figure for numeric obs columns.

        Applies one numeric panel style to several observation columns. Values
        can remain on their native scale or be z-scored independently within
        each column across the finite cells displayed. The transformation is
        plot-local and does not mutate `adata`.

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
          shared_colorbar: Whether numeric panels use one colorbar and one
            common pair of color limits.
          standardization: Value transformation ("none" or "zscore").
            Population z-scores use finite displayed cells and `ddof=0`.
          cbar_title: Optional colorbar label.
          cbar_extend: Which colorbar ends indicate values beyond the displayed
            color limits ("neither", "both", "min", or "max").
          size: Point size forwarded to Scanpy.
          vmin: Lower color limit. Defaults to 0 so zero maps to gray with the
            default expression colormap.
          vmax: Upper color limit.
          center_zero: Whether to derive per-panel symmetric color limits
            around zero when explicit limits are not supplied.

        Example Usage:
          >>> plotter.plot_multi_obs_umap_panel(
          ...     adata,
          ...     obs_keys=["module_score", "age_accel"],
          ...     filename="score_umaps",
          ...     shared_colorbar=True,
          ...     standardization="zscore",
          ...     center_zero=True,
          ...     vmin=-3.0,
          ...     vmax=3.0,
          ... )
        """
        set_matplotlib_publication_parameters()
        if standardization not in {"none", "zscore"}:
            raise ValueError(
                "standardization must be 'none' or 'zscore'; "
                f"received {standardization!r}"
            )
        if standardization == "zscore":
            if not center_zero:
                raise ValueError(
                    "standardization='zscore' requires center_zero=True"
                )
            if (vmin is None) != (vmax is None):
                raise ValueError(
                    "standardized score colors require both vmin and vmax "
                    "or neither"
                )
            if vmin is not None and vmin >= 0.0:
                raise ValueError(
                    "standardized score colors require a negative vmin or "
                    "vmin=None"
                )
            if (
                vmin is not None
                and vmax is not None
                and not np.isclose(abs(vmin), abs(vmax))
            ):
                raise ValueError(
                    "standardized score colors require symmetric vmin/vmax"
                )

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
            cmap if cmap is not None else self._umap_expression_cmap("viridis")
        )
        values_by_key = {
            key: self._numeric_obs_values(
                adata,
                key,
                standardization=standardization,
            )
            for key in valid_keys
        }
        shared_limits = (
            self._shared_obs_color_limits(
                values_by_key,
                vmin=vmin,
                vmax=vmax,
                center_zero=center_zero,
            )
            if shared_colorbar
            else None
        )
        panels: list[UmapPanelSpec] = []
        for key in valid_keys:
            if shared_limits is None:
                panel_vmin = vmin
                panel_vmax = vmax
                if center_zero and (panel_vmin is None or panel_vmax is None):
                    limit = self._symmetric_value_limit(values_by_key[key])
                    if panel_vmin is None:
                        panel_vmin = -limit
                    if panel_vmax is None:
                        panel_vmax = limit
            else:
                panel_vmin, panel_vmax = shared_limits
            panels.append(
                {
                    "obs_key": key,
                    "kind": "numeric",
                    "cmap": self._zero_gray_cmap(
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
        resolved_panels = resolve_umap_panel_specs(adata, panels)
        self._render_umap_panel_grid(
            adata=adata,
            panels=resolved_panels,
            filename=filename,
            basis=basis,
            ncols=ncols,
            panel_w=panel_w,
            panel_h=panel_h,
            size=size,
            row_hspace=row_hspace,
            col_wspace=0.35,
            colorbar_style=colorbar_style,
            numeric_values=values_by_key,
            shared_colorbar=shared_colorbar,
            cbar_title=(
                "Within-column z-score"
                if standardization == "zscore" and cbar_title is None
                else cbar_title
            ),
            cbar_extend=cbar_extend,
        )

    @staticmethod
    def _numeric_obs_values(
        adata: Any,
        obs_key: str,
        *,
        standardization: Literal["none", "zscore"],
    ) -> np.ndarray:
        """Return finite-preserving numeric values for one UMAP panel."""
        values = pd.to_numeric(adata.obs[obs_key], errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        if standardization == "none":
            return values

        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        standardized = np.full(values.shape, np.nan, dtype=float)
        if finite_values.size == 0:
            logger.warning(
                "[plot] %s has no finite values for z-score UMAP", obs_key
            )
            return standardized

        standard_deviation = float(finite_values.std(ddof=0))
        if standard_deviation == 0.0:
            logger.warning(
                "[plot] %s is constant; z-score UMAP renders it as unavailable",
                obs_key,
            )
            return standardized
        if not np.isfinite(standard_deviation):
            logger.warning(
                "[plot] %s has a non-finite standard deviation; z-score "
                "UMAP renders it as unavailable",
                obs_key,
            )
            return standardized

        standardized[finite_mask] = (
            finite_values - float(finite_values.mean())
        ) / standard_deviation
        return standardized

    @staticmethod
    def _shared_obs_color_limits(
        values_by_key: dict[str, np.ndarray],
        *,
        vmin: float | None,
        vmax: float | None,
        center_zero: bool,
    ) -> tuple[float, float]:
        """Resolve one finite color range across numeric UMAP panels."""
        finite_minimum: float | None = None
        finite_maximum: float | None = None
        for values in values_by_key.values():
            finite = values[np.isfinite(values)]
            if not finite.size:
                continue
            candidate_minimum = float(finite.min())
            candidate_maximum = float(finite.max())
            finite_minimum = (
                candidate_minimum
                if finite_minimum is None
                else min(finite_minimum, candidate_minimum)
            )
            finite_maximum = (
                candidate_maximum
                if finite_maximum is None
                else max(finite_maximum, candidate_maximum)
            )

        if finite_minimum is None or finite_maximum is None:
            if vmin is None or vmax is None:
                raise ValueError(
                    "shared color limits require finite values or explicit "
                    "vmin and vmax"
                )
            resolved_vmin = float(vmin)
            resolved_vmax = float(vmax)
        elif center_zero:
            limit = max(abs(finite_minimum), abs(finite_maximum))
            resolved_vmin = -limit if vmin is None else float(vmin)
            resolved_vmax = limit if vmax is None else float(vmax)
        else:
            resolved_vmin = finite_minimum if vmin is None else float(vmin)
            resolved_vmax = finite_maximum if vmax is None else float(vmax)
        if (
            not np.isfinite(resolved_vmin)
            or not np.isfinite(resolved_vmax)
            or resolved_vmin >= resolved_vmax
        ):
            raise ValueError(
                "shared color limits must be finite with vmin < vmax; "
                f"received vmin={resolved_vmin}, vmax={resolved_vmax}"
            )
        return resolved_vmin, resolved_vmax

    @staticmethod
    def _symmetric_value_limit(values: np.ndarray) -> float:
        """Return a non-zero symmetric limit for numeric panel values."""
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
                resolved_kwargs["color_map"] = self._umap_expression_cmap(
                    "viridis"
                )
            else:
                resolved_kwargs[cmap_key] = self._zero_gray_cmap(
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
        values: np.ndarray | None,
        add_colorbar: bool,
        cbar_title: str | None,
        cbar_extend: Literal["neither", "both", "min", "max"],
    ) -> PathCollection | None:
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
            return None

        if kind == "numeric":
            numeric_values = (
                pd.to_numeric(adata.obs[obs_key], errors="coerce").to_numpy(
                    dtype=float,
                    copy=False,
                )
                if values is None
                else np.asarray(values, dtype=float)
            )
            if numeric_values.shape != (xy.shape[0],):
                raise ValueError(
                    f"numeric values for {obs_key!r} must have shape "
                    f"({xy.shape[0]},); received {numeric_values.shape}"
                )
            cmap = self._zero_gray_cmap(
                panel.cmap,
                zero_position=self._zero_cmap_position(
                    vmin=panel.vmin,
                    vmax=panel.vmax,
                    values=numeric_values,
                ),
            )
            collection = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=numeric_values,
                s=size,
                cmap=cmap,
                vmin=panel.vmin,
                vmax=panel.vmax,
                linewidths=0,
                rasterized=True,
            )
            self._style_umap_panel_axis(ax, panel)
            if not np.isfinite(numeric_values).any():
                ax.text(
                    0.5,
                    0.5,
                    "Unavailable",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                )
            if add_colorbar:
                self._add_embedding_colorbar(
                    fig=ax.figure,
                    ax=ax,
                    mappable=collection,
                    colorbar_style=colorbar_style,
                    ticks=panel.cbar_ticks,
                    title=cbar_title,
                    extend=cbar_extend,
                )
            return collection

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
