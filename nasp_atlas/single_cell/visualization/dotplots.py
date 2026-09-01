"""Dotplot statistics, rendering, and public plotting methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad  # type: ignore[import]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore[import]
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.typing import ColorType
from scipy.cluster.hierarchy import (  # type: ignore[import]
    dendrogram as scipy_dendrogram,
)

from nasp_atlas.single_cell.visualization.gene_resolution import (
    _VisualizationGeneMixin,
)
from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DotplotStats:
    """Per-group mean expression and fraction-expressing cells."""

    mean_exp: pd.DataFrame
    frac_exp: pd.DataFrame
    categories: list[str]


class DotplotPlotter(_VisualizationGeneMixin, _PlotterBase):
    """Render marker and ranked-gene dotplots.

    Example Usage:
      >>> plotter = DotplotPlotter(output_dir="path/to/output")
      >>> plotter.plot_marker_dotplot(
      ...     adata,
      ...     groupby="cell_type",
      ...     marker_groups={"Sensors": ["CGAS", "STING1"]},
      ...     filename="sensor_markers",
      ... )
    """

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

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dpi: int = 450,
        dotplot_cmap: Colormap | None = None,
    ) -> None:
        """Initialize dotplot rendering dependencies.

        Args:
          output_dir: Directory where figures are written.
          dpi: Saved PNG resolution.
          dotplot_cmap: Optional dotplot colormap override.
        """
        super().__init__(output_dir, dpi=dpi)
        self.dotplot_cmap = (
            dotplot_cmap
            if dotplot_cmap is not None
            else self._pastelize_cmap("Blues", blend=0.35)
        )

    @staticmethod
    def _compute_dotplot_stats(
        *,
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
        *,
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
            fig=fig,
            fig_w=fig_width,
            fig_h=fig_height,
            x=self.left_margin,
            y=self.bottom_margin,
            w=plot_width,
            h=self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=bottom_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=(-0.5, n_genes - 0.5),
            bar_height=0.5,
        )

        top_bar_ax = self._add_axes(
            fig=fig,
            fig_w=fig_width,
            fig_h=fig_height,
            x=self.left_margin,
            y=scatter_y + plot_height + self.bar_gap,
            w=plot_width,
            h=self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=top_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=(-0.5, n_genes - 0.5),
            bar_height=0.5,
        )

        header_ax = self._add_axes(
            fig=fig,
            fig_w=fig_width,
            fig_h=fig_height,
            x=self.left_margin,
            y=scatter_y
            + plot_height
            + self.bar_gap
            + self.bar_h
            + self.annotation_gap,
            w=plot_width,
            h=self.annotation_height,
        )
        self._draw_dotplot_group_headers(
            hax=header_ax,
            group_labels=group_labels,
            group_positions=group_positions,
            n_genes=n_genes,
        )

    @staticmethod
    def _draw_dotplot_group_headers(
        *,
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
        *,
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
    def _draw_dotplot_colorbar(scat: PathCollection, cax: Axes) -> None:
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

        size_ax = DotplotPlotter._add_axes(
            fig=fig,
            fig_w=fig_w,
            fig_h=fig_h,
            x=legend_left,
            y=size_bottom,
            w=legend_w,
            h=size_legend_h,
        )
        DotplotPlotter._draw_dotplot_size_legend(
            sax=size_ax,
            largest_dot=largest_dot,
            size_exponent=size_exponent,
            dot_edge_color=dot_edge_color,
            dot_edge_lw=dot_edge_lw,
        )

        cbar_ax = DotplotPlotter._add_axes(
            fig=fig,
            fig_w=fig_w,
            fig_h=fig_h,
            x=cbar_left,
            y=cbar_bottom,
            w=cbar_w,
            h=cbar_h,
        )
        DotplotPlotter._draw_dotplot_colorbar(scat=scat, cax=cbar_ax)

    @staticmethod
    def _extract_rank_genes_grouped(
        *,
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

        Example Usage:
          >>> plotter.plot_marker_dotplot(
          ...     adata,
          ...     groupby="cell_type",
          ...     marker_groups={"T cells": ["CD3D", "CD3E"]},
          ...     filename="marker_dotplot",
          ... )
        """
        set_matplotlib_publication_parameters()
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

        categories_order = self._get_dendrogram_order(
            adata=adata,
            groupby=groupby,
        )
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
        """Save a scaled mean expression ranked-genes dot plot.

        Example Usage:
          >>> plotter.plot_rank_genes_dotplot(
          ...     adata,
          ...     groupby="cell_type",
          ...     rank_key="rank_genes_groups",
          ...     filename="ranked_marker_dotplot",
          ... )
        """
        set_matplotlib_publication_parameters()
        out = self.output_dir / filename

        categories_order = self._get_dendrogram_order(
            adata=adata,
            groupby=groupby,
        )
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
            fig=fig,
            fig_w=fig_w,
            fig_h=fig_h,
            x=self.left_margin,
            y=scatter_y,
            w=plot_w,
            h=plot_h,
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
            fig=fig,
            fig_w=fig_w,
            fig_h=fig_h,
            x=self.left_margin,
            y=self.bottom_margin,
            w=plot_w,
            h=self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=bottom_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=xlim,
        )

        top_bar_ax = self._add_axes(
            fig=fig,
            fig_w=fig_w,
            fig_h=fig_h,
            x=self.left_margin,
            y=scatter_y + plot_h + self.bar_gap,
            w=plot_w,
            h=self.bar_h,
        )
        self._draw_dotplot_group_bars(
            ax=top_bar_ax,
            group_positions=group_positions,
            group_colors=group_colors,
            xlim=xlim,
        )

        header_ax = self._add_axes(
            fig=fig,
            fig_w=fig_w,
            fig_h=fig_h,
            x=self.left_margin,
            y=scatter_y
            + plot_h
            + self.bar_gap
            + self.bar_h
            + self.annotation_gap,
            w=plot_w,
            h=self.annotation_height,
        )
        self._draw_dotplot_group_headers(
            hax=header_ax,
            group_labels=group_labels,
            group_positions=group_positions,
            n_genes=n_slots,
            rotation=header_rotation,
        )

        dendro_ax = self._add_axes(
            fig=fig,
            fig_w=fig_w,
            fig_h=fig_h,
            x=self.left_margin + plot_w,
            y=scatter_y,
            w=dendro_w,
            h=plot_h,
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

    @staticmethod
    def _get_dendrogram_order(
        *,
        adata: ad.AnnData,
        groupby: str,
    ) -> list[str]:
        """Return the dendrogram category order, computing it when absent."""
        dendro_key = f"dendrogram_{groupby}"
        if dendro_key not in adata.uns:
            sc.tl.dendrogram(adata, groupby=groupby)

        categories = adata.uns[dendro_key]["categories_ordered"]
        return [str(category) for category in categories]
