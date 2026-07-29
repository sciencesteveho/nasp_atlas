"""Shared visualization style, layout, and colormap helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure
from mpl_toolkits.axes_grid1.inset_locator import (  # type: ignore[import]
    inset_axes,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ColorbarStyle:
    """Inset colorbar geometry and tick styling."""

    height: str | float
    width: str | float
    pad: float
    tick_length: float = 1.5
    tick_pad: float = 0.5
    title_labelpad: float = 4.0

    def with_overrides(
        self,
        *,
        height: str | float | None = None,
        width: str | float | None = None,
        pad: float | None = None,
    ) -> ColorbarStyle:
        """Return a copy with any provided geometry overrides applied."""
        return replace(
            self,
            **{
                key: value
                for key, value in {
                    "height": height,
                    "width": width,
                    "pad": pad,
                }.items()
                if value is not None
            },
        )


class _VisualizationStyleMixin:
    """Shared style, layout and colormap helpers."""

    dpi: int
    legend_w: float
    size_legend_h: float
    cbar_w: float
    cbar_h: float
    legend_inner_gap: float
    bar_h: float
    bar_gap: float
    left_margin: float
    bottom_margin: float
    annotation_height: float
    annotation_gap: float
    output_dir: Path
    expression_cmap: Colormap
    dotplot_cmap: Colormap

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
    def _add_embedding_colorbar(
        fig: Figure | SubFigure,
        ax: Axes,
        colorbar_style: ColorbarStyle,
        mappable: Any | None = None,
        ticks: Sequence[float] | None = None,
        title: str | None = None,
    ) -> None:
        """Add a consistently styled inset colorbar beside a plot."""
        cax = inset_axes(
            ax,
            width=colorbar_style.width,
            height=colorbar_style.height,
            loc="center left",
            bbox_to_anchor=(1.02 + colorbar_style.pad, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(
            mappable if mappable is not None else ax.collections[0],
            cax=cax,
            ticks=ticks,
        )
        cbar.ax.tick_params(
            length=colorbar_style.tick_length,
            pad=colorbar_style.tick_pad,
        )
        if title is not None:
            cbar.ax.set_ylabel(
                title,
                rotation=270,
                labelpad=colorbar_style.title_labelpad,
                va="bottom",
            )

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
    def _add_axes(
        *,
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
                "font.sans-serif": [
                    "Arial",
                    "Nimbus Sans",
                    "DejaVu Sans",
                ],
                "axes.linewidth": 0.25,
                "xtick.major.width": 0.25,
                "ytick.major.width": 0.25,
                "xtick.minor.width": 0.25,
                "ytick.minor.width": 0.25,
            }
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
    def zero_gray_cmap(
        cmap: Colormap | str,
        *,
        zero_position: float | Literal["low", "center", "high"] = "low",
    ) -> ListedColormap:
        """Return a copy of `cmap` with the zero position set to light gray."""
        base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        colors = base(np.linspace(0, 1, 256))
        if zero_position == "low":
            position = 0.0
        elif zero_position == "center":
            position = 0.5
        elif zero_position == "high":
            position = 1.0
        else:
            position = float(np.clip(zero_position, 0.0, 1.0))

        index = round(position * (len(colors) - 1))
        colors[index] = mcolors.to_rgba("#eeeeee")
        name = getattr(base, "name", "cmap")
        return mcolors.ListedColormap(colors, name=f"{name}_zero_gray")

    @staticmethod
    def _zero_cmap_position(
        *,
        vmin: float | None,
        vmax: float | None,
        values: Any | None = None,
    ) -> float | Literal["low"]:
        """Return the color-table position where numeric zero should appear."""
        finite_values = np.asarray([], dtype=float)
        if values is not None:
            numeric_values = np.asarray(values, dtype=float)
            finite_values = numeric_values[np.isfinite(numeric_values)]

        lower = (
            float(vmin)
            if vmin is not None
            else float(finite_values.min())
            if finite_values.size
            else None
        )
        upper = (
            float(vmax)
            if vmax is not None
            else float(finite_values.max())
            if finite_values.size
            else None
        )
        if lower is None or upper is None or lower == upper:
            return "low"
        if lower <= 0 <= upper:
            return float(np.clip((0 - lower) / (upper - lower), 0.0, 1.0))
        return "low"

    @staticmethod
    def _style_embedding_axes(
        *,
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
