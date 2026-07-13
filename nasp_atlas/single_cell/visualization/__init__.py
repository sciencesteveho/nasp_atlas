"""Visualization utilities for single-cell datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import scanpy as sc  # type: ignore[import]

from nasp_atlas.single_cell.visualization.associations import (
    _AssociationPlotMixin,
)
from nasp_atlas.single_cell.visualization.dotplots import _DotplotMixin
from nasp_atlas.single_cell.visualization.heatmaps import _HeatmapMixin
from nasp_atlas.single_cell.visualization.nasp import _NaspPlotMixin
from nasp_atlas.single_cell.visualization.style import ColorbarStyle
from nasp_atlas.single_cell.visualization.summaries import _SummaryPlotMixin
from nasp_atlas.single_cell.visualization.umap import _UmapPlotMixin


__all__ = [
    "ColorbarStyle",
    "SCVisualizer",
]


class SCVisualizer(
    _UmapPlotMixin,
    _HeatmapMixin,
    _DotplotMixin,
    _SummaryPlotMixin,
    _AssociationPlotMixin,
    _NaspPlotMixin,
):
    """Visualization utilities for single-cell data.

    Attributes:
      output_dir: Directory where all figures are written.

    Example Usage:
      >>> import anndata as ad
      >>> from nasp_atlas.single_cell import SCVisualizer
      >>> adata = ad.read_h5ad("path/to/input.h5ad")
      >>> visualizer = SCVisualizer(output_dir="path/to/output")
      >>> visualizer.plot_embedding(
      ...     adata,
      ...     color="cell_type",
      ...     filename="cell_type_umap",
      ... )
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

        self.expression_cmap = self.pastelize_cmap("YlGnBu", blend=0.20)
        self.expression_cmap = self.zero_gray_cmap(self.expression_cmap)
        self.dotplot_cmap = self.pastelize_cmap("Blues", blend=0.35)

        logging.getLogger("matplotlib.category").setLevel(logging.WARNING + 1)
