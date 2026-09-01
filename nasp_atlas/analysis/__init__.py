"""Analysis workflows."""

from nasp_atlas.analysis.clock import ClockConfig
from nasp_atlas.analysis.clock import ClockRegressionStyle
from nasp_atlas.analysis.clock import discover_tissue_h5ads
from nasp_atlas.analysis.clock import metacell_counts_frame
from nasp_atlas.analysis.clock import stratum_indices
from nasp_atlas.analysis.clock import tissue_clock_analysis
from nasp_atlas.analysis.tabula_sapiens import TabulaMixedModelResults
from nasp_atlas.analysis.tabula_sapiens import association_analysis
from nasp_atlas.analysis.tabula_sapiens import plot_global_nasp_visualizations
from nasp_atlas.analysis.tabula_sapiens import (
    plot_nasp_association_visualizations,
)
from nasp_atlas.analysis.tabula_sapiens import (
    plot_tabula_sapiens_mixed_model_inference,
)
from nasp_atlas.analysis.tabula_sapiens import (
    tabula_sapiens_mixed_model_inference,
)
from nasp_atlas.analysis.tabula_sapiens import tabula_sapiens_scoring_analysis
from nasp_atlas.analysis.tabula_sapiens import tabula_sapiens_tissue_analysis


__all__ = [
    "ClockConfig",
    "ClockRegressionStyle",
    "TabulaMixedModelResults",
    "association_analysis",
    "discover_tissue_h5ads",
    "metacell_counts_frame",
    "plot_global_nasp_visualizations",
    "plot_nasp_association_visualizations",
    "plot_tabula_sapiens_mixed_model_inference",
    "stratum_indices",
    "tabula_sapiens_mixed_model_inference",
    "tabula_sapiens_scoring_analysis",
    "tabula_sapiens_tissue_analysis",
    "tissue_clock_analysis",
]
