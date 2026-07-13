"""CELLxGENE metadata visualization API."""

from nasp_atlas.cellxgene.visualization.age import plot_age_makeup
from nasp_atlas.cellxgene.visualization.age import plot_age_ranges
from nasp_atlas.cellxgene.visualization.composition import metadata_barplot
from nasp_atlas.cellxgene.visualization.composition import plot_category_makeup
from nasp_atlas.cellxgene.visualization.composition import plot_stacked_bar
from nasp_atlas.cellxgene.visualization.sankey import metadata_sankey


__all__ = [
    "metadata_barplot",
    "metadata_sankey",
    "plot_age_makeup",
    "plot_age_ranges",
    "plot_category_makeup",
    "plot_stacked_bar",
]
