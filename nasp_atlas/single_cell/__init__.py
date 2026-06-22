"""Single-cell processing and visualization utilities."""

from nasp_atlas.single_cell.config import EmbeddingConfig
from nasp_atlas.single_cell.io import read_h5ad
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import combine_module_scores
from nasp_atlas.single_cell.module_scoring import inverse_module_score_name
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import plot_module_gene_umaps
from nasp_atlas.single_cell.module_scoring import positive_module_score_name
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_module
from nasp_atlas.single_cell.module_scoring import score_scanpy_modules
from nasp_atlas.single_cell.scprocessor import SCProcessor
from nasp_atlas.single_cell.scutils import SCUtils
from nasp_atlas.single_cell.utils import dedupe_stem
from nasp_atlas.single_cell.utils import normalize_h5ad_string_storage
from nasp_atlas.single_cell.utils import snake_case
from nasp_atlas.single_cell.utils import split_anndata_by_obs
from nasp_atlas.single_cell.visualization import SCVisualizer
from nasp_atlas.single_cell.visualization import UmapPanelSpec


__all__ = [
    "EmbeddingConfig",
    "SCProcessor",
    "SCUtils",
    "SCVisualizer",
    "ScorerName",
    "UmapPanelSpec",
    "combine_module_scores",
    "dedupe_stem",
    "inverse_module_score_name",
    "module_score_name",
    "normalize_h5ad_string_storage",
    "plot_module_gene_umaps",
    "positive_module_score_name",
    "read_h5ad",
    "score_aucell_modules",
    "score_scanpy_module",
    "score_scanpy_modules",
    "snake_case",
    "split_anndata_by_obs",
]
