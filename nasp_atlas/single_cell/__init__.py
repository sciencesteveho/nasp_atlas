"""Single-cell processing and visualization utilities."""

from nasp_atlas.single_cell.associations import Aggregation
from nasp_atlas.single_cell.associations import EqtlMergeMode
from nasp_atlas.single_cell.associations import FeatureSpec
from nasp_atlas.single_cell.associations import FeatureType
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import StatisticalUnit
from nasp_atlas.single_cell.associations import aggregate_feature_frame
from nasp_atlas.single_cell.associations import benjamini_hochberg
from nasp_atlas.single_cell.associations import build_cell_feature_frame
from nasp_atlas.single_cell.associations import merge_eqtl_counts
from nasp_atlas.single_cell.associations import metadata_columns
from nasp_atlas.single_cell.associations import (
    partial_correlation_controlling_tissue,
)
from nasp_atlas.single_cell.associations import regress_features_on_continuous
from nasp_atlas.single_cell.associations import resolve_feature_specs
from nasp_atlas.single_cell.associations import (
    summarize_continuous_association_stability,
)
from nasp_atlas.single_cell.associations import summarize_feature_groups
from nasp_atlas.single_cell.associations import test_feature_groups
from nasp_atlas.single_cell.associations import validate_eqtl_table
from nasp_atlas.single_cell.config import EmbeddingConfig
from nasp_atlas.single_cell.context_summary import summarize_module_contexts
from nasp_atlas.single_cell.hypothesis_priorities import MechanisticEdgeSpec
from nasp_atlas.single_cell.hypothesis_priorities import (
    expected_module_coupling_report,
)
from nasp_atlas.single_cell.hypothesis_priorities import rank_nasp_hypotheses
from nasp_atlas.single_cell.io import random_cell_subset
from nasp_atlas.single_cell.io import read_csr_rows
from nasp_atlas.single_cell.io import read_h5ad
from nasp_atlas.single_cell.io import read_h5ad_rows
from nasp_atlas.single_cell.metacells import aggregate_counts_by_coverage
from nasp_atlas.single_cell.metacells import aggregate_metacells
from nasp_atlas.single_cell.module_profiles import EvidenceRole
from nasp_atlas.single_cell.module_profiles import RoleAssignment
from nasp_atlas.single_cell.module_profiles import module_gene_overlap
from nasp_atlas.single_cell.module_profiles import pairwise_module_correlations
from nasp_atlas.single_cell.module_profiles import (
    relative_nasp_evidence_profiles,
)
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import combine_module_scores
from nasp_atlas.single_cell.module_scoring import inverse_module_score_name
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import positive_module_score_name
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_module
from nasp_atlas.single_cell.module_scoring import score_scanpy_modules
from nasp_atlas.single_cell.score_diagnostics import compare_module_scorers
from nasp_atlas.single_cell.scprocessor import SCProcessor
from nasp_atlas.single_cell.scutils import SCUtils
from nasp_atlas.single_cell.umap import UmapPanelSpec
from nasp_atlas.single_cell.utils import dedupe_stem
from nasp_atlas.single_cell.utils import expression_matrix
from nasp_atlas.single_cell.utils import normalize_h5ad_string_storage
from nasp_atlas.single_cell.utils import snake_case
from nasp_atlas.single_cell.utils import split_anndata_by_obs
from nasp_atlas.single_cell.visualization import ColorbarStyle
from nasp_atlas.single_cell.visualization import SCVisualizer


__all__ = [
    "Aggregation",
    "ColorbarStyle",
    "EmbeddingConfig",
    "EqtlMergeMode",
    "EvidenceRole",
    "FeatureSpec",
    "FeatureType",
    "MechanisticEdgeSpec",
    "ObsSchema",
    "RoleAssignment",
    "SCProcessor",
    "SCUtils",
    "SCVisualizer",
    "ScorerName",
    "StatisticalUnit",
    "UmapPanelSpec",
    "aggregate_counts_by_coverage",
    "aggregate_feature_frame",
    "aggregate_metacells",
    "benjamini_hochberg",
    "build_cell_feature_frame",
    "combine_module_scores",
    "compare_module_scorers",
    "dedupe_stem",
    "expected_module_coupling_report",
    "expression_matrix",
    "inverse_module_score_name",
    "merge_eqtl_counts",
    "metadata_columns",
    "module_gene_overlap",
    "module_score_name",
    "normalize_h5ad_string_storage",
    "pairwise_module_correlations",
    "partial_correlation_controlling_tissue",
    "positive_module_score_name",
    "random_cell_subset",
    "rank_nasp_hypotheses",
    "read_csr_rows",
    "read_h5ad",
    "read_h5ad_rows",
    "regress_features_on_continuous",
    "relative_nasp_evidence_profiles",
    "resolve_feature_specs",
    "score_aucell_modules",
    "score_scanpy_module",
    "score_scanpy_modules",
    "snake_case",
    "split_anndata_by_obs",
    "summarize_continuous_association_stability",
    "summarize_feature_groups",
    "summarize_module_contexts",
    "test_feature_groups",
    "validate_eqtl_table",
]
