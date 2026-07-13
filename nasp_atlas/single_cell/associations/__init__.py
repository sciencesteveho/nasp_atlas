"""Donor-aware association testing for module scores and sensor genes.

This module provides the public association analysis API. Implementation lives
in internal domain modules split by feature construction, aggregation,
continuous tests, categorical tests and eQTL joins.
"""

from __future__ import annotations

from nasp_atlas.single_cell.associations.aggregation import (
    aggregate_feature_frame,
)
from nasp_atlas.single_cell.associations.categorical import (
    summarize_feature_groups,
)
from nasp_atlas.single_cell.associations.categorical import test_feature_groups
from nasp_atlas.single_cell.associations.continuous import (
    partial_correlation_controlling_tissue,
)
from nasp_atlas.single_cell.associations.continuous import (
    regress_features_on_continuous,
)
from nasp_atlas.single_cell.associations.core import Aggregation
from nasp_atlas.single_cell.associations.core import EqtlMergeMode
from nasp_atlas.single_cell.associations.core import FeatureSpec
from nasp_atlas.single_cell.associations.core import FeatureType
from nasp_atlas.single_cell.associations.core import ObsSchema
from nasp_atlas.single_cell.associations.core import StatisticalUnit
from nasp_atlas.single_cell.associations.core import benjamini_hochberg
from nasp_atlas.single_cell.associations.core import metadata_columns
from nasp_atlas.single_cell.associations.eqtl import merge_eqtl_counts
from nasp_atlas.single_cell.associations.eqtl import validate_eqtl_table
from nasp_atlas.single_cell.associations.features import (
    build_cell_feature_frame,
)
from nasp_atlas.single_cell.associations.features import resolve_feature_specs
from nasp_atlas.single_cell.associations.stability import (
    summarize_continuous_association_stability,
)


__all__ = [
    "Aggregation",
    "EqtlMergeMode",
    "FeatureSpec",
    "FeatureType",
    "ObsSchema",
    "StatisticalUnit",
    "aggregate_feature_frame",
    "benjamini_hochberg",
    "build_cell_feature_frame",
    "merge_eqtl_counts",
    "metadata_columns",
    "partial_correlation_controlling_tissue",
    "regress_features_on_continuous",
    "resolve_feature_specs",
    "summarize_continuous_association_stability",
    "summarize_feature_groups",
    "test_feature_groups",
    "validate_eqtl_table",
]
