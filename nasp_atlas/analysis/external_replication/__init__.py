"""Cohort intake for donor-paired external replication."""

from nasp_atlas.analysis.external_replication.cohort import CohortIntake
from nasp_atlas.analysis.external_replication.cohort import (
    prepare_cohort_counts,
)
from nasp_atlas.analysis.external_replication.cohort import read_cohort_metadata
from nasp_atlas.analysis.external_replication.cohort import select_cohort
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.analysis.external_replication.specification import (
    PopulationSpec,
)
from nasp_atlas.analysis.external_replication.specification import (
    read_cohort_spec,
)


__all__ = [
    "CohortIntake",
    "CohortSpec",
    "ComparisonSpec",
    "PopulationSpec",
    "prepare_cohort_counts",
    "read_cohort_metadata",
    "read_cohort_spec",
    "select_cohort",
]
