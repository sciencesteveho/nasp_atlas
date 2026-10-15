"""Disease-versus-reference (A/B) analyses of condition cohorts."""

from nasp_atlas.analysis.condition_cohorts.cell_states import (
    contrast_cell_states,
)
from nasp_atlas.analysis.condition_cohorts.cell_states import (
    contrast_state_differences,
)
from nasp_atlas.analysis.condition_cohorts.composition import (
    ConditionComposition,
)
from nasp_atlas.analysis.condition_cohorts.composition import (
    estimate_condition_composition,
)
from nasp_atlas.analysis.condition_cohorts.condition_coupling import (
    ConditionCoupling,
)
from nasp_atlas.analysis.condition_cohorts.condition_coupling import (
    estimate_condition_coupling,
)
from nasp_atlas.analysis.condition_cohorts.condition_effects import (
    estimate_condition_effects,
)
from nasp_atlas.analysis.condition_cohorts.condition_effects import (
    estimate_continuous_effects,
)
from nasp_atlas.analysis.condition_cohorts.design import ConditionDesign
from nasp_atlas.analysis.condition_cohorts.design import audit_condition_design
from nasp_atlas.analysis.condition_cohorts.design import contrast_availability
from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    adjust_fdr_within,
)
from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    donor_regression,
)
from nasp_atlas.analysis.condition_cohorts.donor_regression import ols_hc3
from nasp_atlas.analysis.condition_cohorts.ifn_strata import IfnStrata
from nasp_atlas.analysis.condition_cohorts.ifn_strata import IfnStrataResult
from nasp_atlas.analysis.condition_cohorts.ifn_strata import stratify_by_ifn
from nasp_atlas.analysis.condition_cohorts.intake import ConditionCohortIntake
from nasp_atlas.analysis.condition_cohorts.intake import PreparedConditionCohort
from nasp_atlas.analysis.condition_cohorts.intake import (
    prepare_condition_cohort,
)
from nasp_atlas.analysis.condition_cohorts.intake import select_condition_cohort
from nasp_atlas.analysis.condition_cohorts.specification import CellStateSpec
from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.analysis.condition_cohorts.specification import ConditionSpec
from nasp_atlas.analysis.condition_cohorts.specification import (
    ContinuousContrastSpec,
)
from nasp_atlas.analysis.condition_cohorts.specification import ContrastSpec
from nasp_atlas.analysis.condition_cohorts.specification import CovariateSpec
from nasp_atlas.analysis.condition_cohorts.specification import IfnStrataSpec
from nasp_atlas.analysis.condition_cohorts.specification import (
    read_condition_cohort_spec,
)
from nasp_atlas.analysis.condition_cohorts.target_evidence import TargetEvidence
from nasp_atlas.analysis.condition_cohorts.target_evidence import (
    evidence_matrix,
)
from nasp_atlas.analysis.condition_cohorts.target_evidence import (
    summarize_target_evidence,
)
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits
from nasp_atlas.analysis.condition_cohorts.units import build_condition_units
from nasp_atlas.analysis.condition_cohorts.workflows import (
    audit_condition_cohort,
)
from nasp_atlas.analysis.condition_cohorts.workflows import (
    condition_cohort_analysis,
)


__all__ = [
    "CellStateSpec",
    "ConditionCohortIntake",
    "ConditionCohortSpec",
    "ConditionComposition",
    "ConditionCoupling",
    "ConditionDesign",
    "ConditionSpec",
    "ConditionUnits",
    "ContinuousContrastSpec",
    "ContrastSpec",
    "CovariateSpec",
    "IfnStrata",
    "IfnStrataResult",
    "IfnStrataSpec",
    "PreparedConditionCohort",
    "TargetEvidence",
    "adjust_fdr_within",
    "audit_condition_cohort",
    "audit_condition_design",
    "build_condition_units",
    "condition_cohort_analysis",
    "contrast_availability",
    "contrast_cell_states",
    "contrast_state_differences",
    "donor_regression",
    "estimate_condition_composition",
    "estimate_condition_coupling",
    "estimate_condition_effects",
    "estimate_continuous_effects",
    "evidence_matrix",
    "ols_hc3",
    "prepare_condition_cohort",
    "read_condition_cohort_spec",
    "select_condition_cohort",
    "stratify_by_ifn",
    "summarize_target_evidence",
]
