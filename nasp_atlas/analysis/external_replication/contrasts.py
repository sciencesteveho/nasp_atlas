"""Measurement-aware paired module estimates and discovery eligibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.single_cell.associations import PairedContrastResult
from nasp_atlas.single_cell.associations import paired_feature_contrasts


__all__ = ["estimate_module_contrasts", "resolve_discovery_eligibility"]


def estimate_module_contrasts(
    donor_scores: pd.DataFrame,
    measurement_eligibility: pd.DataFrame,
    comparison: ComparisonSpec,
    *,
    alpha: float = 0.05,
) -> PairedContrastResult:
    """Estimate one frozen cohort/scope, retaining unscored module rows.

    Donor support is evaluated on finite, fully scored donor/context units.
    A failed arm cannot yield an inferential result even when its final score
    varies. Its estimate is retained descriptively, with uncertainty withheld.
    Inputs are not modified; no multiplicity or replication claim is applied.

    Estimates carry `cohort_spec_id`, the scored cohort specification's id. It
    joins to the donor tables' `cohort_id`, which contrast tables built from
    the registry use for the registry key instead. Without donor rows, as for
    an arm that cannot be scored at all, the id is not observable and stays
    missing rather than being guessed.
    """
    keys = ["module_id", "scorer"]
    if measurement_eligibility.duplicated(keys).any():
        raise ValueError(
            "Measurement eligibility must identify each module/scorer"
        )
    required = {
        "cohort_id",
        "donor_id",
        "feature_id",
        "feature_value",
        "n_cells",
        "eligible",
        "eligibility_reason",
        comparison.level_key,
    }
    missing = required.difference(donor_scores.columns)
    if missing:
        raise ValueError(
            "Donor scores are missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    for key in ("cohort_id", "scoring_context_id", "modality"):
        if donor_scores[key].nunique() > 1:
            raise ValueError(f"Contrast input must have one {key}")
    fixed_axis = "population" if comparison.level_key == "tissue" else "tissue"
    if donor_scores[fixed_axis].nunique() > 1:
        raise ValueError(f"Contrast input must fix one {fixed_axis}")
    declared = pd.MultiIndex.from_frame(measurement_eligibility[keys])
    observed = pd.MultiIndex.from_frame(donor_scores[keys])
    if not observed.isin(declared).all():
        raise ValueError("Donor scores include undeclared module/scorer pairs")

    estimates, differences = [], []
    for scorer, eligibility in measurement_eligibility.groupby(
        "scorer", observed=True, sort=False
    ):
        selected = donor_scores.loc[donor_scores.scorer.eq(scorer)].copy()
        supported = selected.eligible & selected.n_cells.ge(
            comparison.minimum_cells
        )
        selected["support_reason"] = selected.eligibility_reason.mask(
            selected.eligible & ~supported, "insufficient_cells"
        )
        selected.loc[~supported, "feature_value"] = np.nan
        paired = paired_feature_contrasts(
            selected,
            pair_keys=("cohort_id", "donor_id"),
            level_key=comparison.level_key,
            target_level=comparison.target,
            reference_level=comparison.reference,
            minimum_pairs=comparison.minimum_pairs,
            alpha=alpha,
        )
        result = eligibility.rename(
            columns={"status": "measurement_status"}
        ).merge(
            paired.estimates.rename(
                columns={
                    "feature_label": "module_id",
                    "status": "inference_status",
                }
            ),
            on="module_id",
            how="left",
            validate="one_to_one",
        )
        result["inference_status"] = result.inference_status.fillna(
            "not_scored"
        )
        result["eligible"] = result.measurement_status.eq(
            "ok"
        ) & result.inference_status.eq("ok")
        result["eligibility_reason"] = np.where(
            result.measurement_status.ne("ok"),
            result.measurement_status,
            result.inference_status,
        )
        unavailable_measurement = result.measurement_status.ne("ok")
        result.loc[
            unavailable_measurement,
            [
                "standard_error",
                "ci_lower",
                "ci_upper",
                "statistic",
                "pvalue",
            ],
        ] = np.nan
        for column in (
            "n_target_donors",
            "n_reference_donors",
            "n_paired_donors",
        ):
            result[column] = result[column].fillna(0).astype(int)
        result["cohort_spec_id"] = (
            str(donor_scores.cohort_id.iloc[0])
            if not donor_scores.empty
            else pd.NA
        )
        result["comparison_axis"] = comparison.level_key
        result["target_level"] = comparison.target
        result["reference_level"] = comparison.reference
        estimates.append(result)
        differences.append(
            _annotate_support_reasons(
                paired.donor_differences, selected, comparison
            ).assign(scorer=scorer)
        )

    if not estimates:
        raise ValueError("Declare at least one module/scorer eligibility row")
    return PairedContrastResult(
        estimates=pd.concat(estimates, ignore_index=True),
        donor_differences=pd.concat(differences, ignore_index=True),
    )


def _annotate_support_reasons(
    donor_differences: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: ComparisonSpec,
) -> pd.DataFrame:
    """Record why each person's target and reference unit did or did not count.

    A unit removed by the support floor and one whose cell scores were not
    finite both leave the difference non-estimable, but they are different
    states: the first is a design exclusion, the second a measurement failure.
    Reported together they are indistinguishable, so each context keeps its own
    reason. A person with no row at all for a context is "context_absent",
    which is distinct from having a row that fell under the floor.
    """
    keys = ["cohort_id", "donor_id", "feature_id"]
    annotated = donor_differences
    for column, level in (
        ("target_support_reason", comparison.target),
        ("reference_support_reason", comparison.reference),
    ):
        support = selected.loc[
            selected[comparison.level_key].eq(level),
            [*keys, "support_reason"],
        ].rename(columns={"support_reason": column})
        annotated = annotated.merge(
            support, on=keys, how="left", validate="many_to_one"
        )
        annotated[column] = annotated[column].fillna("context_absent")
    return annotated


def resolve_discovery_eligibility(
    hypotheses: pd.DataFrame,
    reference_estimates: pd.DataFrame,
    *,
    discovery_artifact: str,
    source_gate_reason: str = "",
    minimum_pairs: int = 3,
) -> pd.DataFrame:
    """Require expected signs in both TS methods without selecting on p-values.

    Input hypotheses describe one primary comparison. The reference estimates
    must come from its declared TS counterpart. A nonempty source_gate_reason
    withholds confirmation regardless of scores; it cannot be cleared by data.
    Returns a resolved registry copy, retaining every planned hypothesis.
    """
    if not hypotheses.analysis_role.eq("primary").all():
        raise ValueError(
            "Discovery eligibility applies only to primary hypotheses"
        )
    if not hypotheses.expected_direction.isin([-1, 1]).all():
        raise ValueError("Register a positive or negative discovery direction")
    if reference_estimates.duplicated(["module_id", "scorer"]).any():
        raise ValueError(
            "Reference must identify one estimate per module/scorer"
        )
    resolved = hypotheses.copy()
    resolved["discovery_artifact"] = discovery_artifact
    for index, hypothesis in hypotheses.iterrows():
        reference = reference_estimates.loc[
            reference_estimates.module_id.eq(hypothesis.module_id)
        ]
        reason = source_gate_reason or _discovery_failure(
            reference, hypothesis, minimum_pairs=minimum_pairs
        )
        resolved.loc[index, "eligibility_status"] = (
            "unavailable" if reason else "eligible"
        )
        resolved.loc[index, "eligibility_reason"] = reason or "eligible"
    return resolved


def _discovery_failure(
    reference: pd.DataFrame,
    hypothesis: pd.Series,
    *,
    minimum_pairs: int,
) -> str:
    """Name why the reference cannot support a registered direction.

    An empty string means it can. A missing module, an unusable measurement
    and thin donor support are pipeline states, while disagreeing signs are
    evidence about the question itself. Reporting them all as one reason would
    let a future gap in the reference read as biological non-replication.
    """
    if set(reference.scorer) != {"scanpy", "aucell"}:
        return "discovery_reference_missing_for_module"
    if (
        not reference.measurement_status.eq("ok").all()
        or not np.isfinite(reference.estimate).all()
    ):
        return "discovery_reference_not_measurable"
    if not reference.n_paired_donors.ge(minimum_pairs).all():
        return "discovery_reference_support_below_minimum"
    if not (reference.estimate * hypothesis.expected_direction > 0).all():
        return "discovery_not_supported_for_registered_contrast"
    return ""
