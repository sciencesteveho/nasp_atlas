"""Sensitivity of paired effects while keeping primary scores fixed."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.single_cell.associations import PairedContrastResult


__all__ = ["fixed_score_sensitivities", "matched_donor_effects"]


def fixed_score_sensitivities(
    donor_scores: pd.DataFrame,
    measurement_eligibility: pd.DataFrame,
    comparison: ComparisonSpec,
    *,
    support_thresholds: Sequence[int],
    exclusions: Mapping[str, Sequence[str]],
    donor_metadata: pd.DataFrame | None = None,
    strata_keys: Sequence[str] = (),
    alpha: float = 0.05,
) -> PairedContrastResult:
    """Re-estimate after donor deletion, fixed support changes and exclusions.

    The same module scores and donor means are used throughout. These are
    influence/support diagnostics, not recalibration or extra confirmation
    tests. All modules/methods remain present, including unsupported variants.
    Optional strata describe people without claiming adjusted associations.
    Inputs are not modified. No multiplicity decision replaces the primary.
    """
    if any(
        threshold < comparison.minimum_cells for threshold in support_thresholds
    ):
        raise ValueError(
            "Fixed-score support checks cannot relax baseline selection"
        )
    baseline = estimate_module_contrasts(
        donor_scores, measurement_eligibility, comparison, alpha=alpha
    )
    people = donor_scores.donor_id.drop_duplicates().sort_values().tolist()
    selections = [("baseline", "baseline", people, comparison)]
    selections.extend(
        (
            f"omit:{donor}",
            "donor_deletion",
            [p for p in people if p != donor],
            comparison,
        )
        for donor in people
    )
    selections.extend(
        (
            f"minimum_cells:{threshold}",
            "cell_support",
            people,
            replace(comparison, minimum_cells=threshold),
        )
        for threshold in support_thresholds
    )
    selections.extend(
        (
            f"exclude:{name}",
            "participant_exclusion",
            [p for p in people if p not in omitted],
            comparison,
        )
        for name, omitted in exclusions.items()
    )
    if strata_keys:
        if donor_metadata is None or donor_metadata.donor_id.duplicated().any():
            raise ValueError("Strata require one metadata record per person")
        metadata = donor_metadata.set_index("donor_id").reindex(people)
        for key in strata_keys:
            # Missing metadata is an explicit descriptive stratum.
            labels = metadata[key].astype("string").fillna("unrecorded")
            for label in labels.drop_duplicates().sort_values():
                members = metadata.index[labels.eq(label)].tolist()
                selections.append(
                    (
                        f"stratum:{key}:{label}",
                        "descriptive_stratum",
                        members,
                        comparison,
                    )
                )

    estimates, differences = [], []
    for variant, kind, members, support in selections:
        selected = donor_scores.loc[donor_scores.donor_id.isin(members)]
        result = (
            baseline
            if kind == "baseline"
            else estimate_module_contrasts(
                selected, measurement_eligibility, support, alpha=alpha
            )
        )
        effect = result.estimates.merge(
            matched_donor_effects(
                baseline.donor_differences, result.donor_differences
            ),
            on=["module_id", "scorer"],
            how="left",
            validate="one_to_one",
        )
        effect["n_matched_donors"] = effect.n_matched_donors.fillna(0).astype(
            int
        )
        effect["matched_donors"] = effect.matched_donors.fillna("[]")
        if kind == "descriptive_stratum":
            effect[
                [
                    "pvalue",
                    "statistic",
                    "standard_error",
                    "ci_lower",
                    "ci_upper",
                ]
            ] = np.nan
            effect["eligible"] = False
            effect["eligibility_reason"] = "descriptive_stratum"
        effect["variant_id"] = variant
        effect["variant_kind"] = kind
        effect["minimum_cells"] = support.minimum_cells
        estimates.append(effect)
        differences.append(result.donor_differences.assign(variant_id=variant))
    return PairedContrastResult(
        pd.concat(estimates, ignore_index=True),
        pd.concat(differences, ignore_index=True),
    )


def matched_donor_effects(
    baseline: pd.DataFrame, variant: pd.DataFrame
) -> pd.DataFrame:
    """Compare descriptive means on each module/method's complete intersection.

    Inputs are the paired kernel's donor_differences, with original module
    identity in feature_label. The shared donor list and both matched means
    separate a measurement change from a change in sampled people. These means
    carry no independent inference; callers keep full variant estimates too.
    """
    keys = ["cohort_id", "donor_id", "feature_label", "scorer"]
    if baseline.duplicated(keys).any() or variant.duplicated(keys).any():
        raise ValueError(
            "Matched comparisons require unique person/module/method rows"
        )
    left = baseline.loc[baseline.status.eq("complete"), [*keys, "difference"]]
    right = variant.loc[variant.status.eq("complete"), [*keys, "difference"]]
    matched = left.merge(
        right,
        on=keys,
        validate="one_to_one",
        suffixes=("_baseline", "_variant"),
    )
    records = []
    for (module, scorer), rows in matched.groupby(
        ["feature_label", "scorer"], sort=False
    ):
        records.append(
            {
                "module_id": module,
                "scorer": scorer,
                "n_matched_donors": len(rows),
                "matched_donors": json.dumps(sorted(rows.donor_id.astype(str))),
                "baseline_matched_estimate": rows.difference_baseline.mean(),
                "variant_matched_estimate": rows.difference_variant.mean(),
            }
        )
    comparisons = (
        variant[["feature_label", "scorer"]]
        .drop_duplicates()
        .rename(columns={"feature_label": "module_id"})
    )
    effects = pd.DataFrame(
        records,
        columns=[
            "module_id",
            "scorer",
            "n_matched_donors",
            "matched_donors",
            "baseline_matched_estimate",
            "variant_matched_estimate",
        ],
    )
    result = comparisons.merge(
        effects, on=["module_id", "scorer"], how="left", validate="one_to_one"
    )
    result["n_matched_donors"] = result.n_matched_donors.fillna(0).astype(int)
    result["matched_donors"] = result.matched_donors.fillna("[]")
    return result
