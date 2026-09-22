"""Fixed-family inference and explicit replication/association decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


__all__ = ["adjust_registered_families", "classify_registered_contrasts"]


def classify_registered_contrasts(
    hypotheses: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Retain every question/method, separating eligibility from support.

    Inputs describe one external cohort. Discovery/source eligibility belongs
    to the frozen primary hypotheses; numerical eligibility belongs to paired
    estimates. Extension questions do not require a TS discovery sign. AUCell
    retains its effect/uncertainty as sensitivity and never receives a primary
    replication decision. Inputs are not modified.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if hypotheses.empty or hypotheses.hypothesis_id.duplicated().any():
        raise ValueError("Require nonempty, uniquely identified hypotheses")
    if hypotheses.cohort_id.nunique() != 1:
        raise ValueError("Classify one cohort at a time")
    keys = [
        "module_id",
        "comparison_axis",
        "target_context",
        "reference_context",
        "scorer",
    ]
    numerical = estimates.rename(
        columns={
            "target_level": "target_context",
            "reference_level": "reference_context",
            "eligible": "numerically_eligible",
            "eligibility_reason": "numerical_reason",
        }
    )
    if numerical.duplicated(keys).any():
        raise ValueError(
            "Paired estimates contain duplicate comparison/method rows"
        )
    planned = hypotheses.rename(
        columns={
            "eligibility_reason": "registration_reason",
        }
    ).merge(pd.DataFrame({"scorer": ["scanpy", "aucell"]}), how="cross")
    result = planned.merge(
        numerical, on=keys, how="left", validate="many_to_one"
    )

    primary_question = result.analysis_role.eq("primary")
    registered = np.where(
        primary_question,
        result.eligibility_status.eq("eligible"),
        result.eligibility_status.ne("unavailable"),
    )
    numerical_ok = result.numerically_eligible.fillna(False).astype(bool)
    result["eligible"] = registered & numerical_ok
    result["eligibility_reason"] = np.select(
        [~registered, ~numerical_ok],
        [
            result.registration_reason,
            result.numerical_reason.fillna("missing_estimate"),
        ],
        default="eligible",
    )
    result.loc[~result.eligible, "pvalue"] = np.nan
    result = adjust_registered_families(result, alpha=alpha)

    primary_method = result.scorer.eq(result.primary_scorer)
    tested = result.eligible & primary_method
    rejected = tested & result.pvalue_adjusted.lt(alpha)
    expected_sign = result.estimate * result.expected_direction > 0
    result["status"] = np.select(
        [
            ~result.eligible,
            ~primary_method,
            rejected & ~primary_question,
            rejected & expected_sign,
            rejected,
        ],
        [
            "unavailable",
            "method_sensitivity",
            "association_detected",
            "supported",
            "opposite_direction",
        ],
        default="inconclusive",
    )
    concordance = _method_concordance(result)
    result["method_concordance"] = result.hypothesis_id.map(concordance)
    result["ci_kind"] = "pointwise"
    return result


def adjust_registered_families(
    results: pd.DataFrame, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Holm-adjust primary methods over all registered members of each family.

    Internal untested p=1 preserves family size; published untested raw and
    adjusted p-values remain missing. Sensitivity p-values are not pooled into
    the primary family. The caller must retain every registered hypothesis.
    """
    adjusted = results.copy()
    adjusted["pvalue_adjusted"] = np.nan
    adjusted["n_registered_family"] = 0
    adjusted["n_tested_family"] = 0
    for family, rows in adjusted.groupby(
        "family_id", observed=True, sort=False
    ):
        primary = rows.loc[rows.scorer.eq(rows.primary_scorer)]
        if primary.hypothesis_id.duplicated().any():
            raise ValueError(f"Duplicate primary tests in family {family}")
        tested = primary.eligible & np.isfinite(primary.pvalue)
        if (
            primary.loc[tested, "pvalue"].lt(0).any()
            or primary.loc[tested, "pvalue"].gt(1).any()
        ):
            raise ValueError(f"Invalid p-values in family {family}")
        internal_pvalues = primary.pvalue.where(tested, 1.0).to_numpy(
            dtype=float
        )
        corrected = np.asarray(
            multipletests(internal_pvalues, alpha=alpha, method="holm")[1],
            dtype=float,
        )
        adjusted.loc[primary.index, "pvalue_adjusted"] = np.where(
            tested, corrected, np.nan
        )
        adjusted.loc[primary.index[~tested], "pvalue"] = np.nan
        adjusted.loc[rows.index, "n_registered_family"] = len(primary)
        adjusted.loc[rows.index, "n_tested_family"] = int(tested.sum())
    return adjusted


def _method_concordance(results: pd.DataFrame) -> dict[str, str]:
    """Compare directions without treating methods as independent studies."""
    concordance = {}
    for hypothesis, rows in results.groupby("hypothesis_id", sort=False):
        methods = rows.set_index("scorer")
        if not methods.loc["scanpy", "eligible"]:
            status = "scanpy_unavailable"
        elif not methods.loc["aucell", "eligible"]:
            status = "aucell_unavailable"
        else:
            scanpy_effect, aucell_effect = methods.reindex(
                ["scanpy", "aucell"]
            ).estimate.to_numpy(dtype=float)
            product = scanpy_effect * aucell_effect
            if product > 0:
                status = "same_direction"
            elif product < 0:
                status = "opposite_direction"
            else:
                status = "zero_effect"
        concordance[str(hypothesis)] = status
    return concordance
