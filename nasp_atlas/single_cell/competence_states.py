"""Describe competence/output states using the tissue-nomination rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_competence_output_states(
    profiles: pd.DataFrame,
    *,
    tissue_key: str = "tissue_in_publication",
    cell_type_key: str = "cell_type",
    donor_key: str = "donor_id",
    min_donors: int = 3,
    high_quantile: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average complete donor profiles and assign descriptive state colors.

    Reproduce the nomination figure's rules, separately for one scorer.
    Complete profiles have finite competence, output, restriction, feedback
    and post axes and at least one cell. Every complete donor has equal
    weight in its tissue/cell-type context. Contexts with at least
    `min_donors` complete donors supply the reference distribution.

    Scores are min(C, O), max(O-C, 0)*O, and max(C-O, 0)*C for active-like,
    responsive-like and exploratory competent-like evidence, respectively.
    A high score is positive and at least its pooled eligible-context
    `high_quantile`. Colors prioritize active, responsive, then competent
    when flags overlap. This is a descriptive nomination heuristic, not
    evidence of activation or an inferential sex/tissue comparison.

    Args:
      profiles: One saved relative evidence profile per donor/tissue/cell
        type, for a single scorer. Input scores are not reranked or mutated.
      tissue_key: Tissue identity column.
      cell_type_key: Cell-type identity column.
      donor_key: Independent donor identity column.
      min_donors: Minimum complete donors supporting a context.
      high_quantile: Reference quantile in [0, 1], default 0.8.

    Returns:
      All observed contexts with means, support, eligibility, continuous
      state scores, nonexclusive threshold flags and exclusive `state`.
      A threshold table with formulas, reference size and support settings.
      Unavailable contexts retain diagnostics and are never colored Other.

    Example Usage:
      >>> contexts, thresholds = summarize_competence_output_states(
      ...     evidence_profiles,
      ...     min_donors=3,
      ... )
    """
    if min_donors < 1 or not 0 <= high_quantile <= 1:
        raise ValueError("Use min_donors >= 1 and high_quantile in [0, 1]")
    keys = [tissue_key, cell_type_key]
    axes = [
        "relative_competence",
        "relative_output",
        "relative_restriction",
        "relative_feedback",
        "relative_post",
    ]
    required = [donor_key, *keys, *axes, "n_cells"]
    if missing := set(required).difference(profiles.columns):
        raise ValueError(
            f"Evidence profiles are missing columns: {sorted(missing)}"
        )
    if profiles[[donor_key, *keys]].isna().any(axis=None):
        raise ValueError(
            "Donor, tissue and cell-type identities must be present"
        )
    if profiles.duplicated([donor_key, *keys]).any():
        raise ValueError("Expected one profile per donor/tissue/cell type")

    selected = profiles.copy()
    selected["complete"] = np.isfinite(
        selected[axes].to_numpy(dtype=float)
    ).all(axis=1) & selected["n_cells"].ge(1).to_numpy(dtype=bool)
    selected.loc[~selected["complete"], axes] = np.nan
    groups = selected.groupby(keys, observed=True, sort=False)
    contexts = groups[axes].mean()
    contexts["n_donors_total"] = groups[donor_key].nunique()
    contexts["n_complete_donors"] = groups["complete"].sum()
    contexts["n_excluded_profiles"] = (
        contexts["n_donors_total"] - contexts["n_complete_donors"]
    )
    contexts["eligible"] = contexts["n_complete_donors"].ge(min_donors)
    contexts["status"] = np.where(
        contexts["eligible"], "ok", "insufficient_complete_donors"
    )
    competence = contexts["relative_competence"]
    output = contexts["relative_output"]
    contexts["active_like"] = np.minimum(competence, output)
    contexts["responsive_like"] = (output - competence).clip(lower=0) * output
    contexts["competent_like_exploratory"] = (competence - output).clip(
        lower=0
    ) * competence

    reference = contexts.loc[contexts["eligible"]]
    contexts["state"] = np.where(contexts["eligible"], "other", "unavailable")
    thresholds = []
    for state, formula in (
        ("active_like", "min(C, O)"),
        ("responsive_like", "max(O - C, 0) * O"),
        ("competent_like_exploratory", "max(C - O, 0) * C"),
    ):
        threshold = float(reference[state].quantile(high_quantile))
        high = (
            contexts["eligible"]
            & contexts[state].gt(0)
            & contexts[state].ge(threshold)
        )
        contexts[f"meets_{state}_threshold"] = high
        contexts.loc[high & contexts["state"].eq("other"), "state"] = state
        thresholds.append(
            {
                "state": state,
                "formula": formula,
                "threshold": threshold,
                "high_quantile": high_quantile,
                "min_complete_donors": min_donors,
                "n_reference_contexts": len(reference),
                "status": "ok" if len(reference) else "no_supported_contexts",
            }
        )
    return contexts.reset_index(), pd.DataFrame(thresholds)
