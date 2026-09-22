"""Metadata-only cell selection for balanced scoring sensitivities."""

from __future__ import annotations

import pandas as pd

from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)


__all__ = ["select_balanced_cells"]


def select_balanced_cells(
    selected_obs: pd.DataFrame,
    comparison: ComparisonSpec,
    *,
    cells_per_context: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample equal cells per person/context without dropping any primary pair.

    Selection uses identities and metadata only, never scores. Sorting cell
    IDs before seeded sampling makes the result independent of input row order.
    The returned frame retains original metadata and IDs; input is not mutated.
    An infeasible registered cap fails instead of changing the donor universe.
    """
    if (
        isinstance(cells_per_context, bool)
        or not isinstance(cells_per_context, int)
        or cells_per_context < comparison.minimum_cells
    ):
        raise ValueError("Balanced cap must be an integer meeting cell support")
    if selected_obs.empty or not selected_obs.index.is_unique:
        raise ValueError("Balanced sampling needs nonempty, unique cell IDs")
    keys = ["cohort_id", "donor_id", comparison.level_key]
    if selected_obs[keys].isna().any(axis=None):
        raise ValueError("Balanced sampling identities must not be missing")
    fixed_axis = "population" if comparison.level_key == "tissue" else "tissue"
    for column in ("cohort_id", "modality", fixed_axis):
        if selected_obs[column].nunique() != 1:
            raise ValueError(f"Balance one {column} at a time")
    contexts = {comparison.target, comparison.reference}
    if set(selected_obs[comparison.level_key]) != contexts:
        raise ValueError("Balance exactly the two registered contexts")

    ordered = selected_obs.sort_index()
    groups = ordered.groupby(keys, observed=True, sort=True)
    counts = groups.size()
    complete = counts.groupby(level=["cohort_id", "donor_id"]).size().eq(2)
    if not complete.all() or counts.lt(cells_per_context).any():
        raise ValueError(
            "Registered balanced cap cannot retain every primary pair"
        )
    return groups.sample(
        n=cells_per_context, random_state=random_state
    ).sort_index()
