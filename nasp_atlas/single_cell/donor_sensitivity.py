"""Donor deletion sensitivity for descriptive module-context rankings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import aggregate_feature_frame_by_keys
from nasp_atlas.single_cell.context_summary import summarize_module_contexts


__all__ = ["DonorSensitivityResults", "donor_sensitivity"]


@dataclass(frozen=True)
class DonorSensitivityResults:
    """Individual donors, ranks, deletion replicates, and paired differences."""

    donor_scores: pd.DataFrame
    context_ranks: pd.DataFrame
    leave_one_donor_out: pd.DataFrame
    paired_tissues: pd.DataFrame


def donor_sensitivity(
    cell_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
) -> DonorSensitivityResults:
    """Assess mean module scores without treating cells as replicates.

    Scores are averaged over cells per donor, tissue and cell type, pooling
    assays. Study qualifies donor identity when present. Ranks use the median
    of donor means, both across all contexts and within matching cell types.
    Each deletion removes one donor from every context and recalculates ranks.
    Unsupported contexts remain explicit. Paired differences compare the same
    cell type in the same donors; they are descriptive, with no p-values.

    The input is not modified. Deletion ranges are sensitivity ranges, not
    confidence intervals. Missing scores never contribute as numeric zero.
    """
    if minimum_cells < 1 or minimum_donors < 2:
        raise ValueError("Require minimum_cells >= 1 and minimum_donors >= 2")

    contexts = [schema.tissue_key, schema.cell_type_key]
    donor_keys = [
        key for key in (schema.study_key, schema.donor_key) if key in cell_frame
    ]
    if schema.donor_key not in donor_keys:
        raise KeyError(f"Missing donor column: {schema.donor_key}")
    if cell_frame[donor_keys].isna().any(axis=None):
        raise ValueError("Donor and study identifiers must not be missing")

    modules = cell_frame.loc[cell_frame.feature_type.eq("module_score")]
    donors = aggregate_feature_frame_by_keys(
        modules,
        unit_keys=[*donor_keys, *contexts],
        statistical_unit="donor_tissue_cell_type",
        aggregation="mean",
        schema=schema,
    )
    donor_index = pd.MultiIndex.from_frame(donors[donor_keys])
    donors["independent_donor"] = pd.factorize(donor_index, sort=True)[0]
    donors["supported"] = donors.n_cells.ge(minimum_cells)
    donors["analysis_value"] = donors.feature_value.where(donors.supported)

    baseline = _context_ranks(donors, contexts, minimum_donors)
    deletions = _donor_deletion_ranks(
        donors,
        baseline,
        donor_keys,
        contexts,
        minimum_donors,
    )

    return DonorSensitivityResults(
        donor_scores=donors,
        context_ranks=baseline,
        leave_one_donor_out=deletions,
        paired_tissues=_paired_tissues(donors, schema, minimum_donors),
    )


def _donor_deletion_ranks(
    donors: pd.DataFrame,
    baseline: pd.DataFrame,
    donor_keys: list[str],
    contexts: list[str],
    minimum_donors: int,
) -> pd.DataFrame:
    """Recalculate context ranks after removing each whole donor."""
    keys = ["feature_label", *contexts, "ranking_scope"]
    replicates: list[pd.DataFrame] = []
    identities = donors[["independent_donor", *donor_keys]].drop_duplicates()
    for identity in identities.to_dict("records"):
        retained = donors.loc[
            donors.independent_donor.ne(identity["independent_donor"])
        ]
        deleted = _context_ranks(retained, contexts, minimum_donors)
        comparison = baseline[[*keys, "context_rank", "median"]].merge(
            deleted[[*keys, "context_rank", "median", "n_finite_donors"]],
            on=keys,
            how="left",
            suffixes=("_baseline", "_without_donor"),
            validate="one_to_one",
        )
        for key in donor_keys:
            comparison[f"omitted_{key}"] = identity[key]
        comparison["rank_change"] = (
            comparison.context_rank_without_donor
            - comparison.context_rank_baseline
        )
        comparison["status"] = np.where(
            comparison.context_rank_without_donor.notna(),
            "ok",
            "insufficient_donors",
        )
        replicates.append(comparison)

    deletion_columns = [
        *keys,
        "context_rank_baseline",
        "median_baseline",
        "context_rank_without_donor",
        "median_without_donor",
        "n_finite_donors",
        *[f"omitted_{key}" for key in donor_keys],
        "rank_change",
        "status",
    ]
    return (
        pd.concat(replicates, ignore_index=True)
        if replicates
        else pd.DataFrame(columns=deletion_columns)
    )


def _context_ranks(
    donors: pd.DataFrame,
    contexts: list[str],
    minimum_donors: int,
) -> pd.DataFrame:
    """Rank all contexts and separately rank tissues within each cell type."""
    ranks = summarize_module_contexts(
        donors,
        context_columns=contexts,
        unit_columns=["independent_donor", *contexts],
        donor_column="independent_donor",
        value_column="analysis_value",
        min_units=minimum_donors,
        min_donors=minimum_donors,
    )

    matched = ranks.copy()
    matched["context_rank"] = (
        matched["median"]
        .where(matched.eligible)
        .groupby([matched.feature_label, matched[contexts[1]]], dropna=False)
        .rank(method="dense", ascending=False)
    )
    matched["context_percentile"] = np.nan

    return pd.concat(
        [
            ranks.assign(ranking_scope="all_contexts"),
            matched.assign(ranking_scope="matched_cell_type"),
        ],
        ignore_index=True,
    )


def _paired_tissues(
    donors: pd.DataFrame,
    schema: ObsSchema,
    minimum_donors: int,
) -> pd.DataFrame:
    """Describe directly paired tissue differences without pooled inference."""
    columns = [
        "feature_label",
        schema.cell_type_key,
        "tissue",
        "reference_tissue",
        "n_paired_donors",
        "mean_difference",
        "median_difference",
        "status",
    ]
    records: list[dict[str, object]] = []
    for (module, cell_type), group in donors.groupby(
        ["feature_label", schema.cell_type_key], observed=True, dropna=False
    ):
        wide = group.pivot(
            index="independent_donor",
            columns=schema.tissue_key,
            values="analysis_value",
        )
        for tissue, reference in combinations(wide.columns, 2):
            differences = (wide[tissue] - wide[reference]).dropna()
            supported = len(differences) >= minimum_donors
            records.append(
                {
                    "feature_label": module,
                    schema.cell_type_key: cell_type,
                    "tissue": tissue,
                    "reference_tissue": reference,
                    "n_paired_donors": len(differences),
                    "mean_difference": differences.mean()
                    if supported
                    else np.nan,
                    "median_difference": differences.median()
                    if supported
                    else np.nan,
                    "status": "ok"
                    if supported
                    else "insufficient_paired_donors",
                }
            )

    return pd.DataFrame(records, columns=columns)
