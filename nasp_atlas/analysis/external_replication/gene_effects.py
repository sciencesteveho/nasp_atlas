"""Paired effects of actual gene-removal scores, with signed-arm gates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
from nasp_compendium.types import GeneModule

from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.scoring import _arm_diagnostics
from nasp_atlas.analysis.external_replication.sensitivity import (
    matched_donor_effects,
)
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.single_cell.associations import PairedContrastResult
from nasp_atlas.single_cell.gene_sensitivity import GeneSensitivityResults
from nasp_atlas.single_cell.module_scoring import ScorerName


__all__ = ["estimate_gene_removal_effects", "paired_gene_expression"]


def estimate_gene_removal_effects(
    result: GeneSensitivityResults,
    expression: ad.AnnData,
    modules: Sequence[GeneModule],
    comparison: ComparisonSpec,
    baseline_differences: pd.DataFrame,
    *,
    scorer: ScorerName,
    scoring_context_id: str,
    alpha: float = 0.05,
) -> tuple[PairedContrastResult, pd.DataFrame, pd.DataFrame]:
    """Infer from returned variant donor means, retaining every failed removal.

    Component scores diagnose information loss independently of final scores.
    The baseline comparison uses only people complete in both measurements.
    Candidate selection is descriptive, not an independent hypothesis test.
    No multiplicity decision or primary result is changed here.
    """
    lookup = {module.module_id: module for module in modules}
    definitions = [
        replace(
            lookup[str(row.module_id)],
            module_id=str(row.variant_id),
            positive_genes=tuple(
                filter(None, str(row.positive_genes).split(";"))
            ),
            inverse_genes=tuple(
                filter(None, str(row.inverse_genes).split(";"))
            ),
            context_dependent_genes=(),
        )
        for row in result.variants.itertuples()
        if row.status == "ok"
    ]
    arms = _arm_diagnostics(expression, result.cell_scores, definitions, scorer)
    eligibility = (
        result.variants[["variant_id", "status"]]
        .rename(columns={"variant_id": "module_id"})
        .assign(scorer=scorer)
    )
    failed_arms = arms.loc[arms.status.ne("ok")].drop_duplicates("module_id")
    for row in failed_arms.itertuples():
        eligibility.loc[eligibility.module_id.eq(row.module_id), "status"] = (
            row.status
        )

    # The public removal API already owns cell-to-donor averaging. Attach the
    # external unit contract without calculating a second set of donor means.
    keys = ["cohort_id", "donor_id", "tissue", "population"]
    obs = expression.obs
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Gene removal requires in-memory observation metadata")
    counts = obs.groupby(keys, observed=True).size().rename("n_cells_total")
    donors = result.donor_scores.merge(
        counts, on=keys, how="left", validate="many_to_one"
    )
    donors = donors.assign(
        modality=obs.modality.unique().item(),
        module_id=donors.variant_id,
        feature_label=donors.variant_id,
        feature_id=donors.variant_id,
        feature_type="module_score",
        scorer=scorer,
        scoring_context_id=scoring_context_id,
        eligible=donors.n_cells.eq(donors.n_cells_total)
        & donors.n_cells.ge(comparison.minimum_cells),
    )
    donors["eligibility_reason"] = np.where(
        donors.eligible, "eligible", "incomplete_cell_support"
    )
    paired = estimate_module_contrasts(
        donors, eligibility, comparison, alpha=alpha
    )
    effects = paired.estimates.rename(
        columns={"module_id": "variant_id"}
    ).merge(result.variants, on="variant_id", validate="one_to_one")
    differences = (
        paired.donor_differences.rename(columns={"feature_label": "variant_id"})
        .merge(
            result.variants[["variant_id", "module_id"]],
            on="variant_id",
            validate="many_to_one",
        )
        .rename(columns={"module_id": "feature_label"})
    )
    matched = []
    for variant_id, group in differences.groupby("variant_id", observed=True):
        matched.append(
            matched_donor_effects(baseline_differences, group).assign(
                variant_id=variant_id
            )
        )
    matches = (
        pd.concat(matched, ignore_index=True)
        if matched
        else pd.DataFrame(
            columns=[
                "module_id",
                "scorer",
                "variant_id",
                "n_matched_donors",
                "matched_donors",
                "baseline_matched_estimate",
                "variant_matched_estimate",
            ]
        )
    )
    effects = effects.merge(
        matches,
        on=["module_id", "scorer", "variant_id"],
        how="left",
        validate="one_to_one",
    )
    effects["n_matched_donors"] = effects.n_matched_donors.fillna(0).astype(int)
    effects["matched_donors"] = effects.matched_donors.fillna("[]")
    return PairedContrastResult(effects, differences), arms, donors


def paired_gene_expression(
    donor_expression: pd.DataFrame, comparison: ComparisonSpec
) -> pd.DataFrame:
    """Return descriptive paired expression/detection differences per person.

    Preserve assay strata and absent/nonfinite genes. No gene-level p-values
    are calculated; the module's sign is not required of every member gene.
    """
    fixed_axis = "population" if comparison.level_key == "tissue" else "tissue"
    keys = [
        "cohort_id",
        "donor_id",
        fixed_axis,
        "assay",
        "module_id",
        "gene",
        "arm",
    ]
    columns = [*keys, "mean_expression", "fraction_detected", "status"]
    target = donor_expression.loc[
        donor_expression[comparison.level_key].eq(comparison.target), columns
    ]
    reference = donor_expression.loc[
        donor_expression[comparison.level_key].eq(comparison.reference), columns
    ]
    result = target.merge(
        reference,
        on=keys,
        how="outer",
        suffixes=("_target", "_reference"),
        validate="one_to_one",
    )
    for measure in ("mean_expression", "fraction_detected"):
        result[f"{measure}_difference"] = (
            result[f"{measure}_target"] - result[f"{measure}_reference"]
        )
    result["comparison_axis"] = comparison.level_key
    result["target_level"] = comparison.target
    result["reference_level"] = comparison.reference
    return result
