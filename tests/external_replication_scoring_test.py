"""Scientific checks for identifier transport and frozen module scoring."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from nasp_compendium.types import GeneModule

from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.contrasts import (
    resolve_discovery_eligibility,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    align_cohort_features,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    apply_feature_alignment,
)
from nasp_atlas.analysis.external_replication.scoring import (
    aggregate_module_scores,
)
from nasp_atlas.analysis.external_replication.scoring import (
    score_prepared_cohort,
)
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.single_cell.associations import paired_feature_contrasts
from nasp_atlas.single_cell.module_scoring import module_score_name


@pytest.mark.parametrize("scorer", ["scanpy", "aucell"])
def test_alias_transport_preserves_explicit_scores_and_paired_estimate(
    scorer,
) -> None:
    """Aliases preserve scores; a changed supplied definition changes them."""
    rng = np.random.default_rng(6)
    values = rng.uniform(0, 5, (24, 400))
    values[:12, 0] += 8
    values[12:, 1] += 8
    ids = [f"ENSG{i:06}" for i in range(400)]
    background = [f"background_{i}" for i in range(397)]
    reference = pd.DataFrame(
        {"symbol": ["RIGI", "STING1", "TASL", *background]}, index=ids
    )
    external = pd.DataFrame(
        {"symbol": ["DDX58", "TMEM173", "CXorf21", *background]}, index=ids
    )
    # This real module ID deliberately has a different supplied definition.
    module = GeneModule(
        module_id="NASP_DNA_SENSING",
        positive_genes=("DDX58",),
        inverse_genes=("STING1",),
        context_dependent_genes=("TASL",),
        gene_id_output="symbols",
    )
    obs = pd.DataFrame(
        {
            "cohort_id": "fixture",
            "donor_id": [f"d{i}" for i in range(4)] * 6,
            "tissue": ["spleen"] * 12 + ["blood"] * 12,
            "population": "mono",
            "modality": "cells",
        },
        index=[f"c{i}" for i in range(24)],
    )
    prepared = ad.AnnData(
        X=sp.csr_matrix(values),
        obs=obs,
        var=external,
        uns={"expression_preparation": {"normalization": "fixture"}},
    )
    alignment = align_cohort_features(
        reference,
        external,
        [module],
        reference_id_column="_index",
        reference_symbol_column="symbol",
        external_id_column="_index",
        external_symbol_column="symbol",
        aliases={"DDX58": ("RIGI",)},
    )
    aligned = apply_feature_alignment(
        prepared, alignment.external_features, alignment.shared_gene_ids
    )
    original = prepared.copy()
    result = score_prepared_cohort(
        aligned,
        [module],
        alignment.coverage,
        scorer=scorer,
        aucell_chunk_size=7,
    )
    canonical = aligned.copy()
    canonical.var["feature_name"] = ["DDX58", "STING1", "TASL", *ids[3:]]
    direct = score_prepared_cohort(
        canonical,
        [module],
        alignment.coverage,
        scorer=scorer,
        aucell_chunk_size=24,
    )
    pd.testing.assert_frame_equal(result.scores, direct.scores)
    assert result.eligibility.status.tolist() == ["ok"]
    np.testing.assert_array_equal(
        sp.csr_matrix(prepared.X).toarray(),
        sp.csr_matrix(original.X).toarray(),
    )
    pd.testing.assert_frame_equal(
        cast(pd.DataFrame, prepared.var), cast(pd.DataFrame, original.var)
    )
    assert prepared.obs.columns.tolist() == obs.columns.tolist()

    changed_module = replace(module, positive_genes=("TASL",))
    changed_alignment = align_cohort_features(
        reference,
        external,
        [changed_module],
        reference_id_column="_index",
        reference_symbol_column="symbol",
        external_id_column="_index",
        external_symbol_column="symbol",
        aliases={"DDX58": ("RIGI",)},
    )
    changed = score_prepared_cohort(
        aligned,
        [changed_module],
        changed_alignment.coverage,
        scorer=scorer,
    )
    column = module_score_name(module, scorer=scorer)
    assert not np.allclose(result.scores[column], changed.scores[column])

    donor = aggregate_module_scores(
        obs,
        result.scores,
        [module],
        scorer=scorer,
        scoring_context_id="fixture",
        minimum_cells=3,
    )
    assert donor.eligible.all()
    contrast = paired_feature_contrasts(
        donor,
        pair_keys=("cohort_id", "donor_id"),
        level_key="tissue",
        target_level="spleen",
        reference_level="blood",
        minimum_pairs=4,
    ).estimates.iloc[0]
    independent_means = result.scores[column].to_numpy().reshape(6, 4)
    expected_difference = independent_means[:3].mean(
        axis=0
    ) - independent_means[3:].mean(axis=0)
    assert contrast["estimate"] == pytest.approx(expected_difference.mean())
    assert contrast["n_paired_donors"] == 4

    comparison = ComparisonSpec(
        level_key="tissue",
        target="spleen",
        reference="blood",
        minimum_cells=3,
        minimum_pairs=4,
    )
    workflow_contrast = estimate_module_contrasts(
        donor, result.eligibility, comparison
    ).estimates.iloc[0]
    assert workflow_contrast["estimate"] == pytest.approx(
        expected_difference.mean()
    )
    assert workflow_contrast["eligible"]

    # A missing gene is not a zero-expression gene or a smaller signature.
    missing = external.iloc[1:]
    incomplete = align_cohort_features(
        reference,
        missing,
        [module],
        reference_id_column="_index",
        reference_symbol_column="symbol",
        external_id_column="_index",
        external_symbol_column="symbol",
        aliases={"DDX58": ("RIGI",)},
    )
    unavailable = score_prepared_cohort(
        aligned,
        [module],
        incomplete.coverage,
        scorer=scorer,
    )
    assert unavailable.scores.empty
    assert unavailable.eligibility.status.tolist() == ["incomplete_signature"]


def test_variable_signed_score_does_not_hide_constant_arm() -> None:
    """AUCell's constant positive arm withholds signed inference."""
    rng = np.random.default_rng(10)
    values = rng.uniform(0, 5, (12, 200))
    values[:, 0] = 100
    values[:6, 1] = 90
    genes = ["POS", "INV", *[f"b{i}" for i in range(198)]]
    var = pd.DataFrame({"feature_name": genes}, index=genes)
    module = GeneModule(
        module_id="SIGNED",
        positive_genes=("POS",),
        inverse_genes=("INV",),
        context_dependent_genes=(),
        gene_id_output="symbols",
    )
    alignment = align_cohort_features(
        var,
        var,
        [module],
        reference_id_column="_index",
        reference_symbol_column="feature_name",
        external_id_column="_index",
        external_symbol_column="feature_name",
        aliases={},
    )
    result = score_prepared_cohort(
        ad.AnnData(sp.csr_matrix(values), var=var),
        [module],
        alignment.coverage,
        scorer="aucell",
    )
    assert result.scores["SIGNED_auc"].std() > 0
    assert result.eligibility.status.tolist() == ["degenerate_arm"]
    assert result.arm_diagnostics.set_index("arm").loc[
        "positive", "score_sd"
    ] == pytest.approx(0)

    obs = pd.DataFrame(
        {
            "cohort_id": "fixture",
            "donor_id": [f"d{i}" for i in range(6)] * 2,
            "tissue": ["target"] * 6 + ["reference"] * 6,
            "population": "mono",
            "modality": "cells",
        },
        index=result.scores.index,
    )
    donor = aggregate_module_scores(
        obs,
        result.scores,
        [module],
        scorer="aucell",
        scoring_context_id="fixture",
        minimum_cells=1,
    )
    eligibility = pd.concat(
        [
            result.eligibility,
            pd.DataFrame(
                [
                    {
                        "module_id": "ABSENT",
                        "scorer": "aucell",
                        "status": "incomplete_signature",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    contrasts = estimate_module_contrasts(
        donor,
        eligibility,
        ComparisonSpec(
            level_key="tissue",
            target="target",
            reference="reference",
            minimum_cells=1,
            minimum_pairs=6,
        ),
    ).estimates.set_index("module_id")
    assert contrasts.loc["SIGNED", "estimate"] != 0
    assert np.isnan(contrasts.loc["SIGNED", "pvalue"])
    assert not contrasts.eligible.any()
    assert contrasts.loc["ABSENT", "n_paired_donors"] == 0


def test_discovery_requires_both_signs_support_and_source_equivalence() -> None:
    """No external estimate can repair a failed harmonized reference gate."""
    hypotheses = pd.DataFrame(
        {
            "module_id": ["A", "B", "C"],
            "analysis_role": "primary",
            "expected_direction": [1, -1, 1],
        }
    )
    reference = pd.DataFrame(
        {
            "module_id": ["A", "B", "C"] * 2,
            "scorer": ["scanpy"] * 3 + ["aucell"] * 3,
            "estimate": [1, -2, 4, 1, 2, 4],
            "measurement_status": "ok",
            "n_paired_donors": [5, 5, 2] * 2,
        }
    )
    resolved = resolve_discovery_eligibility(
        hypotheses,
        reference,
        discovery_artifact="fixture.csv",
    )
    assert resolved.eligibility_status.tolist() == [
        "eligible",
        "unavailable",
        "unavailable",
    ]
    unresolved_mapping = resolve_discovery_eligibility(
        hypotheses,
        reference,
        discovery_artifact="fixture.csv",
        source_gate_reason="unresolved_mapping",
    )
    assert unresolved_mapping.eligibility_status.eq("unavailable").all()
