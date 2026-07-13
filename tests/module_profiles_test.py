"""Tests for relationships among existing NASP module scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nasp_compendium.types import GeneModule

from nasp_atlas.single_cell.module_profiles import module_gene_overlap
from nasp_atlas.single_cell.module_profiles import pairwise_module_correlations
from nasp_atlas.single_cell.module_profiles import (
    relative_nasp_evidence_profiles,
)


def _correlation_frame() -> pd.DataFrame:
    """Build two modules with a shared tissue baseline only."""
    records: list[dict[str, object]] = []
    within_a = [-1.0, -1.0, 1.0, 1.0]
    within_b = [-1.0, 1.0, -1.0, 1.0]
    for tissue, baseline in (("lung", 0.0), ("liver", 10.0)):
        for donor_index, (value_a, value_b) in enumerate(
            zip(within_a, within_b, strict=True)
        ):
            for module, value in (("A", value_a), ("B", value_b)):
                records.append(
                    {
                        "donor_id": f"D{donor_index}",
                        "tissue": tissue,
                        "feature_type": "module_score",
                        "feature_label": module,
                        "feature_value": baseline + value,
                        "statistical_unit": "donor_tissue",
                        "aggregation": "mean",
                    }
                )
    return pd.DataFrame.from_records(records)


def _module(
    module_id: str,
    positive: tuple[str, ...],
    inverse: tuple[str, ...],
    *,
    context: tuple[str, ...] = (),
) -> GeneModule:
    """Build a complete symbol-space signed module for overlap tests."""
    return GeneModule(
        module_id=module_id,
        positive_genes=positive,
        inverse_genes=inverse,
        context_dependent_genes=context,
        gene_id_output="symbols",
    )


def test_pairwise_correlations_align_independent_units() -> None:
    """One donor-tissue value per module contributes to each test."""
    result = pairwise_module_correlations(
        _correlation_frame(),
        unit_columns=["donor_id", "tissue"],
    )

    row = result.iloc[0]
    assert row["n_units"] == 8
    assert row["n_strata"] == 1
    assert row["statistical_unit"] == "donor_tissue"
    assert row["pearson_r"] > 0.9
    assert row["pearson_fdr"] == pytest.approx(row["pearson_pvalue"])
    assert row["spearman_fdr"] == pytest.approx(row["spearman_pvalue"])


def test_pairwise_correlations_remove_stratum_baselines() -> None:
    """Within-tissue centering removes a correlation driven by tissue means."""
    result = pairwise_module_correlations(
        _correlation_frame(),
        unit_columns=["donor_id", "tissue"],
        strata_columns=["tissue"],
        center_within_strata=True,
    )

    row = result.iloc[0]
    assert row["n_units"] == 8
    assert row["n_strata"] == 2
    assert row["pearson_r"] == pytest.approx(0.0, abs=1e-12)
    assert row["centered_within_strata"]


def test_pairwise_correlations_can_limit_tested_pairs() -> None:
    """Explicit pairs avoid unrelated feature-feature tests."""
    frame = _correlation_frame()
    extra = frame[frame["feature_label"] == "A"].copy()
    extra["feature_label"] = "C"
    frame = pd.concat([frame, extra], ignore_index=True)

    result = pairwise_module_correlations(
        frame,
        unit_columns=["donor_id", "tissue"],
        module_pairs=[("A", "C")],
    )

    assert list(zip(result["module_a"], result["module_b"], strict=True)) == [
        ("A", "C")
    ]


def test_pairwise_correlations_reject_duplicate_units() -> None:
    """Duplicate module-unit rows fail instead of inflating sample size."""
    frame = _correlation_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="one row per module"):
        pairwise_module_correlations(
            frame,
            unit_columns=["donor_id", "tissue"],
        )


def test_module_gene_overlap_preserves_shared_gene_direction() -> None:
    """Overlap annotations separate concordant from cross-arm genes."""
    module_a = _module(
        "A",
        ("A", "B"),
        ("C",),
        context=("CTX",),
    )
    module_b = _module(
        "B",
        ("B", "C", "D"),
        ("A", "E"),
        context=("CTX",),
    )

    row = module_gene_overlap([module_a, module_b]).iloc[0]

    assert row["n_concordant_shared"] == 1
    assert row["n_cross_arm_shared"] == 2
    assert row["concordant_shared_genes"] == "B"
    assert row["cross_arm_shared_genes"] == "A;C"
    assert row["signed_gene_jaccard"] == pytest.approx(3 / 5)
    assert row["shared_sign_concordance"] == pytest.approx(-1 / 3)
    assert row["signed_cosine"] == pytest.approx(-1 / np.sqrt(15))
    assert row["n_context_shared"] == 1


def test_module_gene_overlap_rejects_mixed_identifier_spaces() -> None:
    """Symbol and var-name modules cannot be compared silently."""
    module_a = _module("A", ("CGAS",), ())
    module_b = GeneModule(
        module_id="B",
        positive_genes=("ENSG_CGAS",),
        inverse_genes=(),
        context_dependent_genes=(),
        gene_id_output="var_names",
    )

    with pytest.raises(ValueError, match="same gene identifier"):
        module_gene_overlap([module_a, module_b])


def test_relative_profiles_distinguish_extreme_evidence_states() -> None:
    """Strict relative flags distinguish active, responsive, and competent."""
    scores = pd.DataFrame(
        {
            "sensor": [0.0, 4.0, 4.0, 0.0, 2.0],
            "output": [4.0, 4.0, 0.0, 0.0, 2.0],
            "restriction": [0.0, 0.0, 4.0, 4.0, 2.0],
            "feedback": [0.0, 0.0, 0.0, 4.0, 2.0],
            "post": [0.0, 0.0, 0.0, 4.0, 2.0],
        },
        index=["responsive", "active", "competent", "post_only", "middle"],
    )
    roles = {
        "sensor": "competence",
        "output": "output",
        "restriction": "restriction",
        "feedback": "feedback",
        "post": "post",
    }

    profiles = relative_nasp_evidence_profiles(scores, module_roles=roles)

    assert profiles.loc["responsive", "relative_nasp_responsive"]
    assert profiles.loc["active", "relative_nasp_active"]
    assert profiles.loc["competent", "relative_nasp_competent"]
    assert profiles.loc["post_only", "relative_post_without_nasp_evidence"]
    assert profiles.loc["competent", "relative_restriction_output_mismatch"]
    assert not profiles.loc["middle", "relative_nasp_active"]


def test_relative_profiles_rank_modules_within_reference_strata() -> None:
    """Percentiles compare units within caller-supplied biological strata."""
    scores = pd.DataFrame(
        {
            "tissue": ["lung", "lung", "liver", "liver"],
            "sensor": [1.0, 2.0, 100.0, 200.0],
        },
        index=["L1", "L2", "V1", "V2"],
    )

    profiles = relative_nasp_evidence_profiles(
        scores,
        module_roles={"sensor": "competence"},
        reference_columns=["tissue"],
    )

    assert profiles["percentile__sensor"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert profiles["relative_competence"].equals(
        profiles["percentile__sensor"]
    )


def test_relative_profiles_give_tied_scores_neutral_rank() -> None:
    """An uninformative constant module cannot create high or low flags."""
    scores = pd.DataFrame({"restriction": [2.0, 2.0, 2.0]})

    profiles = relative_nasp_evidence_profiles(
        scores,
        module_roles={"restriction": "restriction"},
    )

    assert profiles["percentile__restriction"].tolist() == [0.5, 0.5, 0.5]
    assert not profiles["relative_nasp_restricted"].any()
