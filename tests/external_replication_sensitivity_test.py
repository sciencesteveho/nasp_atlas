"""Influential people and changing support remain visible in paired effects."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.analysis.external_replication.contrast_drivers import (
    select_contrast_drivers,
)
from nasp_atlas.analysis.external_replication.sampling import (
    select_balanced_cells,
)
from nasp_atlas.analysis.external_replication.sensitivity import (
    fixed_score_sensitivities,
)
from nasp_atlas.analysis.external_replication.sensitivity import (
    matched_donor_effects,
)
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)


def test_fixed_scores_reveal_influence_and_retain_unsupported_deletions() -> (
    None
):
    """A sixth influential person reverses the mean, not independent support."""
    effects = [1.0, 2.0, 3.0, 4.0, 5.0, -20.0]
    donors = pd.DataFrame(
        [
            {
                "cohort_id": "fixture",
                "donor_id": f"d{index}",
                "tissue": tissue,
                "population": "monocyte",
                "modality": "cells",
                "module_id": "A",
                "feature_type": "module_score",
                "feature_id": "A_score",
                "feature_label": "A",
                "scorer": "scanpy",
                "scoring_context_id": "fixed",
                "feature_value": value,
                "n_cells": 15 if index == 5 else 40,
                "eligible": True,
                "eligibility_reason": "eligible",
            }
            for index, effect in enumerate(effects)
            for tissue, value in (("spleen", effect), ("blood", 0.0))
        ]
    )
    original = donors.copy(deep=True)
    eligibility = pd.DataFrame(
        {"module_id": ["A"], "scorer": ["scanpy"], "status": ["ok"]}
    )
    result = fixed_score_sensitivities(
        donors,
        eligibility,
        ComparisonSpec(level_key="tissue", target="spleen", reference="blood"),
        support_thresholds=[20, 50],
        exclusions={"shared_person": ["d5"]},
        donor_metadata=pd.DataFrame(
            {"donor_id": [f"d{i}" for i in range(6)], "site": ["a"] * 5 + ["b"]}
        ),
        strata_keys=["site"],
    )
    estimates = result.estimates.set_index("variant_id")
    assert estimates.loc["baseline", "estimate"] == pytest.approx(-5 / 6)
    for variant in ("omit:d5", "minimum_cells:20", "exclude:shared_person"):
        row = estimates.loc[variant]
        assert row.estimate == 3
        assert row.n_paired_donors == 5
        assert row.eligibility_reason == "insufficient_pairs"
        assert np.isnan(row.pvalue)
        assert row.baseline_matched_estimate == 3
        assert json.loads(row.matched_donors) == ["d0", "d1", "d2", "d3", "d4"]
    assert estimates.loc["minimum_cells:50", "n_paired_donors"] == 0
    assert np.isnan(estimates.loc["minimum_cells:50", "estimate"])
    assert estimates.loc["minimum_cells:50", "matched_donors"] == "[]"
    assert (
        estimates.loc["stratum:site:a", "eligibility_reason"]
        == "descriptive_stratum"
    )
    pd.testing.assert_frame_equal(donors, original)

    baseline = result.donor_differences.query("variant_id == 'baseline'")
    changed = baseline.loc[baseline.donor_id.isin(["d0", "d1", "d2"])].copy()
    changed["difference"] += 10
    matched = matched_donor_effects(baseline, changed).iloc[0]
    assert matched.n_matched_donors == 3
    assert matched.baseline_matched_estimate == 2
    assert matched.variant_matched_estimate == 12


def test_balancing_keeps_every_pair_without_using_scores_or_row_order() -> None:
    """Metadata-only sampling cannot silently replace weakly sampled people."""
    rows = [
        {
            "cohort_id": "fixture",
            "donor_id": f"d{donor}",
            "tissue": tissue,
            "population": "monocyte",
            "modality": "cells",
            "irrelevant_score": cell,
        }
        for donor in range(6)
        for tissue in ("spleen", "blood")
        for cell in range(5 + donor)
    ]
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    comparison = ComparisonSpec(
        level_key="tissue",
        target="spleen",
        reference="blood",
        minimum_cells=2,
    )
    sampled = select_balanced_cells(
        obs, comparison, cells_per_context=3, random_state=42
    )
    reordered = obs.sample(frac=1, random_state=99).assign(
        irrelevant_score=-1000
    )
    repeated = select_balanced_cells(
        reordered, comparison, cells_per_context=3, random_state=42
    )
    assert sampled.index.equals(repeated.index)
    assert sampled.groupby(["donor_id", "tissue"]).size().eq(3).all()
    assert sampled.donor_id.nunique() == 6
    assert len(sampled) == 36
    with pytest.raises(ValueError, match="cannot retain every"):
        select_balanced_cells(
            obs, comparison, cells_per_context=6, random_state=42
        )


def test_contrast_drivers_follow_the_difference_not_the_abundance() -> None:
    """Select expression differences without implying score contributions."""
    paired = pd.DataFrame(
        [
            # Abundant everywhere, but nearly equal between contexts.
            {
                "module_id": "M",
                "gene": "HOUSEKEEPER",
                "arm": "positive",
                "mean_expression_difference": 0.02,
            },
            # Modest expression, with the largest paired difference.
            {
                "module_id": "M",
                "gene": "DRIVER",
                "arm": "positive",
                "mean_expression_difference": 0.90,
            },
            # An inverse member falling in the target raises the module score.
            {
                "module_id": "M",
                "gene": "RESTRICTOR",
                "arm": "inverse",
                "mean_expression_difference": -0.40,
            },
            # Direction is undeclared, so it can never be selected.
            {
                "module_id": "M",
                "gene": "AMBIGUOUS",
                "arm": "context_dependent",
                "mean_expression_difference": 5.00,
            },
        ]
    )
    drivers = select_contrast_drivers(paired, drivers_per_module=2)
    assert drivers["M"] == ("DRIVER", "RESTRICTOR")

    tied = paired.copy()
    tied.loc[tied.gene.eq("HOUSEKEEPER"), "mean_expression_difference"] = 0.90
    assert select_contrast_drivers(tied.sample(frac=1, random_state=42))[
        "M"
    ] == (
        "DRIVER",
        "HOUSEKEEPER",
    )


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_contrast_driver_selection_rejects_missing_scored_genes(
    missing: float,
) -> None:
    """A missing donor must not silently change the genes selected."""
    paired = pd.DataFrame(
        {
            "module_id": ["M", "M"],
            "gene": ["A", "A"],
            "arm": ["positive", "positive"],
            "mean_expression_difference": [1.0, missing],
        }
    )
    with pytest.raises(ValueError, match="finite paired differences"):
        select_contrast_drivers(paired)
