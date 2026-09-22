"""Numerical experiments for donor-paired feature inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import aggregate_feature_frame_by_keys
from nasp_atlas.single_cell.associations import paired_feature_contrasts


@pytest.mark.parametrize("level_key", ["tissue", "population"])
def test_paired_inference_uses_people_and_preserves_missing_pairs(
    level_key,
) -> None:
    """Repeated cells and an unmatched outlier cannot alter paired inference."""
    cells = pd.DataFrame(
        {
            "donor_id": ["d1", "d2", "d3", "d4"] * 2 + ["outlier"],
            level_key: ["target"] * 4 + ["reference"] * 4 + ["target"],
            "feature_value": [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1000.0],
            "feature_type": "module_score",
            "feature_id": "example_score",
            "feature_label": "example",
        }
    )
    original = cells.copy(deep=True)
    results = []
    for frame in (cells, pd.concat([cells, cells.iloc[:2]], ignore_index=True)):
        donor = aggregate_feature_frame_by_keys(
            frame,
            unit_keys=["donor_id", level_key],
            statistical_unit="donor_context",
            aggregation="mean",
            schema=ObsSchema(tissue_key="tissue", cell_type_key="population"),
        )
        results.append(
            paired_feature_contrasts(
                donor.sample(frac=1, random_state=7),
                pair_keys=("donor_id",),
                level_key=level_key,
                target_level="target",
                reference_level="reference",
                minimum_pairs=4,
            )
        )
    estimate = results[0].estimates.iloc[0]
    assert estimate["n_paired_donors"] == 4
    assert estimate["estimate"] == pytest.approx(2.5)
    assert estimate["standard_error"] == pytest.approx(np.sqrt(5 / 12))
    assert estimate["ci_lower"] == pytest.approx(0.445739743)
    assert estimate["ci_upper"] == pytest.approx(4.554260257)
    assert estimate["degrees_of_freedom"] == 3
    pd.testing.assert_frame_equal(results[0].estimates, results[1].estimates)
    support = results[0].donor_differences.set_index("donor_id")
    assert support.loc["outlier", "status"] == "missing_reference"
    assert pd.isna(support.loc["outlier", "difference"])
    pd.testing.assert_frame_equal(cells, original)

    with pytest.raises(ValueError, match="Duplicate"):
        paired_feature_contrasts(
            pd.concat([donor, donor.iloc[:1]]),
            pair_keys=("donor_id",),
            level_key=level_key,
            target_level="target",
            reference_level="reference",
        )


def test_unavailable_pairs_keep_descriptive_estimates_and_reasons() -> None:
    """Constant differences, low support and missing scores stay distinct."""
    rows = []
    for feature in ("constant", "low_support", "missing_side", "nonfinite"):
        for donor in range(6):
            for level in ("target", "reference"):
                if feature == "missing_side" and level == "target":
                    continue
                if feature == "low_support" and donor == 5:
                    continue
                value = 2.0 if feature == "constant" else float(donor)
                if level == "reference":
                    value = 0.0
                if feature == "nonfinite" and donor == 5:
                    value = np.inf
                rows.append(
                    {
                        "feature_type": "module_score",
                        "feature_id": feature,
                        "feature_label": feature,
                        "feature_value": value,
                        "donor_id": str(donor),
                        "context": level,
                    }
                )
    result = paired_feature_contrasts(
        pd.DataFrame(rows),
        pair_keys=("donor_id",),
        level_key="context",
        target_level="target",
        reference_level="reference",
    )
    effects = result.estimates.set_index("feature_id")
    assert effects.loc["constant", "estimate"] == 2
    assert effects.loc["constant", "status"] == "degenerate_variance"
    assert effects.loc["low_support", "estimate"] == 2
    assert effects.loc["low_support", "n_paired_donors"] == 5
    assert effects.loc["low_support", "status"] == "insufficient_pairs"
    assert pd.isna(effects.loc["missing_side", "estimate"])
    assert effects["pvalue"].isna().all()
    assert effects["ci_lower"].isna().all()
    assert "nonfinite_pair" in result.donor_differences["status"].to_list()
