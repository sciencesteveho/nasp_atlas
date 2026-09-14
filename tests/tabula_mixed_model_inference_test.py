"""Tests for the prespecified Tabula Sapiens mixed-model planner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.analysis.tabula_sapiens import mixed_models
from nasp_atlas.single_cell.associations import ObsSchema


def _combined_study_cell_frame(*, n_studies: int) -> pd.DataFrame:
    """Return cells spanning study, donor, context, assay, and condition."""
    records: list[dict[str, object]] = []
    for study_index in range(n_studies):
        for donor_index in range(2):
            for tissue in ("liver", "lung"):
                for cell_type in ("B cell", "T cell"):
                    for assay in ("10x", "smartseq"):
                        for replicate in range(2):
                            records.append(
                                {
                                    "feature_type": "module_score",
                                    "feature_id": "NASP_TEST_auc",
                                    "feature_label": "NASP_TEST",
                                    "feature_value": float(
                                        study_index
                                        + donor_index
                                        + (cell_type == "T cell")
                                        + replicate / 10.0
                                    ),
                                    "dataset_id": f"S{study_index}",
                                    "donor_id": f"D{donor_index}",
                                    "tissue_in_publication": tissue,
                                    "cell_type": cell_type,
                                    "assay": assay,
                                    "sex": (
                                        "female" if donor_index == 0 else "male"
                                    ),
                                    "disease": (
                                        "normal"
                                        if donor_index == 0
                                        else "disease"
                                    ),
                                    "age_years": float(
                                        30 + study_index * 10 + donor_index * 5
                                    ),
                                }
                            )
    return pd.DataFrame.from_records(records)


def test_filtered_cells_cannot_produce_estimates() -> None:
    """Unsupported cell aggregates produce no apparently estimable effects."""
    results = mixed_models.tabula_sapiens_mixed_model_inference(
        _combined_study_cell_frame(n_studies=1),
        schema=ObsSchema(),
        minimum_cells=3,
    )

    assert not results.contrasts.estimable.any()
    assert not results.variance_components.estimable.any()
    diagnostic = results.diagnostics.iloc[0]
    assert diagnostic.n_aggregates_before_minimum_cells == 16
    assert diagnostic.n_aggregates_after_minimum_cells == 0


def test_tabula_wrapper_rejects_nonfinite_detection_threshold() -> None:
    """Invalid expressing thresholds fail before observational aggregation."""
    with pytest.raises(ValueError, match="finite real number"):
        mixed_models.tabula_sapiens_mixed_model_inference(
            _combined_study_cell_frame(n_studies=1),
            schema=ObsSchema(),
            detection_threshold=np.nan,
            minimum_cells=2,
        )


def _paired_signal_cells() -> pd.DataFrame:
    """Return a paired signal with donor labels reused between studies."""
    rng = np.random.default_rng(31)
    rows = []
    for study in range(2):
        for donor in range(12):
            offset = rng.normal(0, 0.8) + 3 * study
            for tissue in ("liver", "lung"):
                for cell_type in ("B cell", "T cell"):
                    for _ in range(3):
                        rows.append(
                            {
                                "feature_type": "module_score",
                                "feature_id": "M_score",
                                "feature_label": "M",
                                "feature_value": offset
                                + (tissue == "lung")
                                + 0.5 * (cell_type == "T cell")
                                + rng.normal(0, 0.1),
                                "dataset_id": f"S{study}",
                                "donor_id": f"D{donor}",
                                "tissue_in_publication": tissue,
                                "cell_type": cell_type,
                            }
                        )
    return pd.DataFrame(rows)


def test_unpaired_donors_cannot_change_the_paired_tissue_estimate() -> None:
    """Unpaired outliers cannot drive a within-donor tissue effect."""
    cells = _paired_signal_cells()
    unpaired = cells.loc[
        cells.donor_id.eq("D0") & cells.tissue_in_publication.eq("liver")
    ].copy()
    unpaired["donor_id"] = "unpaired"
    unpaired["feature_value"] = 1000.0
    estimates = []
    for frame in (cells, pd.concat([cells, unpaired], ignore_index=True)):
        result = mixed_models.tabula_sapiens_mixed_model_inference(
            frame,
            schema=ObsSchema(),
            minimum_cells=3,
        )
        paired = result.contrasts.loc[
            result.contrasts.estimand.eq("paired_tissue")
        ].set_index("level")
        assert paired.estimable.all()
        assert paired.n_paired_units.eq(24).all()
        estimates.append(paired.estimate)

    pd.testing.assert_series_equal(estimates[0], estimates[1])
    assert abs(estimates[0].loc["liver"]) == pytest.approx(1.0, abs=0.05)


def test_rare_condition_does_not_hide_a_supported_condition_effect() -> None:
    """A rare condition cannot block a supported condition comparison."""
    cells = _paired_signal_cells()
    cells["donor_id"] = np.repeat([f"D{i}" for i in range(24)], 12)
    cells["disease"] = np.repeat(["normal", "disease"] * 12, 12)
    cells.loc[cells.disease.eq("disease"), "feature_value"] += 2.0
    rare = cells.loc[cells.donor_id.eq("D0")].copy()
    rare["donor_id"] = "rare"
    rare["disease"] = "rare_condition"

    result = mixed_models.tabula_sapiens_mixed_model_inference(
        pd.concat([cells, rare], ignore_index=True),
        schema=ObsSchema(),
        condition_reference="normal",
        minimum_cells=3,
    )

    condition = result.contrasts.loc[
        result.contrasts.estimand.eq("condition_by_cell_type")
    ]
    supported = condition.loc[condition.level.eq("disease")]
    unsupported = condition.loc[condition.level.eq("rare_condition")]
    assert not supported.empty and supported.estimable.all()
    assert supported.estimate.gt(0).all()
    assert not unsupported.empty and not unsupported.estimable.any()
    assert unsupported.estimate.isna().all()
