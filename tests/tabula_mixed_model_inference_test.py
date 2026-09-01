"""Tests for the prespecified Tabula Sapiens mixed-model planner."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.analysis.tabula_sapiens import mixed_models
from nasp_atlas.single_cell.associations import MixedModelContrast
from nasp_atlas.single_cell.associations import MixedModelInferenceResult
from nasp_atlas.single_cell.associations import MixedModelSpec
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


def _record_model_calls(
    monkeypatch,
) -> list[tuple[pd.DataFrame, MixedModelSpec, Sequence[MixedModelContrast]]]:
    """Replace model fitting with a recorder that preserves result types."""
    calls: list[
        tuple[pd.DataFrame, MixedModelSpec, Sequence[MixedModelContrast]]
    ] = []

    def record(
        frame: pd.DataFrame,
        *,
        spec: MixedModelSpec,
        contrasts: Sequence[MixedModelContrast] = (),
        feature_keys: Sequence[str] = (),
    ) -> MixedModelInferenceResult:
        """Record one model family and return its stable empty tables."""
        del feature_keys
        calls.append((frame.copy(), spec, contrasts))
        empty = MixedModelInferenceResult.empty()
        return MixedModelInferenceResult(
            contrasts=empty.contrasts,
            fixed_effects=empty.fixed_effects,
            term_tests=empty.term_tests,
            variance_components=empty.variance_components,
            diagnostics=pd.DataFrame(
                {
                    "analysis": [spec.analysis],
                    "feature_type": ["module_score"],
                    "feature_id": ["NASP_TEST_auc"],
                    "feature_label": ["NASP_TEST"],
                    "group_key": [spec.group_key],
                    "status": ["ok"],
                    "reason": [""],
                }
            ),
        )

    monkeypatch.setattr(mixed_models, "mixed_model_inference", record)
    return calls


def test_three_studies_use_nested_random_structure_and_all_model_families(
    monkeypatch,
) -> None:
    """Supported studies trigger study, donor, and context random intercepts."""
    calls = _record_model_calls(monkeypatch)

    results = mixed_models.tabula_sapiens_mixed_model_inference(
        _combined_study_cell_frame(n_studies=3),
        schema=ObsSchema(),
        minimum_cells=2,
        minimum_donors=3,
    )
    specs = {spec.analysis: spec for _, spec, _ in calls}
    assert set(specs) == {
        "adjusted_context",
        "paired_tissue",
        "age_by_cell_type",
        "condition_by_cell_type",
    }
    assert all(spec.group_key == "dataset_id" for spec in specs.values())
    assert all(
        len(spec.variance_component_keys) == 2 for spec in specs.values()
    )
    main_frame, main_spec, main_contrasts = calls[0]
    assert main_frame["unit_id"].nunique() == 48
    assert main_frame[main_spec.independent_unit_key].nunique() == 6
    assert "dataset_id" not in main_spec.categorical_keys
    assert {contrast.estimand for contrast in main_contrasts} == {
        "adjusted_cell_type",
        "assay_batch_effects",
    }
    paired_call = next(
        call for call in calls if call[1].analysis == "paired_tissue"
    )
    assert {contrast.estimand for contrast in paired_call[2]} == {
        "paired_tissue"
    }
    availability = results.availability.set_index("analysis")
    assert bool(availability.loc["multi_study_structure", "estimable"])
    assert availability.loc["multi_study_structure", "status"] == "estimated"


def test_two_studies_use_fixed_study_adjustment_with_donor_random_intercept(
    monkeypatch,
) -> None:
    """Too few studies use fixed adjustment, not a study variance claim."""
    calls = _record_model_calls(monkeypatch)

    results = mixed_models.tabula_sapiens_mixed_model_inference(
        _combined_study_cell_frame(n_studies=2),
        schema=ObsSchema(),
        minimum_cells=2,
        minimum_donors=3,
    )

    specs = {spec.analysis: spec for _, spec, _ in calls}
    main = specs["adjusted_context"]
    assert main.group_key == main.independent_unit_key
    assert "dataset_id" in main.categorical_keys
    assert len(main.variance_component_keys) == 1
    availability = results.availability.set_index("analysis")
    assert not bool(availability.loc["multi_study_structure", "estimable"])
    assert (
        availability.loc["multi_study_structure", "reason"]
        == "too_few_studies:2<3"
    )


def test_sparse_cell_types_are_excluded_only_from_interaction_fits(
    monkeypatch,
) -> None:
    """Rare strata stay requested but cannot rank-defeat supported effects."""
    frame = _combined_study_cell_frame(n_studies=3)
    rare = frame.loc[
        frame["dataset_id"].eq("S0")
        & frame["donor_id"].eq("D0")
        & frame["cell_type"].eq("B cell")
    ].copy()
    rare["cell_type"] = "Rare cell"
    frame = pd.concat((frame, rare), ignore_index=True)
    calls = _record_model_calls(monkeypatch)

    mixed_models.tabula_sapiens_mixed_model_inference(
        frame,
        schema=ObsSchema(),
        minimum_cells=2,
        minimum_donors=3,
    )

    calls_by_analysis = {
        spec.analysis: (model_frame, contrasts)
        for model_frame, spec, contrasts in calls
    }
    main_frame, _ = calls_by_analysis["adjusted_context"]
    assert "Rare cell" in set(main_frame["cell_type"])
    for analysis in ("age_by_cell_type", "condition_by_cell_type"):
        model_frame, contrasts = calls_by_analysis[analysis]
        assert "Rare cell" not in set(model_frame["cell_type"])
        assert contrasts[0].by_levels is not None
        assert "Rare cell" in contrasts[0].by_levels


def test_paired_tissue_model_excludes_single_tissue_donors(
    monkeypatch,
) -> None:
    """Tissue effects are fitted only from donors with within-donor support."""
    frame = _combined_study_cell_frame(n_studies=3)
    unpaired = frame.loc[
        frame["dataset_id"].eq("S0")
        & frame["donor_id"].eq("D0")
        & frame["tissue_in_publication"].eq("liver")
    ].copy()
    unpaired["donor_id"] = "unpaired"
    unpaired["feature_value"] = 1_000.0
    frame = pd.concat((frame, unpaired), ignore_index=True)
    calls = _record_model_calls(monkeypatch)

    mixed_models.tabula_sapiens_mixed_model_inference(
        frame,
        schema=ObsSchema(),
        minimum_cells=2,
        minimum_donors=3,
    )

    paired_frame = next(
        model_frame
        for model_frame, spec, _ in calls
        if spec.analysis == "paired_tissue"
    )
    assert "unpaired" not in set(paired_frame["donor_id"])


def test_factorized_identifiers_do_not_collide_on_types_or_delimiters() -> None:
    """Internal repeated-unit codes preserve exact metadata tuples."""
    frame = pd.DataFrame(
        {
            "study": ["A", "A|donor=B", "typed", "typed"],
            "donor": ["B|donor=C", "C", 1, "1"],
        }
    )

    identifiers = mixed_models._factorized_identifier(
        frame,
        keys=["study", "donor"],
        prefix="unit_",
    )

    assert identifiers.nunique() == 4


def test_condition_pairs_do_not_require_sparse_unrelated_arms(
    monkeypatch,
) -> None:
    """A sparse third condition cannot suppress a supported two-arm effect."""
    frame = _combined_study_cell_frame(n_studies=3)
    rare = frame.loc[
        frame["dataset_id"].eq("S0")
        & frame["donor_id"].eq("D0")
        & frame["cell_type"].eq("B cell")
    ].copy()
    rare["donor_id"] = "rare_donor"
    rare["disease"] = "aaa_rare"
    frame = pd.concat((frame, rare), ignore_index=True)
    calls = _record_model_calls(monkeypatch)

    mixed_models.tabula_sapiens_mixed_model_inference(
        frame,
        schema=ObsSchema(),
        condition_reference=None,
        minimum_cells=2,
        minimum_donors=3,
    )

    condition_calls = [
        (model_frame, contrasts)
        for model_frame, spec, contrasts in calls
        if spec.analysis == "condition_by_cell_type"
    ]
    assert len(condition_calls) == 2
    calls_by_level = {
        contrasts[0].levels[0]: (model_frame, contrasts[0])
        for model_frame, contrasts in condition_calls
    }
    assert set(calls_by_level) == {"aaa_rare", "normal"}
    assert all(
        contrast.reference == "disease"
        for _, contrast in calls_by_level.values()
    )
    supported_frame, _ = calls_by_level["normal"]
    assert supported_frame["feature_value"].notna().all()
    assert set(supported_frame["cell_type"]) == {"B cell", "T cell"}
    unsupported_frame, _ = calls_by_level["aaa_rare"]
    assert unsupported_frame["feature_value"].isna().all()


def test_all_minimum_cell_filtered_features_keep_stable_diagnostics() -> None:
    """Fully filtered inputs remain readable and report their exclusion."""
    results = mixed_models.tabula_sapiens_mixed_model_inference(
        _combined_study_cell_frame(n_studies=1),
        schema=ObsSchema(),
        minimum_cells=3,
        minimum_donors=3,
    )
    empty = mixed_models.TabulaMixedModelResults.empty()

    for table in (
        results.contrasts,
        results.fixed_effects,
        results.term_tests,
        results.variance_components,
        results.diagnostics,
    ):
        assert len(table.columns) > 0
        assert "condition_comparison" in table
        assert "detection_threshold" in table
        assert "configured_condition_reference" in table
        assert "minimum_studies" in table
        assert "minimum_repeated_contexts" in table
    assert list(results.diagnostics.columns) == list(empty.diagnostics.columns)

    assert len(results.diagnostics) == 1
    diagnostic = results.diagnostics.iloc[0]
    assert diagnostic["analysis"] == "input_support"
    assert diagnostic["status"] == "no_observations_after_minimum_cells"
    assert diagnostic["n_aggregates_before_minimum_cells"] == 16
    assert diagnostic["n_aggregates_excluded_minimum_cells"] == 16
    assert diagnostic["n_aggregates_after_minimum_cells"] == 0
    availability = results.availability.set_index("analysis")
    requested = availability.drop(index="multi_study_structure")
    assert set(requested["reason"]) == {"all_aggregates_below_minimum_cells"}


def test_tabula_wrapper_rejects_nonfinite_detection_threshold() -> None:
    """Invalid expressing thresholds fail before observational aggregation."""
    with pytest.raises(ValueError, match="finite real number"):
        mixed_models.tabula_sapiens_mixed_model_inference(
            _combined_study_cell_frame(n_studies=1),
            schema=ObsSchema(),
            detection_threshold=np.nan,
            minimum_cells=2,
        )
