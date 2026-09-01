"""Tests for reusable repeated-measures mixed-model inference."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.single_cell.associations import MixedModelContrast
from nasp_atlas.single_cell.associations import MixedModelSpec
from nasp_atlas.single_cell.associations import mixed_model_inference


def _feature_row(
    *,
    donor: str,
    cell_type: str,
    tissue: str,
    age: float,
    value: float,
) -> dict[str, object]:
    """Return one long-format module-score observation."""
    return {
        "feature_type": "module_score",
        "feature_id": "NASP_TEST_auc",
        "feature_label": "NASP_TEST",
        "donor id": donor,
        "cell type": cell_type,
        "tissue label": tissue,
        "age centered": age,
        "feature_value": value,
    }


def _interaction_frame(*, seed: int = 17) -> pd.DataFrame:
    """Return paired donor contexts with known categorical and age effects."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    ages = np.linspace(-1.0, 1.0, 30)
    donor_effects = rng.normal(scale=1.5, size=ages.size)
    age_design = np.column_stack((np.ones(ages.size), ages))
    projection = (
        age_design
        @ np.linalg.lstsq(
            age_design,
            donor_effects,
            rcond=None,
        )[0]
    )
    donor_effects -= projection
    for index, age in enumerate(ages):
        donor_effect = donor_effects[index]
        for cell_type, cell_effect, age_slope in (
            ("B cell", 0.0, 0.4),
            ("T cell", 1.2, 1.1),
        ):
            for tissue, tissue_effect in (("liver", 0.0), ("lung", 0.3)):
                value = (
                    3.0
                    + donor_effect
                    + cell_effect
                    + age_slope * age
                    + tissue_effect
                    + rng.normal(scale=0.04)
                )
                rows.append(
                    _feature_row(
                        donor=f"D{index:02d}",
                        cell_type=cell_type,
                        tissue=tissue,
                        age=float(age),
                        value=float(value),
                    )
                )
    return pd.DataFrame(rows)


def _interaction_spec() -> MixedModelSpec:
    """Return the repeated-donor model used by interaction tests."""
    return MixedModelSpec(
        analysis="age_and_context",
        group_key="donor id",
        continuous_keys=("age centered",),
        categorical_keys=("cell type", "tissue label"),
        interactions=(("age centered", "cell type"),),
    )


def test_marginal_contrasts_recover_repeated_donor_effects() -> None:
    """Mixed estimates recover paired effects and cell-specific slopes."""
    frame = _interaction_frame()
    contrasts = (
        MixedModelContrast(
            mode="categorical_vs_mean",
            predictor_key="cell type",
            estimand="adjusted_cell_type",
        ),
        MixedModelContrast(
            mode="categorical_pairwise",
            predictor_key="tissue label",
            estimand="paired_tissue",
        ),
        MixedModelContrast(
            mode="continuous_slope",
            predictor_key="age centered",
            by_key="cell type",
            estimand="age_slope_by_cell_type",
            effect_scale="score_per_year",
        ),
    )

    result = mixed_model_inference(
        frame,
        spec=_interaction_spec(),
        contrasts=contrasts,
    )

    assert result.diagnostics.iloc[0]["status"] == "ok"
    estimates = result.contrasts.set_index(["estimand", "level"])
    assert estimates.loc[
        ("adjusted_cell_type", "T cell"), "estimate"
    ] == pytest.approx(0.6, abs=0.04)
    assert estimates.loc[
        ("adjusted_cell_type", "B cell"), "estimate"
    ] == pytest.approx(-0.6, abs=0.04)
    assert estimates.loc[("paired_tissue", "liver"), "estimate"] == (
        pytest.approx(-0.3, abs=0.04)
    )
    assert estimates.loc[
        ("age_slope_by_cell_type", "B cell"), "estimate"
    ] == pytest.approx(0.4, abs=0.04)
    assert estimates.loc[
        ("age_slope_by_cell_type", "T cell"), "estimate"
    ] == pytest.approx(1.1, abs=0.04)
    paired = estimates.loc[("paired_tissue", "liver")]
    assert paired["n_paired_units"] == 30
    assert paired["contrast"] == "liver vs lung"
    assert (
        estimates.loc[("adjusted_cell_type", "T cell"), "contrast"]
        == "T cell vs marginal mean"
    )
    assert (
        estimates.loc[("age_slope_by_cell_type", "T cell"), "contrast"]
        == "age centered slope within T cell"
    )
    assert result.contrasts["pvalue_fdr"].notna().all()


def test_marginal_mean_excludes_levels_below_donor_support() -> None:
    """Rare levels stay visible but cannot distort the supported-level mean."""
    frame = _interaction_frame()
    rare = (
        frame.loc[
            frame["donor id"].isin(("D00", "D01"))
            & frame["tissue label"].eq("liver")
        ]
        .groupby("donor id", observed=True, sort=False)
        .head(1)
        .copy()
    )
    rare["cell type"] = "Rare cell"
    rare["feature_value"] = 100.0
    model_frame = pd.concat((frame, rare), ignore_index=True)
    spec = MixedModelSpec(
        analysis="supported_marginal_mean",
        group_key="donor id",
        continuous_keys=("age centered",),
        categorical_keys=("cell type", "tissue label"),
    )
    contrast = MixedModelContrast(
        mode="categorical_vs_mean",
        predictor_key="cell type",
        estimand="adjusted_cell_type",
    )

    result = mixed_model_inference(
        model_frame,
        spec=spec,
        contrasts=(contrast,),
    )

    effects = result.contrasts.set_index("level")
    assert effects.loc["T cell", "estimate"] == pytest.approx(0.6, abs=0.05)
    assert effects.loc["B cell", "estimate"] == pytest.approx(-0.6, abs=0.05)
    assert effects.loc["Rare cell", "status"] == "non_estimable"
    assert effects.loc["Rare cell", "reason"] == "insufficient_level_support"
    assert effects.loc["T cell", "n_reference_units"] == 30


def test_condition_contrasts_report_simple_effects_within_cell_type() -> None:
    """Condition-by-cell-type output reports simple condition effects."""
    rng = np.random.default_rng(23)
    donor_effects = rng.normal(scale=1.0, size=20)
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(("control", "disease")):
        for pair_index, donor_effect in enumerate(donor_effects):
            donor = f"{condition}_{pair_index:02d}"
            for cell_type, condition_effect in (
                ("B cell", 0.5),
                ("T cell", 1.8),
            ):
                value = (
                    2.0
                    + donor_effect
                    + (cell_type == "T cell")
                    + condition_index * condition_effect
                    + rng.normal(scale=0.03)
                )
                row = _feature_row(
                    donor=donor,
                    cell_type=cell_type,
                    tissue="blood",
                    age=0.0,
                    value=float(value),
                )
                row["condition status"] = condition
                rows.append(row)
    frame = pd.DataFrame(rows)
    spec = MixedModelSpec(
        analysis="condition_by_cell_type",
        group_key="donor id",
        categorical_keys=("condition status", "cell type"),
        interactions=(("condition status", "cell type"),),
    )
    contrast = MixedModelContrast(
        mode="categorical_vs_reference",
        predictor_key="condition status",
        reference="control",
        levels=("disease",),
        by_key="cell type",
        estimand="condition_effect_by_cell_type",
    )

    result = mixed_model_inference(frame, spec=spec, contrasts=(contrast,))

    effects = result.contrasts.set_index("by_level")
    assert effects.loc["B cell", "estimate"] == pytest.approx(0.5, abs=0.04)
    assert effects.loc["T cell", "estimate"] == pytest.approx(1.8, abs=0.04)
    assert (effects["level"] == "disease").all()
    assert (effects["reference"] == "control").all()
    assert (effects["n_level_units"] == 20).all()
    assert (effects["n_reference_units"] == 20).all()


def test_missing_and_nonfinite_inputs_remain_in_fit_diagnostics() -> None:
    """Diagnostics count dropped missing and infinite rows separately."""
    frame = _interaction_frame().iloc[:48].copy()
    frame.loc[frame.index[0], "feature_value"] = np.nan
    frame.loc[frame.index[1], "age centered"] = np.inf
    frame["tissue label"] = frame["tissue label"].astype(object)
    frame.loc[frame.index[2], "tissue label"] = -np.inf

    result = mixed_model_inference(
        frame,
        spec=_interaction_spec(),
        contrasts=(
            MixedModelContrast(
                mode="continuous_slope",
                predictor_key="age centered",
                estimand="age_slope",
            ),
        ),
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["n_input_observations"] == 48
    assert diagnostic["n_observations"] == 45
    assert diagnostic["n_dropped_missing"] == 1
    assert diagnostic["n_dropped_nonfinite"] == 2
    assert diagnostic["minimum_independent_units"] == 3
    assert diagnostic["minimum_focal_units"] == 3
    assert diagnostic["alpha"] == pytest.approx(0.05)
    assert diagnostic["max_iterations"] == 1_000
    assert diagnostic["status"] == "ok"


def test_categorical_metadata_is_scanned_without_dtype_failure() -> None:
    """Pandas categorical group and predictor columns remain valid inputs."""
    frame = _interaction_frame().iloc[:48].copy()
    for key in ("donor id", "cell type", "tissue label"):
        frame[key] = frame[key].astype("category")

    result = mixed_model_inference(
        frame,
        spec=_interaction_spec(),
        contrasts=(
            MixedModelContrast(
                mode="categorical_vs_mean",
                predictor_key="cell type",
                estimand="adjusted_cell_type",
            ),
        ),
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["n_dropped_nonfinite"] == 0
    assert diagnostic["status"] == "ok"
    assert result.contrasts["estimable"].all()


def test_mixed_type_group_labels_remain_distinct() -> None:
    """Numeric and string identifiers with the same display stay distinct."""
    frame = pd.DataFrame(
        [
            _feature_row(
                donor=donor,
                cell_type="T cell",
                tissue=tissue,
                age=0.0,
                value=float(index + offset),
            )
            for index, donor in enumerate((1, "1", 2, "2"))
            for offset, tissue in enumerate(("liver", "lung"))
        ]
    )

    result = mixed_model_inference(
        frame,
        spec=MixedModelSpec(
            analysis="mixed_identifier_types",
            group_key="donor id",
            categorical_keys=("tissue label",),
        ),
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["n_groups"] == 4
    assert diagnostic["n_independent_units"] == 4


def test_insufficient_groups_produce_visible_nonestimable_results() -> None:
    """Too few donors produce NaN contrasts with an actionable model status."""
    frame = _interaction_frame().loc[
        lambda table: table["donor id"].isin(("D00", "D01"))
    ]
    contrast = MixedModelContrast(
        mode="categorical_vs_mean",
        predictor_key="cell type",
        estimand="adjusted_cell_type",
    )

    result = mixed_model_inference(
        frame,
        spec=_interaction_spec(),
        contrasts=(contrast,),
    )

    assert result.diagnostics.iloc[0]["status"] == "insufficient_groups"
    assert (result.contrasts["status"] == "insufficient_groups").all()
    assert not result.contrasts["estimable"].any()
    assert result.contrasts["estimate"].isna().all()
    assert result.variance_components["variance"].isna().all()


def test_repeated_independent_units_require_a_covariance_term() -> None:
    """Nested repeated donors cannot be treated as residual independence."""
    frame = _interaction_frame().copy()
    frame["study"] = frame["donor id"].str[-2:].astype(int).mod(5).astype(str)
    result = mixed_model_inference(
        frame,
        spec=MixedModelSpec(
            analysis="nested_without_donor_effect",
            group_key="study",
            independent_unit_key="donor id",
            continuous_keys=("age centered",),
            categorical_keys=("cell type",),
        ),
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["status"] == "unmodeled_repeated_independent_unit"
    assert diagnostic["independent_unit_covariance"] == "unmodeled"
    assert diagnostic["max_observations_per_independent_unit"] > 1


def test_singleton_variance_component_is_not_reported_as_decomposed() -> None:
    """A row-level random effect is confounded with residual noise."""
    frame = _interaction_frame().copy()
    frame["row component"] = [f"R{index}" for index in range(len(frame))]
    result = mixed_model_inference(
        frame,
        spec=MixedModelSpec(
            analysis="singleton_variance_component",
            group_key="donor id",
            continuous_keys=("age centered",),
            categorical_keys=("cell type",),
            variance_component_keys=("row component",),
        ),
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["status"] == "unidentifiable_variance_component"
    assert diagnostic["reason"] == (
        "variance_component_all_singletons:row component"
    )
    assert not result.variance_components["estimable"].any()


def test_duplicate_variance_component_partitions_are_rejected() -> None:
    """Equivalent nested classifications cannot split the same variance."""
    frame = _interaction_frame().copy()
    frame["study"] = frame["donor id"].str[-2:].astype(int).mod(5).astype(str)
    frame["donor clone"] = frame["donor id"]
    result = mixed_model_inference(
        frame,
        spec=MixedModelSpec(
            analysis="duplicate_variance_components",
            group_key="study",
            independent_unit_key="donor id",
            continuous_keys=("age centered",),
            categorical_keys=("cell type",),
            variance_component_keys=("donor id", "donor clone"),
        ),
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["status"] == "unidentifiable_variance_component"
    assert diagnostic["reason"] == (
        "variance_components_confounded:donor id|donor clone"
    )


def test_pairwise_contrast_requires_overlapping_donor_support() -> None:
    """A tissue contrast remains non-estimable with too few paired donors."""
    rng = np.random.default_rng(29)
    rows: list[dict[str, object]] = []
    for donor_index in range(12):
        donor_effect = rng.normal(scale=1.0)
        tissues: list[str] = []
        if donor_index < 8:
            tissues.append("liver")
        if donor_index >= 4:
            tissues.append("lung")
        for tissue in tissues:
            rows.append(
                _feature_row(
                    donor=f"D{donor_index:02d}",
                    cell_type="T cell",
                    tissue=tissue,
                    age=0.0,
                    value=float(
                        2.0
                        + donor_effect
                        + (0.4 if tissue == "lung" else 0.0)
                        + rng.normal(scale=0.05)
                    ),
                )
            )
    frame = pd.DataFrame(rows)
    spec = MixedModelSpec(
        analysis="paired_tissue",
        group_key="donor id",
        categorical_keys=("tissue label",),
        minimum_focal_units=5,
    )
    contrast = MixedModelContrast(
        mode="categorical_pairwise",
        predictor_key="tissue label",
        estimand="paired_tissue",
    )

    result = mixed_model_inference(frame, spec=spec, contrasts=(contrast,))

    row = result.contrasts.iloc[0]
    assert row["n_level_units"] == 8
    assert row["n_reference_units"] == 8
    assert row["n_paired_units"] == 4
    assert row["status"] == "non_estimable"
    assert row["reason"] == "insufficient_paired_support"
    assert np.isnan(row["estimate"])


def test_variance_fractions_decompose_conditional_variability() -> None:
    """Donor and residual variance fractions are nonnegative and sum to one."""
    result = mixed_model_inference(
        _interaction_frame(),
        spec=_interaction_spec(),
    )

    variance = result.variance_components.set_index("component")
    assert set(variance.index) == {"donor id", "residual"}
    assert variance["estimable"].all()
    assert (variance["variance"] >= 0).all()
    assert variance["variance_fraction"].sum() == pytest.approx(1.0)
    assert (
        variance.loc["donor id", "variance_fraction"]
        > variance.loc["residual", "variance_fraction"]
    )


def test_nested_variance_component_is_named_and_included_in_fractions() -> None:
    """Nested donor variance is reported beside study and residual variance."""
    rng = np.random.default_rng(37)
    rows: list[dict[str, object]] = []
    for study_index in range(10):
        study_effect = rng.normal(scale=1.2)
        for donor_index in range(4):
            donor_effect = rng.normal(scale=0.6)
            donor = f"S{study_index:02d}_D{donor_index:02d}"
            for repeat in range(3):
                row = _feature_row(
                    donor=donor,
                    cell_type="T cell",
                    tissue=f"context_{repeat}",
                    age=0.0,
                    value=float(
                        4.0
                        + study_effect
                        + donor_effect
                        + rng.normal(scale=0.08)
                    ),
                )
                row["study id"] = f"S{study_index:02d}"
                rows.append(row)
    spec = MixedModelSpec(
        analysis="nested_variance",
        group_key="study id",
        independent_unit_key="donor id",
        variance_component_keys=("donor id",),
    )

    result = mixed_model_inference(pd.DataFrame(rows), spec=spec)

    variance = result.variance_components.set_index("component")
    assert result.diagnostics.iloc[0]["status"] == "ok"
    assert set(variance.index) == {"study id", "donor id", "residual"}
    assert variance["estimable"].all()
    assert variance["variance_fraction"].sum() == pytest.approx(1.0)
    assert variance.loc["donor id", "component_type"] == "variance_component"
    assert result.diagnostics.iloc[0]["variance_component_keys"] == "donor id"


def test_single_level_terms_remain_visible_without_blocking_other_tests() -> (
    None
):
    """Zero-degree categorical terms are explicit while other Wald tests run."""
    frame = _interaction_frame().assign(**{"cell type": "T cell"})
    spec = MixedModelSpec(
        analysis="single_level_covariate",
        group_key="donor id",
        continuous_keys=("age centered",),
        categorical_keys=("cell type", "tissue label"),
    )

    result = mixed_model_inference(frame, spec=spec)

    tests = result.term_tests.set_index("term")
    assert result.diagnostics.iloc[0]["status"] == "ok"
    assert tests.loc["C(cell type)", "status"] == "non_estimable"
    assert tests.loc["C(cell type)", "reason"] == "zero_degree_term"
    assert tests.loc["C(cell type)", "degrees_of_freedom"] == 0
    assert pd.isna(tests.loc["C(cell type)", "pvalue_fdr"])
    assert tests.loc["C(tissue label)", "status"] == "ok"
    assert np.isfinite(tests.loc["C(tissue label)", "pvalue_fdr"])


def test_unrepresentable_feature_keys_are_rejected() -> None:
    """Feature identities cannot be silently dropped from stable outputs."""
    frame = _interaction_frame().assign(gene="NASP_TEST")

    with pytest.raises(ValueError, match="unsupported metadata columns"):
        mixed_model_inference(
            frame,
            spec=_interaction_spec(),
            feature_keys=("gene",),
        )


def test_model_construction_failure_is_isolated_to_its_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One invalid feature records failure without aborting later features."""
    mixed_models = importlib.import_module(
        "nasp_atlas.single_cell.associations.mixed_models"
    )
    real_mixedlm = mixed_models.smf.mixedlm
    call_count = 0

    def fail_first_construction(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("synthetic design failure")
        return real_mixedlm(*args, **kwargs)

    monkeypatch.setattr(mixed_models.smf, "mixedlm", fail_first_construction)
    failed = _interaction_frame().assign(
        feature_id="construction_failure",
        feature_label="construction_failure",
    )
    valid = _interaction_frame(seed=31).assign(
        feature_id="valid_feature",
        feature_label="valid_feature",
    )

    result = mixed_model_inference(
        pd.concat((failed, valid), ignore_index=True),
        spec=_interaction_spec(),
    )

    diagnostics = result.diagnostics.set_index("feature_id")
    assert (
        diagnostics.loc["construction_failure", "status"]
        == "model_construction_failed"
    )
    assert (
        diagnostics.loc["construction_failure", "reason"]
        == "model_construction_failed:ValueError"
    )
    assert diagnostics.loc["valid_feature", "status"] == "ok"
