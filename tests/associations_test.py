"""Tests for the donor-aware NASP association system."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from nasp_atlas.analysis.tabula_sapiens import association_analysis
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import aggregate_feature_frame
from nasp_atlas.single_cell.associations import benjamini_hochberg
from nasp_atlas.single_cell.associations import build_cell_feature_frame
from nasp_atlas.single_cell.associations import merge_eqtl_counts
from nasp_atlas.single_cell.associations import (
    partial_correlation_controlling_tissue,
)
from nasp_atlas.single_cell.associations import regress_features_on_continuous
from nasp_atlas.single_cell.associations import resolve_feature_specs
from nasp_atlas.single_cell.associations import summarize_feature_groups
from nasp_atlas.single_cell.associations import (
    test_feature_groups as run_feature_group_tests,
)
from nasp_atlas.single_cell.associations import validate_eqtl_table


SENSOR_GENES = ("CGAS", "IFIH1", "DDX58")
METADATA_COLUMNS = (
    "donor_id",
    "tissue_in_publication",
    "cell_type",
    "sex",
    "assay",
    "development_stage",
    "age_years",
)


def _synthetic_adata(*, seed: int = 0) -> ad.AnnData:
    """Build a donor-structured synthetic AnnData for association tests.

    The matrix carries the three sensor genes with an age-graded CGAS signal so
    that donor-aware regression recovers a positive age association. Six donors
    span three ages across two tissues, giving multiple cells per donor-tissue
    unit.

    Args:
      seed: Seed for the Poisson expression draws.

    Returns:
      AnnData with sparse CSR expression, var indexed by gene symbol with a
      `feature_name` column, and the full obs metadata schema.
    """
    rng = np.random.default_rng(seed)
    donors = [f"D{i}" for i in range(6)]
    tissues = ["lung", "liver"]
    donor_age = {
        donor: 30.0 + 10.0 * index for index, donor in enumerate(donors)
    }
    donor_sex = {
        donor: ("male" if index % 2 else "female")
        for index, donor in enumerate(donors)
    }
    rows: list[list[float]] = []
    records: list[tuple[str, ...]] = []
    for donor in donors:
        for tissue in tissues:
            for cell in range(8):
                age_fraction = donor_age[donor] / 100.0
                rows.append(
                    [
                        float(rng.poisson(1 + age_fraction * 6)),
                        float(rng.poisson(2)),
                        float(rng.poisson(2)),
                    ]
                )
                records.append(
                    (
                        f"{donor}_{tissue}_{cell}",
                        donor,
                        tissue,
                        "Tcell",
                        donor_sex[donor],
                        "10x",
                        "human adult stage",
                        float(donor_age[donor]),
                    )
                )
    obs = pd.DataFrame(records, columns=["obs_name", *METADATA_COLUMNS])
    obs = obs.set_index("obs_name")
    obs.index = pd.Index(obs.index.astype(str), dtype=object)
    for column in obs.columns.drop("age_years"):
        obs[column] = obs[column].astype(object)
    var = pd.DataFrame(index=pd.Index(list(SENSOR_GENES), dtype=object))
    var["feature_name"] = pd.Series(
        list(SENSOR_GENES),
        index=var.index,
        dtype=object,
    )
    adata = ad.AnnData(X=sp.csr_matrix(np.array(rows)), obs=obs, var=var)
    adata.layers["log1p"] = adata.X.copy().multiply(2.0)
    return adata


def _synthetic_scores(adata: ad.AnnData, *, seed: int = 1) -> pd.DataFrame:
    """Build a score table aligned to an AnnData by obs name.

    NASP_DNA_SENSING and IFN_I_OUTPUT track donor age, while
    NASP_RNA_SENSING is random so that its age association is null.

    Args:
      adata: AnnData whose obs names and ages the scores align to.
      seed: Seed for the random restriction score.

    Returns:
      DataFrame indexed by obs name with the signed module score columns
      and the metadata columns needed by the association workflow.
    """
    rng = np.random.default_rng(seed)
    ages = adata.obs["age_years"].to_numpy(dtype=float)
    ifn = (ages - ages.mean()) / 10.0 + rng.normal(scale=0.02, size=ages.size)
    restriction = rng.normal(size=ages.size)
    scores = pd.DataFrame(
        {
            "NASP_DNA_SENSING_score": ifn,
            "NASP_RNA_SENSING_score": restriction,
            "IFN_I_OUTPUT_score": ifn
            + rng.normal(
                scale=0.05,
                size=ages.size,
            ),
        },
        index=adata.obs_names.astype(str),
    )
    for column in METADATA_COLUMNS:
        scores[column] = adata.obs[column].to_numpy()
    return scores


def _write_inputs(
    tmp_path: Path, adata: ad.AnnData, scores: pd.DataFrame
) -> tuple[Path, Path]:
    """Persist an AnnData and score CSV and return their paths.

    Args:
      tmp_path: Temporary directory for the written inputs.
      adata: AnnData to write as h5ad.
      scores: Score table to write as a gzipped CSV indexed by obs_name.

    Returns:
      Tuple of the h5ad path and the score CSV path.
    """
    h5ad_path = tmp_path / "synthetic.h5ad"
    score_path = tmp_path / "scores.csv.gz"
    adata.write_h5ad(h5ad_path)
    scores.to_csv(score_path, index=True, index_label="obs_name")
    return h5ad_path, score_path


def test_benjamini_hochberg_matches_reference() -> None:
    """BH adjustment matches hand-calculated values and preserves NaN."""
    pvalues = np.array([0.01, 0.04, 0.03, 0.002, np.nan])

    adjusted = benjamini_hochberg(pvalues)

    np.testing.assert_allclose(
        adjusted,
        [0.02, 0.04, 0.04, 0.008, np.nan],
        equal_nan=True,
    )


def test_resolve_feature_specs_splits_sources() -> None:
    """Modules resolve to score columns and sensors to var names."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)

    specs, skipped = resolve_feature_specs(
        adata,
        scores,
        module_ids=["NASP_DNA_SENSING"],
        sensor_group="nucleic_acid_sensors",
        scorer="scanpy",
        gene_symbol_column="feature_name",
    )

    sources = {spec.feature_id: spec.source for spec in specs}
    assert sources["NASP_DNA_SENSING_score"] == "scores"
    assert sources["CGAS"] == "expression"
    assert skipped == []


def test_resolve_feature_specs_reports_missing() -> None:
    """Missing genes are returned as skip records, not silently dropped."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)

    specs, skipped = resolve_feature_specs(
        adata,
        scores,
        gene_symbols=["CGAS", "NOT_A_GENE"],
        scorer="scanpy",
        gene_symbol_column="feature_name",
    )

    resolved = {spec.feature_id for spec in specs}
    assert "CGAS" in resolved
    assert any(record["requested"] == "NOT_A_GENE" for record in skipped)


def test_resolve_feature_specs_missing_score_column() -> None:
    """A requested score column absent from the table is skipped."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)

    specs, skipped = resolve_feature_specs(
        adata,
        scores,
        score_keys=["NASP_DNA_SENSING_score", "NASP_ABSENT_score"],
    )

    resolved = {spec.feature_id for spec in specs}
    assert "NASP_DNA_SENSING_score" in resolved
    assert any(
        "NASP_ABSENT_score" in record["skip_reason"] for record in skipped
    )


def test_build_cell_feature_frame_is_long_without_unit() -> None:
    """The cell frame is tidy-long and carries no statistical_unit column."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    specs, _ = resolve_feature_specs(
        adata,
        scores,
        module_ids=["NASP_DNA_SENSING"],
        gene_symbols=["CGAS"],
        gene_symbol_column="feature_name",
    )

    cell_frame = build_cell_feature_frame(
        adata, scores, specs, schema=ObsSchema()
    )

    assert "statistical_unit" not in cell_frame.columns
    assert set(cell_frame["feature_id"]) == {
        "NASP_DNA_SENSING_score",
        "CGAS",
    }
    assert len(cell_frame) == adata.n_obs * len(specs)


def test_module_features_use_scores_not_anndata() -> None:
    """Module feature values come from the score table, not expression."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    specs, _ = resolve_feature_specs(
        adata,
        scores,
        module_ids=["NASP_DNA_SENSING"],
    )

    cell_frame = build_cell_feature_frame(
        adata, scores, specs, schema=ObsSchema()
    )
    module_values = (
        cell_frame.set_index("obs_name")["feature_value"]
        .reindex(adata.obs_names.astype(str))
        .to_numpy()
    )

    np.testing.assert_allclose(
        module_values, scores["NASP_DNA_SENSING_score"].to_numpy()
    )


def test_sensor_features_use_anndata_expression() -> None:
    """Gene features read from the configured AnnData expression source."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    specs, _ = resolve_feature_specs(
        adata, scores, gene_symbols=["CGAS"], gene_symbol_column="feature_name"
    )

    cell_frame = build_cell_feature_frame(
        adata, scores, specs, schema=ObsSchema()
    )
    gene_values = (
        cell_frame.set_index("obs_name")["feature_value"]
        .reindex(adata.obs_names.astype(str))
        .to_numpy()
    )
    expected = np.asarray(adata[:, "CGAS"].layers["log1p"].todense()).ravel()

    np.testing.assert_allclose(gene_values, expected)


@pytest.mark.parametrize(
    "unit,expected_units",
    [
        ("donor", 6),
        ("donor_tissue", 12),
        ("donor_tissue_cell_type", 12),
    ],
)
def test_aggregate_feature_frame_units(unit: str, expected_units: int) -> None:
    """Aggregation collapses cells to the requested unit count per feature."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    specs, _ = resolve_feature_specs(
        adata,
        scores,
        module_ids=["NASP_DNA_SENSING"],
    )
    cell_frame = build_cell_feature_frame(
        adata, scores, specs, schema=ObsSchema()
    )

    unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit=unit,
        aggregation="mean",
        schema=ObsSchema(),
    )

    assert (unit_frame["statistical_unit"] == unit).all()
    assert len(unit_frame) == expected_units


@pytest.mark.parametrize("unit", ["cell", "metacell"])
def test_aggregate_feature_frame_annotates_existing_units(unit: str) -> None:
    """Rows already representing units retain values and gain unit metadata."""
    cell_frame = pd.DataFrame(
        {
            "obs_name": ["cell_a", "cell_b"],
            "feature_type": ["gene", "gene"],
            "feature_id": ["CGAS", "CGAS"],
            "feature_label": ["CGAS", "CGAS"],
            "feature_value": [2.0, np.nan],
        }
    )
    original = cell_frame.copy()

    unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit=unit,
        aggregation="mean",
        schema=ObsSchema(),
    )

    pd.testing.assert_frame_equal(cell_frame, original)
    np.testing.assert_allclose(
        unit_frame["feature_value"],
        [2.0, np.nan],
        equal_nan=True,
    )
    assert unit_frame["unit_id"].tolist() == ["cell_a", "cell_b"]
    assert unit_frame["n_cells"].tolist() == [1.0, 0.0]
    assert unit_frame["n_cells_total"].tolist() == [1.0, 1.0]
    assert (unit_frame["statistical_unit"] == unit).all()


def test_aggregation_does_not_carry_heterogeneous_cell_type() -> None:
    """Cell type becomes missing when a donor-tissue unit mixes cell types."""
    adata = _synthetic_adata()
    adata.obs["cell_type"] = np.resize(
        np.array(["Tcell", "Bcell"], dtype=object),
        adata.n_obs,
    )
    scores = _synthetic_scores(adata)
    specs, _ = resolve_feature_specs(
        adata,
        scores,
        module_ids=["NASP_DNA_SENSING"],
    )
    cell_frame = build_cell_feature_frame(
        adata, scores, specs, schema=ObsSchema()
    )

    unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit="donor_tissue",
        aggregation="mean",
        schema=ObsSchema(),
    )

    assert unit_frame["cell_type"].isna().all()
    assert (unit_frame["n_cells"] == 8).all()
    assert (unit_frame["n_cells_total"] == 8).all()


def test_regression_recovers_age_signal() -> None:
    """Donor-tissue regression is significant for the age-graded module."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    specs, _ = resolve_feature_specs(
        adata,
        scores,
        module_ids=["NASP_DNA_SENSING", "NASP_RNA_SENSING"],
    )
    cell_frame = build_cell_feature_frame(
        adata, scores, specs, schema=ObsSchema()
    )
    unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit="donor_tissue",
        aggregation="mean",
        schema=ObsSchema(),
    )

    result = regress_features_on_continuous(
        unit_frame, predictor_key="age_years"
    )

    assert (result["statistical_unit"] == "donor_tissue").all()
    assert (result["analysis_role"] == "inferential").all()
    ifn = result.loc[result["feature_id"] == "NASP_DNA_SENSING_score"].iloc[0]
    assert ifn["pearson_r"] > 0.9
    assert ifn["pearson_pvalue"] < 0.01


def test_regression_skips_constant_response() -> None:
    """A constant feature is skipped rather than reported with NaN tests."""
    frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 3,
            "feature_id": ["constant"] * 3,
            "feature_label": ["constant"] * 3,
            "feature_value": [1.0, 1.0, 1.0],
            "age_years": [30.0, 40.0, 50.0],
            "statistical_unit": ["donor"] * 3,
            "aggregation": ["mean"] * 3,
        }
    )

    result = regress_features_on_continuous(
        frame,
        predictor_key="age_years",
    )

    assert result.iloc[0]["skipped"]
    assert result.iloc[0]["skip_reason"] == "constant_response"


def test_feature_group_tests_compare_two_supported_groups() -> None:
    """Two supported groups produce Welch and Mann-Whitney tests."""
    unit_frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 6,
            "feature_id": ["NASP_TEST_score"] * 6,
            "feature_label": ["NASP_TEST"] * 6,
            "feature_value": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
            "sex": ["female"] * 3 + ["male"] * 3,
            "statistical_unit": ["donor"] * 6,
            "aggregation": ["mean"] * 6,
        }
    )

    result = run_feature_group_tests(unit_frame, group_key="sex")

    row = result.iloc[0]
    assert row["parametric_test"] == "welch_t"
    assert row["nonparametric_test"] == "mann_whitney_u"
    assert {row["group_a"], row["group_b"]} == {"female", "male"}
    assert not row["skipped"]


def test_summarize_feature_groups_reports_group_support() -> None:
    """Group summaries report means and independent donor support."""
    unit_frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 6,
            "feature_id": ["NASP_TEST_score"] * 6,
            "feature_label": ["NASP_TEST"] * 6,
            "feature_value": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
            "donor_id": [f"D{index}" for index in range(6)],
            "sex": ["female"] * 3 + ["male"] * 3,
            "statistical_unit": ["donor"] * 6,
            "aggregation": ["mean"] * 6,
        }
    )

    summary = summarize_feature_groups(
        unit_frame,
        group_key="sex",
        schema=ObsSchema(),
    ).set_index("group")

    assert summary.loc["female", "mean"] == pytest.approx(2.0)
    assert summary.loc["male", "mean"] == pytest.approx(6.0)
    assert (summary["n_units"] == 3).all()
    assert (summary["n_donors"] == 3).all()
    assert (summary["statistical_unit"] == "donor").all()


def test_summarize_feature_groups_empty_declares_columns() -> None:
    """An empty grouping still returns a frame with the declared columns."""
    empty = pd.DataFrame(
        columns=[
            "feature_type",
            "feature_id",
            "feature_label",
            "feature_value",
            "donor_id",
            "sex",
            "statistical_unit",
            "aggregation",
        ]
    )

    summary = summarize_feature_groups(
        empty, group_key="sex", schema=ObsSchema()
    )

    assert summary.empty
    assert summary.columns.tolist() == [
        "feature_type",
        "feature_id",
        "feature_label",
        "group_key",
        "group",
        "statistical_unit",
        "aggregation",
        "analysis_role",
        "mean",
        "median",
        "std",
        "sem",
        "n_units",
        "n_donors",
    ]


def test_partial_correlation_controls_tissue() -> None:
    """Partial correlation returns one residualized row per feature."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    specs, _ = resolve_feature_specs(
        adata,
        scores,
        module_ids=["NASP_DNA_SENSING"],
    )
    cell_frame = build_cell_feature_frame(
        adata, scores, specs, schema=ObsSchema()
    )
    unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit="donor_tissue",
        aggregation="mean",
        schema=ObsSchema(),
    )

    partial = partial_correlation_controlling_tissue(
        unit_frame, predictor_key="age_years", schema=ObsSchema()
    )

    assert len(partial) == 1
    assert partial.iloc[0]["feature_id"] == "NASP_DNA_SENSING_score"


def test_validate_eqtl_table_returns_available_value_columns() -> None:
    """A valid eQTL table reports each supported burden column in order."""
    eqtl_table = pd.DataFrame(
        {
            "gene_symbol": ["CGAS"],
            "total_eqtls": [12],
            "tissue_specific_eqtls": [3],
        }
    )

    value_columns = validate_eqtl_table(eqtl_table, merge_mode="gene")

    assert value_columns == ["total_eqtls", "tissue_specific_eqtls"]


def test_merge_eqtl_counts_joins_gene_burden() -> None:
    """Gene-mode merging annotates matches and preserves unmatched genes."""
    unit_frame = pd.DataFrame(
        {
            "feature_label": ["CGAS", "IFIH1"],
            "feature_value": [1.5, 2.5],
        }
    )
    eqtl_table = pd.DataFrame(
        {
            "gene_symbol": ["CGAS"],
            "total_eqtls": [12],
        }
    )

    merged = merge_eqtl_counts(
        unit_frame, eqtl_table, merge_mode="gene", schema=ObsSchema()
    ).set_index("feature_label")

    assert merged.loc["CGAS", "total_eqtls"] == 12
    assert np.isnan(merged.loc["IFIH1", "total_eqtls"])
    assert (merged["eqtl_merge_mode"] == "gene").all()


def test_validate_eqtl_table_reports_missing_columns() -> None:
    """Validation raises when a merge mode's key column is absent."""
    bad_table = pd.DataFrame({"total_eqtls": [1]})

    with pytest.raises(KeyError, match="gene_symbol"):
        validate_eqtl_table(bad_table, merge_mode="gene")


def test_orchestration_end_to_end(tmp_path: Path) -> None:
    """The workflow writes donor-aware tables recording the statistical unit."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    h5ad_path, score_path = _write_inputs(tmp_path, adata, scores)
    output_dir = tmp_path / "assoc"

    association_analysis(
        h5ad_path=h5ad_path,
        score_csv_path=score_path,
        output_dir=output_dir,
        sensor_group="nucleic_acid_sensors",
        statistical_unit="donor_tissue",
        aggregation="mean",
    )

    tables = output_dir / "association_tables"
    regression = pd.read_csv(tables / "association_regression_results.csv")
    assert "donor_tissue" in set(regression["statistical_unit"])
    assert "donor_tissue_cell_type" in set(regression["statistical_unit"])
    assert (regression["analysis_role"] == "inferential").all()
    assert {
        "pooled_donor_tissue",
        "within_tissue",
        "within_tissue_cell_type",
    } <= set(regression["analysis_scope"])
    ifn = regression.loc[
        regression["feature_id"] == "NASP_DNA_SENSING_score"
    ].iloc[0]
    assert ifn["pearson_r"] > 0.9
    skipped = pd.read_csv(tables / "association_skipped_features.csv")
    assert {
        "feature_type",
        "requested",
        "skip_reason",
    } <= set(skipped.columns)
    profiles = pd.read_csv(tables / "nasp_evidence_profiles.csv")
    assert {
        "unit_id",
        "relative_competence",
        "output_minus_competence_gap",
    } <= set(profiles.columns)
    coupling = pd.read_csv(tables / "nasp_module_coupling.csv")
    assert {
        "module_a",
        "module_b",
        "spearman_r",
        "signed_gene_jaccard",
    } <= set(coupling.columns)
    sensor_output = pd.read_csv(tables / "nasp_sensor_output_coupling.csv")
    assert {
        "gene",
        "output_module",
        "spearman_r",
        "gene_in_output_module",
    } <= set(sensor_output.columns)
    context_ranking = pd.read_csv(tables / "nasp_module_context_ranking.csv")
    assert {
        "feature_label",
        "n_finite_donors",
        "eligible",
        "context_percentile",
    } <= set(context_ranking.columns)
    hypotheses = pd.read_csv(tables / "nasp_hypothesis_priorities.csv")
    assert {
        "hypothesis",
        "priority_score",
        "interpretation",
        "experimental_follow_up",
    } <= set(hypotheses.columns)
    mechanistic_edges = pd.read_csv(tables / "nasp_mechanistic_edges.csv")
    assert {
        "source_module",
        "target_module",
        "mechanistic_edge",
        "spearman_r",
    } <= set(mechanistic_edges.columns)
    assert {
        "dna_sensing_to_ifn_output",
        "rna_sensing_to_ifn_output",
    } <= set(mechanistic_edges["mechanistic_edge"])
    stability = pd.read_csv(tables / "association_age_stability.csv")
    assert {
        "analysis_scope",
        "n_tested_strata",
        "dominant_direction",
        "direction_consistency_fraction",
    } <= set(stability.columns)
    nasp_plots = tables.parent / "association_plots" / "nasp"
    expected_plots = {
        "nasp_module_coupling_heatmap.png",
        "nasp_competence_output_state_map.png",
        "nasp_ranked_hypotheses.png",
        "nasp_sensor_output_mismatch.png",
        "nasp_age_effects_by_cell_type.png",
        "nasp_age_effect_consistency_across_cell_types.png",
        "nasp_mechanistic_edge_network.png",
    }
    written_plots = {path.name: path for path in nasp_plots.glob("*.png")}
    assert expected_plots <= set(written_plots)
    assert all(
        written_plots[name].stat().st_size > 0 for name in expected_plots
    )


def test_orchestration_writes_eqtl_annotations(tmp_path: Path) -> None:
    """Valid feature-level eQTL burden is written without invalid regression."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    h5ad_path, score_path = _write_inputs(tmp_path, adata, scores)
    eqtl_path = tmp_path / "eqtl.csv"
    pd.DataFrame(
        {"module_id": ["NASP_DNA_SENSING"], "total_eqtls": [7]}
    ).to_csv(eqtl_path, index=False)
    output_dir = tmp_path / "assoc"

    association_analysis(
        h5ad_path=h5ad_path,
        score_csv_path=score_path,
        output_dir=output_dir,
        sensor_group=None,
        eqtl_table_path=eqtl_path,
        eqtl_merge_mode="module",
        max_plots=0,
    )

    annotations = pd.read_csv(
        output_dir / "association_tables" / "association_eqtl_annotations.csv"
    )
    matched = annotations["feature_label"] == "NASP_DNA_SENSING"
    assert (annotations.loc[matched, "total_eqtls"] == 7).all()


def test_orchestration_rejects_duplicated_obs_name(tmp_path: Path) -> None:
    """A score table with duplicated obs names fails fast."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    scores = pd.concat([scores, scores.iloc[[0]]])
    h5ad_path = tmp_path / "synthetic.h5ad"
    score_path = tmp_path / "scores.csv.gz"
    adata.write_h5ad(h5ad_path)
    scores.to_csv(score_path, index=True, index_label="obs_name")

    with pytest.raises(ValueError, match="not unique"):
        association_analysis(
            h5ad_path=h5ad_path,
            score_csv_path=score_path,
            output_dir=tmp_path / "assoc",
            sensor_group=None,
        )


def test_orchestration_rejects_unaligned_scores(tmp_path: Path) -> None:
    """A score table sharing no obs names with the AnnData fails fast."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    scores.index = [f"other_{name}" for name in scores.index]
    h5ad_path = tmp_path / "synthetic.h5ad"
    score_path = tmp_path / "scores.csv.gz"
    adata.write_h5ad(h5ad_path)
    scores.to_csv(score_path, index=True, index_label="obs_name")

    with pytest.raises(ValueError, match="no obs_name"):
        association_analysis(
            h5ad_path=h5ad_path,
            score_csv_path=score_path,
            output_dir=tmp_path / "assoc",
            sensor_group=None,
        )


def test_orchestration_does_not_rescore(tmp_path: Path, monkeypatch) -> None:
    """The association workflow never calls the scanpy scoring entry points."""

    def _fail(*args, **kwargs):
        raise AssertionError("scoring must not run during association")

    monkeypatch.setattr(
        "nasp_atlas.single_cell.module_scoring.sc.tl.score_genes",
        _fail,
    )
    monkeypatch.setattr(
        "nasp_atlas.single_cell.module_scoring.aucell",
        _fail,
    )

    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    h5ad_path, score_path = _write_inputs(tmp_path, adata, scores)

    association_analysis(
        h5ad_path=h5ad_path,
        score_csv_path=score_path,
        output_dir=tmp_path / "assoc",
        sensor_group="nucleic_acid_sensors",
        statistical_unit="donor_tissue",
    )


def test_orchestration_cell_level_marked_descriptive(tmp_path: Path) -> None:
    """Cell-level descriptive plots are not marked inferential."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    h5ad_path, score_path = _write_inputs(tmp_path, adata, scores)
    output_dir = tmp_path / "assoc"

    association_analysis(
        h5ad_path=h5ad_path,
        score_csv_path=score_path,
        output_dir=output_dir,
        sensor_group="nucleic_acid_sensors",
        statistical_unit="donor_tissue",
        run_cell_level_descriptive_plots=True,
    )

    manifest = pd.read_csv(
        output_dir / "association_tables" / "association_plot_manifest.csv"
    )
    cell_rows = manifest.loc[manifest["statistical_unit"] == "cell"]
    assert not cell_rows.empty
