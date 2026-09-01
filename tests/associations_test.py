"""Tests for the donor-aware NASP association system."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from nasp_atlas.analysis.tabula_sapiens import association_analysis
from nasp_atlas.analysis.tabula_sapiens import workflows
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import aggregate_feature_frame
from nasp_atlas.single_cell.associations import (
    associate_features_with_eqtl_counts,
)
from nasp_atlas.single_cell.associations import benjamini_hochberg
from nasp_atlas.single_cell.associations import build_cell_feature_frame
from nasp_atlas.single_cell.associations import build_eqtl_association_frames
from nasp_atlas.single_cell.associations import merge_eqtl_counts
from nasp_atlas.single_cell.associations import (
    partial_correlation_controlling_tissue,
)
from nasp_atlas.single_cell.associations import prepare_eqtl_table
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


def test_regression_preserves_missing_categorical_stratum() -> None:
    """Missing categorical strata remain an explicit NA comparison group."""
    frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 6,
            "feature_id": ["NASP_TEST_score"] * 6,
            "feature_label": ["NASP_TEST"] * 6,
            "feature_value": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "age_years": [30.0, 40.0, 50.0] * 2,
            "sex": pd.Categorical(["female"] * 3 + [None] * 3),
            "statistical_unit": ["donor"] * 6,
            "aggregation": ["mean"] * 6,
        }
    )

    result = regress_features_on_continuous(
        frame,
        predictor_key="age_years",
        stratify_key="sex",
    )

    assert set(result["stratum"]) == {"female", "NA"}
    assert (result["n_units"] == 3.0).all()


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


def test_prepare_eqtl_table_accepts_long_sensor_schema() -> None:
    """Long sensor counts retain gene-tissue keys and source provenance."""
    source = pd.DataFrame(
        {
            "gene": ["CGAS", "CGAS", "IFIH1", "IFIH1"],
            "tissue": ["Liver", "Lung", "Liver", "Lung"],
            "tissue_abbrev": ["LIVER", "LUNG", "LIVER", "LUNG"],
            "n_significant_eqtls": [2, 5, 3, 8],
            "gene_total_eqtls": [7, 7, 11, 11],
        }
    )

    prepared = prepare_eqtl_table(source, merge_mode="gene_tissue")

    assert prepared.loc[0, "gene_symbol"] == "CGAS"
    assert prepared.loc[0, "tissue_specific_eqtls"] == 2
    assert prepared.loc[0, "eqtl_tissue_abbrev"] == "LIVER"
    assert (
        prepared["eqtl_source_schema"] == "nasp_sensor_eqtl_counts_long"
    ).all()


def test_prepare_eqtl_table_accepts_wide_gene_totals() -> None:
    """Wide sensor counts validate tissue sums and expose gene totals."""
    source = pd.DataFrame(
        {
            "gene": ["CGAS", "IFIH1"],
            "gene_total_eqtls": [7, 11],
            "LIVER": [2, 3],
            "LUNG": [5, 8],
        }
    )

    prepared = prepare_eqtl_table(source, merge_mode="gene")

    assert prepared[["gene_symbol", "total_eqtls"]].to_dict(
        orient="records"
    ) == [
        {"gene_symbol": "CGAS", "total_eqtls": 7},
        {"gene_symbol": "IFIH1", "total_eqtls": 11},
    ]


def test_prepare_eqtl_table_rejects_wide_tissue_join() -> None:
    """Wide tissue abbreviations require the long table's tissue labels."""
    source = pd.DataFrame(
        {
            "gene": ["CGAS"],
            "gene_total_eqtls": [7],
            "LIVER": [2],
            "LUNG": [5],
        }
    )

    with pytest.raises(KeyError, match="use the long table"):
        prepare_eqtl_table(source, merge_mode="gene_tissue")


def test_prepare_eqtl_table_sums_sensor_counts_by_tissue() -> None:
    """Tissue mode sums significant pairs across the source sensor genes."""
    source = pd.DataFrame(
        {
            "gene": ["CGAS", "CGAS", "IFIH1", "IFIH1"],
            "tissue": ["Liver", "Lung", "Liver", "Lung"],
            "n_significant_eqtls": [2, 5, 3, 8],
            "gene_total_eqtls": [7, 7, 11, 11],
        }
    )

    prepared = prepare_eqtl_table(source, merge_mode="tissue").set_index(
        "tissue"
    )

    assert prepared.loc["Liver", "tissue_specific_eqtls"] == 5
    assert prepared.loc["Lung", "tissue_specific_eqtls"] == 13
    assert (prepared["eqtl_n_genes"] == 2).all()


def test_prepare_eqtl_table_rejects_inconsistent_gene_totals() -> None:
    """Long sensor counts fail when tissue counts do not equal gene totals."""
    source = pd.DataFrame(
        {
            "gene": ["CGAS", "CGAS"],
            "tissue": ["Liver", "Lung"],
            "n_significant_eqtls": [2, 5],
            "gene_total_eqtls": [8, 8],
        }
    )

    with pytest.raises(ValueError, match="do not sum to gene totals"):
        prepare_eqtl_table(source, merge_mode="gene_tissue")


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


def test_merge_eqtl_counts_joins_gene_and_normalized_tissue() -> None:
    """Gene-tissue mode matches spelling-equivalent atlas tissue labels."""
    unit_frame = pd.DataFrame(
        {
            "feature_type": ["gene_expression", "module_score"],
            "feature_id": ["ENSG_CGAS", "NASP_DNA_SENSING_score"],
            "feature_label": ["CGAS", "NASP_DNA_SENSING"],
            "tissue_in_publication": ["liver", "liver"],
            "feature_value": [1.5, 2.5],
        }
    )
    eqtl_table = pd.DataFrame(
        {
            "gene": ["CGAS"],
            "tissue": ["Liver"],
            "n_significant_eqtls": [2],
            "gene_total_eqtls": [2],
        }
    )

    merged = merge_eqtl_counts(
        unit_frame,
        eqtl_table,
        merge_mode="gene_tissue",
        schema=ObsSchema(),
    )

    gene_row = merged.loc[merged["feature_label"] == "CGAS"].iloc[0]
    module_row = merged.loc[merged["feature_label"] == "NASP_DNA_SENSING"].iloc[
        0
    ]
    assert gene_row["tissue_specific_eqtls"] == 2
    assert gene_row["eqtl_source_tissue"] == "Liver"
    assert bool(gene_row["eqtl_matched"])
    assert not bool(module_row["eqtl_matched"])


def test_eqtl_association_frames_average_donors_before_plotting() -> None:
    """Plot frames use genes, rather than repeated donors, as support."""
    unit_frame = pd.DataFrame(
        {
            "feature_type": ["gene_expression"] * 8,
            "feature_id": ["ENSG_CGAS"] * 4 + ["ENSG_IFIH1"] * 4,
            "feature_label": ["CGAS"] * 4 + ["IFIH1"] * 4,
            "donor_id": ["D1", "D2"] * 4,
            "tissue_in_publication": ["Liver", "Liver", "Lung", "Lung"] * 2,
            "feature_value": [1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
            "statistical_unit": ["donor_tissue"] * 8,
            "aggregation": ["mean"] * 8,
        }
    )
    eqtl_table = pd.DataFrame(
        {
            "gene": ["CGAS", "CGAS", "IFIH1", "IFIH1"],
            "tissue": ["Liver", "Lung", "Liver", "Lung"],
            "n_significant_eqtls": [10, 20, 30, 40],
            "gene_total_eqtls": [30, 30, 70, 70],
        }
    )
    merged = merge_eqtl_counts(
        unit_frame,
        eqtl_table,
        merge_mode="gene_tissue",
        schema=ObsSchema(),
    )

    frames = build_eqtl_association_frames(
        merged,
        merge_mode="gene_tissue",
        schema=ObsSchema(),
    )
    plot_frame = frames[
        "eqtl_tissue_count_within_tissue_across_genes"
    ].sort_values(["tissue_in_publication", "source_feature_label"])

    assert plot_frame["statistical_unit"].eq("gene").all()
    assert plot_frame["n_source_units"].eq(2).all()
    assert plot_frame["unit_id"].tolist() == ["CGAS", "IFIH1"] * 2
    assert plot_frame["feature_value"].tolist() == [2.0, 6.0, 3.0, 7.0]


def test_eqtl_tissue_association_uses_tissues_as_units() -> None:
    """Tissue-level eQTL tests do not count repeated donors as support."""
    unit_frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 6,
            "feature_id": ["NASP_DNA_SENSING_score"] * 6,
            "feature_label": ["NASP_DNA_SENSING"] * 6,
            "donor_id": ["D1", "D2"] * 3,
            "tissue_in_publication": [
                "Liver",
                "Liver",
                "Lung",
                "Lung",
                "Spleen",
                "Spleen",
            ],
            "feature_value": [1.0, 1.2, 2.0, 2.2, 3.0, 3.2],
            "statistical_unit": ["donor_tissue"] * 6,
            "aggregation": ["mean"] * 6,
        }
    )
    eqtl_table = pd.DataFrame(
        {
            "tissue": ["Liver", "Lung", "Spleen"],
            "tissue_specific_eqtls": [10, 20, 30],
        }
    )
    merged = merge_eqtl_counts(
        unit_frame,
        eqtl_table,
        merge_mode="tissue",
        schema=ObsSchema(),
    )

    result = associate_features_with_eqtl_counts(
        merged,
        merge_mode="tissue",
        schema=ObsSchema(),
    )

    assert result.iloc[0]["statistical_unit"] == "tissue"
    assert result.iloc[0]["n"] == 3
    assert result.iloc[0]["pearson_r"] > 0.99


def test_eqtl_tissue_association_rejects_cell_units() -> None:
    """Tissue-level counts cannot be tested against repeated cell rows."""
    merged = pd.DataFrame(
        {
            "feature_type": ["module_score"],
            "feature_id": ["NASP_DNA_SENSING_score"],
            "feature_label": ["NASP_DNA_SENSING"],
            "feature_value": [1.0],
            "tissue_in_publication": ["Liver"],
            "tissue_specific_eqtls": [10.0],
            "statistical_unit": ["cell"],
            "aggregation": ["mean"],
        }
    )

    with pytest.raises(ValueError, match="require donor_tissue or tissue"):
        associate_features_with_eqtl_counts(
            merged,
            merge_mode="tissue",
            schema=ObsSchema(),
        )


def test_eqtl_gene_total_association_uses_genes_as_units() -> None:
    """Wide-table gene totals compare genes without repeated-donor support."""
    unit_frame = pd.DataFrame(
        {
            "feature_type": ["gene_expression"] * 6,
            "feature_id": [
                "ENSG1",
                "ENSG1",
                "ENSG2",
                "ENSG2",
                "ENSG3",
                "ENSG3",
            ],
            "feature_label": [
                "CGAS",
                "CGAS",
                "IFIH1",
                "IFIH1",
                "DDX58",
                "DDX58",
            ],
            "donor_id": ["D1", "D2"] * 3,
            "feature_value": [1.0, 1.2, 2.0, 2.2, 3.0, 3.2],
            "statistical_unit": ["donor"] * 6,
            "aggregation": ["mean"] * 6,
        }
    )
    eqtl_table = pd.DataFrame(
        {
            "gene": ["CGAS", "IFIH1", "DDX58"],
            "gene_total_eqtls": [10, 20, 30],
            "LIVER": [4, 8, 12],
            "LUNG": [6, 12, 18],
        }
    )
    merged = merge_eqtl_counts(
        unit_frame,
        eqtl_table,
        merge_mode="gene",
        schema=ObsSchema(),
    )

    result = associate_features_with_eqtl_counts(
        merged,
        merge_mode="gene",
        schema=ObsSchema(),
    )

    assert result.iloc[0]["statistical_unit"] == "gene"
    assert result.iloc[0]["n"] == 3
    assert result.iloc[0]["pearson_r"] > 0.99


def test_validate_eqtl_table_reports_missing_columns() -> None:
    """Validation raises when a merge mode's key column is absent."""
    bad_table = pd.DataFrame({"total_eqtls": [1]})

    with pytest.raises(KeyError, match="gene_symbol"):
        validate_eqtl_table(bad_table, merge_mode="gene")


def test_orchestration_end_to_end(tmp_path: Path) -> None:
    """The workflow emits mixed-model inference and retained NASP summaries."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    h5ad_path, score_path = _write_inputs(tmp_path, adata, scores)
    output_dir = tmp_path / "assoc"
    stale_tables = output_dir / "association_tables"
    stale_regressions = output_dir / "association_plots" / "regressions"
    stale_boxplots = output_dir / "association_plots" / "boxplots"
    stale_mixed = output_dir / "association_plots" / "mixed_models"
    stale_nasp = output_dir / "association_plots" / "nasp"
    for directory in (
        stale_tables,
        stale_regressions,
        stale_boxplots,
        stale_mixed,
        stale_nasp,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (stale_tables / "association_age_stability.csv").write_text("stale\n")
    (stale_regressions / "old_regression.png").write_bytes(b"stale")
    (stale_boxplots / "old_group.png").write_bytes(b"stale")
    (stale_mixed / "nasp_mixed_condition_effects_by_cell_type.png").write_bytes(
        b"stale"
    )
    (stale_nasp / "nasp_age_effects_by_cell_type.png").write_bytes(b"stale")

    association_analysis(
        h5ad_path=h5ad_path,
        score_csv_path=score_path,
        output_dir=output_dir,
        sensor_group="nucleic_acid_sensors",
        statistical_unit="donor_tissue",
        aggregation="mean",
        mixed_model_min_cells=5,
    )

    tables = output_dir / "association_tables"
    mixed_table_names = {
        "association_mixed_model_contrasts.csv",
        "association_mixed_model_fixed_effects.csv",
        "association_mixed_model_term_tests.csv",
        "association_mixed_model_variance_components.csv",
        "association_mixed_model_diagnostics.csv",
        "association_mixed_model_availability.csv",
    }
    written_table_names = {path.name for path in tables.glob("*.csv")}
    assert mixed_table_names <= written_table_names
    provenance = pd.read_csv(tables / "association_provenance.csv").iloc[0]
    assert Path(provenance["h5ad_path"]) == h5ad_path.resolve()
    assert Path(provenance["score_csv_path"]) == score_path.resolve()
    assert provenance["scorer"] == "scanpy"
    assert provenance["expression_source"] == "X"
    assert provenance["statistical_unit"] == "donor_tissue"
    assert provenance["aggregation"] == "mean"
    assert json.loads(provenance["requested_module_ids"]) == []
    assert set(json.loads(provenance["resolved_module_labels"])) == {
        "IFN_I_OUTPUT",
        "NASP_DNA_SENSING",
        "NASP_RNA_SENSING",
    }
    assert set(json.loads(provenance["resolved_gene_labels"])) == set(
        SENSOR_GENES
    )
    assert len(provenance["compendium_marker_panel_sha256"]) == 64

    contrasts = pd.read_csv(tables / "association_mixed_model_contrasts.csv")
    fixed_effects = pd.read_csv(
        tables / "association_mixed_model_fixed_effects.csv"
    )
    term_tests = pd.read_csv(tables / "association_mixed_model_term_tests.csv")
    variance = pd.read_csv(
        tables / "association_mixed_model_variance_components.csv"
    )
    diagnostics = pd.read_csv(
        tables / "association_mixed_model_diagnostics.csv"
    )
    availability = pd.read_csv(
        tables / "association_mixed_model_availability.csv"
    ).set_index("analysis")

    model_unit = "donor_id x tissue_in_publication x cell_type x assay"
    provenance_columns = {
        "age_center_years",
        "aggregation",
        "detection_threshold",
        "configured_condition_reference",
        "minimum_cells_per_observation",
        "minimum_donors",
        "minimum_studies",
        "minimum_repeated_contexts",
        "observational_unit",
        "independent_unit",
    }
    for result in (
        contrasts,
        fixed_effects,
        term_tests,
        variance,
        diagnostics,
    ):
        assert provenance_columns <= set(result.columns)
        assert set(result["aggregation"]) == {"mean"}
        assert set(result["minimum_cells_per_observation"]) == {5}
        assert set(result["minimum_donors"]) == {3}
        assert set(result["minimum_studies"]) == {3}
        assert set(result["minimum_repeated_contexts"]) == {3}
        assert set(result["detection_threshold"]) == {0.0}
        assert set(result["configured_condition_reference"]) == {"normal"}
        assert set(result["observational_unit"]) == {model_unit}
        assert set(result["independent_unit"]) == {"donor"}
        assert set(result["age_center_years"]) == {55.0}

    assert set(contrasts["estimand"]) == {
        "adjusted_cell_type",
        "age_by_cell_type",
        "assay_batch_effects",
        "paired_tissue",
    }
    estimable_contrasts = contrasts.loc[contrasts["estimable"]]
    assert set(estimable_contrasts["estimand"]) == {
        "age_by_cell_type",
        "paired_tissue",
    }
    paired_tissue = contrasts.loc[contrasts["estimand"] == "paired_tissue"]
    assert (paired_tissue["n_paired_units"] == 6).all()
    assert set(variance["component"]) == {"donor", "residual"}
    unavailable_variance = variance.loc[~variance["estimable"]]
    assert not unavailable_variance.empty
    assert set(unavailable_variance["status"]) == {"invalid_covariance"}
    assert set(unavailable_variance["reason"]) == {
        "hessian_not_positive_definite"
    }

    assert set(diagnostics["analysis"]) == {
        "adjusted_context",
        "age_by_cell_type",
        "paired_tissue",
    }
    assert diagnostics["converged"].all()
    assert set(diagnostics["status"]) == {"ok", "invalid_covariance"}
    assert set(
        diagnostics.loc[
            diagnostics["status"].eq("invalid_covariance"), "reason"
        ]
    ) == {"hessian_not_positive_definite"}
    assert set(diagnostics["n_input_observations"]) == {12.0}
    assert set(diagnostics["n_observations"]) == {12.0}
    assert set(diagnostics["n_dropped_missing"]) == {0.0}
    assert set(diagnostics["n_dropped_nonfinite"]) == {0.0}
    assert set(diagnostics["n_groups"]) == {6.0}
    assert set(diagnostics["n_independent_units"]) == {6.0}
    assert set(diagnostics["optimizer_methods"]) == {"lbfgs|powell"}
    assert diagnostics["statsmodels_version"].notna().all()
    assert diagnostics["statsmodels_version"].astype(str).str.len().gt(0).all()

    expected_availability = {
        "adjusted_cell_type",
        "condition_by_cell_type",
        "age_by_cell_type",
        "paired_tissue",
        "assay_batch_effects",
        "variance_decomposition",
        "multi_study_structure",
    }
    assert set(availability.index) == expected_availability
    assert set(availability["minimum_cells"]) == {5}
    assert set(availability["minimum_donors"]) == {3}
    assert set(availability["minimum_studies"]) == {3}
    assert set(availability["minimum_repeated_contexts"]) == {3}
    assert set(availability["detection_threshold"]) == {0.0}
    assert set(availability["configured_condition_reference"]) == {"normal"}
    assert set(availability["observational_unit"]) == {model_unit}
    assert set(availability["independent_unit"]) == {"donor"}
    assert set(availability.loc[availability["estimable"], "status"]) == {
        "partially_estimable"
    }
    assert set(availability.index[availability["estimable"]]) == {
        "age_by_cell_type",
        "paired_tissue",
        "variance_decomposition",
    }
    assert (
        availability.loc["adjusted_cell_type", "reason"]
        == "single_predictor_level:cell_type"
    )
    assert (
        availability.loc["condition_by_cell_type", "reason"]
        == "missing_predictor:disease"
    )
    assert (
        availability.loc["assay_batch_effects", "reason"]
        == "single_predictor_level:assay"
    )
    assert (
        availability.loc["multi_study_structure", "reason"]
        == "missing_study_column:dataset_id"
    )

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
    context_summary = pd.read_csv(tables / "nasp_context_summary.csv")
    assert {
        "tissue_in_publication",
        "cell_type",
        "relative_competence",
        "n_donors",
    } <= set(context_summary.columns)
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

    deprecated_tables = {
        "association_age_stability.csv",
        "association_partial_correlation_age.csv",
        "association_group_test_results.csv",
        "association_group_summary.csv",
    }
    assert deprecated_tables.isdisjoint(written_table_names)

    plot_root = tables.parent / "association_plots"
    mixed_plots = plot_root / "mixed_models"
    expected_mixed_plots = {
        "nasp_mixed_age_slopes_by_cell_type.png",
        "nasp_mixed_paired_tissue_effects.png",
        "nasp_mixed_variance_decomposition.png",
    }
    written_mixed_plots = {
        path.name: path for path in mixed_plots.glob("*.png")
    }
    assert set(written_mixed_plots) == expected_mixed_plots
    assert all(path.stat().st_size > 0 for path in written_mixed_plots.values())

    expected_nasp_plots = {
        "nasp_module_coupling_heatmap.png",
        "nasp_competence_output_state_map.png",
        "nasp_ranked_hypotheses.png",
        "nasp_sensor_output_mismatch.png",
        "nasp_mechanistic_edge_network.png",
    }
    plot_manifest = pd.read_csv(tables / "association_plot_manifest.csv")
    assert set(plot_manifest["kind"]) == {
        "mixed_model_inference",
        "nasp_summary",
    }
    mixed_manifest = plot_manifest.loc[
        plot_manifest["kind"].eq("mixed_model_inference")
    ]
    nasp_manifest = plot_manifest.loc[plot_manifest["kind"].eq("nasp_summary")]
    assert set(mixed_manifest["analysis_scope"]) == {
        Path(filename).stem for filename in expected_mixed_plots
    }
    assert set(nasp_manifest["analysis_scope"]) == {
        Path(filename).stem for filename in expected_nasp_plots
    }
    assert set(mixed_manifest["statistical_unit"]) == {model_unit}
    assert set(nasp_manifest["statistical_unit"]) == {"donor_tissue_cell_type"}
    assert set(plot_manifest["aggregation"]) == {"mean"}
    assert plot_manifest["predictor"].isna().all()
    manifest_paths = [Path(path) for path in plot_manifest["path"]]
    assert {path.name for path in manifest_paths} == {
        *expected_mixed_plots,
        *expected_nasp_plots,
    }
    assert all(path.is_file() for path in manifest_paths)

    nasp_plots = tables.parent / "association_plots" / "nasp"
    written_nasp_plots = {path.name: path for path in nasp_plots.glob("*.png")}
    assert set(written_nasp_plots) == expected_nasp_plots
    assert all(path.stat().st_size > 0 for path in written_nasp_plots.values())

    deprecated_plot_names = {
        "nasp_age_effects_by_cell_type.png",
        "nasp_sensor_age_effects_by_cell_type.png",
        "nasp_age_effect_consistency_across_cell_types.png",
        "nasp_sensor_age_effect_consistency_across_cell_types.png",
    }
    all_plot_names = {path.name for path in plot_root.rglob("*.png")}
    assert deprecated_plot_names.isdisjoint(all_plot_names)
    assert not any((plot_root / "regressions").glob("*.png"))
    assert not any((plot_root / "boxplots").glob("*.png"))
    assert not any((plot_root / "barplots").glob("*.png"))


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


def test_orchestration_runs_gene_tissue_eqtl_associations(
    tmp_path: Path,
) -> None:
    """Workflow tests tissue eQTL burden without donor inflation."""
    adata = _synthetic_adata()
    scores = _synthetic_scores(adata)
    h5ad_path, score_path = _write_inputs(tmp_path, adata, scores)
    eqtl_path = tmp_path / "eqtl_long.csv"
    pd.DataFrame(
        {
            "gene": [
                "CGAS",
                "CGAS",
                "IFIH1",
                "IFIH1",
                "DDX58",
                "DDX58",
            ],
            "tissue": ["Liver", "Lung"] * 3,
            "n_significant_eqtls": [1, 2, 3, 4, 5, 6],
            "gene_total_eqtls": [3, 3, 7, 7, 11, 11],
        }
    ).to_csv(eqtl_path, index=False)
    output_dir = tmp_path / "assoc"

    association_analysis(
        h5ad_path=h5ad_path,
        score_csv_path=score_path,
        output_dir=output_dir,
        sensor_group="nucleic_acid_sensors",
        eqtl_table_path=eqtl_path,
        eqtl_merge_mode="gene_tissue",
        max_plots=0,
        plot_nasp_visualizations=False,
    )

    tables = output_dir / "association_tables"
    regression = pd.read_csv(tables / "association_regression_results.csv")
    eqtl_results = regression.loc[
        regression["analysis_scope"]
        == "eqtl_tissue_count_within_tissue_across_genes"
    ]
    assert set(eqtl_results["stratum"]) == {"liver", "lung"}
    assert (eqtl_results["statistical_unit"] == "gene").all()
    assert (eqtl_results["n"] == 3).all()
    assert (eqtl_results["eqtl_predictor_transform"] == "none").all()
    annotations = pd.read_csv(tables / "association_eqtl_annotations.csv")
    matched_genes = annotations["feature_type"].eq("gene_expression")
    assert annotations.loc[matched_genes, "eqtl_matched"].all()


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


def test_orchestration_rejects_nonfinite_detection_threshold_before_io(
    tmp_path: Path,
) -> None:
    """Invalid thresholds fail before any large association input is read."""
    with pytest.raises(ValueError, match="finite real number"):
        association_analysis(
            h5ad_path=tmp_path / "missing.h5ad",
            score_csv_path=tmp_path / "missing.csv",
            output_dir=tmp_path / "assoc",
            detection_threshold=np.inf,
        )


def test_empty_workflow_outputs_keep_mixed_and_manifest_headers(
    tmp_path: Path,
) -> None:
    """Unavailable workflow stages still write readable public CSV schemas."""
    workflows._write_empty_association_tables(tmp_path)
    expected = workflows.TabulaMixedModelResults.empty()
    table_names = {
        "association_mixed_model_contrasts.csv": expected.contrasts,
        "association_mixed_model_fixed_effects.csv": expected.fixed_effects,
        "association_mixed_model_term_tests.csv": expected.term_tests,
        "association_mixed_model_variance_components.csv": (
            expected.variance_components
        ),
        "association_mixed_model_diagnostics.csv": expected.diagnostics,
        "association_mixed_model_availability.csv": expected.availability,
    }
    for filename, schema in table_names.items():
        observed = pd.read_csv(tmp_path / filename)
        assert list(observed.columns) == list(schema.columns)
        assert observed.empty

    regression = pd.read_csv(tmp_path / "association_regression_results.csv")
    annotations = pd.read_csv(tmp_path / "association_eqtl_annotations.csv")
    assert regression.columns.tolist() == [
        "feature_type",
        "feature_id",
        "feature_label",
        "predictor",
        "statistical_unit",
        "aggregation",
        "stratify_key",
        "stratum",
        "analysis_role",
        "n_units",
        "fdr_method",
        "n",
        "pearson_r",
        "pearson_pvalue",
        "spearman_r",
        "spearman_pvalue",
        "slope",
        "intercept",
        "ols_pvalue",
        "skipped",
        "skip_reason",
        "pearson_pvalue_fdr",
        "spearman_pvalue_fdr",
        "ols_pvalue_fdr",
        "analysis_scope",
        "response_estimand",
        "eqtl_fdr_family",
        "eqtl_source_schema",
        "eqtl_count_unit",
        "eqtl_predictor_transform",
        "eqtl_table_path",
    ]
    assert annotations.columns.tolist() == [
        "feature_type",
        "feature_id",
        "feature_label",
        "feature_value",
        "statistical_unit",
        "aggregation",
        "unit_id",
        "n_cells",
        "n_cells_total",
        "eqtl_matched",
        "eqtl_merge_mode",
        "eqtl_source_schema",
        "eqtl_count_unit",
        "eqtl_predictor_transform",
        "eqtl_table_path",
    ]
    assert regression.empty
    assert annotations.empty

    manifest = pd.read_csv(tmp_path / "association_plot_manifest.csv")
    assert manifest.columns.tolist() == [
        "kind",
        "feature_id",
        "predictor",
        "statistical_unit",
        "aggregation",
        "stratum",
        "analysis_scope",
        "path",
    ]
    assert manifest.empty


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
