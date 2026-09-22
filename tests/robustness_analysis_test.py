"""Scientific regression checks for donor and gene robustness analyses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from nasp_compendium import GeneModules
from nasp_compendium.types import GeneModule

from nasp_atlas.analysis import association_analysis
from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.gene_effects import (
    estimate_gene_removal_effects,
)
from nasp_atlas.analysis.external_replication.scoring import (
    aggregate_module_scores,
)
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.analysis.tabula_sapiens.scoring import score_table_provenance
from nasp_atlas.single_cell import ObsSchema
from nasp_atlas.single_cell import donor_sensitivity
from nasp_atlas.single_cell import gene_removal_sensitivity
from nasp_atlas.single_cell import module_gene_diagnostics
from nasp_atlas.single_cell import score_aucell_modules
from nasp_atlas.single_cell import score_scanpy_module
from nasp_atlas.single_cell import score_scanpy_modules
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import inverse_module_score_name
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import positive_module_score_name
from nasp_atlas.single_cell.visualization.robustness import RobustnessPlotter


def test_donor_deletion_reveals_a_rank_reversal(tmp_path: Path) -> None:
    """Removing a donor changes ranks across all their contexts together."""
    records = []
    for donor, muscle_score in zip(
        ["d1", "d2", "d3"], [1, 10, 11], strict=True
    ):
        for tissue, score in [("Muscle", muscle_score), ("Lung", 8)]:
            records.append(
                {
                    "obs_name": f"{donor}_{tissue}",
                    "donor_id": donor,
                    "tissue_in_publication": tissue,
                    "cell_type": "endothelial",
                    "feature_type": "module_score",
                    "feature_id": "M_score",
                    "feature_label": "M",
                    "feature_value": score,
                }
            )
    frame = pd.DataFrame(records)

    result = donor_sensitivity(
        frame, schema=ObsSchema(), minimum_cells=1, minimum_donors=2
    )

    deletion = result.leave_one_donor_out.query(
        "omitted_donor_id == 'd2' and tissue_in_publication == 'Muscle' "
        "and ranking_scope == 'matched_cell_type'"
    ).iloc[0]
    assert deletion.context_rank_baseline == 1
    assert deletion.context_rank_without_donor == 2
    assert deletion.n_finite_donors == 2
    assert result.paired_tissues.n_paired_donors.tolist() == [3]
    assert result.paired_tissues.mean_difference.iloc[0] == pytest.approx(2 / 3)
    paths = RobustnessPlotter(tmp_path).plot_tables(
        {"nasp_leave_one_donor_out": result.leave_one_donor_out}
    )
    displayed = pd.read_csv(
        next(path for path in paths if path.suffix == ".csv")
    )
    muscle = displayed.loc[displayed.tissue_in_publication.eq("Muscle")].iloc[0]
    assert muscle.point == 1
    assert muscle.low == 1 and muscle.high == 2


def test_cell_replication_does_not_increase_donor_support() -> None:
    """Duplicating cell measurements preserves donor ranks and support."""
    frame = pd.DataFrame(
        {
            "obs_name": ["a", "b", "c"],
            "donor_id": ["d1", "d2", "d3"],
            "tissue_in_publication": "Muscle",
            "cell_type": "endothelial",
            "feature_type": "module_score",
            "feature_id": "M_score",
            "feature_label": "M",
            "feature_value": [1.0, 2.0, 3.0],
        }
    )
    duplicated = pd.concat([frame, frame.assign(obs_name=["d", "e", "f"])])

    result = donor_sensitivity(
        duplicated, schema=ObsSchema(), minimum_cells=1, minimum_donors=3
    )

    assert result.context_ranks.n_donors.eq(3).all()
    assert result.context_ranks["median"].eq(2).all()
    assert result.leave_one_donor_out.status.eq("insufficient_donors").all()
    assert result.leave_one_donor_out.context_rank_without_donor.isna().all()


def test_gene_diagnostics_distinguish_absent_undetected_and_assay_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One-gene reads preserve missingness and assay differences."""
    obs = pd.DataFrame(
        [
            {
                "donor_id": f"d{donor}",
                "tissue_in_publication": "Muscle",
                "cell_type": "endothelial",
                "assay": assay,
            }
            for donor in range(3)
            for assay in ["cells", "nuclei"]
        ],
        index=[f"cell_{index}" for index in range(6)],
    )
    values = np.array(
        [[1, 0], [2, 0], [2, 0], [4, 0], [3, 0], [6, 0]], dtype=float
    )
    adata = ad.AnnData(
        sp.csr_matrix(values),
        obs=obs,
        var=pd.DataFrame({"feature_name": ["G", "ZERO"]}, index=["g", "zero"]),
    )
    module = GeneModule("M", ("G", "ZERO", "ABSENT"), (), (), "symbols")
    scores = pd.DataFrame({"M_score": values[:, 0]}, index=obs.index)
    original_toarray = sp.csr_matrix.toarray

    def bounded_toarray(
        matrix: sp.csr_matrix,
        order: str | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reject accidental full expression matrix materialization."""
        assert matrix.shape != adata.shape
        return original_toarray(matrix, order=order, out=out)

    monkeypatch.setattr(sp.csr_matrix, "toarray", bounded_toarray)

    result = module_gene_diagnostics(
        adata,
        scores,
        [module],
        schema=ObsSchema(),
        minimum_cells=1,
        minimum_donors=3,
    )

    donor = result.donor_expression
    assert donor.loc[donor.gene.eq("ABSENT"), "mean_expression"].isna().all()
    assert donor.loc[donor.gene.eq("ABSENT"), "status"].eq("absent").all()
    assert donor.loc[donor.gene.eq("ZERO"), "status"].eq("undetected").all()
    contexts = result.context_summary.query("gene == 'G'").set_index("assay")
    assert contexts.loc["nuclei", "median_mean_expression"] == 4
    assert contexts.loc["cells", "median_mean_expression"] == 2
    assert contexts.gene_score_spearman.eq(1).all()
    assert contexts.n_donors.eq(3).all()
    assert (
        RobustnessPlotter(tmp_path).plot_tables(
            {"nasp_gene_context_diagnostics": result.context_summary},
            minimum_donors=4,
        )
        == []
    )


@pytest.mark.parametrize("scorer", ["scanpy", "aucell"])
@pytest.mark.parametrize("signed", [False, True])
def test_gene_removal_matches_independent_rescoring(
    scorer: ScorerName, signed: bool
) -> None:
    """Derived signatures match explicit rescoring without input mutation."""
    rng = np.random.default_rng(19)
    values = rng.uniform(0, 4, size=(32, 120))
    values[:, 0] = rng.uniform(8, 50, size=32)
    values[:, 1:3] = rng.uniform(0, 14, size=(32, 2))
    genes = ["SHARED", "LEFT", "RIGHT", *[f"bg{i}" for i in range(117)]]
    obs = pd.DataFrame(
        {
            "donor_id": [f"d{i // 4}" for i in range(32)],
            "tissue_in_publication": ["Muscle", "Lung"] * 16,
            "cell_type": "endothelial",
            "assay": "cells",
        },
        index=[f"cell_{i}" for i in range(32)],
    )
    adata = ad.AnnData(
        sp.csr_matrix(values),
        obs=obs,
        var=pd.DataFrame({"feature_name": genes}, index=genes),
    )
    modules = [
        GeneModule(
            "A", ("SHARED", "LEFT"), ("bg0",) if signed else (), (), "symbols"
        ),
        GeneModule(
            "B", ("SHARED", "RIGHT"), ("bg1",) if signed else (), (), "symbols"
        ),
    ]
    original_obs = adata.obs.copy()
    baseline = adata.copy()
    if scorer == "scanpy":
        for module in modules:
            score_scanpy_module(baseline, module, expression_layer=None)
        scores = baseline.obs[["A_score", "B_score"]]
    else:
        _, scores, _ = score_aucell_modules(
            baseline,
            ["A", "B"],
            gene_modules=modules,
            expression_layer=None,
            chunk_size=8,
        )
    diagnostics = module_gene_diagnostics(
        adata,
        scores,
        modules,
        schema=ObsSchema(),
        scorer=scorer,
        minimum_cells=1,
    )

    result = gene_removal_sensitivity(
        adata,
        scores,
        modules,
        diagnostics,
        schema=ObsSchema(),
        scorer=scorer,
        module_pairs=[("A", "B")],
        dominant_per_arm=signed,
        minimum_cells=1,
        aucell_chunk_size=8,
    )

    selected = result.variants.query(
        "module_id == 'A' and mode == 'shared_genes'"
    ).iloc[0]
    assert selected.removed_genes == "SHARED"
    if signed:
        drivers = result.variants.query("mode == 'dominant_gene'")
        assert set(drivers.selection_arm) == {"positive", "inverse"}
        inverse = drivers.loc[drivers.selection_arm.eq("inverse")]
        assert inverse.status.eq("unscorable_arm").all()
        assert set(inverse.removed_genes) == {"bg0", "bg1"}
    expected_module = replace(modules[0], positive_genes=("LEFT",))
    expected = adata.copy()
    if scorer == "scanpy":
        column = score_scanpy_module(
            expected, expected_module, expression_layer=None
        )
        expected_values = expected.obs[column]
    else:
        _, expected_scores, _ = score_aucell_modules(
            expected,
            ["A"],
            gene_modules=[expected_module],
            expression_layer=None,
            chunk_size=8,
        )
        expected_values = expected_scores["A_auc"]
    keys = ["donor_id", "tissue_in_publication", "cell_type"]
    expected_means = (
        obs.assign(value=expected_values).groupby(keys).value.mean()
    )
    actual = (
        result.donor_scores.loc[
            result.donor_scores.variant_id.eq(selected.variant_id)
        ]
        .set_index(keys)
        .feature_value
    )
    np.testing.assert_allclose(actual.sort_index(), expected_means.sort_index())
    expected_components = (
        expected.obs if scorer == "scanpy" else expected_scores
    )
    variant_module = replace(expected_module, module_id=selected.variant_id)
    for score_name in (
        module_score_name,
        positive_module_score_name,
        inverse_module_score_name,
    ):
        expected_column = score_name(expected_module, scorer=scorer)
        variant_column = score_name(variant_module, scorer=scorer)
        if expected_column is not None:
            np.testing.assert_allclose(
                result.cell_scores[variant_column],
                expected_components[expected_column].reindex(obs.index),
            )
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    np.testing.assert_allclose(adata.X.toarray(), values)
    assert result.overlap_coupling.n_donors.eq(8).all()

    # Use the actual returned means in the external paired estimator; compare
    # with independently rescored cell means, including the donor intersection.
    external = adata.copy()
    names = {"tissue_in_publication": "tissue", "cell_type": "population"}
    external.obs = obs.rename(columns=names).assign(
        cohort_id="fixture", modality="cells"
    )
    external_result = replace(
        result,
        donor_scores=result.donor_scores.rename(columns=names).assign(
            cohort_id="fixture"
        ),
    )
    comparison = ComparisonSpec(
        level_key="tissue",
        target="Muscle",
        reference="Lung",
        minimum_cells=1,
        minimum_pairs=6,
    )
    baseline_donors = aggregate_module_scores(
        external.obs,
        scores,
        modules,
        scorer=scorer,
        scoring_context_id="baseline",
        minimum_cells=1,
    )
    baseline_effect = estimate_module_contrasts(
        baseline_donors,
        pd.DataFrame(
            {"module_id": ["A", "B"], "scorer": scorer, "status": "ok"}
        ),
        comparison,
    )
    effect, arms, _ = estimate_gene_removal_effects(
        external_result,
        external,
        modules,
        comparison,
        baseline_effect.donor_differences,
        scorer=scorer,
        scoring_context_id="removal",
    )
    measured = effect.estimates.loc[
        effect.estimates.variant_id.eq(selected.variant_id)
    ].iloc[0]
    means = expected_means.unstack("tissue_in_publication")
    expected_difference = means.Muscle - means.Lung
    assert measured.estimate == pytest.approx(expected_difference.mean())
    assert measured.standard_error == pytest.approx(
        expected_difference.std(ddof=1) / np.sqrt(8)
    )
    assert measured.n_matched_donors == 8
    assert measured.variant_matched_estimate == pytest.approx(measured.estimate)
    assert measured.baseline_matched_estimate == pytest.approx(
        baseline_effect.estimates.loc[
            baseline_effect.estimates.module_id.eq("A"), "estimate"
        ].item()
    )
    assert arms.status.eq("ok").all()
    if signed:
        failed = effect.estimates.loc[
            effect.estimates.selection_arm.eq("inverse")
        ]
        assert failed.eligibility_reason.eq("unscorable_arm").all()
        assert failed.pvalue.isna().all()
        # A variable final score cannot rescue a constant retained arm.
        component = positive_module_score_name(variant_module, scorer=scorer)
        degenerate = replace(
            external_result,
            cell_scores=external_result.cell_scores.assign(**{component: 0.0}),
        )
        invalid, _, _ = estimate_gene_removal_effects(
            degenerate,
            external,
            modules,
            comparison,
            baseline_effect.donor_differences,
            scorer=scorer,
            scoring_context_id="constant-arm",
        )
        invalid_effect = invalid.estimates.loc[
            invalid.estimates.variant_id.eq(selected.variant_id)
        ].iloc[0]
        assert invalid_effect.estimate == pytest.approx(measured.estimate)
        assert invalid_effect.eligibility_reason == "degenerate_arm"
        assert np.isnan(invalid_effect.pvalue)


def test_removing_the_only_signed_gene_is_unscorable() -> None:
    """Deletion cannot silently turn a lost score arm into a zero score."""
    obs = pd.DataFrame(
        {
            "donor_id": ["a", "b", "c"],
            "tissue_in_publication": "Muscle",
            "cell_type": "E",
        },
        index=["a", "b", "c"],
    )
    adata = ad.AnnData(
        np.array([[1.0], [2.0], [3.0]]),
        obs=obs,
        var=pd.DataFrame({"feature_name": ["G"]}, index=["G"]),
    )
    module = GeneModule("M", ("G",), (), (), "symbols")
    scores = pd.DataFrame({"M_score": [1.0, 2.0, 3.0]}, index=obs.index)
    diagnostics = module_gene_diagnostics(
        adata, scores, [module], schema=ObsSchema(), minimum_cells=1
    )

    result = gene_removal_sensitivity(
        adata,
        scores,
        [module],
        diagnostics,
        schema=ObsSchema(),
        minimum_cells=1,
    )

    assert result.variants.status.tolist() == ["unscorable_arm"]
    assert result.donor_scores.empty


def test_all_analysis_stages_share_the_saved_scoring_population(
    tmp_path: Path,
) -> None:
    """Public orchestration publishes both scopes and all robustness stages."""
    module_ids = [
        "NASP_DNA_SENSING",
        "IFN_I_OUTPUT",
        "NASP_FEEDBACK",
        "NASP_RESTRICTION",
        "NASP_RNA_SENSING",
        "SIGNALING_CONTEXT_IFN_JAK_STAT",
    ]
    modules = [GeneModules.modules(module_id) for module_id in module_ids]
    genes = sorted(
        {
            gene
            for module in modules
            for gene in module.positive_genes + module.inverse_genes
        }
    ) + [f"background_{index}" for index in range(200)]
    rng = np.random.default_rng(29)
    obs = pd.DataFrame(
        [
            {
                "donor_id": f"donor_{donor}",
                "tissue_in_publication": tissue,
                "cell_type": cell_type,
                "assay": "cells",
            }
            for donor in range(6)
            for tissue in ["Muscle", "Lung"]
            for cell_type in ["E", "M"]
            for _ in range(2)
        ],
        index=pd.Index([f"cell_{index}" for index in range(48)], dtype=object),
        dtype=object,
    )
    adata = ad.AnnData(
        sp.csr_matrix(rng.uniform(0, 5, size=(len(obs), len(genes)))),
        obs=obs,
        var=pd.DataFrame(
            {"feature_name": genes},
            index=pd.Index(genes, dtype=object),
            dtype=object,
        ),
    )
    input_path = tmp_path / "input.h5ad"
    adata.write_h5ad(input_path)
    score_scanpy_modules(adata, module_ids, expression_layer=None)
    score_columns = [f"{module_id}_score" for module_id in module_ids]
    scores = adata.obs[score_columns].copy()
    provenance = score_table_provenance(
        n_obs=adata.n_obs,
        scorers=["scanpy"],
        subset_fraction=None,
        random_state=42,
        module_ids=module_ids,
    )
    provenance.index = scores.index
    scores = scores.join(provenance)
    score_path = tmp_path / "scores.csv.gz"
    scores.to_csv(score_path, index_label="obs_name")
    output_dir = tmp_path / "analysis"

    association_analysis(
        h5ad_path=input_path,
        score_csv_path=score_path,
        output_dir=output_dir,
        module_ids=module_ids,
        sensor_group=None,
        mixed_models_combined=True,
        mixed_models_per_tissue=True,
        run_donor_sensitivity=True,
        run_gene_diagnostics=True,
        run_gene_removal_sensitivity=True,
        run_mechanism_diagnostics=True,
        mixed_model_min_cells=1,
        plot_nasp_visualizations=True,
    )

    scopes = pd.read_csv(
        output_dir / "association_tables" / "association_mixed_model_scopes.csv"
    )
    assert not scopes.empty and scopes.status.eq("completed").all()
    combined = scopes.loc[scopes.scope.eq("combined_input")]
    tissue = scopes.loc[scopes.scope.eq("per_tissue")]
    assert combined.n_cells.tolist() == [48]
    assert tissue.n_cells.sum() == 48
    robustness_dir = output_dir / "robustness"
    stages = pd.read_csv(robustness_dir / "robustness_manifest.csv")
    assert not stages.empty and stages.status.eq("completed").all()
    donors = pd.read_csv(robustness_dir / "nasp_donor_scores.csv")
    assert donors.donor_id.nunique() == 6
    assert donors.n_cells.sum() == 48 * len(module_ids)
    variants = pd.read_csv(robustness_dir / "nasp_gene_removal_variants.csv")
    assert variants.status.eq("ok").any()
    changes = pd.read_csv(
        robustness_dir / "nasp_gene_removal_context_changes.csv"
    )
    assert changes.n_donors_removed.eq(6).all()
    assert changes.status.eq("ok").all()
    assert stages.visualizations.eq("completed").all()
    artifacts = [
        Path(path)
        for stage_artifacts in stages.artifacts
        for path in stage_artifacts.split(";")
    ]
    figures = [path for path in artifacts if path.suffix in {".png", ".pdf"}]
    assert figures and all(path.stat().st_size > 0 for path in figures)
    coupling = pd.read_csv(
        output_dir / "mechanisms" / "regulator_output_coupling.csv"
    )
    assert coupling.status.eq("ok").any()
    assert coupling.loc[coupling.status.eq("ok"), "n_donors"].eq(6).all()
    assert coupling.loc[coupling.gene_in_output, "spearman_r"].isna().all()
    components = pd.read_csv(
        output_dir / "mechanisms" / "mechanism_components.csv"
    )
    assert components.n_donors.eq(6).all()
    assert components.median_mean_expression.gt(0).all()
