"""Real source-to-score checkpoint and donor inference round trip."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from nasp_compendium import GeneModules

from nasp_atlas.analysis.external_replication import CohortSpec
from nasp_atlas.analysis.external_replication import ComparisonSpec
from nasp_atlas.analysis.external_replication import PopulationSpec
from nasp_atlas.analysis.external_replication import read_cohort_metadata
from nasp_atlas.analysis.external_replication import select_cohort
from nasp_atlas.analysis.external_replication.balanced_scoring import (
    rescore_balanced_comparison,
)
from nasp_atlas.analysis.external_replication.balanced_scoring import (
    validate_balanced_artifacts,
)
from nasp_atlas.analysis.external_replication.chemistry_workflows import (
    rescore_pooled_chemistry,
)
from nasp_atlas.analysis.external_replication.chemistry_workflows import (
    validate_chemistry_artifacts,
)
from nasp_atlas.analysis.external_replication.contrast_drivers import (
    rescore_without_contrast_drivers,
)
from nasp_atlas.analysis.external_replication.contrast_drivers import (
    validate_contrast_driver_removal,
)
from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.extension_workflows import (
    analyze_wells_extension,
)
from nasp_atlas.analysis.external_replication.extension_workflows import (
    validate_extension_artifacts,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    align_cohort_features,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    panel_aliases,
)
from nasp_atlas.analysis.external_replication.final_reports import (
    publish_final_report,
)
from nasp_atlas.analysis.external_replication.final_reports import (
    validate_final_report,
)
from nasp_atlas.analysis.external_replication.gene_workflows import (
    analyze_gene_sensitivity,
)
from nasp_atlas.analysis.external_replication.gene_workflows import (
    validate_gene_sensitivity,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registered_inputs import (
    RegisteredPrimary,
)
from nasp_atlas.analysis.external_replication.registration import (
    freeze_registration,
)
from nasp_atlas.analysis.external_replication.reports import (
    publish_primary_report,
)
from nasp_atlas.analysis.external_replication.reports import (
    validate_primary_report,
)
from nasp_atlas.analysis.external_replication.robustness_reports import (
    publish_robustness_report,
)
from nasp_atlas.analysis.external_replication.robustness_reports import (
    validate_robustness_report,
)
from nasp_atlas.analysis.external_replication.scoring import (
    aggregate_module_scores,
)
from nasp_atlas.analysis.external_replication.sensitivity_workflows import (
    analyze_fixed_score_sensitivity,
)
from nasp_atlas.analysis.external_replication.workflows import (
    analyze_primary_cohort,
)
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)


def test_frozen_panel_scores_resume_and_reject_changed_inputs(tmp_path) -> None:
    """Saved scores preserve pairing and reject changed measurements."""
    rng = np.random.default_rng(17)
    counts = rng.poisson(3, (52, 400))
    counts[:12, 0] += 20
    counts[12:24, 1] += 20
    counts[28:40, 0] += 10
    counts[40:52, 1] += 10
    obs = pd.DataFrame(
        {
            "person": (
                [f"d{i}" for i in range(6)] * 4
                + ["d6"] * 4
                + [f"d{i}" for i in range(6)] * 4
            ),
            "site": ["spleen"] * 12
            + ["blood"] * 12
            + ["spleen"] * 2
            + ["blood"] * 2
            + ["spleen"] * 12
            + ["blood"] * 12,
            "label": ["mono"] * 28 + ["nonclassical"] * 24,
            "library": ["lib_a", "lib_b"] * 26,
            "chemistry": (["primary"] * 24 + ["other"] * 4 + ["primary"] * 24),
        },
        index=pd.Index([f"c{i}" for i in range(52)], dtype=object),
        dtype=object,
    )
    var = pd.DataFrame(
        {
            "symbol": [
                "RIGI",
                "LMNB1",
                "GAPDH",
                *[f"b{i}" for i in range(397)],
            ],
        },
        index=pd.Index([f"ENSG{i:06}" for i in range(400)], dtype=object),
        dtype=object,
    )
    source = tmp_path / "source.h5ad"
    data = ad.AnnData(
        X=sp.csr_matrix(np.full(counts.shape, 99)),
        obs=obs,
        var=var,
        layers={"counts": sp.csr_matrix(counts)},
    )
    data.write_h5ad(source)
    panel = tmp_path / "panel.tsv"
    panel_text = (
        "gene_symbol\tmodule_id\tscoring_direction\taliases\n"
        "DDX58\tNASP_DNA_SENSING\tpositive\tRIGI\n"
        "LMNB1\tNASP_DNA_SENSING\tinverse\t\n"
        "GAPDH\tNASP_DNA_SENSING\tcontext_dependent\t\n"
    )
    panel.write_text(panel_text)
    catalog = GeneModules(panel_path=panel)
    modules = [catalog.get_module("NASP_DNA_SENSING", output="symbols")]
    spec = CohortSpec(
        cohort_id="fixture",
        input_path=source,
        counts_source="layers/counts",
        gene_id_column="_index",
        gene_symbol_column="symbol",
        obs_columns=(
            ("donor_id", "person"),
            ("tissue", "site"),
            ("assay", "chemistry"),
        ),
        obs_constants=(("modality", "cells"),),
        filters=(("chemistry", ("primary",)),),
        populations=(
            PopulationSpec(
                name="monocyte",
                column="label",
                labels=("mono",),
            ),
        ),
        library_keys=("library",),
        donor_metadata=(),
        comparison=ComparisonSpec(
            level_key="tissue",
            target="spleen",
            reference="blood",
            minimum_cells=2,
            minimum_pairs=6,
        ),
        source_notes="Synthetic source with deliberately incompatible X.",
    )
    metadata, source_var = read_cohort_metadata(spec)
    selected = select_cohort(metadata, spec).selected_obs.iloc[::-1]
    alignment = align_cohort_features(
        source_var,
        source_var,
        modules,
        reference_id_column="_index",
        reference_symbol_column="symbol",
        external_id_column="_index",
        external_symbol_column="symbol",
        aliases=panel_aliases(catalog.panel),
    )
    output = tmp_path / "scores"
    score_cohort(
        spec,
        selected,
        alignment,
        modules,
        reference=True,
        panel_path=panel,
        output_dir=output,
        aucell_chunk_size=7,
    )
    manifest = validate_scoring_artifacts(output)
    scores = pd.read_csv(output / "scores.csv.gz", index_col="obs_name")
    eligibility = pd.read_csv(output / "score_eligibility.csv")
    assert scores.index.tolist() == selected.index.tolist()
    assert eligibility.status.eq("ok").all()
    donors = aggregate_module_scores(
        selected,
        scores,
        modules,
        scorer="scanpy",
        scoring_context_id=str(manifest["scoring_context_id"]),
        minimum_cells=2,
    )
    estimate = estimate_module_contrasts(
        donors,
        eligibility.loc[eligibility.scorer.eq("scanpy")],
        spec.comparison,
    ).estimates.iloc[0]
    cell_scores = scores.loc[
        obs.index[:24], "NASP_DNA_SENSING_score"
    ].to_numpy()
    means = cell_scores.reshape(4, 6)
    differences = means[:2].mean(axis=0) - means[2:].mean(axis=0)
    assert estimate["estimate"] == pytest.approx(differences.mean())
    assert estimate["standard_error"] == pytest.approx(
        differences.std(ddof=1) / np.sqrt(6)
    )
    assert estimate["n_paired_donors"] == 6

    timestamp = (output / "scores.csv.gz").stat().st_mtime_ns
    inference_change = replace(
        spec, comparison=replace(spec.comparison, minimum_pairs=5)
    )
    score_cohort(
        inference_change,
        selected,
        alignment,
        modules,
        reference=True,
        panel_path=panel,
        output_dir=output,
        aucell_chunk_size=7,
    )
    assert (output / "scores.csv.gz").stat().st_mtime_ns == timestamp

    # Continue through saved inference/reporting, then reject corruption.
    hypotheses = pd.DataFrame(
        {
            "cohort_id": ["fixture", "fixture"],
            "hypothesis_id": ["paired_dna", "nonclassical_dna"],
            "family_id": ["primary", "extension"],
            "analysis_role": ["primary", "extension"],
            "module_id": ["NASP_DNA_SENSING", "NASP_DNA_SENSING"],
            "expected_direction": [1, 0],
            "primary_scorer": ["scanpy", "scanpy"],
            "comparison_axis": ["tissue", "tissue"],
            "target_context": ["spleen", "spleen"],
            "reference_context": ["blood", "blood"],
            "eligibility_status": ["eligible", "pending"],
            "eligibility_reason": [
                "eligible",
                "measurement_and_support_pending",
            ],
        }
    )
    discovery_path = tmp_path / "discovery.csv"
    pd.DataFrame([estimate]).to_csv(discovery_path, index=False)
    registration_dir = tmp_path / "registration"
    extension_spec = replace(
        spec,
        populations=(
            PopulationSpec(
                name="nonclassical_monocyte",
                column="label",
                labels=("nonclassical",),
            ),
        ),
    )
    extension_intake = select_cohort(metadata, extension_spec)
    frozen_extension = asdict(extension_spec)
    frozen_extension["input_path"] = str(extension_spec.input_path)
    registration = freeze_registration(
        hypotheses,
        {
            "fixture_discovery_contrasts": file_identity(discovery_path),
            "fixture_discovery_score_manifest": file_identity(
                output / "stage_manifest.json"
            ),
            "cohorts": {
                "fixture": {
                    "cohort_spec": {"cohort_id": "fixture"},
                    "balanced_cells_per_donor_context": 2,
                },
                "wells": {
                    "cohort_spec": {"cohort_id": "fixture"},
                    "extension_spec": frozen_extension,
                    "extension_selected_cells_sha256": request_identity(
                        {
                            "cells": extension_intake.selected_obs.index.astype(
                                str
                            ).tolist()
                        }
                    ),
                },
            },
            "sensitivities": {
                "cross_release_linked_ids": [],
                "primary_shared_donor_exclusion": [],
                "support_thresholds": [20, 50],
                "wells_chemistry": (
                    "all released chemistries as separate rescored sensitivity"
                ),
            },
            "limitations": ["Synthetic test, not biological evidence."],
        },
        output_dir=registration_dir,
    )
    intake = select_cohort(metadata, spec)
    prepared = RegisteredPrimary(
        registration=registration,
        spec=spec,
        intake=replace(intake, selected_obs=selected),
        reference_spec=spec,
        reference_intake=replace(intake, selected_obs=selected),
        alignment=alignment,
        modules=tuple(modules),
        panel_path=panel,
        hypotheses=hypotheses.loc[hypotheses.analysis_role.eq("primary")],
        target_sum=10000.0,
        random_state=42,
        aucell_chunk_size=7,
        aucell_num_workers=1,
        alpha=0.05,
    )
    analysis_dir = analyze_primary_cohort(
        prepared,
        score_dir=output,
        output_dir=tmp_path / "analysis",
    )
    classified = pd.read_csv(analysis_dir / "replication_contrasts.csv")
    primary = classified.loc[classified.scorer.eq("scanpy")].iloc[0]
    assert primary.estimate == pytest.approx(differences.mean())
    assert primary.status == "supported"
    chemistry_dir = rescore_pooled_chemistry(
        prepared,
        primary_analysis_dir=analysis_dir,
        output_dir=tmp_path / "chemistry",
    )
    validate_chemistry_artifacts(chemistry_dir)
    chemistry = pd.read_csv(chemistry_dir / "analysis" / "contrasts.csv")
    assert chemistry.n_paired_donors.eq(7).all()
    assert chemistry.n_matched_donors.eq(6).all()
    assert chemistry.loc[
        chemistry.scorer.eq("scanpy"), "baseline_matched_estimate"
    ].item() == pytest.approx(differences.mean())
    chemistry_pairs = pd.read_csv(
        chemistry_dir / "analysis" / "donor_differences.csv"
    )
    scanpy_pairs = chemistry_pairs.loc[chemistry_pairs.scorer.eq("scanpy")]
    assert chemistry.loc[
        chemistry.scorer.eq("scanpy"), "estimate"
    ].item() == pytest.approx(scanpy_pairs.difference.mean())
    assert chemistry.loc[
        chemistry.scorer.eq("scanpy"), "variant_matched_estimate"
    ].item() == pytest.approx(
        scanpy_pairs.loc[scanpy_pairs.donor_id.ne("d6"), "difference"].mean()
    )

    extension_dir = analyze_wells_extension(
        prepared, output_dir=tmp_path / "extension"
    )
    validate_extension_artifacts(extension_dir)
    extension = pd.read_csv(
        extension_dir / "analysis" / "extension_contrasts.csv"
    )
    assert set(extension.scorer) == {"scanpy", "aucell"}
    assert extension.loc[extension.scorer.eq("scanpy"), "status"].item() == (
        "association_detected"
    )
    assert (
        extension.loc[
            extension.scorer.eq("scanpy"), "n_registered_family"
        ].item()
        == 1
    )
    assert extension.n_paired_donors.eq(6).all()

    gene_dir = analyze_gene_sensitivity(
        prepared,
        primary_analysis_dir=analysis_dir,
        primary_score_dir=output,
        output_dir=tmp_path / "gene_sensitivity",
    )
    validate_gene_sensitivity(gene_dir)
    gene_effects = pd.read_csv(gene_dir / "scanpy_removal_contrasts.csv")
    assert set(gene_effects.selection_arm) == {"positive", "inverse"}
    assert gene_effects.eligibility_reason.eq("unscorable_arm").all()
    assert gene_effects.pvalue.isna().all()
    gene_differences = pd.read_csv(
        gene_dir / "scanpy_gene_paired_differences.csv"
    )
    assert (
        gene_differences.loc[
            gene_differences.gene.eq("DDX58"), "mean_expression_difference"
        ]
        .gt(0)
        .all()
    )

    # Removing the only member of either arm cannot create an unsigned result.
    driver_dir = rescore_without_contrast_drivers(
        prepared,
        primary_analysis_dir=analysis_dir,
        gene_dir=gene_dir,
        output_dir=tmp_path / "drivers",
        drivers_per_module=1,
    )
    validate_contrast_driver_removal(driver_dir)
    driver_effects = pd.read_csv(driver_dir / "analysis" / "contrasts.csv")
    assert driver_effects.eligibility_reason.eq("unscorable_arm").all()
    assert driver_effects.estimate.isna().all()
    assert driver_effects.pvalue.isna().all()
    assert driver_effects.n_paired_donors.eq(0).all()

    # Gene selection must reject changed diagnostics before any rescore.
    gene_path = gene_dir / "scanpy_gene_paired_differences.csv"
    original_genes = gene_path.read_bytes()
    gene_path.write_bytes(original_genes + b"\n")
    with pytest.raises(ValueError, match="Changed gene artifact"):
        rescore_without_contrast_drivers(
            prepared,
            primary_analysis_dir=analysis_dir,
            gene_dir=gene_dir,
            output_dir=tmp_path / "corrupt_drivers",
        )
    gene_path.write_bytes(original_genes)

    balanced_dir = rescore_balanced_comparison(
        prepared,
        cohort_key="fixture",
        primary_analysis_dir=analysis_dir,
        primary_score_dir=output,
        output_dir=tmp_path / "balanced",
    )
    validate_balanced_artifacts(balanced_dir)
    balanced_effects = pd.read_csv(balanced_dir / "external_contrasts.csv")
    scanpy_balanced = balanced_effects.loc[
        balanced_effects.scorer.eq("scanpy")
    ].iloc[0]
    assert scanpy_balanced.estimate == pytest.approx(differences.mean())
    assert scanpy_balanced.n_matched_donors == 6
    assert scanpy_balanced.baseline_matched_estimate == pytest.approx(
        differences.mean()
    )
    rescore_balanced_comparison(
        prepared,
        cohort_key="fixture",
        primary_analysis_dir=analysis_dir,
        primary_score_dir=output,
        output_dir=balanced_dir,
    )
    report_dir = publish_primary_report(
        analysis_dir=analysis_dir,
        registration_dir=registration_dir,
        cohort_key="fixture",
        output_dir=tmp_path / "report",
        title="Synthetic primary report",
        reproduction_command="synthetic fixture",
    )
    validate_primary_report(report_dir)
    assert "1 supported" in (report_dir / "report.md").read_text()
    fixed_dir = analyze_fixed_score_sensitivity(
        analysis_dir=analysis_dir,
        score_dir=output,
        registration_dir=registration_dir,
        cohort_key="fixture",
        output_dir=tmp_path / "fixed",
    )
    complete_report = publish_robustness_report(
        primary_report_dir=report_dir,
        fixed_dir=fixed_dir,
        balanced_dir=balanced_dir,
        gene_dir=gene_dir,
        variants_dir=chemistry_dir,
        registration_dir=registration_dir,
        output_dir=tmp_path / "robustness_report",
        title="Synthetic robustness",
        cohort_key="fixture",
        reproduction_command="synthetic fixture",
    )
    validate_robustness_report(complete_report)
    assert "1 supported" in (complete_report / "report.md").read_text()
    assert "unscorable_arm" in (complete_report / "report.md").read_text()
    displayed = pd.read_csv(
        complete_report / "figures" / "sensitivity_effects_displayed.csv"
    )
    assert (
        displayed.loc[
            displayed.variant_id.eq("minimum_cells:50"), "n_paired_donors"
        ]
        .eq(0)
        .all()
    )
    assert (
        displayed.loc[
            displayed.variant_id.eq("donor_deletion_range"), "interval_kind"
        ]
        .eq("effect_range_not_CI")
        .all()
    )
    assert (
        complete_report / "figures" / "4_gene_support.pdf"
    ).stat().st_size > 0
    assert (
        complete_report / "figures" / "5_sensitivity_effects.png"
    ).stat().st_size > 0
    final_report = publish_final_report(
        robustness_report_dir=complete_report,
        extension_dir=extension_dir,
        registration_dir=registration_dir,
        cohort_key="fixture",
        output_dir=tmp_path / "final_report",
        title="Synthetic complete report",
        reproduction_command="synthetic fixture",
    )
    validate_final_report(final_report, cohort_key="fixture")
    final_text = (final_report / "report.md").read_text()
    assert "Complete fixture report" in final_text
    assert "Registered extension: AUCell sensitivity" in final_text
    assert (
        final_report / "figures" / "6_population_extension.pdf"
    ).stat().st_size > 0
    assert (output / "scores.csv.gz").stat().st_mtime_ns == timestamp
    panel.write_text(panel_text.replace("DDX58", "GAPDH"))
    with pytest.raises(ValueError, match="Incompatible"):
        rescore_balanced_comparison(
            prepared,
            cohort_key="fixture",
            primary_analysis_dir=analysis_dir,
            primary_score_dir=output,
            output_dir=balanced_dir,
        )
    panel.write_text(panel_text)
    table = analysis_dir / "replication_contrasts.csv"
    table.write_text(table.read_text().replace("supported", "inconclusive"))
    with pytest.raises(ValueError, match="Changed analysis artifact"):
        publish_primary_report(
            analysis_dir=analysis_dir,
            registration_dir=registration_dir,
            cohort_key="fixture",
            output_dir=tmp_path / "corrupt_report",
            title="Must fail",
            reproduction_command="synthetic fixture",
        )

    panel.write_text(panel_text.replace("DDX58", "GAPDH"))
    with pytest.raises(ValueError, match="Incompatible"):
        score_cohort(
            spec,
            selected,
            alignment,
            modules,
            reference=True,
            panel_path=panel,
            output_dir=output,
            aucell_chunk_size=7,
        )
    panel.write_text(panel_text)
    counts[0, 0] += 10
    data.layers["counts"] = sp.csr_matrix(counts)
    data.write_h5ad(source)
    with pytest.raises(ValueError, match="Incompatible"):
        score_cohort(
            spec,
            selected,
            alignment,
            modules,
            reference=True,
            panel_path=panel,
            output_dir=output,
            aucell_chunk_size=7,
        )

    score_path = output / "scores.csv.gz"
    score_path.write_bytes(score_path.read_bytes()[:20])
    with pytest.raises(ValueError, match="changed scoring artifact"):
        validate_scoring_artifacts(output)
