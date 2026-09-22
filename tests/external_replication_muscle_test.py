"""Muscle subtype extension reuses primary scores and keeps unsupported rows."""

from __future__ import annotations

import json

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from nasp_compendium import GeneModules

from nasp_atlas.analysis.external_replication import read_cohort_metadata
from nasp_atlas.analysis.external_replication import select_cohort
from nasp_atlas.analysis.external_replication.extension_workflows import (
    analyze_muscle_extension,
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
from nasp_atlas.analysis.external_replication.muscle_variants import (
    rescore_muscle_variants,
)
from nasp_atlas.analysis.external_replication.muscle_variants import (
    validate_muscle_variants,
)
from nasp_atlas.analysis.external_replication.registered_inputs import (
    RegisteredPrimary,
)
from nasp_atlas.analysis.external_replication.registration import (
    freeze_registration,
)
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.analysis.external_replication.specification import (
    PopulationSpec,
)
from nasp_atlas.analysis.external_replication.workflows import (
    analyze_primary_cohort,
)
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.single_cell.visualization.replication import (
    plot_subtype_extension,
)


@pytest.fixture
def muscle_cohort(tmp_path) -> dict[str, object]:
    """Write a muscle-like source holding cells, doublets and nuclei.

    Cells d0-d7 are the primary population. d8 appears only as released
    doublets, so relaxing that filter changes donor membership. Nuclei cover
    seven people of whom three are also primary donors, the overlap the real
    muscle release has, and carry no vascular signal so the modality's effect
    differs from the primary's.
    """
    rng = np.random.default_rng(3)
    rows = []
    for donor in range(8):
        # CapEC is too sparse in three people to support a subtype contrast.
        n_capec = 12 if donor < 5 else 3
        for label, level2, n_cells in (
            ("ArtEC", "Artery", 12),
            ("CapEC", "Capillary", n_capec),
            ("VenEC", "Vein", 12),
            ("CD4+T", "CD4+T", 12),
        ):
            rows += [
                {
                    "person": f"d{donor}",
                    "level1": label,
                    "level2": level2 if label != "CD4+T" else "CD4+T",
                    "library": f"lib{donor}_{cell % 2}",
                    "batch": "cells",
                    "is_doublet": False,
                }
                for cell in range(n_cells)
            ]
    for label, level2 in (("ArtEC", "Artery"), ("CD4+T", "CD4+T")):
        rows += [
            {
                "person": "d8",
                "level1": label,
                "level2": level2,
                "library": f"lib8_{cell % 2}",
                "batch": "cells",
                "is_doublet": True,
            }
            for cell in range(12)
        ]
    nucleus_donors = ["d0", "d1", "d2", "n0", "n1", "n2", "n3"]
    for donor in nucleus_donors:
        for label, level2 in (("ArtEC", "Artery"), ("CD4+T", "CD4+T")):
            rows += [
                {
                    "person": donor,
                    "level1": label,
                    "level2": level2,
                    "library": f"nuc{donor}_{cell % 2}",
                    "batch": "nuclei",
                    "is_doublet": False,
                }
                for cell in range(12)
            ]
    obs = pd.DataFrame(
        rows,
        index=pd.Index([f"c{i}" for i in range(len(rows))], dtype=object),
        dtype=object,
    )
    obs["is_doublet"] = obs.is_doublet.astype(bool)
    counts = rng.poisson(3, (len(obs), 400))
    vascular = obs.level1.isin(["ArtEC", "VenEC"]) & obs.batch.eq("cells")
    counts[vascular.to_numpy(), 0] += 20
    counts[:, 2:12] += 15
    var = pd.DataFrame(
        {"symbol": ["RIGI", "GAPDH", *[f"b{i}" for i in range(398)]]},
        index=pd.Index([f"ENSG{i:06}" for i in range(400)], dtype=object),
        dtype=object,
    )
    source = tmp_path / "muscle.hdf"
    ad.AnnData(
        X=sp.csr_matrix(np.zeros_like(counts)),
        obs=obs,
        var=var,
        layers={"counts": sp.csr_matrix(counts)},
    ).write_h5ad(source)
    panel = tmp_path / "panel.tsv"
    panel.write_text(
        "gene_symbol\tmodule_id\tscoring_direction\taliases\n"
        "DDX58\tNASP_RNA_SENSING\tpositive\tRIGI\n"
        "GAPDH\tNASP_RNA_SENSING\tpositive\t\n"
    )
    catalog = GeneModules(panel_path=panel)
    modules = (catalog.get_module("NASP_RNA_SENSING", output="symbols"),)
    spec = CohortSpec(
        cohort_id="fixture_muscle",
        input_path=source,
        counts_source="layers/counts",
        gene_id_column="_index",
        gene_symbol_column="symbol",
        obs_columns=(("donor_id", "person"), ("modality", "batch")),
        obs_constants=(
            ("tissue", "muscle"),
            ("assay", "3'v3"),
        ),
        filters=(("batch", ("cells",)), ("is_doublet", (False,))),
        populations=(
            PopulationSpec(
                name="vascular_endothelium",
                column="level1",
                labels=("ArtEC", "CapEC", "VenEC"),
            ),
            PopulationSpec(name="CD4_T", column="level2", labels=("CD4+T",)),
        ),
        library_keys=("library",),
        donor_metadata=(),
        comparison=ComparisonSpec(
            level_key="population",
            target="vascular_endothelium",
            reference="CD4_T",
            minimum_cells=10,
            minimum_pairs=6,
        ),
        source_notes="Synthetic muscle-like source with three subtypes.",
    )
    metadata, features = read_cohort_metadata(spec)
    intake = select_cohort(metadata, spec)
    alignment = align_cohort_features(
        features,
        features,
        modules,
        reference_id_column="_index",
        reference_symbol_column="symbol",
        external_id_column="_index",
        external_symbol_column="symbol",
        aliases=panel_aliases(catalog.panel),
    )
    score_dir = score_cohort(
        spec,
        intake.selected_obs,
        alignment,
        modules,
        reference=False,
        panel_path=panel,
        output_dir=tmp_path / "primary_scores",
        aucell_chunk_size=50,
    )
    return {
        "obs": obs,
        "panel": panel,
        "modules": modules,
        "spec": spec,
        "intake": intake,
        "alignment": alignment,
        "score_dir": score_dir,
    }


def test_subtype_extension_uses_primary_scores_and_keeps_unsupported_rows(
    muscle_cohort,
    tmp_path,
) -> None:
    """Subtype effects match independent donor means; thin subtypes stay."""
    obs = muscle_cohort["obs"]
    panel = muscle_cohort["panel"]
    modules = muscle_cohort["modules"]
    spec = muscle_cohort["spec"]
    intake = muscle_cohort["intake"]
    alignment = muscle_cohort["alignment"]
    score_dir = muscle_cohort["score_dir"]

    subtypes = ["ArtEC", "CapEC", "VenEC"]
    hypotheses = pd.DataFrame(
        {
            "cohort_id": "muscle",
            "hypothesis_id": [f"muscle_extension_{name}" for name in subtypes],
            "family_id": "muscle_extension",
            "analysis_role": "extension",
            "module_id": "NASP_RNA_SENSING",
            "expected_direction": 0,
            "primary_scorer": "scanpy",
            "comparison_axis": "population",
            "target_context": subtypes,
            "reference_context": "CD4_T",
            "eligibility_status": "pending",
            "eligibility_reason": "measurement_and_support_pending",
        }
    )
    registration = freeze_registration(
        hypotheses,
        {
            "cohorts": {
                "muscle": {
                    "extension": {
                        "source_column": "level1",
                        "subtypes": subtypes,
                    }
                }
            }
        },
        output_dir=tmp_path / "registration",
    )
    prepared = RegisteredPrimary(
        registration=registration,
        spec=spec,
        intake=intake,
        reference_spec=spec,
        reference_intake=intake,
        alignment=alignment,
        modules=modules,
        panel_path=panel,
        hypotheses=hypotheses,
        target_sum=10000.0,
        random_state=42,
        aucell_chunk_size=50,
        aucell_num_workers=1,
        alpha=0.05,
    )

    output = analyze_muscle_extension(
        prepared,
        cohort_key="muscle",
        primary_score_dir=score_dir,
        output_dir=tmp_path / "extension",
    )

    validate_extension_artifacts(output)
    assert not (output / "scores").exists()
    results = pd.read_csv(output / "analysis" / "extension_contrasts.csv")
    scanpy = results.loc[results.scorer.eq("scanpy")].set_index(
        "target_context"
    )
    assert scanpy.n_registered_family.eq(3).all()
    assert scanpy.loc["CapEC", "n_paired_donors"] == 5
    assert scanpy.loc["CapEC", "status"] == "unavailable"
    assert np.isnan(scanpy.loc["CapEC", "pvalue_adjusted"])
    assert scanpy.loc["ArtEC", "n_paired_donors"] == 8

    cell_scores = pd.read_csv(
        score_dir / "scores.csv.gz", index_col="obs_name"
    )["NASP_RNA_SENSING_score"]
    donor_means = cell_scores.groupby([obs.person, obs.level1]).mean().unstack()
    expected = (donor_means["VenEC"] - donor_means["CD4+T"]).mean()
    assert scanpy.loc["VenEC", "estimate"] == pytest.approx(expected)

    frequencies = pd.read_csv(output / "analysis" / "subtype_frequencies.csv")
    shares = frequencies.groupby("donor_id").fraction_of_vascular_cells.sum()
    assert np.allclose(shares, 1.0)

    differences = pd.read_csv(output / "analysis" / "donor_differences.csv")
    figure, axes = plot_subtype_extension(
        results,
        differences,
        frequencies,
        module_ids=["NASP_RNA_SENSING"],
        subtypes=subtypes,
        title="Subtype support",
    )
    figure.canvas.draw()
    for column in range(len(subtypes)):
        upper, lower = axes[column], axes[column + len(subtypes)]
        # A person's point and composition bar must align on the page.
        for donor_position in (0, 4):
            assert upper.transData.transform((donor_position, 0))[0] == (
                pytest.approx(lower.transData.transform((donor_position, 0))[0])
            )
    plt.close(figure)


def test_muscle_variants_compare_on_the_true_donor_intersection(
    muscle_cohort,
    tmp_path,
) -> None:
    """Each variant keeps its own people; matched means use the real overlap."""
    spec = muscle_cohort["spec"]
    hypotheses = pd.DataFrame(
        {
            "cohort_id": ["muscle"],
            "hypothesis_id": ["muscle_primary_vascular_rna_sensing"],
            "family_id": ["muscle_primary"],
            "analysis_role": ["primary"],
            "module_id": ["NASP_RNA_SENSING"],
            "expected_direction": [1],
            "primary_scorer": ["scanpy"],
            "comparison_axis": ["population"],
            "target_context": ["vascular_endothelium"],
            "reference_context": ["CD4_T"],
            "eligibility_status": ["eligible"],
            "eligibility_reason": ["eligible"],
        }
    )
    registration = freeze_registration(
        hypotheses,
        {
            "sensitivities": {
                "muscle_QC": ["as_released", "scrublet_score_at_most_0.4"],
                "muscle_modality": (
                    "nuclei scored separately; overlapping donor intersection "
                    "descriptive"
                ),
            }
        },
        output_dir=tmp_path / "variant_registration",
    )
    prepared = RegisteredPrimary(
        registration=registration,
        spec=spec,
        intake=muscle_cohort["intake"],
        reference_spec=spec,
        reference_intake=muscle_cohort["intake"],
        alignment=muscle_cohort["alignment"],
        modules=muscle_cohort["modules"],
        panel_path=muscle_cohort["panel"],
        hypotheses=hypotheses,
        target_sum=10000.0,
        random_state=42,
        aucell_chunk_size=50,
        aucell_num_workers=1,
        alpha=0.05,
    )
    analysis_dir = analyze_primary_cohort(
        prepared,
        score_dir=muscle_cohort["score_dir"],
        output_dir=tmp_path / "primary_analysis",
    )
    primary_before = pd.read_csv(analysis_dir / "module_contrasts.csv")

    output = rescore_muscle_variants(
        prepared,
        primary_analysis_dir=analysis_dir,
        output_dir=tmp_path / "variants",
    )

    validate_muscle_variants(output)
    contrasts = pd.read_csv(output / "analysis" / "contrasts.csv")
    scanpy = contrasts.loc[contrasts.scorer.eq("scanpy")].set_index(
        "variant_id"
    )
    # Nuclei are seven people, only three of whom are primary donors.
    assert scanpy.loc["nuclei", "n_paired_donors"] == 7
    assert scanpy.loc["nuclei", "n_matched_donors"] == 3
    assert json.loads(scanpy.loc["nuclei", "matched_donors"]) == [
        "d0",
        "d1",
        "d2",
    ]
    assert scanpy.loc["nuclei", "modality"] == "nuclei"
    # Relaxing the released doublet flag admits a ninth person.
    assert scanpy.loc["as_released", "n_paired_donors"] == 9
    assert scanpy.loc["as_released", "n_matched_donors"] == 8
    assert scanpy.loc["as_released", "modality"] == "cells"

    differences = pd.read_csv(analysis_dir / "donor_differences.csv")
    shared = differences.loc[
        differences.scorer.eq("scanpy")
        & differences.status.eq("complete")
        & differences.donor_id.isin(["d0", "d1", "d2"])
    ]
    assert scanpy.loc["nuclei", "baseline_matched_estimate"] == pytest.approx(
        shared.difference.mean()
    )
    primary_estimate = primary_before.loc[
        primary_before.scorer.eq("scanpy"), "estimate"
    ].item()
    assert shared.difference.mean() != pytest.approx(primary_estimate)
    assert scanpy.loc["nuclei", "variant_matched_estimate"] != pytest.approx(
        primary_estimate
    )
    after = pd.read_csv(analysis_dir / "module_contrasts.csv")
    assert primary_before.estimate.equals(after.estimate)
