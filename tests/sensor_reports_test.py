"""Worked donor-paired sensor report with aliases, zeros and missing genes."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import scipy.stats as stats
import yaml

from nasp_atlas.analysis.sensor_reports import analyze_sensor_reference
from nasp_atlas.single_cell.associations import resolve_feature_specs
from nasp_atlas.single_cell.sensor_effects import paired_sensor_effects
from nasp_atlas.single_cell.sensor_effects import unique_sensor_measurements


def test_sensor_report_uses_declared_counts_and_people(tmp_path) -> None:
    """Aliases, full-library normalization and pairing survive export."""
    records, counts = [], []
    for donor, target_count in enumerate((2, 4, 6)):
        for tissue, value in (("target", target_count), ("reference", 1)):
            for _ in range(2 + donor):
                records.append(
                    {
                        "person": str(donor),
                        "site": tissue,
                        "label": "mono",
                        "chemistry": "v3",
                        "library": f"{donor}_{tissue}",
                    }
                )
                counts.append([value, 0, 100])
    obs = pd.DataFrame(
        records,
        index=pd.Index([f"c{i}" for i in range(len(records))], dtype=object),
        dtype=object,
    )
    var = pd.DataFrame(
        {"symbol": ["RIGI", "ZERO", "BACKGROUND"]},
        index=pd.Index(["e1", "e2", "e3"], dtype=object),
        dtype=object,
    )
    source = tmp_path / "source.h5ad"
    ad.AnnData(
        X=sp.csr_matrix(np.full((len(obs), 3), 99)),
        obs=obs,
        var=var,
        layers={"counts": sp.csr_matrix(counts)},
    ).write_h5ad(source)
    panel = tmp_path / "panel.tsv"
    panel.write_text(
        "gene_symbol\tmodule_id\tscoring_direction\tsensor\taliases\nDDX58\tNASP_RNA_SENSING\tpositive\trna_sensor\tRIGI\nDDX58\tNASP_DNA_SENSING\tpositive\trna_sensor\tRIGI\nZERO\tNASP_RNA_SENSING\tpositive\trna_sensor\t\nMISSING\tNASP_DNA_SENSING\tpositive\tdna_sensor\t\n"
    )
    spec = tmp_path / "cohort.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "cohort_id": "fixture",
                "input_path": source.name,
                "counts_source": "layers/counts",
                "gene_id_column": "_index",
                "gene_symbol_column": "symbol",
                "obs_columns": {
                    "donor_id": "person",
                    "tissue": "site",
                    "assay": "chemistry",
                },
                "obs_constants": {"modality": "cells"},
                "filters": {},
                "populations": {
                    "mono": {"column": "label", "labels": ["mono"]}
                },
                "library_keys": ["library"],
                "donor_metadata": [],
                "comparison": {
                    "level_key": "tissue",
                    "target": "target",
                    "reference": "reference",
                    "minimum_cells": 2,
                    "minimum_pairs": 3,
                },
                "source_notes": "Synthetic counts; deliberately unrelated X.",
            }
        )
    )
    output = analyze_sensor_reference(
        spec_path=spec, panel_path=panel, output_dir=tmp_path / "report"
    )
    effects = pd.read_csv(output / "sensor_effects.csv").set_index("gene")
    differences = np.log1p(
        np.array([2, 4, 6]) / np.array([102, 104, 106]) * 10000
    ) - np.log1p(10000 / 101)
    expected_se = differences.std(ddof=1) / np.sqrt(3)
    effect = effects.loc["DDX58"]
    assert effect.n_paired_donors == 3
    assert effect.estimate == pytest.approx(differences.mean())
    assert effect.standard_error == pytest.approx(expected_se)
    assert effect.ci_lower == pytest.approx(
        differences.mean() - stats.t.ppf(0.975, 2) * expected_se
    )
    assert effect.pvalue_bh == pytest.approx(min(1, effect.pvalue * 3))
    assert effect.detection_difference == 0
    assert effects.loc["ZERO", "estimate"] == 0
    assert effects.loc["ZERO", "status"] == "degenerate_variance"
    assert pd.isna(effects.loc["MISSING", "estimate"])
    pairs = pd.read_csv(output / "sensor_donor_differences.csv")
    assert (
        pairs.loc[pairs.gene.eq("MISSING"), "target_support_reason"]
        .eq("absent")
        .all()
    )
    manifest = json.loads((output / "sensor_manifest.json").read_text())
    assert manifest["status"] == "completed_sensor_report"
    assert (output / "paired_sensor_effects.png").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        analyze_sensor_reference(
            spec_path=spec, panel_path=panel, output_dir=output
        )

    donors = pd.read_csv(output / "sensor_donor_expression.csv.gz")
    catalog = pd.read_csv(output / "sensor_catalog.csv")
    keys = ["cohort_id", "donor_id", "tissue", "population", "assay"]
    duplicated = pd.concat([donors, donors.loc[donors.gene.eq("DDX58")]])
    unique = unique_sensor_measurements(duplicated, catalog, context_keys=keys)
    pd.testing.assert_frame_equal(
        unique.sort_values([*keys, "gene"]).reset_index(drop=True),
        unique_sensor_measurements(donors, catalog, context_keys=keys)
        .sort_values([*keys, "gene"])
        .reset_index(drop=True),
    )
    duplicated.iloc[-1, duplicated.columns.get_loc("mean_expression")] += 1
    with pytest.raises(ValueError, match="Conflicting"):
        unique_sensor_measurements(duplicated, catalog, context_keys=keys)
    low = unique.copy()
    mask = low.donor_id.eq(0) & low.tissue.eq("target") & low.gene.eq("DDX58")
    low.loc[mask, "n_finite_cells"] = 1
    result = paired_sensor_effects(
        low,
        catalog,
        pair_keys=["cohort_id", "donor_id"],
        level_key="tissue",
        target="target",
        reference="reference",
        minimum_cells=2,
        minimum_pairs=3,
    )
    row = result.estimates.set_index("gene").loc["DDX58"]
    assert row.n_paired_donors == 2
    assert row.status == "insufficient_pairs"
    assert row.estimate == pytest.approx(differences[1:].mean())
    assert pd.isna(row.ci_lower)
    low.loc[low.tissue.eq("target"), "assay"] = "other"
    with pytest.raises(ValueError, match="one assay per person"):
        paired_sensor_effects(
            low,
            catalog,
            pair_keys=["cohort_id", "donor_id"],
            level_key="tissue",
            target="target",
            reference="reference",
        )


def test_association_sensor_alias_and_unavailable_requests() -> None:
    """Resolve canonical sensors through aliases; report absent sensors."""
    data = ad.AnnData(
        shape=(0, 1),
        var=pd.DataFrame({"feature_name": ["RIGI"]}, index=["ENSG00000107201"]),
    )
    specs, skipped = resolve_feature_specs(
        data, pd.DataFrame(), sensor_group="nucleic_acid_sensors"
    )
    assert [(spec.feature_id, spec.feature_label) for spec in specs] == [
        ("ENSG00000107201", "DDX58")
    ]
    assert skipped
    assert all(row["requested"] != "DDX58" for row in skipped)
