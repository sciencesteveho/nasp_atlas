"""Harmonized-reference resolution before external expression scoring."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from nasp_atlas.analysis.external_replication import read_cohort_metadata
from nasp_atlas.analysis.external_replication import read_cohort_spec
from nasp_atlas.analysis.external_replication import select_cohort
from nasp_atlas.analysis.external_replication.discovery import (
    finalize_cohort_registration,
)
from nasp_atlas.analysis.external_replication.discovery import (
    reconstruct_discovery_reference,
)
from nasp_atlas.analysis.external_replication.discovery import (
    validate_discovery_artifacts,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registration import (
    freeze_registration,
)
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)


def test_discovery_resolves_and_freezes_before_external_scoring(
    tmp_path,
) -> None:
    """A pending cohort gains a TS decision without any external scoring."""
    n_donors = 4
    cells_per_context = 4
    obs_rows = []
    for context in ("vascular", "CD4_T"):
        for donor in range(n_donors):
            for cell in range(cells_per_context):
                obs_rows.append(
                    {
                        "person": f"d{donor}",
                        "label": context,
                        "chemistry": "v3",
                        "library": f"d{donor}_{context}_{cell % 2}",
                    }
                )
    obs = pd.DataFrame(
        obs_rows,
        index=pd.Index(
            [f"cell_{index}" for index in range(len(obs_rows))], dtype=object
        ),
        dtype=object,
    )
    genes = ["RIGI", "LMNB1", *[f"BG{index}" for index in range(198)]]
    counts = np.random.default_rng(17).poisson(3, (len(obs), len(genes)))
    counts[:, 2:12] += 40
    target = obs.label.eq("vascular").to_numpy()
    counts[target, 0] = 80
    counts[target, 1] = 1
    counts[~target, 0] = 1
    counts[~target, 1] = 80
    var = pd.DataFrame(
        {"symbol": genes},
        index=pd.Index(
            [f"ENSG{index:06}" for index in range(len(genes))], dtype=object
        ),
        dtype=object,
    )
    source = tmp_path / "source.h5ad"
    ad.AnnData(
        X=sp.csr_matrix(np.zeros_like(counts)),
        obs=obs,
        var=var,
        layers={"counts": sp.csr_matrix(counts)},
    ).write_h5ad(source)
    panel = tmp_path / "panel.tsv"
    panel.write_text(
        "gene_symbol\tmodule_id\tscoring_direction\taliases\n"
        "DDX58\tNASP_DNA_SENSING\tpositive\tRIGI\n"
        "LMNB1\tNASP_DNA_SENSING\tinverse\t\n"
    )
    external_path = tmp_path / "external.yaml"
    reference_path = tmp_path / "reference.yaml"
    _write_spec(
        external_path,
        cohort_id="fixture_external",
        source_name=source.name,
        cells_per_context=cells_per_context,
        n_donors=n_donors,
    )
    _write_spec(
        reference_path,
        cohort_id="fixture_reference",
        source_name=source.name,
        cells_per_context=cells_per_context,
        n_donors=n_donors,
    )

    external = read_cohort_spec(external_path)
    reference = read_cohort_spec(reference_path)
    metadata, _ = read_cohort_metadata(external)
    selected = select_cohort(metadata, external).selected_obs
    reference_metadata, _ = read_cohort_metadata(reference)
    reference_selected = select_cohort(
        reference_metadata, reference
    ).selected_obs
    frozen_external = asdict(external)
    frozen_external["input_path"] = str(external.input_path)
    frozen_reference = asdict(reference)
    frozen_reference["input_path"] = str(reference.input_path)
    questions = pd.DataFrame(
        {
            "cohort_id": ["muscle"],
            "hypothesis_id": ["fixture_dna"],
            "family_id": ["fixture_primary"],
            "analysis_role": ["primary"],
            "comparison_axis": ["population"],
            "target_context": ["vascular"],
            "reference_context": ["CD4_T"],
            "module_id": ["NASP_DNA_SENSING"],
            "expected_direction": [1],
            "primary_scorer": ["scanpy"],
            "discovery_artifact": ["pending"],
            "eligibility_status": ["pending"],
            "eligibility_reason": ["discovery_pending"],
        }
    )
    source_audit_v1 = tmp_path / "source_audit_v1.md"
    source_audit_v1.write_text("Pending fixture source audit.\n")
    compendium_install = json.loads(
        importlib.metadata.distribution("nasp_compendium").read_text(
            "direct_url.json"
        )
        or "null"
    )
    base_dir = tmp_path / "registration_v1"
    freeze_registration(
        questions,
        {
            "cohorts": {
                "muscle": {
                    "cohort_spec": frozen_external,
                    "reference_spec": frozen_reference,
                    "source": file_identity(source),
                    "reference_source": file_identity(source),
                    "source_gate": "pending",
                    "selected_cells_sha256": request_identity(
                        {"cells": selected.index.astype(str).tolist()}
                    ),
                    "reference_selected_cells_sha256": request_identity(
                        {"cells": reference_selected.index.astype(str).tolist()}
                    ),
                }
            },
            "panel": file_identity(panel),
            "compendium_install": compendium_install,
            "scoring": {
                "target_sum": 10_000.0,
                "random_state": 42,
                "aucell_chunk_size": 7,
                "aucell_num_workers": 1,
            },
            "inference": {"alpha": 0.05},
            "source_audit": file_identity(source_audit_v1),
        },
        output_dir=base_dir,
    )
    source_gate = tmp_path / "source_gate.json"
    source_gate.write_text(
        json.dumps(
            {
                "cohort_key": "muscle",
                "status": "resolved",
                "discovery_equivalence": "resolved",
                "discovery_reason": "",
            }
        )
        + "\n"
    )

    discovery_dir = reconstruct_discovery_reference(
        registration_dir=base_dir,
        cohort_key="muscle",
        spec_path=external_path,
        reference_path=reference_path,
        panel_path=panel,
        source_gate_path=source_gate,
        output_dir=tmp_path / "discovery",
    )
    validate_discovery_artifacts(discovery_dir)
    resolved = pd.read_csv(discovery_dir / "analysis" / "hypotheses.csv")
    assert resolved.eligibility_status.tolist() == ["eligible"]
    assert not (discovery_dir / "external_scores").exists()

    source_audit_v2 = tmp_path / "source_audit_v2.md"
    source_audit_v2.write_text("Resolved fixture source audit.\n")
    registration_v2 = finalize_cohort_registration(
        base_registration_dir=base_dir,
        cohort_key="muscle",
        discovery_dir=discovery_dir,
        source_audit_path=source_audit_v2,
        output_dir=tmp_path / "registration_v2",
    )
    finalized = read_registration(registration_v2)
    assert finalized.specification["cohorts"]["muscle"]["source_gate"] == (
        "resolved"
    )
    assert finalized.hypotheses.eligibility_status.tolist() == ["eligible"]


def _write_spec(
    path,
    *,
    cohort_id: str,
    source_name: str,
    cells_per_context: int,
    n_donors: int,
) -> None:
    """Write one strict fixture cohort specification."""
    path.write_text(
        yaml.safe_dump(
            {
                "cohort_id": cohort_id,
                "input_path": source_name,
                "counts_source": "layers/counts",
                "gene_id_column": "_index",
                "gene_symbol_column": "symbol",
                "obs_columns": {
                    "donor_id": "person",
                    "assay": "chemistry",
                },
                "obs_constants": {
                    "tissue": "muscle",
                    "modality": "cells",
                },
                "filters": {},
                "populations": {
                    "vascular": {
                        "column": "label",
                        "labels": ["vascular"],
                    },
                    "CD4_T": {
                        "column": "label",
                        "labels": ["CD4_T"],
                    },
                },
                "library_keys": ["library"],
                "donor_metadata": [],
                "comparison": {
                    "level_key": "population",
                    "target": "vascular",
                    "reference": "CD4_T",
                    "minimum_cells": cells_per_context,
                    "minimum_pairs": n_donors,
                },
                "source_notes": "Synthetic harmonized reference fixture.",
            },
            sort_keys=False,
        )
    )
