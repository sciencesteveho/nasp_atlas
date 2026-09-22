"""Reconstruct a harmonized TS reference before external effect scoring."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from nasp_compendium import GeneModules

from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import scoring
from nasp_atlas.analysis.external_replication.cohort import read_cohort_metadata
from nasp_atlas.analysis.external_replication.cohort import select_cohort
from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.contrasts import (
    resolve_discovery_eligibility,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    align_cohort_features,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    panel_aliases,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registered_inputs import (
    _read_frozen_cohort_inputs,
)
from nasp_atlas.analysis.external_replication.registration import (
    freeze_registration,
)
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)
from nasp_atlas.analysis.external_replication.registration import (
    registered_questions,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.scoring import (
    aggregate_module_scores,
)
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell.associations import paired


__all__ = [
    "finalize_cohort_registration",
    "reconstruct_discovery_reference",
    "validate_discovery_artifacts",
]


def reconstruct_discovery_reference(
    *,
    registration_dir: Path,
    cohort_key: str,
    spec_path: Path,
    reference_path: Path,
    panel_path: Path,
    source_gate_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Score the frozen TS population and resolve discovery eligibility.

    This stage may consume a registration whose external source gate is still
    pending. It validates the frozen source, specifications, exact cell sets,
    panel and compendium before reading expression. The external matrix is used
    only for feature reconciliation; no external expression is scored here.
    """
    frozen = _read_frozen_cohort_inputs(
        registration_dir=registration_dir,
        cohort_key=cohort_key,
        spec_path=spec_path,
        reference_path=reference_path,
        panel_path=panel_path,
    )
    source_gate = _read_source_gate(source_gate_path, cohort_key=cohort_key)
    external_obs, external_features = read_cohort_metadata(frozen.spec)
    reference_obs, reference_features = read_cohort_metadata(frozen.reference)
    external_intake = select_cohort(external_obs, frozen.spec)
    reference_intake = select_cohort(reference_obs, frozen.reference)
    for selected, key in (
        (external_intake.selected_obs, "selected_cells_sha256"),
        (reference_intake.selected_obs, "reference_selected_cells_sha256"),
    ):
        identity = request_identity(
            {"cells": selected.index.astype(str).tolist()}
        )
        if identity != frozen.cohort[key]:
            raise ValueError(f"Selected cells differ from registration: {key}")

    questions = registered_questions(
        frozen.registration, cohort_key, analysis_role="primary"
    )
    if questions.empty:
        raise ValueError(f"No registered primary questions for {cohort_key}")
    catalog = GeneModules(panel_path=panel_path)
    modules = tuple(
        catalog.get_module(module_id, output="symbols")
        for module_id in dict.fromkeys(questions.module_id)
    )
    alignment = align_cohort_features(
        reference_features,
        external_features,
        modules,
        reference_id_column=frozen.reference.gene_id_column,
        reference_symbol_column=frozen.reference.gene_symbol_column,
        external_id_column=frozen.spec.gene_id_column,
        external_symbol_column=frozen.spec.gene_symbol_column,
        aliases=panel_aliases(catalog.panel),
    )
    if dry_run:
        return output_dir
    score_dir = score_cohort(
        frozen.reference,
        reference_intake.selected_obs,
        alignment,
        modules,
        reference=True,
        panel_path=panel_path,
        output_dir=output_dir / "scores",
        target_sum=float(frozen.settings["target_sum"]),
        random_state=int(frozen.settings["random_state"]),
        aucell_chunk_size=int(frozen.settings["aucell_chunk_size"]),
        aucell_num_workers=int(frozen.settings["aucell_num_workers"]),
    )
    request = {
        "base_registration_identity": frozen.registration.identity,
        "cohort_key": cohort_key,
        "source_gate": file_identity(source_gate_path),
        "score_manifest": file_identity(score_dir / "stage_manifest.json"),
        "alpha": float(frozen.inference["alpha"]),
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (contrasts, scoring, paired)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    analysis_dir = output_dir / "analysis"
    manifest_path = analysis_dir / "discovery_manifest.json"
    if manifest_path.exists():
        validate_discovery_artifacts(output_dir, request=request)
        return output_dir

    score_record = validate_scoring_artifacts(score_dir)
    scores = pd.read_csv(score_dir / "scores.csv.gz", index_col="obs_name")
    eligibility = pd.read_csv(score_dir / "score_eligibility.csv")
    donors = pd.concat(
        [
            aggregate_module_scores(
                reference_intake.selected_obs,
                scores,
                modules,
                scorer=scorer,
                scoring_context_id=str(score_record["scoring_context_id"]),
                minimum_cells=frozen.reference.comparison.minimum_cells,
            )
            for scorer in ("scanpy", "aucell")
        ],
        ignore_index=True,
    )
    effects = estimate_module_contrasts(
        donors,
        eligibility,
        frozen.reference.comparison,
        alpha=float(frozen.inference["alpha"]),
    )
    discovery_path = (analysis_dir / "discovery_contrasts.csv").resolve()
    source_reason = (
        ""
        if source_gate["discovery_equivalence"] == "resolved"
        else str(source_gate["discovery_reason"])
    )
    resolved = resolve_discovery_eligibility(
        questions,
        effects.estimates,
        discovery_artifact=str(discovery_path),
        source_gate_reason=source_reason,
        minimum_pairs=frozen.reference.comparison.minimum_pairs,
    )
    tables = {
        "discovery_contrasts.csv": effects.estimates.assign(
            cohort_id=frozen.reference.cohort_id,
            reference_kind="harmonized",
        ),
        "donor_differences.csv": effects.donor_differences,
        "donor_scores.csv.gz": donors,
        "selected_cells.csv.gz": reference_intake.selected_obs.rename_axis(
            "obs_name"
        ).reset_index(),
        "donor_support.csv": reference_intake.donor_support,
        "preparation_support.csv": reference_intake.preparation_support,
        "module_coverage.csv": alignment.coverage,
        "reference_features.csv": alignment.reference_features,
        "external_features.csv": alignment.external_features,
        "hypotheses.csv": resolved,
    }
    analysis_dir.mkdir(exist_ok=False)
    for name, table in tables.items():
        table.to_csv(analysis_dir / name, index=False)
    record = {
        "status": "completed_discovery_reference",
        "request": request,
        "identity": request_identity(request),
        "artifacts": {
            name: file_identity(analysis_dir / name) for name in tables
        },
    }
    partial = analysis_dir / "discovery_manifest.json.partial"
    partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    partial.replace(manifest_path)
    return output_dir


def finalize_cohort_registration(
    *,
    base_registration_dir: Path,
    cohort_key: str,
    discovery_dir: Path,
    source_audit_path: Path,
    output_dir: Path,
) -> Path:
    """Freeze resolved source/discovery evidence as a new registration."""
    base = read_registration(base_registration_dir)
    discovery = validate_discovery_artifacts(discovery_dir)
    request = discovery["request"]
    if (
        request["base_registration_identity"] != base.identity
        or request["cohort_key"] != cohort_key
    ):
        raise ValueError("Discovery evidence belongs to another registration")
    source_identity = request["source_gate"]
    if not isinstance(source_identity, dict):
        raise ValueError("Discovery request has no source-gate identity")
    source_gate_path = Path(source_identity["path"])
    source_gate = _read_source_gate(source_gate_path, cohort_key=cohort_key)
    if source_gate["status"] != "resolved":
        raise ValueError(f"Unresolved technical source gate for {cohort_key}")

    resolved = pd.read_csv(
        discovery_dir / "analysis" / "hypotheses.csv", keep_default_na=False
    )
    cohorts = base.specification["cohorts"]
    if not isinstance(cohorts, dict) or not isinstance(
        cohorts.get(cohort_key), dict
    ):
        raise ValueError(f"Missing frozen cohort record: {cohort_key}")
    cohort_record = cohorts[cohort_key]
    cohort_spec = cohort_record["cohort_spec"]
    if not isinstance(cohort_spec, dict):
        raise ValueError(f"Invalid frozen cohort specification: {cohort_key}")
    primary = registered_questions(base, cohort_key, analysis_role="primary")
    immutable = [
        column
        for column in primary.columns
        if column
        not in {
            "discovery_artifact",
            "eligibility_status",
            "eligibility_reason",
        }
    ]
    pd.testing.assert_frame_equal(
        primary[immutable].reset_index(drop=True),
        resolved[immutable].reset_index(drop=True),
        check_dtype=False,
    )
    if not resolved.eligibility_status.isin(["eligible", "unavailable"]).all():
        raise ValueError("Discovery hypotheses remain unresolved")

    hypotheses = base.hypotheses.copy().set_index("hypothesis_id")
    updates = resolved.set_index("hypothesis_id")
    columns = [
        "discovery_artifact",
        "eligibility_status",
        "eligibility_reason",
    ]
    hypotheses.loc[updates.index, columns] = updates[columns]
    hypotheses = hypotheses.reset_index()

    specification: dict[str, Any] = deepcopy(base.specification)
    cohort = specification["cohorts"][cohort_key]
    cohort["source_gate"] = "resolved"
    specification[f"{cohort_key}_discovery_contrasts"] = file_identity(
        discovery_dir / "analysis" / "discovery_contrasts.csv"
    )
    specification[f"{cohort_key}_discovery_score_manifest"] = file_identity(
        discovery_dir / "scores" / "stage_manifest.json"
    )
    specification[f"{cohort_key}_discovery_manifest"] = file_identity(
        discovery_dir / "analysis" / "discovery_manifest.json"
    )
    specification[f"{cohort_key}_source_gate"] = file_identity(source_gate_path)
    specification["source_audit"] = file_identity(source_audit_path)
    specification["registration_predecessor"] = {
        "identity": base.identity,
        "manifest": file_identity(
            base_registration_dir / "registration_manifest.json"
        ),
    }
    freeze_registration(hypotheses, specification, output_dir=output_dir)
    return output_dir


def validate_discovery_artifacts(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, Any]:
    """Require intact scores and complete reference-resolution artifacts."""
    analysis_dir = output_dir / "analysis"
    manifest_path = analysis_dir / "discovery_manifest.json"
    record: dict[str, Any] = read_stage_manifest(
        manifest_path, stage="discovery reference"
    )
    if record["status"] != "completed_discovery_reference":
        raise ValueError(f"Incomplete discovery reference: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(f"Incompatible discovery reference: {output_dir}")
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed discovery request: {manifest_path}")
    score_manifest = output_dir / "scores" / "stage_manifest.json"
    if file_identity(score_manifest) != record["request"]["score_manifest"]:
        raise ValueError(f"Changed discovery scores: {score_manifest}")
    validate_scoring_artifacts(score_manifest.parent)
    for name, identity in record["artifacts"].items():
        path = analysis_dir / name
        if file_identity(path)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed discovery artifact: {path}")
    hypotheses = pd.read_csv(
        analysis_dir / "hypotheses.csv", keep_default_na=False
    )
    if hypotheses.hypothesis_id.duplicated().any():
        raise ValueError("Duplicate resolved discovery hypotheses")
    return record


def _read_source_gate(path: Path, *, cohort_key: str) -> dict[str, Any]:
    """Read the human-reviewed source and annotation decision."""
    record = read_stage_manifest(path, stage="source gate")
    if record.get("cohort_key") != cohort_key:
        raise ValueError(f"Source gate belongs to another cohort: {path}")
    if record.get("status") not in {"resolved", "unresolved"}:
        raise ValueError("Source gate status must be resolved or unresolved")
    equivalence = record.get("discovery_equivalence")
    if equivalence not in {"resolved", "unresolved"}:
        raise ValueError("discovery_equivalence must be resolved or unresolved")
    if equivalence == "unresolved" and not record.get("discovery_reason"):
        raise ValueError("Unresolved discovery equivalence requires a reason")
    return record
