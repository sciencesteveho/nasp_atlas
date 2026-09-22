"""Publish fixed-score diagnostics independently of rescoring sensitivities."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import sensitivity
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)
from nasp_atlas.analysis.external_replication.registration import (
    registered_questions,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    validate_analysis_artifacts,
)
from nasp_atlas.analysis.external_replication.sensitivity import (
    fixed_score_sensitivities,
)
from nasp_atlas.analysis.external_replication.specification import (
    ComparisonSpec,
)
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell.associations import paired


__all__ = ["analyze_fixed_score_sensitivity", "validate_fixed_sensitivity"]


def analyze_fixed_score_sensitivity(
    *,
    analysis_dir: Path,
    score_dir: Path,
    registration_dir: Path,
    cohort_key: str,
    output_dir: Path,
) -> Path:
    """Run registered deletions/support/overlap checks on intact primary scores.

    This completes only the fixed-score portion of robustness. Balanced scores,
    gene removal, chemistry/QC and modality rescoring are separate measurements.
    The primary table is consumed read-only and its decisions are preserved.
    """
    analysis = validate_analysis_artifacts(analysis_dir)
    score_manifest = validate_scoring_artifacts(score_dir)
    registration = read_registration(registration_dir)
    primary_request = analysis["request"]
    cohorts = registration.specification["cohorts"]
    settings = registration.specification["sensitivities"]
    if (
        not isinstance(primary_request, dict)
        or not isinstance(cohorts, dict)
        or not isinstance(settings, dict)
    ):
        raise ValueError("Missing registered analysis settings")
    if primary_request["registration_identity"] != registration.identity:
        raise ValueError("Sensitivity registration differs from primary")
    cohort_id = cohorts[cohort_key]["cohort_spec"]["cohort_id"]
    if primary_request["cohort_id"] != cohort_id:
        raise ValueError("Sensitivity cohort differs from primary analysis")
    score_identity = file_identity(score_dir / "stage_manifest.json")
    if score_identity != primary_request["score_manifest"]:
        raise ValueError("Sensitivity scores differ from primary measurement")
    request = {
        "primary_analysis": file_identity(
            analysis_dir / "analysis_manifest.json"
        ),
        "registration_identity": registration.identity,
        "calculation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (sensitivity, contrasts, paired)
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("pandas", "numpy", "scipy")
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    if output_dir.exists():
        validate_fixed_sensitivity(output_dir, request=request)
        return output_dir

    questions = registered_questions(
        registration, cohort_key, analysis_role="primary"
    )
    donors = pd.read_csv(analysis_dir / "donor_scores.csv.gz")
    donors = donors.loc[donors.module_id.isin(questions.module_id)]
    eligibility = pd.read_csv(score_dir / "score_eligibility.csv")
    eligibility = eligibility.loc[
        eligibility.module_id.isin(questions.module_id)
    ]
    selected = pd.read_csv(analysis_dir / "selected_cells.csv.gz")
    strata = ("institute", "cmv") if cohort_key == "wells" else ("assay",)
    exclusions = {
        "primary_cohort_overlap": settings["primary_shared_donor_exclusion"]
    }
    if cohort_key == "muscle":
        exclusions["any_wells_participant"] = settings[
            "cross_release_linked_ids"
        ]
    metadata = selected[["donor_id", *strata]].drop_duplicates()
    result = fixed_score_sensitivities(
        donors,
        eligibility,
        ComparisonSpec(**primary_request["comparison"]),
        support_thresholds=settings["support_thresholds"],
        exclusions=exclusions,
        donor_metadata=metadata,
        strata_keys=strata,
        alpha=primary_request["alpha"],
    )
    estimates = result.estimates.merge(
        questions[["module_id", "hypothesis_id"]],
        on="module_id",
        validate="many_to_one",
    ).assign(
        cohort_id=cohort_id,
        scoring_context_id=str(score_manifest["scoring_context_id"]),
        modality=selected.modality.unique().item(),
    )
    tables = {
        "sensitivity_contrasts.csv": estimates,
        "sensitivity_donor_differences.csv": result.donor_differences,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    manifest = {
        "status": "completed_fixed_score_sensitivity_only",
        "request": request,
        "identity": request_identity(request),
        "artifacts": {
            name: file_identity(output_dir / name) for name in tables
        },
        "remaining_robustness": [
            "balanced_rescoring",
            "gene_diagnostics_and_removal",
            "cohort_specific_chemistry_QC_or_modality",
        ],
    }
    partial = output_dir / "sensitivity_manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    partial.replace(output_dir / "sensitivity_manifest.json")
    return output_dir


def validate_fixed_sensitivity(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Validate fixed-score outputs without claiming all robustness is done."""
    manifest = read_stage_manifest(
        output_dir / "sensitivity_manifest.json",
        stage="fixed-score sensitivity",
    )
    if manifest["status"] != "completed_fixed_score_sensitivity_only":
        raise ValueError(f"Incomplete fixed-score sensitivity: {output_dir}")
    if request is not None and manifest["request"] != request:
        raise ValueError(
            f"Incompatible sensitivity: {output_dir}; use a new directory"
        )
    if manifest["identity"] != request_identity(manifest["request"]):
        raise ValueError(f"Changed sensitivity request: {output_dir}")
    for name, identity in manifest["artifacts"].items():
        if file_identity(output_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(
                f"Changed sensitivity artifact: {output_dir / name}"
            )
    return manifest
