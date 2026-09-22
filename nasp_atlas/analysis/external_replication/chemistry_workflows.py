"""Wells all-chemistry rescoring with matched-person primary comparisons."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.external_replication import balanced_scoring
from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import sensitivity
from nasp_atlas.analysis.external_replication.balanced_scoring import (
    _paired_scores,
)
from nasp_atlas.analysis.external_replication.cohort import read_cohort_metadata
from nasp_atlas.analysis.external_replication.cohort import select_cohort
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registered_inputs import (
    RegisteredPrimary,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    validate_analysis_artifacts,
)
from nasp_atlas.analysis.external_replication.sensitivity import (
    matched_donor_effects,
)
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell.associations import paired


__all__ = ["rescore_pooled_chemistry", "validate_chemistry_artifacts"]


def rescore_pooled_chemistry(
    prepared: RegisteredPrimary,
    *,
    primary_analysis_dir: Path,
    output_dir: Path,
) -> Path:
    """Remove only the registered assay restriction and rescore distinct cells.

    Library and assay identities remain in the intake audit. The estimand still
    uses one mean difference per person; chemistry pooling is a changed
    measurement, never extra independent replication or a primary decision.
    """
    settings = prepared.registration.specification["sensitivities"]
    if (
        not isinstance(settings, dict)
        or settings.get("wells_chemistry")
        != "all released chemistries as separate rescored sensitivity"
    ):
        raise ValueError("Pooled Wells chemistry sensitivity is not registered")
    cohorts = prepared.registration.specification["cohorts"]
    if (
        not isinstance(cohorts, dict)
        or prepared.spec.cohort_id
        != cohorts["wells"]["cohort_spec"]["cohort_id"]
    ):
        raise ValueError(
            "This chemistry stage requires the registered Wells cohort"
        )
    primary = validate_analysis_artifacts(primary_analysis_dir)
    original = primary["request"]
    if (
        not isinstance(original, dict)
        or original["registration_identity"] != prepared.registration.identity
        or original["cohort_id"] != prepared.spec.cohort_id
    ):
        raise ValueError(
            "Chemistry sensitivity and primary registration differ"
        )
    assay_column = dict(prepared.spec.obs_columns)["assay"]
    if not any(key == assay_column for key, _ in prepared.spec.filters):
        raise ValueError("Primary selection has no assay filter to relax")
    spec = replace(
        prepared.spec,
        filters=tuple(
            (key, values)
            for key, values in prepared.spec.filters
            if key != assay_column
        ),
    )
    metadata, _ = read_cohort_metadata(spec)
    intake = select_cohort(metadata, spec)
    score_dir = score_cohort(
        spec,
        intake.selected_obs,
        prepared.alignment,
        prepared.modules,
        reference=False,
        panel_path=prepared.panel_path,
        output_dir=output_dir / "scores",
        target_sum=prepared.target_sum,
        random_state=prepared.random_state,
        aucell_chunk_size=prepared.aucell_chunk_size,
        aucell_num_workers=prepared.aucell_num_workers,
    )
    request = {
        "registration_identity": prepared.registration.identity,
        "primary_analysis": file_identity(
            primary_analysis_dir / "analysis_manifest.json"
        ),
        "score_manifest": file_identity(score_dir / "stage_manifest.json"),
        "selection": "primary population and tissues, all released assays",
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (balanced_scoring, contrasts, sensitivity, paired)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    analysis_dir = output_dir / "analysis"
    if analysis_dir.exists():
        validate_chemistry_artifacts(output_dir, request=request)
        return output_dir
    donors, effects = _paired_scores(
        score_dir,
        intake.selected_obs,
        spec,
        prepared.modules,
        alpha=prepared.alpha,
    )
    baseline = pd.read_csv(primary_analysis_dir / "donor_differences.csv")
    matched = matched_donor_effects(baseline, effects.donor_differences)
    estimates = effects.estimates.merge(
        matched, on=["module_id", "scorer"], how="left", validate="one_to_one"
    ).assign(
        variant_id="pooled_chemistry",
        variant_kind="chemistry",
        cohort_id=spec.cohort_id,
    )
    tables = {
        "contrasts.csv": estimates,
        "donor_differences.csv": effects.donor_differences,
        "donor_scores.csv.gz": donors,
        "selected_cells.csv.gz": intake.selected_obs.rename_axis(
            "obs_name"
        ).reset_index(),
        "donor_support.csv": intake.donor_support,
        "preparation_support.csv": intake.preparation_support,
        "cohort_audit.csv": intake.selection_audit,
    }
    analysis_dir.mkdir(exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(analysis_dir / name, index=False)
    record = {
        "status": "completed_pooled_chemistry_sensitivity",
        "request": request,
        "identity": request_identity(request),
        "artifacts": {
            name: file_identity(analysis_dir / name) for name in tables
        },
    }
    partial = analysis_dir / "chemistry_manifest.json.partial"
    partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    partial.replace(analysis_dir / "chemistry_manifest.json")
    return output_dir


def validate_chemistry_artifacts(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Require intact completed score and chemistry sensitivity artifacts."""
    analysis_dir = output_dir / "analysis"
    record = read_stage_manifest(
        analysis_dir / "chemistry_manifest.json",
        stage="pooled-chemistry sensitivity",
    )
    if record["status"] != "completed_pooled_chemistry_sensitivity":
        raise ValueError(f"Incomplete chemistry sensitivity: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(f"Incompatible chemistry sensitivity: {output_dir}")
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed chemistry request: {output_dir}")
    score_manifest = output_dir / "scores" / "stage_manifest.json"
    if file_identity(score_manifest) != record["request"]["score_manifest"]:
        raise ValueError(f"Changed chemistry scores: {score_manifest}")
    validate_scoring_artifacts(score_manifest.parent)
    for name, identity in record["artifacts"].items():
        if file_identity(analysis_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed chemistry artifact: {name}")
    return record
