"""Freeze questions and scientific settings before external scoring."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)


REGISTRATION_MANIFEST = "registration_manifest.json"

__all__ = [
    "Registration",
    "freeze_registration",
    "read_registration",
    "registered_questions",
]


@dataclass(frozen=True)
class Registration:
    """Frozen questions, settings, and their shared content identity."""

    hypotheses: pd.DataFrame
    specification: dict[str, object]
    identity: str


def registered_questions(
    registration: Registration, cohort_key: str, *, analysis_role: str
) -> pd.DataFrame:
    """Return one cohort's frozen questions of one role, keyed by registry key.

    The registry labels questions with the cohort key; a cohort specification's
    own `cohort_id` names its measurements and can differ from that key.
    """
    hypotheses = registration.hypotheses
    return hypotheses.loc[
        hypotheses.cohort_id.eq(cohort_key)
        & hypotheses.analysis_role.eq(analysis_role)
    ].copy()


def freeze_registration(
    hypotheses: pd.DataFrame,
    specification: Mapping[str, object],
    *,
    output_dir: Path,
) -> Registration:
    """Freeze a new registration; never overwrite a previous or partial freeze.

    The caller resolves input/panel identities and cohort discovery/source
    gates before calling. Pending gates for a later cohort remain explicit.
    Directory creation claims one writer; completion is published last by an
    atomic rename. A failed attempt requires a new directory, preserving its
    evidence instead of silently revising a preregistered question.
    """
    _validate_questions(hypotheses)
    resolved = json.loads(json.dumps(specification, allow_nan=False))
    output_dir.mkdir(parents=True, exist_ok=False)
    questions_path = output_dir / "hypotheses.csv"
    specification_path = output_dir / "resolved_spec.json"
    hypotheses.to_csv(questions_path, index=False)
    specification_path.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    )
    artifacts = {
        "hypotheses": {
            "filename": questions_path.name,
            "sha256": file_identity(questions_path)["sha256"],
        },
        "specification": {
            "filename": specification_path.name,
            "sha256": file_identity(specification_path)["sha256"],
        },
    }
    record = {
        "status": "frozen",
        "created_utc": datetime.now(UTC).isoformat(),
        "identity": request_identity(artifacts),
        "artifacts": artifacts,
    }
    temporary = output_dir / (REGISTRATION_MANIFEST + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output_dir / REGISTRATION_MANIFEST)
    return read_registration(output_dir)


def read_registration(output_dir: Path) -> Registration:
    """Load only an intact completed freeze, retaining unavailable questions."""
    manifest = output_dir / REGISTRATION_MANIFEST
    record = read_stage_manifest(manifest, stage="registration")
    if record["status"] != "frozen":
        raise ValueError(f"Incomplete registration: {manifest}")
    artifacts = record["artifacts"]
    if request_identity(artifacts) != record["identity"]:
        raise ValueError(f"Changed registration identity: {manifest}")
    paths = {}
    for role, artifact in artifacts.items():
        path = output_dir / artifact["filename"]
        if file_identity(path)["sha256"] != artifact["sha256"]:
            raise ValueError(f"Changed registration artifact: {path}")
        paths[role] = path
    hypotheses = pd.read_csv(paths["hypotheses"], keep_default_na=False)
    _validate_questions(hypotheses)
    return Registration(
        hypotheses,
        json.loads(paths["specification"].read_text()),
        record["identity"],
    )


def _validate_questions(hypotheses: pd.DataFrame) -> None:
    """Reject ambiguous identities, directions or accidental family mixing."""
    required = {
        "cohort_id",
        "hypothesis_id",
        "family_id",
        "analysis_role",
        "comparison_axis",
        "target_context",
        "reference_context",
        "module_id",
        "expected_direction",
        "primary_scorer",
        "eligibility_status",
        "eligibility_reason",
    }
    if missing := required.difference(hypotheses.columns):
        raise KeyError(f"Missing hypothesis fields: {sorted(missing)}")
    if hypotheses.empty or hypotheses.hypothesis_id.duplicated().any():
        raise ValueError("Register nonempty, uniquely identified questions")
    identity = ["cohort_id", "hypothesis_id", "family_id", "module_id"]
    if hypotheses[identity].isna().any(axis=None):
        raise ValueError("Hypothesis identities must not be missing")
    if not hypotheses.analysis_role.isin(["primary", "extension"]).all():
        raise ValueError("Declare primary or extension role for every question")
    if not hypotheses.primary_scorer.eq("scanpy").all():
        raise ValueError("This registered analysis uses Scanpy as primary")
    primary = hypotheses.analysis_role.eq("primary")
    if (
        not hypotheses.loc[primary, "expected_direction"].isin([-1, 1]).all()
        or not hypotheses.loc[~primary, "expected_direction"].eq(0).all()
    ):
        raise ValueError(
            "Primary directions are signed; extensions are two-sided"
        )
    families = hypotheses.groupby("family_id", observed=True)
    if families[["cohort_id", "analysis_role"]].nunique().gt(1).any(axis=None):
        raise ValueError(
            "Each family must belong to one cohort and analysis role"
        )
    if hypotheses.target_context.eq(hypotheses.reference_context).any():
        raise ValueError("Target and reference contexts must differ")
