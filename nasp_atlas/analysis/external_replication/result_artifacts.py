"""Publish and validate the saved numerical inputs to a replication report."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity


__all__ = [
    "publish_stage_tables",
    "read_stage_manifest",
    "validate_analysis_artifacts",
    "write_analysis_artifacts",
]


def write_analysis_artifacts(
    tables: Mapping[str, pd.DataFrame],
    request: Mapping[str, object],
    *,
    output_dir: Path,
) -> Path:
    """Claim a new analysis directory and publish completion after all tables.

    Exclusive directory creation prevents concurrent writers. A failed partial
    directory remains inspectable and requires a new output directory. Existing
    completed output is reusable only if its dependencies and contents match.
    """
    if output_dir.exists():
        validate_analysis_artifacts(output_dir, request=request)
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    artifacts = {name: file_identity(output_dir / name) for name in tables}
    manifest = {
        "status": "completed_primary_analysis",
        "request": dict(request),
        "identity": request_identity(request),
        "artifacts": artifacts,
    }
    partial = output_dir / "analysis_manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    partial.replace(output_dir / "analysis_manifest.json")
    return output_dir


def publish_stage_tables(
    analysis_dir: Path,
    request: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    *,
    status: str,
    manifest_name: str,
) -> None:
    """Write every table, then publish the completion manifest atomically.

    Exclusive directory creation claims a single writer; a failed attempt
    leaves an inspectable directory and needs a new output location.
    """
    analysis_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(analysis_dir / name, index=False)
    record = {
        "status": status,
        "request": dict(request),
        "identity": request_identity(request),
        "artifacts": {
            name: file_identity(analysis_dir / name) for name in tables
        },
    }
    partial = analysis_dir / f"{manifest_name}.partial"
    partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    partial.replace(analysis_dir / manifest_name)


def read_stage_manifest(path: Path, *, stage: str) -> dict[str, Any]:
    """Read a completion manifest or explain why the stage is unusable.

    A stage publishes its manifest last, so a missing or unreadable one means
    the output is an incomplete or failed attempt and cannot be reused.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Missing or invalid {stage} manifest: {path}. If this output is "
            "an incomplete or failed attempt, do not reuse it; write to a new "
            "output directory. Otherwise check that the path is correct."
        ) from error


def validate_analysis_artifacts(
    output_dir: Path, *, request: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Validate numerical provenance and content before reporting or reuse."""
    manifest_path = output_dir / "analysis_manifest.json"
    record = read_stage_manifest(manifest_path, stage="primary analysis")
    if record["status"] != "completed_primary_analysis":
        raise ValueError(f"Incomplete primary analysis: {output_dir}")
    if request is not None and record["request"] != dict(request):
        raise ValueError(
            f"Incompatible analysis: {output_dir}; use a new directory"
        )
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed analysis request: {manifest_path}")
    for name, identity in record["artifacts"].items():
        path = output_dir / name
        if file_identity(path)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed analysis artifact: {path}")
    results = pd.read_csv(output_dir / "replication_contrasts.csv")
    if results.duplicated(["hypothesis_id", "scorer"]).any():
        raise ValueError(f"Duplicate classified contrasts: {output_dir}")
    return record
