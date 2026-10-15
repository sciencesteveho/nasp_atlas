"""Content identities and manifest-last publication for analysis stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def file_identity(path: Path) -> dict[str, str | int]:
    """Hash source bytes; fail if a writer changes the file during reading."""
    before = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()

    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Input changed during hashing: {path}")

    return {
        "path": str(path.resolve()),
        "bytes": after.st_size,
        "sha256": digest,
    }


def request_identity(request: Mapping[str, object]) -> str:
    """Identify a JSON-compatible scientific request without hidden defaults."""
    payload = json.dumps(request, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


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

    publish_stage_manifest(
        analysis_dir,
        request,
        list(tables),
        status=status,
        manifest_name=manifest_name,
    )


def publish_stage_manifest(
    analysis_dir: Path,
    request: Mapping[str, object],
    artifacts: Sequence[str],
    *,
    status: str,
    manifest_name: str,
) -> None:
    """Hash already written artifacts and publish their manifest last.

    Use this when a stage writes non-tabular artifacts, such as an h5ad,
    into a directory it claimed with `mkdir(exist_ok=False)`. Artifact names
    are relative to `analysis_dir`.
    """
    record = {
        "status": status,
        "request": dict(request),
        "identity": request_identity(request),
        "artifacts": {
            name: file_identity(analysis_dir / name) for name in artifacts
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


def read_published_table(
    stage_dir: Path,
    name: str,
    *,
    stage: str,
) -> pd.DataFrame:
    """Read one artifact of a completed stage after checking its bytes.

    Use this when a consumer does not know the producing request, for
    example when combining finished runs: the table must be listed in the
    stage manifest and still hash to its recorded value.
    """
    manifest = read_stage_manifest(
        stage_dir / f"{stage}_manifest.json", stage=stage
    )

    recorded = manifest["artifacts"].get(name)
    if recorded is None:
        raise ValueError(f"{name} is not a published {stage} artifact")

    path = stage_dir / name
    if file_identity(path)["sha256"] != recorded["sha256"]:
        raise ValueError(f"Changed {stage} artifact: {path}")

    return _read_table(path)


def validate_stage(
    stage_dir: Path,
    request: Mapping[str, object],
    *,
    stage: str,
) -> dict[str, Any]:
    """Return a completed stage's manifest if it answers this request.

    The saved request must equal `request` and every artifact must still
    hash to its recorded value; otherwise the directory belongs to another
    request or was modified, and the caller must choose a new directory.
    """
    manifest = read_stage_manifest(
        stage_dir / f"{stage}_manifest.json", stage=stage
    )

    if manifest["request"] != json.loads(json.dumps(dict(request))):
        raise ValueError(
            f"{stage_dir} holds a {stage} stage for a different request; "
            "choose a new output directory"
        )

    for name, identity in manifest["artifacts"].items():
        if file_identity(stage_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed {stage} artifact: {stage_dir / name}")

    return manifest


def reuse_or_publish_tables(
    stage_dir: Path,
    request: Mapping[str, object],
    build: Callable[[], Mapping[str, pd.DataFrame]],
    *,
    stage: str,
) -> dict[str, pd.DataFrame]:
    """Return a stage's tables, building and publishing them only once.

    A completed directory is reused after `validate_stage`; a missing one is
    built and published with `publish_stage_tables`. Table names are file
    names, such as "arm_support.csv".
    """
    if stage_dir.exists():
        manifest = validate_stage(stage_dir, request, stage=stage)
        return {
            name: _read_table(stage_dir / name)
            for name in manifest["artifacts"]
        }

    tables = dict(build())
    publish_stage_tables(
        stage_dir,
        request,
        tables,
        status=f"completed_{stage}",
        manifest_name=f"{stage}_manifest.json",
    )

    return tables


def _read_table(path: Path) -> pd.DataFrame:
    """Read a published table; a table with no columns reads as empty.

    pandas writes an empty DataFrame as a file without a header, which
    `read_csv` rejects, yet an empty table is a valid stage result, for
    example when no contrast was estimable.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
