"""Validate score checkpoints against their complete workflow request."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


__all__ = ["score_checkpoint_matches", "write_score_checkpoint"]


def score_checkpoint_matches(
    score_path: Path,
    request: Mapping[str, object],
) -> bool:
    """Accept completed scores matching input, selection and settings."""
    checkpoint = score_path.with_name(score_path.name + ".manifest.json")
    if not score_path.is_file() or not checkpoint.is_file():
        return False

    try:
        saved = json.loads(checkpoint.read_text())
    except (OSError, ValueError):
        return False

    return (
        isinstance(saved, dict)
        and saved.get("status") == "completed"
        and saved.get("request") == dict(request)
        and saved.get("score_sha256") == _file_hash(score_path)
    )


def write_score_checkpoint(
    score_path: Path,
    request: Mapping[str, object],
) -> None:
    """Publish completion only after the scoring workflow has returned."""
    header = pd.read_csv(score_path, nrows=1)
    if (
        header.empty
        or "obs_name" not in header
        or "scoring_scorers" not in header
    ):
        raise ValueError(f"Incomplete score table: {score_path}")

    checkpoint = score_path.with_name(score_path.name + ".manifest.json")
    record = {
        "status": "completed",
        "request": dict(request),
        "score_sha256": _file_hash(score_path),
        "columns": header.columns.tolist(),
    }

    with NamedTemporaryFile(
        dir=checkpoint.parent, suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)

    try:
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(checkpoint)
    finally:
        temporary.unlink(missing_ok=True)


def _file_hash(path: Path) -> str:
    """Hash compressed bytes without loading the score table into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
