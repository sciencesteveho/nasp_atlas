"""Tests for workflow command-line parsing."""

from __future__ import annotations

import sys
from pathlib import Path

from run_scripts import score_modules


def test_score_modules_cli_passes_none_for_default_expression_sources(
    monkeypatch,
) -> None:
    """The CLI passes normal None values for X and representation defaults."""
    captured: dict[str, object] = {}

    def capture_analysis(**kwargs: object) -> dict[str, Path]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_modules.py",
            "--h5ad-path",
            "input.h5ad",
            "--single-tissue-use-x",
            "--detection-threshold",
            "0.25",
        ],
    )
    monkeypatch.setattr(
        score_modules,
        "tabula_sapiens_tissue_analysis",
        capture_analysis,
    )

    score_modules.main()

    assert captured["expression_layer"] is None
    assert captured["single_tissue_use_rep"] is None
    assert captured["detection_threshold"] == 0.25
