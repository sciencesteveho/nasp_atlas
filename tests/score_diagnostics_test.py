"""Tests for module-score diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nasp_atlas.single_cell import compare_module_scorers


def test_compare_module_scorers_reports_rank_agreement() -> None:
    """Monotone scorer outputs have perfect Spearman agreement."""
    scores = pd.DataFrame(
        {
            "NASP_DNA_SENSING_score": [0.0, 1.0, 2.0, 3.0],
            "NASP_DNA_SENSING_auc": [0.0, 0.1, 0.4, 0.9],
        }
    )

    result = compare_module_scorers(scores, ["NASP_DNA_SENSING"])

    row = result.iloc[0]
    assert row["n_complete_cells"] == 4
    assert row["spearman_r"] == 1.0
    assert row["median_absolute_percentile_difference"] == 0.0
    assert not row["skipped"]


def test_compare_module_scorers_marks_constant_scores_uninformative() -> None:
    """Constant scorer outputs are retained with an explicit skip reason."""
    scores = pd.DataFrame(
        {
            "NASP_FEEDBACK_score": np.ones(4),
            "NASP_FEEDBACK_auc": np.ones(4),
        }
    )

    result = compare_module_scorers(scores, ["NASP_FEEDBACK"])

    row = result.iloc[0]
    assert row["skipped"]
    assert row["skip_reason"] == "constant_score"
