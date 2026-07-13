"""Tests for donor-aware module-score context summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.single_cell.context_summary import summarize_module_contexts


def _unit_frame() -> pd.DataFrame:
    """Build module scores spanning two tissues and three donors."""
    return pd.DataFrame(
        {
            "unit_id": [
                "lung-d1",
                "lung-d2",
                "lung-d3",
                "liver-d1",
                "liver-d2",
                "liver-d3",
            ]
            * 2,
            "donor_id": ["d1", "d2", "d3"] * 4,
            "tissue": (["lung"] * 3 + ["liver"] * 3) * 2,
            "feature_type": ["module_score"] * 12,
            "feature_label": ["DNA sensing"] * 6 + ["RNA sensing"] * 6,
            "feature_value": [
                -2.0,
                0.0,
                2.0,
                4.0,
                6.0,
                np.inf,
                4.0,
                5.0,
                6.0,
                0.0,
                1.0,
                2.0,
            ],
            "n_cells": [10, 20, 30, 40, 50, 60] * 2,
        }
    )


def test_context_summary_reports_coverage_and_raw_scale_statistics() -> None:
    """Finite coverage and quartiles retain the input scorer scale."""
    result = summarize_module_contexts(
        _unit_frame(),
        context_columns=["tissue"],
        min_units=2,
        min_donors=2,
    )

    liver = result[
        (result["feature_label"] == "DNA sensing")
        & (result["tissue"] == "liver")
    ].iloc[0]
    assert liver["n_units"] == 3
    assert liver["n_finite_units"] == 2
    assert liver["n_donors"] == 3
    assert liver["n_finite_donors"] == 2
    assert liver["finite_coverage"] == pytest.approx(2 / 3)
    assert liver["mean"] == 5.0
    assert liver["median"] == 5.0
    assert liver["std"] == pytest.approx(np.sqrt(2.0))
    assert liver["q25"] == 4.5
    assert liver["q75"] == 5.5
    assert liver["iqr"] == 1.0
    assert liver["median_contributing_cells"] == 45.0


def test_context_summary_ranks_contexts_within_each_module() -> None:
    """The highest eligible median is rank one independently per module."""
    result = summarize_module_contexts(
        _unit_frame(),
        context_columns=["tissue"],
        min_units=2,
        min_donors=2,
    )

    dna = result[result["feature_label"] == "DNA sensing"].set_index("tissue")
    rna = result[result["feature_label"] == "RNA sensing"].set_index("tissue")
    assert dna.loc["liver", "context_rank"] == 1.0
    assert dna.loc["liver", "context_percentile"] == 1.0
    assert dna.loc["lung", "context_percentile"] == 0.0
    assert rna.loc["lung", "context_rank"] == 1.0
    assert rna.loc["lung", "context_percentile"] == 1.0


def test_context_summary_does_not_rank_ineligible_contexts() -> None:
    """Contexts with too few finite donors remain visible but unranked."""
    frame = _unit_frame()
    liver_dna = (frame["feature_label"] == "DNA sensing") & (
        frame["tissue"] == "liver"
    )
    frame.loc[liver_dna, "feature_value"] = [10.0, np.nan, np.nan]

    result = summarize_module_contexts(
        frame,
        context_columns=["tissue"],
        min_units=2,
        min_donors=2,
    )

    dna = result[result["feature_label"] == "DNA sensing"].set_index("tissue")
    assert not dna.loc["liver", "eligible"]
    assert np.isnan(dna.loc["liver", "context_rank"])
    assert dna.loc["lung", "context_rank"] == 1.0
    assert dna.loc["lung", "context_percentile"] == 0.5


def test_context_summary_rejects_duplicate_module_units() -> None:
    """Duplicate module-unit rows fail instead of inflating context support."""
    frame = _unit_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="one row per module"):
        summarize_module_contexts(frame, context_columns=["tissue"])


def test_context_summary_reports_missing_required_columns() -> None:
    """Missing donor metadata raises an actionable schema error."""
    frame = _unit_frame().drop(columns="donor_id")

    with pytest.raises(KeyError, match="donor_id"):
        summarize_module_contexts(frame, context_columns=["tissue"])


def test_context_summary_filters_non_module_features() -> None:
    """Default feature filtering excludes gene-expression rows."""
    frame = _unit_frame()
    gene_row = frame.iloc[[0]].copy()
    gene_row["feature_type"] = "gene_expression"
    gene_row["feature_label"] = "CGAS"
    frame = pd.concat([frame, gene_row], ignore_index=True)

    result = summarize_module_contexts(
        frame,
        context_columns=["tissue"],
        min_units=2,
        min_donors=2,
    )

    assert set(result["feature_label"]) == {"DNA sensing", "RNA sensing"}
