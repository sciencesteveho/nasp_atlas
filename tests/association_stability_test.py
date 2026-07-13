"""Tests for cross-stratum continuous association stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.single_cell.associations import (
    summarize_continuous_association_stability,
)


def _regression_results() -> pd.DataFrame:
    """Build realistic continuous-regression results across three tissues."""
    records: list[dict[str, object]] = []
    feature_values = {
        "DNA": {
            "slopes": [2.0, 1.0, -0.5],
            "fdr": [0.04, 0.01, 0.20],
            "spearman": [0.8, 0.7, -0.2],
            "pearson": [0.9, 0.6, -0.3],
            "skipped": [False, False, False],
        },
        "RNA": {
            "slopes": [-2.0, -1.0, 100.0],
            "fdr": [0.03, 0.02, 0.001],
            "spearman": [-0.6, -0.4, 1.0],
            "pearson": [-0.5, -0.3, 1.0],
            "skipped": [False, False, True],
        },
        "IFN": {
            "slopes": [0.5, np.nan, np.nan],
            "fdr": [0.01, np.nan, np.nan],
            "spearman": [0.5, np.nan, np.nan],
            "pearson": [0.4, np.nan, np.nan],
            "skipped": [False, True, True],
        },
    }
    strata = ["lung", "liver", "blood"]
    for feature_id, values in feature_values.items():
        for index, stratum in enumerate(strata):
            records.append(
                {
                    "feature_type": "module",
                    "feature_id": feature_id,
                    "feature_label": feature_id,
                    "predictor": "age_years",
                    "statistical_unit": "donor_tissue",
                    "aggregation": "mean",
                    "stratify_key": "tissue",
                    "stratum": stratum,
                    "analysis_role": "inferential",
                    "fdr_method": "benjamini_hochberg",
                    "slope": values["slopes"][index],
                    "spearman_r": values["spearman"][index],
                    "pearson_r": values["pearson"][index],
                    "ols_pvalue_fdr": values["fdr"][index],
                    "skipped": values["skipped"][index],
                }
            )
    return pd.DataFrame.from_records(records)


def test_stability_summary_counts_directions_and_effect_strength() -> None:
    """The summary captures cross-tissue support and direction consistency."""
    summary = summarize_continuous_association_stability(
        _regression_results()
    ).set_index("feature_id")

    dna = summary.loc["DNA"]
    assert dna["n_strata"] == 3
    assert dna["n_tested_strata"] == 3
    assert dna["n_significant_strata"] == 2
    assert dna["median_slope"] == pytest.approx(1.0)
    assert dna["min_slope"] == pytest.approx(-0.5)
    assert dna["max_slope"] == pytest.approx(2.0)
    assert dna["n_positive_strata"] == 2
    assert dna["n_negative_strata"] == 1
    assert dna["dominant_direction"] == "positive"
    assert dna["direction_consistency_fraction"] == pytest.approx(2 / 3)
    assert dna["median_abs_spearman_r"] == pytest.approx(0.7)
    assert dna["median_abs_pearson_r"] == pytest.approx(0.6)
    assert dna["best_stratum"] == "liver"
    assert dna["best_fdr"] == pytest.approx(0.01)


def test_stability_summary_ignores_skipped_strata() -> None:
    """Skipped regressions cannot inflate slope or significance summaries."""
    summary = summarize_continuous_association_stability(
        _regression_results()
    ).set_index("feature_id")

    rna = summary.loc["RNA"]
    assert rna["n_tested_strata"] == 2
    assert rna["n_significant_strata"] == 2
    assert rna["median_slope"] == pytest.approx(-1.5)
    assert rna["max_slope"] == pytest.approx(-1.0)
    assert rna["n_positive_strata"] == 0
    assert rna["n_negative_strata"] == 2
    assert rna["dominant_direction"] == "negative"
    assert rna["direction_consistency_fraction"] == pytest.approx(1.0)
    assert rna["best_stratum"] == "liver"
    assert rna["best_fdr"] == pytest.approx(0.02)


def test_stability_summary_marks_too_few_tested_strata() -> None:
    """A lone tested tissue remains descriptive rather than cross-tissue."""
    summary = summarize_continuous_association_stability(
        _regression_results(), min_tested_strata=2
    ).set_index("feature_id")

    ifn = summary.loc["IFN"]
    assert ifn["n_tested_strata"] == 1
    assert ifn["median_slope"] == pytest.approx(0.5)
    assert ifn["dominant_direction"] == "positive"
    assert ifn["direction_consistency_fraction"] == pytest.approx(1.0)
    assert not ifn["meets_min_tested_strata"]
    assert ifn["skipped"]
    assert ifn["skip_reason"] == "too_few_tested_strata"


def test_stability_summary_allows_missing_correlation_columns() -> None:
    """Correlation summaries are NaN when those optional inputs are absent."""
    results = _regression_results().drop(columns=["spearman_r", "pearson_r"])

    summary = summarize_continuous_association_stability(results)

    assert summary["median_abs_spearman_r"].isna().all()
    assert summary["median_abs_pearson_r"].isna().all()


def test_stability_summary_rejects_duplicate_strata() -> None:
    """Duplicate feature-stratum regressions fail rather than double count."""
    results = _regression_results()
    results = pd.concat([results, results.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate stratum rows"):
        summarize_continuous_association_stability(results)


def test_stability_summary_returns_typed_empty_schema() -> None:
    """An empty regression table returns the complete output schema."""
    empty = _regression_results().iloc[0:0]

    summary = summarize_continuous_association_stability(empty)

    assert summary.empty
    assert summary.columns.tolist() == [
        "feature_type",
        "feature_id",
        "feature_label",
        "predictor",
        "statistical_unit",
        "aggregation",
        "stratify_key",
        "analysis_role",
        "fdr_method",
        "n_strata",
        "n_tested_strata",
        "n_significant_strata",
        "median_slope",
        "min_slope",
        "max_slope",
        "n_positive_strata",
        "n_negative_strata",
        "dominant_direction",
        "direction_consistency_fraction",
        "median_abs_spearman_r",
        "median_abs_pearson_r",
        "best_stratum",
        "best_fdr",
        "fdr_column",
        "fdr_threshold",
        "min_tested_strata",
        "meets_min_tested_strata",
        "skipped",
        "skip_reason",
    ]
