"""Tests for mixed-model contrast and variance visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection

from nasp_atlas.single_cell.visualization import MixedModelPlotter


def _effect_results() -> pd.DataFrame:
    """Return estimable and non-estimable planned mixed-model contrasts."""
    return pd.DataFrame(
        {
            "analysis": ["condition_by_cell_type"] * 3,
            "feature_type": ["module_score"] * 3,
            "feature_id": ["MODULE_A", "MODULE_B", "MODULE_MISSING"],
            "feature_label": ["MODULE_A", "MODULE_B", "MODULE_MISSING"],
            "estimand": ["condition_effect_by_cell_type"] * 3,
            "contrast": ["disease_vs_normal"] * 3,
            "level": ["disease"] * 3,
            "reference": ["normal"] * 3,
            "by": ["cell_type"] * 3,
            "by_level": ["macrophage", "T_cell", "fibroblast"],
            "estimate": [0.5, -0.2, 0.0],
            "standard_error": [0.1, 0.08, np.nan],
            "ci_low": [0.3, -0.35, 0.0],
            "ci_high": [0.7, -0.05, 0.0],
            "pvalue": [0.001, 0.08, np.nan],
            "pvalue_fdr": [0.01, 0.2, 1.0],
            "n_independent_units": [8, 11, 2],
            "n_level_units": [4, 5, 1],
            "n_reference_units": [4, 6, 1],
            "n_paired_units": [4, np.nan, np.nan],
            "estimable": [True, True, False],
            "status": ["ok", "ok", "insufficient_support"],
            "reason": ["", "", "too few independent units"],
            "effect_scale": ["Adjusted module-score difference"] * 3,
        }
    )


def _variance_results() -> pd.DataFrame:
    """Return complete, missing, and numeric-zero variance components."""
    return pd.DataFrame(
        {
            "analysis": ["variance_decomposition"] * 6,
            "feature_type": ["module_score"] * 6,
            "feature_id": ["MODULE_A"] * 3 + ["MODULE_B"] * 3,
            "feature_label": ["MODULE_A"] * 3 + ["MODULE_B"] * 3,
            "component": ["donor", "study", "residual"] * 2,
            "variance": [0.3, np.nan, 0.7, 0.0, 0.2, 0.8],
            "variance_fraction": [0.3, np.nan, 0.7, 0.0, 0.2, 0.8],
            "n_observations": [24] * 3 + [30] * 3,
            "n_independent_units": [8] * 3 + [10] * 3,
            "estimable": [True, False, True, True, True, True],
            "status": ["ok", "not_available", "ok", "ok", "ok", "ok"],
            "reason": ["", "study metadata unavailable", "", "", "", ""],
        }
    )


def test_effect_plot_preserves_estimates_intervals_and_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Plotted contrasts retain estimates, CIs, references, and unit support."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        MixedModelPlotter(
            output_dir=tmp_path, dpi=450
        ).plot_mixed_model_effects(
            _effect_results(),
            estimand="condition_effect_by_cell_type",
            analysis="condition_by_cell_type",
            filename="condition_effects",
        )

        axis = figures[0].axes[0]
        point_offsets = np.concatenate(
            [
                collection.get_offsets()
                for collection in axis.collections
                if isinstance(collection, PathCollection)
            ]
        )
        np.testing.assert_allclose(
            np.sort(point_offsets[:, 0].astype(float)),
            [-0.2, 0.5],
        )

        interval_segments = [
            segment
            for collection in axis.collections
            if isinstance(collection, LineCollection)
            for segment in collection.get_segments()
        ]
        interval_endpoints = sorted(
            (float(segment[0, 0]), float(segment[1, 0]))
            for segment in interval_segments
        )
        assert interval_endpoints == [(-0.35, -0.05), (0.3, 0.7)]

        row_labels = "\n".join(
            label.get_text() for label in axis.get_yticklabels()
        )
        assert "Disease vs normal" in row_labels
        assert "n=4 paired" in row_labels
        assert "n=5/6 units" in row_labels
        assert "FDR=0.010" in row_labels
        assert "MODULE MISSING" not in row_labels
        assert (tmp_path / "condition_effects.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_effect_plot_selects_top_rows_deterministically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The requested top-N limit retains the strongest reproducible effects."""
    effects = _effect_results().iloc[:2].copy()
    third = effects.iloc[[0]].copy()
    third["feature_id"] = "MODULE_C"
    third["feature_label"] = "MODULE_C"
    third["estimate"] = 0.1
    third["ci_low"] = -0.1
    third["ci_high"] = 0.3
    third["pvalue_fdr"] = 0.8
    effects = pd.concat([effects, third], ignore_index=True)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        MixedModelPlotter(
            output_dir=tmp_path, dpi=450
        ).plot_mixed_model_effects(
            effects,
            estimand="condition_effect_by_cell_type",
            filename="top_effects",
            max_effects=2,
        )

        row_labels = "\n".join(
            label.get_text() for label in figures[0].axes[0].get_yticklabels()
        )
        assert "MODULE A" in row_labels
        assert "MODULE B" in row_labels
        assert "MODULE C" not in row_labels
    finally:
        for figure in figures:
            close_figure(figure)


def test_variance_plot_preserves_fractions_and_marks_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Variance bars preserve fractions and distinguish missing from zero."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        MixedModelPlotter(
            output_dir=tmp_path, dpi=450
        ).plot_mixed_model_variance(
            _variance_results(),
            analysis="variance_decomposition",
            filename="variance_components",
            component_order=("donor", "study", "residual"),
        )

        axis = figures[0].axes[0]
        component_widths = sorted(
            float(patch.get_width())
            for patch in axis.patches
            if patch.get_hatch() != "////"
        )
        np.testing.assert_allclose(component_widths, [0.0, 0.2, 0.3, 0.7, 0.8])

        annotations = [text.get_text() for text in axis.texts]
        assert sum("Missing: Study" in text for text in annotations) == 1
        assert not any("Missing: Donor" in text for text in annotations)
        row_labels = "\n".join(
            label.get_text() for label in axis.get_yticklabels()
        )
        assert "n=8 independent" in row_labels
        assert "n=10 independent" in row_labels
        assert (tmp_path / "variance_components.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_variance_plot_prioritizes_estimable_features_before_truncation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Failed alphabetic features cannot displace an estimable result."""
    failed = pd.DataFrame(
        {
            "analysis": ["variance_decomposition"] * 31,
            "feature_type": ["module_score"] * 31,
            "feature_id": [f"A_FAILED_{index:02d}" for index in range(31)],
            "feature_label": [f"A_FAILED_{index:02d}" for index in range(31)],
            "component": ["donor"] * 31,
            "variance": [np.nan] * 31,
            "variance_fraction": [np.nan] * 31,
            "n_observations": [12] * 31,
            "n_independent_units": [6] * 31,
            "estimable": [False] * 31,
            "status": ["fit_failed"] * 31,
            "reason": ["fit failed"] * 31,
        }
    )
    successful = _variance_results().iloc[[0]].copy()
    successful["feature_id"] = "Z_SUCCESS"
    successful["feature_label"] = "Z_SUCCESS"
    variance = pd.concat((failed, successful), ignore_index=True)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        MixedModelPlotter(
            output_dir=tmp_path,
            dpi=450,
        ).plot_mixed_model_variance(
            variance,
            analysis="variance_decomposition",
            filename="prioritized_variance",
            max_features=1,
        )

        labels = [
            label.get_text() for label in figures[0].axes[0].get_yticklabels()
        ]
        assert len(labels) == 1
        assert "Z SUCCESS" in labels[0]
    finally:
        for figure in figures:
            close_figure(figure)


def test_effect_plot_rejects_nonpositive_row_height(tmp_path: Path) -> None:
    """A nonpositive effect-row height is rejected before rendering."""
    with pytest.raises(ValueError, match="row_height"):
        MixedModelPlotter(
            output_dir=tmp_path,
            dpi=450,
        ).plot_mixed_model_effects(
            _effect_results(),
            estimand="condition_effect_by_cell_type",
            filename="invalid_effect_geometry",
            row_height=0.0,
        )


def test_variance_plot_rejects_oversized_bar_height(tmp_path: Path) -> None:
    """A variance bar taller than its row is rejected before rendering."""
    with pytest.raises(ValueError, match="bar_height"):
        MixedModelPlotter(
            output_dir=tmp_path,
            dpi=450,
        ).plot_mixed_model_variance(
            _variance_results(),
            analysis="variance_decomposition",
            filename="invalid_variance_geometry",
            bar_height=1.1,
        )
