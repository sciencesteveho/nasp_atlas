"""Tests for mixed-model contrast and variance visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def test_effect_plot_preserves_estimates_and_intervals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Plotted contrasts retain estimates and CIs, excluding failed fits."""
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

        assert (tmp_path / "condition_effects.png").stat().st_size > 0
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
        widths = np.array([patch.get_width() for patch in axis.patches])
        for fraction in (0.0, 0.2, 0.3, 0.7, 0.8):
            assert np.isclose(widths, fraction).any()

        annotations = [text.get_text().lower() for text in axis.texts]
        assert any(
            "missing" in text and "study" in text for text in annotations
        )
        assert not any(
            "missing" in text and "donor" in text for text in annotations
        )
        assert (tmp_path / "variance_components.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)
