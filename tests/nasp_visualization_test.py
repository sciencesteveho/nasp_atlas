"""Tests for mechanistic NASP table visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nasp_atlas.single_cell.visualization import NaspPlotter


def test_mechanistic_edge_barplot_maps_correlations_to_edge_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Each edge label receives its median plotted Spearman correlation."""
    edges = pd.DataFrame(
        [
            {
                "source_module": "NASP_DNA_SENSING",
                "target_module": "IFN_I_OUTPUT",
                "analysis": "within_context_centered",
                "spearman_r": correlation,
            }
            for correlation in (0.2, 0.6)
        ]
        + [
            {
                "source_module": "IFN_I_OUTPUT",
                "target_module": "NASP_FEEDBACK",
                "analysis": "within_context_centered",
                "spearman_r": -0.3,
            }
        ]
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_mechanistic_edge_barplot(
            edges,
            filename="mechanistic_edge_bars",
        )

        ax = figures[0].axes[0]
        tick_positions = ax.get_yticks()
        tick_labels = [label.get_text() for label in ax.get_yticklabels()]
        label_by_position = dict(zip(tick_positions, tick_labels, strict=True))
        plotted: dict[str, float] = {}
        for bar in ax.patches:
            center = bar.get_y() + bar.get_height() / 2.0
            position = min(tick_positions, key=lambda tick: abs(tick - center))
            assert np.isclose(center, position)
            plotted[label_by_position[position]] = bar.get_width()

        expected = {
            "DNA sensing → IFN-I output": 0.4,
            "IFN-I output → NASP feedback": -0.3,
        }
        assert set(plotted) == set(expected)
        for edge_label, correlation in expected.items():
            assert plotted[edge_label] == pytest.approx(correlation)
    finally:
        for figure in figures:
            close_figure(figure)


def test_mechanistic_edge_barplot_retains_unestimable_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Edges without finite correlation remain explicitly labeled."""
    edges = pd.DataFrame(
        [
            {
                "source_module": "NASP_DNA_SENSING",
                "target_module": "IFN_I_OUTPUT",
                "analysis": "within_context_centered",
                "spearman_r": np.nan,
            },
            {
                "source_module": "IFN_I_OUTPUT",
                "target_module": "NASP_FEEDBACK",
                "analysis": "within_context_centered",
                "spearman_r": 0.4,
            },
        ]
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_mechanistic_edge_barplot(
            edges,
            filename="mechanistic_edge_missing",
        )

        ax = figures[0].axes[0]
        edge_labels = {label.get_text() for label in ax.get_yticklabels()}
        assert "DNA sensing → IFN-I output" in edge_labels
        assert any(text.get_text() == "not estimable" for text in ax.texts)
    finally:
        for figure in figures:
            close_figure(figure)
