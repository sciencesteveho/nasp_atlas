"""Tests for mechanistic NASP table visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import nasp_atlas.single_cell.visualization.nasp as nasp_visualization
from nasp_atlas.analysis import plot_global_nasp_visualizations
from nasp_atlas.analysis import plot_nasp_association_visualizations
from nasp_atlas.analysis.tabula_sapiens import (
    visualizations as tabula_visualizations,
)
from nasp_atlas.single_cell.visualization import NaspPlotter


def _visualization_tables() -> dict[str, pd.DataFrame]:
    """Build supported one- and two-tissue visualization inputs."""
    modules = ["NASP_DNA_SENSING", "IFN_I_OUTPUT", "NASP_FEEDBACK"]
    coupling_records: list[dict[str, object]] = []
    edge_records: list[dict[str, object]] = []
    for tissue, offset in (("liver", 0.0), ("lung", -0.15)):
        pairs = [
            (modules[0], modules[1], 0.75 + offset, 0.01),
            (modules[0], modules[2], 0.25 + offset, 0.20),
            (modules[1], modules[2], 0.55 + offset, 0.04),
        ]
        for module_a, module_b, correlation, fdr in pairs:
            coupling_records.append(
                {
                    "tissue": tissue,
                    "module_a": module_a,
                    "module_b": module_b,
                    "analysis": "within_context_centered",
                    "n_units": 8,
                    "spearman_r": correlation,
                    "spearman_fdr": fdr,
                    "skipped": False,
                }
            )
        edge_records.extend(
            [
                {
                    "tissue": tissue,
                    "source_module": modules[0],
                    "target_module": modules[1],
                    "analysis": "within_context_centered",
                    "spearman_r": 0.75 + offset,
                    "spearman_fdr": 0.01,
                    "skipped": False,
                },
                {
                    "tissue": tissue,
                    "source_module": modules[1],
                    "target_module": modules[2],
                    "analysis": "within_context_centered",
                    "spearman_r": 0.55 + offset,
                    "spearman_fdr": 0.04,
                    "skipped": False,
                },
            ]
        )

    contexts = pd.DataFrame(
        {
            "tissue": ["liver", "liver", "lung", "lung"],
            "cell_type": ["macrophage", "fibroblast"] * 2,
            "relative_competence": [0.8, 0.3, 0.6, 0.2],
            "relative_output": [0.7, 0.8, 0.4, 0.75],
            "relative_restriction": [0.5, 0.4, 0.7, 0.3],
            "relative_feedback": [0.6, 0.5, 0.6, 0.4],
            "relative_post": [0.4, 0.9, 0.3, 0.8],
            "output_minus_competence_gap": [-0.1, 0.5, -0.2, 0.55],
            "n_units": [6, 5, 6, 4],
            "n_donors": [4, 3, 4, 3],
        }
    )
    hypotheses = pd.DataFrame(
        {
            "tissue": ["liver", "liver", "lung", "lung", "liver"],
            "cell_type": [
                "macrophage",
                "fibroblast",
                "macrophage",
                "fibroblast",
                "fibroblast",
            ],
            "hypothesis": [
                "active_like",
                "responsive_like",
                "restricted_buffered",
                "post_without_nasp",
                "feedback_dominant",
            ],
            "priority_score": [0.72, 0.44, 0.36, 0.48, 0.31],
        }
    )
    sensor_records: list[dict[str, object]] = []
    for tissue, offset in (("liver", 0.0), ("lung", -0.1)):
        for gene, output, correlation, fdr in (
            ("CGAS", "IFN_I_OUTPUT", 0.72, 0.01),
            ("IFIH1", "IFN_I_OUTPUT", 0.48, 0.04),
            ("DDX58", "NFKB_CYTOKINE_OUTPUT", -0.35, 0.20),
        ):
            sensor_records.append(
                {
                    "tissue": tissue,
                    "gene": gene,
                    "output_module": output,
                    "analysis": "within_context_centered",
                    "spearman_r": correlation + offset,
                    "spearman_fdr": fdr,
                    "gene_in_output_module": False,
                }
            )

    regression_records: list[dict[str, object]] = []
    for tissue in ("liver", "lung"):
        for cell_type, shift in (("macrophage", 0.02), ("fibroblast", -0.01)):
            for module, slope, fdr in (
                ("NASP_DNA_SENSING", 0.04 + shift, 0.02),
                ("IFN_I_OUTPUT", -0.03 + shift, 0.08),
            ):
                regression_records.append(
                    {
                        "feature_type": "module_score",
                        "feature_label": module,
                        "stratum": f"{tissue}::{cell_type}",
                        "analysis_scope": "within_tissue_cell_type",
                        "slope": slope,
                        "ols_pvalue_fdr": fdr,
                        "skipped": False,
                    }
                )
            for gene, slope, fdr in (
                ("CGAS", 0.01 + shift, 0.03),
                ("IFIH1", -0.02 + shift, 0.12),
            ):
                regression_records.append(
                    {
                        "feature_type": "gene_expression",
                        "feature_label": gene,
                        "stratum": f"{tissue}::{cell_type}",
                        "analysis_scope": "within_tissue_cell_type",
                        "slope": slope,
                        "ols_pvalue_fdr": fdr,
                        "skipped": False,
                    }
                )
    stability = pd.DataFrame(
        {
            "feature_type": (["module_score"] * 2 + ["gene_expression"] * 2)
            * 2,
            "feature_label": [
                "NASP_DNA_SENSING",
                "IFN_I_OUTPUT",
                "CGAS",
                "IFIH1",
            ]
            * 2,
            "analysis_scope": ["within_tissue"] * 4
            + ["within_tissue_cell_type"] * 4,
            "median_slope": [
                0.04,
                -0.02,
                0.015,
                -0.01,
                0.035,
                -0.015,
                0.012,
                -0.008,
            ],
            "min_slope": [
                0.02,
                -0.05,
                -0.001,
                -0.025,
                0.01,
                -0.04,
                -0.002,
                -0.02,
            ],
            "max_slope": [
                0.06,
                0.01,
                0.03,
                0.005,
                0.06,
                0.02,
                0.025,
                0.004,
            ],
            "direction_consistency_fraction": [
                1.0,
                0.75,
                0.9,
                0.8,
                0.75,
                0.5,
                0.8,
                0.7,
            ],
            "n_significant_strata": [2, 1, 2, 1, 3, 1, 2, 1],
            "meets_min_tested_strata": [True] * 8,
        }
    )
    return {
        "module_coupling": pd.DataFrame.from_records(coupling_records),
        "context_summary": contexts,
        "hypothesis_priorities": hypotheses,
        "sensor_output_coupling": pd.DataFrame.from_records(sensor_records),
        "regression_results": pd.DataFrame.from_records(regression_records),
        "age_stability": stability,
        "mechanistic_edges": pd.DataFrame.from_records(edge_records),
    }


def test_plotter_writes_module_coupling_heatmap(tmp_path: Path) -> None:
    """A supported coupling table produces a symmetric heatmap."""
    tables = _visualization_tables()
    plotter = NaspPlotter(tmp_path)

    plotter.plot_module_coupling_heatmap(
        tables["module_coupling"],
        filename="coupling",
        facet_column="tissue",
    )

    assert (tmp_path / "coupling.png").stat().st_size > 0


def test_module_coupling_heatmap_uses_requested_title(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A caller title replaces the automatic analysis-qualified title."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_module_coupling_heatmap(
            _visualization_tables()["module_coupling"],
            filename="titled_coupling",
            title="Module coupling",
        )

        assert figures[0].axes[0].get_title() == "Module coupling"
    finally:
        for figure in figures:
            close_figure(figure)


def test_plotter_writes_competence_output_state_map(tmp_path: Path) -> None:
    """Supported contexts produce a donor-scaled competence/output map."""
    tables = _visualization_tables()
    plotter = NaspPlotter(tmp_path)

    plotter.plot_competence_output_state_map(
        tables["context_summary"],
        filename="states",
        label_columns=["cell_type"],
        facet_column="tissue",
    )

    assert (tmp_path / "states.png").stat().st_size > 0


def test_competence_state_map_can_title_panels_by_tissue_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Suppressing the shared title leaves each tissue facet as its title."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_competence_output_state_map(
            _visualization_tables()["context_summary"],
            filename="tissue_titled_states",
            label_columns=["cell_type"],
            facet_column="tissue",
            title=None,
            adjust_labels=False,
        )

        titles = [axis.get_title() for axis in figures[0].axes[:2]]
        assert titles == ["liver", "lung"]
    finally:
        for figure in figures:
            close_figure(figure)


def test_competence_state_map_scales_context_panels_and_tissue_facets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Default state-map geometry scales from the tuned Liver panel."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    def contexts(tissues: tuple[str, ...], per_tissue: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "tissue": tissue,
                    "cell_type": f"cell_{index:02d}",
                    "relative_competence": index / per_tissue,
                    "relative_output": (per_tissue - index) / per_tissue,
                    "output_minus_competence_gap": 0.0,
                    "n_donors": 3,
                }
                for tissue in tissues
                for index in range(per_tissue)
            ]
        )

    try:
        NaspPlotter(tmp_path).plot_competence_output_state_map(
            contexts(("liver",), 24),
            filename="single_tissue_states",
            label_columns=["cell_type"],
            facet_column="tissue",
            adjust_labels=False,
        )
        NaspPlotter(tmp_path).plot_competence_output_state_map(
            contexts(("liver", "lung"), 30),
            filename="multi_tissue_states",
            label_columns=["cell_type"],
            facet_column="tissue",
            adjust_labels=False,
        )

        np.testing.assert_allclose(figures[0].get_size_inches(), (3.2, 3.2))
        panel_side = 3.2 * np.sqrt(30 / 24)
        np.testing.assert_allclose(
            figures[1].get_size_inches(),
            (2 * panel_side, panel_side),
        )
    finally:
        for figure in figures:
            close_figure(figure)


def test_competence_state_map_scales_support_to_requested_size_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Observed donor support spans the caller's point-area range."""
    tables = _visualization_tables()
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_competence_output_state_map(
            tables["context_summary"],
            filename="states_with_sizes",
            label_columns=["cell_type"],
            support_size_range=(10.0, 30.0),
        )

        sizes = figures[0].axes[0].collections[0].get_sizes()
        np.testing.assert_allclose([sizes.min(), sizes.max()], [10.0, 30.0])
    finally:
        for figure in figures:
            close_figure(figure)


def test_competence_state_map_forwards_label_adjustment_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Caller-selected repulsion controls configure adjusted labels."""
    adjustment: dict[str, object] = {}

    def capture_adjustment(texts, **kwargs) -> None:
        """Record the external label-adjustment request."""
        adjustment.update(kwargs)

    monkeypatch.setattr(nasp_visualization, "adjust_text", capture_adjustment)

    NaspPlotter(tmp_path).plot_competence_output_state_map(
        _visualization_tables()["context_summary"],
        filename="states_with_adjustment",
        label_columns=["cell_type"],
        label_force=(0.3, 0.4),
        label_static_force=(0.5, 0.6),
        label_explode_force=(0.7, 0.8),
        label_expand=(1.2, 1.4),
        label_max_move=(4, 6),
    )

    assert adjustment["force_text"] == (0.3, 0.4)
    assert adjustment["force_static"] == (0.5, 0.6)
    assert adjustment["force_explode"] == (0.7, 0.8)
    assert adjustment["expand"] == (1.2, 1.4)
    assert adjustment["max_move"] == (4, 6)


def test_plotter_writes_ranked_hypothesis_panels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Ranked panels retain scientific titles and show scoring formulas."""
    tables = _visualization_tables()
    plotter = NaspPlotter(tmp_path)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        plotter.plot_ranked_nasp_hypotheses(
            tables["hypothesis_priorities"],
            filename="hypotheses",
            label_columns=["tissue", "cell_type"],
        )

        assert (tmp_path / "hypotheses.png").stat().st_size > 0
        axes = figures[0].axes
        assert [ax.get_title() for ax in axes[:5]] == [
            "Active-like",
            "Responsive-like",
            "Restricted/buffered",
            "Post without NASP",
            "Feedback-dominant",
        ]
        assert [ax.get_xlabel() for ax in axes[:5]] == [
            "min(competence, output)",
            "max(output - competence, 0) * output",
            ("max(restriction - output, 0)\n* mean(restriction, competence)"),
            "max(post - max(competence, output), 0) * post",
            "max(feedback - output, 0) * feedback",
        ]
    finally:
        for figure in figures:
            close_figure(figure)


def test_ranked_hypothesis_panel_uses_custom_priority_basis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A caller-defined hypothesis labels its plotted priority quantity."""
    priorities = pd.DataFrame(
        {
            "context": ["macrophage"],
            "hypothesis": ["custom_response"],
            "priority_score": [0.6],
            "priority_basis": ["caller-defined priority"],
        }
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_ranked_nasp_hypotheses(
            priorities,
            filename="custom_hypothesis",
            label_columns=["context"],
        )

        assert figures[0].axes[0].get_xlabel() == "caller-defined priority"
    finally:
        for figure in figures:
            close_figure(figure)


def test_plotter_writes_sensor_output_mismatch_map(tmp_path: Path) -> None:
    """Sensor/output coupling produces an FDR-sized dot matrix."""
    tables = _visualization_tables()
    plotter = NaspPlotter(tmp_path)

    plotter.plot_sensor_output_mismatch(
        tables["sensor_output_coupling"],
        filename="sensor_output",
        facet_column="tissue",
    )

    assert (tmp_path / "sensor_output.png").stat().st_size > 0


def test_sensor_output_mismatch_scales_requested_axis_spacing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dot-center distances follow the requested independent axis spacing."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_sensor_output_mismatch(
            _visualization_tables()["sensor_output_coupling"],
            filename="default_sensor_output",
            column_spacing=1.0,
            row_spacing=1.0,
        )
        NaspPlotter(tmp_path).plot_sensor_output_mismatch(
            _visualization_tables()["sensor_output_coupling"],
            filename="aligned_sensor_output",
            column_spacing=0.5,
            row_spacing=0.6,
        )

        default_ax = figures[0].axes[0]
        ax = figures[1].axes[0]
        default_distance = (
            default_ax.transData.transform((1.0, 0.0))[0]
            - default_ax.transData.transform((0.0, 0.0))[0]
        )
        compact_distance = (
            ax.transData.transform((1.0, 0.0))[0]
            - ax.transData.transform((0.0, 0.0))[0]
        )
        default_row_distance = abs(
            default_ax.transData.transform((0.0, 1.0))[1]
            - default_ax.transData.transform((0.0, 0.0))[1]
        )
        compact_row_distance = abs(
            ax.transData.transform((0.0, 1.0))[1]
            - ax.transData.transform((0.0, 0.0))[1]
        )
        assert compact_distance < default_distance
        np.testing.assert_allclose(
            compact_distance / default_distance,
            0.5,
            rtol=0.05,
        )
        assert compact_row_distance < default_row_distance
        np.testing.assert_allclose(
            compact_row_distance / default_row_distance,
            0.6,
            rtol=0.05,
        )
    finally:
        for figure in figures:
            close_figure(figure)


def test_sensor_output_mismatch_honors_requested_maximum_dot_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The strongest FDR evidence uses the requested maximum marker area."""
    coupling = _visualization_tables()["sensor_output_coupling"].copy()
    coupling.loc[coupling["gene"] == "CGAS", "spearman_fdr"] = 1e-8
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_sensor_output_mismatch(
            coupling,
            filename="sized_sensor_output",
            max_dot_size=20.0,
        )

        sizes = figures[0].axes[0].collections[0].get_sizes()
        assert sizes.max() == 20.0
    finally:
        for figure in figures:
            close_figure(figure)


def test_plotter_writes_age_effect_figures(tmp_path: Path) -> None:
    """Age regressions and stability produce dot and range figures."""
    tables = _visualization_tables()
    plotter = NaspPlotter(tmp_path)

    plotter.plot_age_effect_dotplot(
        tables["regression_results"],
        filename="module_age_dots",
        feature_type="module_score",
    )
    plotter.plot_age_effect_dotplot(
        tables["regression_results"],
        filename="sensor_age_dots",
        feature_type="gene_expression",
        feature_labels=["CGAS", "IFIH1"],
    )
    plotter.plot_age_effect_consistency(
        tables["age_stability"],
        filename="module_age_consistency",
        analysis_scope="within_tissue_cell_type",
        feature_type="module_score",
    )
    plotter.plot_age_effect_consistency(
        tables["age_stability"],
        filename="sensor_age_consistency",
        analysis_scope="within_tissue_cell_type",
        feature_type="gene_expression",
        feature_labels=["CGAS", "IFIH1"],
    )

    assert (tmp_path / "module_age_dots.png").stat().st_size > 0
    assert (tmp_path / "sensor_age_dots.png").stat().st_size > 0
    assert (tmp_path / "module_age_consistency.png").stat().st_size > 0
    assert (tmp_path / "sensor_age_consistency.png").stat().st_size > 0


def test_age_effect_dotplot_default_size_scales_with_matrix_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Default dimensions preserve the tuned pitch as contexts increase."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    def regression_grid(size: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature_type": "module_score",
                    "feature_label": f"MODULE_{feature_index:02d}",
                    "stratum": f"tissue::cell_{context_index:02d}",
                    "analysis_scope": "within_tissue_cell_type",
                    "slope": 0.01,
                    "ols_pvalue_fdr": 0.05,
                }
                for context_index in range(size)
                for feature_index in range(size)
            ]
        )

    try:
        for size in (10, 20):
            NaspPlotter(tmp_path).plot_age_effect_dotplot(
                regression_grid(size),
                filename=f"age_effects_{size}",
            )

        np.testing.assert_allclose(figures[0].get_size_inches(), (2.0, 2.4))
        np.testing.assert_allclose(figures[1].get_size_inches(), (4.0, 4.8))
    finally:
        for figure in figures:
            close_figure(figure)


def test_age_effect_dotplot_scales_requested_axis_spacing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dot-center distances follow requested independent axis spacing."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_age_effect_dotplot(
            _visualization_tables()["regression_results"],
            filename="default_age_effect_spacing",
            column_spacing=1.0,
            row_spacing=1.0,
            figsize=(4.0, 4.0),
        )
        NaspPlotter(tmp_path).plot_age_effect_dotplot(
            _visualization_tables()["regression_results"],
            filename="compact_age_effect_spacing",
            column_spacing=0.5,
            row_spacing=0.6,
            figsize=(4.0, 4.0),
        )

        default_ax = figures[0].axes[0]
        compact_ax = figures[1].axes[0]
        default_column_distance = (
            default_ax.transData.transform((1.0, 0.0))[0]
            - default_ax.transData.transform((0.0, 0.0))[0]
        )
        compact_column_distance = (
            compact_ax.transData.transform((1.0, 0.0))[0]
            - compact_ax.transData.transform((0.0, 0.0))[0]
        )
        default_row_distance = abs(
            default_ax.transData.transform((0.0, 1.0))[1]
            - default_ax.transData.transform((0.0, 0.0))[1]
        )
        compact_row_distance = abs(
            compact_ax.transData.transform((0.0, 1.0))[1]
            - compact_ax.transData.transform((0.0, 0.0))[1]
        )
        np.testing.assert_allclose(
            compact_column_distance / default_column_distance,
            0.5,
        )
        np.testing.assert_allclose(
            compact_row_distance / default_row_distance,
            0.6,
        )
    finally:
        for figure in figures:
            close_figure(figure)


def test_age_effect_dotplot_applies_requested_maximum_dot_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Strongest FDR evidence uses caller-selected maximum marker area."""
    regressions = _visualization_tables()["regression_results"].copy()
    regressions.loc[
        regressions["feature_label"] == "NASP_DNA_SENSING",
        "ols_pvalue_fdr",
    ] = 1e-8
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_age_effect_dotplot(
            regressions,
            filename="sized_age_effects",
            max_dot_size=20.0,
        )

        sizes = figures[0].axes[0].collections[0].get_sizes()
        assert sizes.max() == 20.0
    finally:
        for figure in figures:
            close_figure(figure)


def test_age_effect_dotplot_resizes_colorbar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Larger colorbar controls produce a larger rendered colorbar."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        for label, height, width in (
            ("small", "30%", "4%"),
            ("large", "70%", "10%"),
        ):
            NaspPlotter(tmp_path).plot_age_effect_dotplot(
                _visualization_tables()["regression_results"],
                filename=f"{label}_age_effect_colorbar",
                cbar_height=height,
                cbar_width=width,
                figsize=(4.0, 4.0),
            )

        for figure in figures:
            figure.canvas.draw()
        small_bounds = figures[0].axes[-1].get_window_extent()
        large_bounds = figures[1].axes[-1].get_window_extent()
        assert large_bounds.height > small_bounds.height
        assert large_bounds.width > small_bounds.width
    finally:
        for figure in figures:
            close_figure(figure)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("column_spacing", 0.0, "column_spacing"),
        ("row_spacing", -1.0, "row_spacing"),
        ("tick_label_pad", -0.1, "tick_label_pad"),
        ("max_dot_size", 6.0, "max_dot_size"),
    ],
)
def test_age_effect_dotplot_rejects_invalid_sizing_controls(
    tmp_path: Path,
    parameter: str,
    value: float,
    message: str,
) -> None:
    """Age-effect sizing controls reject nonpositive or undersized values."""
    with pytest.raises(ValueError, match=message):
        NaspPlotter(tmp_path).plot_age_effect_dotplot(
            _visualization_tables()["regression_results"],
            filename="invalid_age_effect_sizing",
            **{parameter: value},
        )


@pytest.mark.parametrize(
    ("feature_type", "feature_labels", "expected_labels"),
    [
        (
            "module_score",
            ["SIGNALING_CONTEXT_TLR", "INFLAMMASOME", "NASP_DNA_SENSING"],
            ["DNA sensing", "Inflammasome", "Signaling context TLR"],
        ),
        (
            "gene_expression",
            ["ZBP1", "CGAS", "IFIH1"],
            ["CGAS", "IFIH1", "ZBP1"],
        ),
    ],
)
def test_age_effect_dotplot_alphabetizes_displayed_features(
    tmp_path: Path,
    monkeypatch,
    feature_type: str,
    feature_labels: list[str],
    expected_labels: list[str],
) -> None:
    """Module and gene columns are alphabetical by their displayed labels."""
    regressions = pd.DataFrame(
        {
            "feature_type": [feature_type] * 3,
            "feature_label": feature_labels,
            "stratum": ["liver::macrophage"] * 3,
            "analysis_scope": ["within_tissue_cell_type"] * 3,
            "slope": [-0.02, 0.03, 0.01],
            "ols_pvalue_fdr": [0.01, 0.02, 0.03],
        }
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_age_effect_dotplot(
            regressions,
            filename=f"alphabetical_{feature_type}_age_effects",
            feature_type=feature_type,
        )

        labels = [
            label.get_text() for label in figures[0].axes[0].get_xticklabels()
        ]
        assert labels == expected_labels
    finally:
        for figure in figures:
            close_figure(figure)


@pytest.mark.parametrize(
    ("feature_type", "feature_labels", "expected_labels"),
    [
        (
            "module_score",
            ["SIGNALING_CONTEXT_TLR", "INFLAMMASOME", "NASP_DNA_SENSING"],
            ["Inflammasome", "DNA sensing", "Signaling context TLR"],
        ),
        (
            "gene_expression",
            ["ZBP1", "CGAS", "IFIH1"],
            ["CGAS", "IFIH1", "ZBP1"],
        ),
    ],
)
def test_age_effect_consistency_orders_by_signed_median_slope(
    tmp_path: Path,
    monkeypatch,
    feature_type: str,
    feature_labels: list[str],
    expected_labels: list[str],
) -> None:
    """Rows descend from the rightmost to leftmost median age effect."""
    stability = pd.DataFrame(
        {
            "feature_type": [feature_type] * 3,
            "feature_label": feature_labels,
            "analysis_scope": ["within_tissue"] * 3,
            "median_slope": [-0.02, 0.03, 0.01],
            "min_slope": [-0.04, 0.01, -0.01],
            "max_slope": [-0.01, 0.05, 0.03],
            "direction_consistency_fraction": [0.8, 1.0, 0.7],
            "n_significant_strata": [2, 3, 1],
        }
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_age_effect_consistency(
            stability,
            filename=f"ordered_{feature_type}_age_consistency",
            feature_type=feature_type,
        )

        ax = figures[0].axes[0]
        labels = [label.get_text() for label in ax.get_yticklabels()]
        median_slopes = ax.collections[0].get_offsets()[:, 0]
        assert labels == expected_labels
        assert np.all(np.diff(median_slopes) <= 0.0)
        assert ax.yaxis_inverted()
    finally:
        for figure in figures:
            close_figure(figure)


def test_plotter_writes_mechanistic_edge_barplot(tmp_path: Path) -> None:
    """Curated edges produce a saved Spearman-correlation barplot."""
    NaspPlotter(tmp_path).plot_mechanistic_edge_barplot(
        _visualization_tables()["mechanistic_edges"],
        filename="mechanistic_edge_bars",
    )

    assert (tmp_path / "mechanistic_edge_bars.png").stat().st_size > 0


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


def test_mechanistic_edge_barplot_honors_requested_figure_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The barplot uses caller-selected physical dimensions."""
    requested_figsize = (4.1, 2.7)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_mechanistic_edge_barplot(
            _visualization_tables()["mechanistic_edges"],
            filename="sized_mechanistic_edge_bars",
            figsize=requested_figsize,
        )

        np.testing.assert_allclose(
            figures[0].get_size_inches(),
            requested_figsize,
        )
    finally:
        for figure in figures:
            close_figure(figure)


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"row_spacing": 0.0}, "row_spacing must be a positive finite"),
        ({"bar_height": 0.0}, "bar_height must be a positive finite"),
        (
            {"row_spacing": 0.5, "bar_height": 0.6},
            "bar_height must not exceed row_spacing",
        ),
        (
            {"tick_label_pad": -1.0},
            "tick_label_pad must be a nonnegative finite",
        ),
        (
            {"figsize": (np.inf, 3.0)},
            "figsize must contain two positive finite",
        ),
    ],
)
def test_mechanistic_edge_barplot_rejects_invalid_geometry(
    tmp_path: Path,
    geometry: dict[str, object],
    message: str,
) -> None:
    """Invalid bar geometry fails with an actionable parameter name."""
    with pytest.raises(ValueError, match=message):
        NaspPlotter(tmp_path).plot_mechanistic_edge_barplot(
            _visualization_tables()["mechanistic_edges"],
            filename="invalid_mechanistic_edge_bars",
            **geometry,
        )


def test_mechanistic_edge_barplot_rejects_invalid_correlations(
    tmp_path: Path,
) -> None:
    """Spearman values outside their valid range fail before plotting."""
    edges = _visualization_tables()["mechanistic_edges"].copy()
    edges["spearman_r"] = 1.2

    with pytest.raises(ValueError, match="between -1 and 1"):
        NaspPlotter(tmp_path).plot_mechanistic_edge_barplot(
            edges,
            filename="invalid_mechanistic_edge_correlations",
        )


def test_plotter_writes_mechanistic_edge_network(tmp_path: Path) -> None:
    """Expected edge annotations produce a correlation-overlay network."""
    tables = _visualization_tables()
    plotter = NaspPlotter(tmp_path)

    plotter.plot_mechanistic_edge_network(
        tables["mechanistic_edges"],
        filename="network",
        facet_column="tissue",
    )

    assert (tmp_path / "network.png").stat().st_size > 0


def test_mechanistic_edge_network_defaults_render_full_topology(
    tmp_path: Path,
) -> None:
    """Default geometry renders the full expected topology without clipping."""
    pairs = [
        ("NASP_DNA_SENSING", "SIGNALING_CONTEXT_TBK1_IRF"),
        ("NASP_RNA_SENSING", "SIGNALING_CONTEXT_TBK1_IRF"),
        ("NASP_DNA_SENSING", "IFN_I_OUTPUT"),
        ("NASP_RNA_SENSING", "IFN_I_OUTPUT"),
        ("NASP_DNA_SENSING", "NFKB_CYTOKINE_OUTPUT"),
        ("NASP_RNA_SENSING", "NFKB_CYTOKINE_OUTPUT"),
        ("SIGNALING_CONTEXT_TBK1_IRF", "IFN_I_OUTPUT"),
        ("SIGNALING_CONTEXT_TLR", "NFKB_CYTOKINE_OUTPUT"),
        ("SIGNALING_CONTEXT_NFKB", "NFKB_CYTOKINE_OUTPUT"),
        ("SIGNALING_CONTEXT_IFN_JAK_STAT", "IFN_I_OUTPUT"),
        ("NASP_RNA_SENSING", "ISR"),
        ("NASP_DNA_SENSING", "INFLAMMASOME"),
        ("MITOCHONDRIAL_NA_SENSING", "NASP_DNA_SENSING"),
        ("MITOCHONDRIAL_NA_SENSING", "INFLAMMASOME"),
        ("TE_DEREPRESSION", "NASP_DNA_SENSING"),
        ("TE_DEREPRESSION", "NASP_RNA_SENSING"),
        ("CGAMP_TRANSPORT", "IFN_I_OUTPUT"),
        ("IFN_I_OUTPUT", "NASP_FEEDBACK"),
        ("NFKB_CYTOKINE_OUTPUT", "NASP_FEEDBACK"),
        ("IFN_I_OUTPUT", "INFLAMMAGING"),
        ("NFKB_CYTOKINE_OUTPUT", "SASP"),
    ]
    edges = pd.DataFrame(
        [
            {
                "source_module": source,
                "target_module": target,
                "analysis": "within_context_centered",
                "spearman_r": correlation,
                "spearman_fdr": 0.01,
            }
            for (source, target), correlation in zip(
                pairs,
                np.linspace(-0.8, 0.8, len(pairs)),
                strict=True,
            )
        ]
    )
    NaspPlotter(tmp_path).plot_mechanistic_edge_network(
        edges,
        filename="default_full_network",
    )

    assert (tmp_path / "default_full_network.png").stat().st_size > 0


def test_mechanistic_edge_network_honors_requested_figure_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The saved network uses caller-selected physical dimensions."""
    requested_figsize = (4.25, 3.75)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        NaspPlotter(tmp_path).plot_mechanistic_edge_network(
            _visualization_tables()["mechanistic_edges"],
            filename="compact_network",
            figsize=requested_figsize,
        )

        np.testing.assert_allclose(
            figures[0].get_size_inches(),
            requested_figsize,
        )
    finally:
        for figure in figures:
            close_figure(figure)


def test_mechanistic_edge_network_reports_layer_span_filtering(
    tmp_path: Path,
) -> None:
    """Layer-span filtering reports dropped edges and orphaned modules."""
    edges = pd.DataFrame(
        [
            {
                "source_module": "NASP_DNA_SENSING",
                "target_module": "SIGNALING_CONTEXT_TBK1_IRF",
                "analysis": "within_context_centered",
                "spearman_r": 0.7,
                "spearman_fdr": 0.01,
            },
            {
                "source_module": "MITOCHONDRIAL_NA_SENSING",
                "target_module": "INFLAMMASOME",
                "analysis": "within_context_centered",
                "spearman_r": 0.5,
                "spearman_fdr": 0.03,
            },
        ]
    )

    with pytest.warns(UserWarning) as caught:
        NaspPlotter(tmp_path).plot_mechanistic_edge_network(
            edges,
            filename="adjacent_layer_network",
            max_layer_span=1,
        )

    message = str(caught[0].message)
    assert "max_layer_span=1 dropped 1 of 2 edges" in message
    assert "modules no longer shown" in message
    assert (tmp_path / "adjacent_layer_network.png").stat().st_size > 0


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"node_size": (0.0, 0.1)}, "node_size must contain positive"),
        ({"label_gutter": 1.0}, "label_gutter must be in"),
        ({"max_layer_span": 0}, "max_layer_span must be at least 1"),
        ({"layer_spacing": 0.0}, "layer_spacing must be positive"),
        (
            {"within_layer_spacing": -1.0},
            "within_layer_spacing must be positive",
        ),
        (
            {"node_corner_radius": 0.14},
            "node_corner_radius must be between",
        ),
        (
            {"figsize": (0.0, 3.0)},
            "figsize must contain two positive finite",
        ),
        (
            {"figsize": (np.inf, 3.0)},
            "figsize must contain two positive finite",
        ),
    ],
)
def test_mechanistic_edge_network_rejects_invalid_geometry(
    tmp_path: Path,
    geometry: dict[str, object],
    message: str,
) -> None:
    """Invalid network geometry fails with an actionable parameter name."""
    with pytest.raises(ValueError, match=message):
        NaspPlotter(tmp_path).plot_mechanistic_edge_network(
            _visualization_tables()["mechanistic_edges"],
            filename="invalid_network_geometry",
            **geometry,
        )


def test_mechanistic_edge_network_rejects_boxes_that_clip_labels(
    tmp_path: Path,
) -> None:
    """Undersized nodes fail rather than silently clipping labels."""
    targets = [
        "CGAMP_TRANSPORT",
        "SIGNALING_CONTEXT_IFN_JAK_STAT",
        "SIGNALING_CONTEXT_NFKB",
        "SIGNALING_CONTEXT_TBK1_IRF",
        "SIGNALING_CONTEXT_TLR",
    ]
    edges = pd.DataFrame(
        [
            {
                "source_module": "NASP_DNA_SENSING",
                "target_module": target,
                "analysis": "within_context_centered",
                "spearman_r": 0.5,
                "spearman_fdr": 0.01,
            }
            for target in targets
        ]
    )

    with pytest.raises(ValueError) as caught:
        NaspPlotter(tmp_path).plot_mechanistic_edge_network(
            edges,
            filename="undersized_nodes",
            node_size=(0.265, 0.05),
        )

    message = str(caught.value)
    assert "requested node_size (0.265, 0.05)" in message
    assert "drawn node size (0.265, 0.05)" in message
    assert "too small to fit module label" in message


def test_visualization_orchestrators_write_tissue_and_global_sets(
    tmp_path: Path,
) -> None:
    """Per-tissue and global helpers emit their documented figure sets."""
    tables = _visualization_tables()
    tissue_dir = tmp_path / "tissue"
    global_dir = tmp_path / "global"
    liver_tables = {
        key: table.loc[table["tissue"] == "liver"].copy()
        if "tissue" in table
        else table.copy()
        for key, table in tables.items()
    }

    plot_nasp_association_visualizations(
        output_dir=tissue_dir,
        **liver_tables,
    )
    plot_global_nasp_visualizations(
        output_dir=global_dir,
        tissue_key="tissue",
        **tables,
    )

    expected_tissue_plots = {
        "nasp_module_coupling_heatmap.png",
        "nasp_competence_output_state_map.png",
        "nasp_ranked_hypotheses.png",
        "nasp_sensor_output_mismatch.png",
        "nasp_age_effects_by_cell_type.png",
        "nasp_sensor_age_effects_by_cell_type.png",
        "nasp_age_effect_consistency_across_cell_types.png",
        "nasp_sensor_age_effect_consistency_across_cell_types.png",
        "nasp_mechanistic_edge_network.png",
    }
    expected_global_plots = {
        "global_nasp_module_coupling_consensus.png",
        "global_nasp_competence_output_states.png",
        "global_nasp_ranked_hypotheses.png",
        "global_nasp_sensor_output_consensus.png",
        "global_nasp_age_effects.png",
        "global_nasp_sensor_age_effects.png",
        "global_nasp_age_effect_consistency.png",
        "global_nasp_sensor_age_effect_consistency.png",
        "global_nasp_mechanistic_edge_consensus.png",
    }
    tissue_plots = {path.name: path for path in tissue_dir.glob("*.png")}
    global_plots = {path.name: path for path in global_dir.glob("*.png")}
    assert expected_tissue_plots <= set(tissue_plots)
    assert expected_global_plots <= set(global_plots)
    assert all(
        tissue_plots[name].stat().st_size > 0 for name in expected_tissue_plots
    )
    assert all(
        global_plots[name].stat().st_size > 0 for name in expected_global_plots
    )


def test_association_visualizations_facet_complete_atlas_by_tissue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The score-modules call path recognizes and facets multi-tissue input."""
    captured: dict[str, object] = {}

    class CapturePlotter:
        def __init__(self, *, output_dir: str | Path) -> None:
            captured["output_dir"] = output_dir

        def plot_competence_output_state_map(
            self,
            contexts: pd.DataFrame,
            **kwargs: object,
        ) -> None:
            captured["contexts"] = contexts
            captured.update(kwargs)

    monkeypatch.setattr(
        tabula_visualizations,
        "NaspPlotter",
        CapturePlotter,
    )
    empty = pd.DataFrame()
    plot_nasp_association_visualizations(
        output_dir=tmp_path,
        module_coupling=empty,
        context_summary=_visualization_tables()["context_summary"],
        hypothesis_priorities=empty,
        sensor_output_coupling=empty,
        regression_results=empty,
        age_stability=empty,
        mechanistic_edges=empty,
        tissue_key="tissue",
    )

    assert captured["facet_column"] == "tissue"
    assert captured["label_columns"] == ["cell_type"]
