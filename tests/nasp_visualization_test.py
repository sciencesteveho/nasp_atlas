"""Tests for mechanistic NASP table visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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


def test_plotter_writes_ranked_hypothesis_panels(tmp_path: Path) -> None:
    """The hypothesis report renders on representative supported contexts."""
    NaspPlotter(tmp_path).plot_ranked_nasp_hypotheses(
        _visualization_tables()["hypothesis_priorities"],
        filename="hypotheses",
        label_columns=["tissue", "cell_type"],
    )

    assert (tmp_path / "hypotheses.png").stat().st_size > 0


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

    with pytest.warns(UserWarning):
        NaspPlotter(tmp_path).plot_mechanistic_edge_network(
            edges,
            filename="adjacent_layer_network",
            max_layer_span=1,
        )

    assert (tmp_path / "adjacent_layer_network.png").stat().st_size > 0
