"""Tests for mechanistic NASP table visualizations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nasp_atlas.analysis import plot_global_nasp_visualizations
from nasp_atlas.analysis import plot_nasp_association_visualizations
from nasp_atlas.single_cell.visualization import SCVisualizer


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
    stability = pd.DataFrame(
        {
            "feature_label": ["NASP_DNA_SENSING", "IFN_I_OUTPUT"] * 2,
            "analysis_scope": ["within_tissue"] * 2
            + ["within_tissue_cell_type"] * 2,
            "median_slope": [0.04, -0.02, 0.035, -0.015],
            "min_slope": [0.02, -0.05, 0.01, -0.04],
            "max_slope": [0.06, 0.01, 0.06, 0.02],
            "direction_consistency_fraction": [1.0, 0.75, 0.75, 0.5],
            "n_significant_strata": [2, 1, 3, 1],
            "meets_min_tested_strata": [True] * 4,
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


def test_visualizer_writes_module_coupling_heatmap(tmp_path: Path) -> None:
    """A supported coupling table produces a symmetric heatmap."""
    tables = _visualization_tables()
    visualizer = SCVisualizer(tmp_path)

    visualizer.plot_module_coupling_heatmap(
        tables["module_coupling"],
        filename="coupling",
        facet_column="tissue",
    )

    assert (tmp_path / "coupling.png").stat().st_size > 0


def test_visualizer_writes_competence_output_state_map(tmp_path: Path) -> None:
    """Supported contexts produce a donor-scaled competence/output map."""
    tables = _visualization_tables()
    visualizer = SCVisualizer(tmp_path)

    visualizer.plot_competence_output_state_map(
        tables["context_summary"],
        filename="states",
        label_columns=["cell_type"],
        facet_column="tissue",
    )

    assert (tmp_path / "states.png").stat().st_size > 0


def test_visualizer_writes_ranked_hypothesis_panels(tmp_path: Path) -> None:
    """Ranked context hypotheses produce one panel per hypothesis."""
    tables = _visualization_tables()
    visualizer = SCVisualizer(tmp_path)

    visualizer.plot_ranked_nasp_hypotheses(
        tables["hypothesis_priorities"],
        filename="hypotheses",
        label_columns=["tissue", "cell_type"],
    )

    assert (tmp_path / "hypotheses.png").stat().st_size > 0


def test_visualizer_writes_sensor_output_mismatch_map(tmp_path: Path) -> None:
    """Sensor/output coupling produces an FDR-sized dot matrix."""
    tables = _visualization_tables()
    visualizer = SCVisualizer(tmp_path)

    visualizer.plot_sensor_output_mismatch(
        tables["sensor_output_coupling"],
        filename="sensor_output",
        facet_column="tissue",
    )

    assert (tmp_path / "sensor_output.png").stat().st_size > 0


def test_visualizer_writes_age_effect_figures(tmp_path: Path) -> None:
    """Age regressions and stability produce dot and range figures."""
    tables = _visualization_tables()
    visualizer = SCVisualizer(tmp_path)

    visualizer.plot_age_effect_dotplot(
        tables["regression_results"],
        filename="age_dots",
    )
    visualizer.plot_age_effect_consistency(
        tables["age_stability"],
        filename="age_consistency",
    )

    assert (tmp_path / "age_dots.png").stat().st_size > 0
    assert (tmp_path / "age_consistency.png").stat().st_size > 0


def test_visualizer_writes_mechanistic_edge_network(tmp_path: Path) -> None:
    """Expected edge annotations produce a correlation-overlay network."""
    tables = _visualization_tables()
    visualizer = SCVisualizer(tmp_path)

    visualizer.plot_mechanistic_edge_network(
        tables["mechanistic_edges"],
        filename="network",
        facet_column="tissue",
    )

    assert (tmp_path / "network.png").stat().st_size > 0


def test_visualization_orchestrators_write_tissue_and_global_sets(
    tmp_path: Path,
) -> None:
    """Per-tissue and global helpers emit their complete fixed figure sets."""
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

    tissue_plots = list(tissue_dir.glob("*.png"))
    global_plots = list(global_dir.glob("*.png"))
    assert len(tissue_plots) == 7
    assert len(global_plots) == 7
    assert all(path.stat().st_size > 0 for path in tissue_plots)
    assert all(path.stat().st_size > 0 for path in global_plots)
