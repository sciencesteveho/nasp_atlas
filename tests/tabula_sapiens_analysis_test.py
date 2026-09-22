"""Tests for the Tabula Sapiens analysis workflow."""

from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path

import anndata as ad  # type: ignore[import]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from nasp_compendium import GeneModules  # type: ignore[import]
from nasp_compendium.types import GeneModule  # type: ignore[import]

from nasp_atlas.analysis.tabula_sapiens.compendium_symbols import (
    COMPENDIUM_SYMBOL_COLUMN,
)
from nasp_atlas.analysis.tabula_sapiens.compendium_symbols import (
    add_compendium_symbol_column,
)
from nasp_atlas.analysis.tabula_sapiens.scoring import (
    plot_reference_score_umaps,
)
from nasp_atlas.analysis.tabula_sapiens.scoring import (
    plot_scorer_concordance_by_source,
)
from nasp_atlas.single_cell.visualization import SummaryPlotter
from nasp_atlas.single_cell.visualization import UmapPlotter


tabula_sapiens = importlib.import_module("nasp_atlas.analysis.tabula_sapiens")
tabula_sapiens_workflows = importlib.import_module(
    "nasp_atlas.analysis.tabula_sapiens.workflows"
)


def test_compendium_aliases_recover_renamed_genes_without_guessing(
    tmp_path,
) -> None:
    """A curated gene under a newer dataset symbol must still be matched.

    GENEA appears only as its curated alias NEWA (as DDX58 appears as RIGI).
    GENEC and GENED both list SHARED, so neither may claim that feature.
    """
    template = {
        "module_class": "rna_sensing_core",
        "sensor_family": "RLR",
        "activation_tier": "Early",
        "scoring_direction": "positive",
        "cell_type_breadth": "Broad",
        "detectability": "high",
        "doi": "10.0000/test",
        "sensor": "rna_sensor",
    }
    panel = pd.DataFrame(
        [
            {**template, "gene_symbol": gene, "aliases": aliases}
            for gene, aliases in (
                ("GENEA", "NEWA"),
                ("GENEB", ""),
                ("GENEC", "SHARED"),
                ("GENED", "SHARED"),
            )
        ]
    ).assign(module_id="TEST_SENSING")
    panel_path = tmp_path / "marker_genes.tsv"
    panel.to_csv(panel_path, sep="\t", index=False)
    adata = ad.AnnData(
        var=pd.DataFrame(
            {"feature_name": ["NEWA", "GENEB", "SHARED", "OTHER"]},
            index=["G1", "G2", "G3", "G4"],
        )
    )

    applied = add_compendium_symbol_column(
        adata, source_column="feature_name", panel_path=panel_path
    )
    module = GeneModules.modules(
        "TEST_SENSING",
        panel_path=panel_path,
        adata=adata,
        gene_symbol_column=COMPENDIUM_SYMBOL_COLUMN,
        output="var_names",
    )

    assert applied == {"GENEA": "NEWA"}
    assert set(module.positive_genes) == {"G1", "G2"}
    assert set(module.missing_positive_genes) == {"GENEC", "GENED"}
    assert adata.var.feature_name.tolist() == [
        "NEWA",
        "GENEB",
        "SHARED",
        "OTHER",
    ]


@pytest.mark.parametrize(
    "scorer,suffix", [("scanpy", "_score"), ("aucell", "_auc")]
)
def test_reference_umaps_group_databases_without_changing_scores(
    tmp_path, monkeypatch, scorer, suffix
) -> None:
    """Each external source retains its panel values and original embedding."""
    identifiers = [
        "REACTOME_R_HSA_1834949",
        "REACTOME_R_HSA_1834941",
        "HALLMARK_INFLAMMATORY_RESPONSE",
    ]
    scores = {
        f"{identifier}{suffix}": np.array([0.1, 0.2, 0.3]) + i
        for i, identifier in enumerate(identifiers)
    }
    embedding = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    adata = ad.AnnData(
        obs=pd.DataFrame(scores, index=["a", "b", "c"]),
        obsm={"X_umap": embedding},
    )
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    try:
        plot_reference_score_umaps(
            adata,
            list(scores),
            scorer=scorer,
            plotter=UmapPlotter(tmp_path, dpi=60),
        )
        shown = [
            [
                axis.collections[0]
                for axis in figure.axes
                if axis.collections
                and axis.collections[0].get_offsets().shape == (3, 2)
            ]
            for figure in figures
        ]
        assert [len(panels) for panels in shown] == [2, 1]
        for collection, values in zip(
            [panel for panels in shown for panel in panels],
            scores.values(),
            strict=True,
        ):
            np.testing.assert_allclose(collection.get_array(), values)
            np.testing.assert_allclose(collection.get_offsets(), embedding)
    finally:
        for figure in figures:
            close(figure)


def test_concordance_source_split_preserves_within_source_pairs(
    tmp_path, monkeypatch
) -> None:
    """Split matrices retain off-diagonal pairs and unavailable correlations."""
    identifiers = [
        "NASP_DNA_SENSING",
        "IFN_I_OUTPUT",
        "REACTOME_R_HSA_1834949",
        "HALLMARK_INFLAMMATORY_RESPONSE",
    ]
    values = np.arange(16, dtype=float).reshape(4, 4) / 16
    values[2, 3] = np.nan
    correlations = pd.DataFrame(
        [
            {
                "scanpy_module_id": scanpy,
                "aucell_module_id": aucell,
                "spearman_r": values[row, column],
            }
            for row, aucell in enumerate(identifiers)
            for column, scanpy in enumerate(identifiers)
        ]
    )
    original = correlations.copy(deep=True)
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    try:
        plot_scorer_concordance_by_source(
            correlations,
            identifiers,
            plotter=SummaryPlotter(tmp_path, dpi=60),
        )
        assert len(figures) == 2
        for figure, expected in zip(
            figures,
            (values[:2, :2], values[2:, 2:]),
            strict=True,
        ):
            np.testing.assert_allclose(
                figure.axes[0].images[0].get_array().filled(np.nan),
                expected,
            )
        pd.testing.assert_frame_equal(correlations, original)
    finally:
        for figure in figures:
            close(figure)


def _write_completed_score_table(
    output_dir: str | Path,
    filename: str,
    scorers: str = "scanpy,aucell",
) -> None:
    """Write the minimum score table needed by orchestration tests."""
    score_dir = Path(output_dir)
    score_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "scoring_scorers": [scorers],
            "NASP_DNA_SENSING_score": [0.1],
            "NASP_DNA_SENSING_auc": [0.2],
        },
        index=["cell_a"],
    ).to_csv(score_dir / filename, index_label="obs_name")


def test_scoring_workflow_preserves_scores_and_cell_alignment(
    tmp_path, monkeypatch
) -> None:
    """Persisted scores retain their values and cell identities."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"cell_type": ["T cell", "B cell"]},
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame({"feature_name": ["CGAS"]}, index=["CGAS"]),
    )
    module = GeneModule("NASP_DNA_SENSING", ("CGAS",), (), (), "symbols")

    def scanpy_scores(adata_arg, *args, **kwargs):
        """Stand in for expensive scoring with distinguishable cell values."""
        adata_arg.obs["NASP_DNA_SENSING_score"] = [-2.0, 1.0]
        return [module]

    def aucell_scores(adata_arg, *args, **kwargs):
        """Return deliberately reversed rows to exercise score alignment."""
        scores = pd.DataFrame(
            {"NASP_DNA_SENSING_auc": [0.5, -0.25]},
            index=["cell_b", "cell_a"],
        )
        return adata_arg.copy(), scores, [module]

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows, "score_scanpy_modules", scanpy_scores
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows, "score_aucell_modules", aucell_scores
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.GeneModules,
        "sensors",
        lambda *args, **kwargs: [],
    )
    for owner, name in (
        (tabula_sapiens_workflows, "plot_tabula_sapiens_metadata_umaps"),
        (tabula_sapiens_workflows.UmapPlotter, "plot_multi_gene_umap_panel"),
        (tabula_sapiens_workflows.UmapPlotter, "plot_multi_obs_umap_panel"),
        (
            tabula_sapiens_workflows.HeatmapPlotter,
            "plot_grouped_obs_score_heatmap",
        ),
        (
            tabula_sapiens_workflows.SummaryPlotter,
            "plot_scorer_concordance_heatmap",
        ),
    ):
        monkeypatch.setattr(owner, name, lambda *args, **kwargs: None)

    tabula_sapiens.tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
        module_ids=["NASP_DNA_SENSING"],
        score_scanpy=True,
        score_aucell=True,
    )

    scores = pd.read_csv(
        tmp_path / "tabula_sapiens_module_scores.csv.gz", index_col="obs_name"
    ).reindex(["cell_a", "cell_b"])
    np.testing.assert_allclose(scores["NASP_DNA_SENSING_score"], [-2.0, 1.0])
    np.testing.assert_allclose(scores["NASP_DNA_SENSING_auc"], [-0.25, 0.5])
    assert scores.cell_type.tolist() == ["T cell", "B cell"]


def test_tabula_sapiens_saves_scanpy_scores_before_aucell_failure(
    tmp_path,
    monkeypatch,
) -> None:
    """Completed score steps are persisted before later scoring failures."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame({"feature_name": ["CGAS"]}, index=["gene_a"]),
    )
    module = GeneModule(
        module_id="NASP_DNA_SENSING",
        positive_genes=("CGAS",),
        inverse_genes=(),
        context_dependent_genes=(),
        gene_id_output="symbols",
    )

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "plot_tabula_sapiens_metadata_umaps",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.GeneModules,
        "sensors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.UmapPlotter,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.UmapPlotter,
        "plot_multi_obs_umap_panel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "plot_grouped_obs_score_heatmap",
        lambda *args, **kwargs: None,
    )

    def fake_scanpy_scores(adata_arg, *args, **kwargs):
        adata_arg.obs["NASP_DNA_SENSING_score"] = [1.0, 2.0]
        return [module]

    def fail_aucell(*args, **kwargs):
        raise RuntimeError("AUCell failed")

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "score_scanpy_modules",
        fake_scanpy_scores,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "score_aucell_modules",
        fail_aucell,
    )

    with pytest.raises(RuntimeError, match="AUCell failed"):
        tabula_sapiens.tabula_sapiens_scoring_analysis(
            h5ad_path=tmp_path / "input.h5ad",
            output_dir=tmp_path,
            module_ids=["NASP_DNA_SENSING"],
            score_scanpy=True,
            score_aucell=True,
        )

    scores = pd.read_csv(
        tmp_path / "tabula_sapiens_module_scores.csv.gz",
        index_col="obs_name",
    )
    assert "NASP_DNA_SENSING_score" in scores.columns
    assert "NASP_DNA_SENSING_auc" not in scores.columns
    assert scores["scoring_scorers"].unique().tolist() == ["scanpy"]
    assert scores["scoring_requested_scorers"].unique().tolist() == [
        "scanpy,aucell"
    ]
    assert scores["NASP_DNA_SENSING_score"].tolist() == [1.0, 2.0]


def test_tissue_analysis_resume_reuses_complete_score_table(
    tmp_path,
    monkeypatch,
) -> None:
    """Resume reuses a completed checkpoint from the same scoring request."""
    h5ad_path = tmp_path / "liver.h5ad"
    h5ad_path.touch()
    scoring_dir = tmp_path / "results" / "liver" / "scoring"
    _write_completed_score_table(
        scoring_dir,
        "tabula_sapiens_module_scores.csv.gz",
    )
    association_scorers: list[str] = []
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "tabula_sapiens_scoring_analysis",
        lambda **kwargs: _write_completed_score_table(
            kwargs["output_dir"], kwargs["score_table_filename"]
        ),
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "association_analysis",
        lambda **kwargs: association_scorers.append(kwargs["scorer"]),
    )

    tabula_sapiens.tabula_sapiens_tissue_analysis(
        h5ad_path=h5ad_path,
        output_dir=tmp_path / "results",
        run_name="liver",
    )
    association_scorers.clear()
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "tabula_sapiens_scoring_analysis",
        lambda **kwargs: pytest.fail("complete scores should be reused"),
    )

    tabula_sapiens.tabula_sapiens_tissue_analysis(
        h5ad_path=h5ad_path,
        output_dir=tmp_path / "results",
        run_name="liver",
        resume_from_scores=True,
    )

    assert Counter(association_scorers) == Counter(["scanpy", "aucell"])


def test_tissue_analysis_resume_rebuilds_partial_score_table(
    tmp_path,
    monkeypatch,
) -> None:
    """A Scanpy-only checkpoint is rebuilt before both analyses resume."""
    h5ad_path = tmp_path / "liver.h5ad"
    h5ad_path.touch()
    scoring_dir = tmp_path / "results" / "liver" / "scoring"
    filename = "tabula_sapiens_module_scores.csv.gz"
    _write_completed_score_table(scoring_dir, filename, scorers="scanpy")

    def capture_scoring(**kwargs) -> None:
        _write_completed_score_table(
            kwargs["output_dir"],
            str(kwargs["score_table_filename"]),
        )

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "tabula_sapiens_scoring_analysis",
        capture_scoring,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "association_analysis",
        lambda **kwargs: None,
    )

    tabula_sapiens.tabula_sapiens_tissue_analysis(
        h5ad_path=h5ad_path,
        output_dir=tmp_path / "results",
        run_name="liver",
        resume_from_scores=True,
    )

    completed = pd.read_csv(scoring_dir / filename)
    assert completed.scoring_scorers.eq("scanpy,aucell").all()
