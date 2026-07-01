"""Tests for the Tabula Sapiens analysis workflow."""

from __future__ import annotations

import importlib
import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import pytest
from nasp_compendium.types import GeneModule


tabula_sapiens = importlib.import_module("nasp_atlas.analysis.tabula_sapiens")


def test_tabula_sapiens_saves_combined_scores_and_plots_score_umaps(
    tmp_path,
    monkeypatch,
) -> None:
    """The batch workflow persists final scores and requests score UMAPs."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {
                "cell_type": ["T cell", "B cell"],
                "assay": ["10x 3' v3", "Smart-seq2"],
                "sex": ["female", "male"],
                "development_stage": [
                    "50-year-old human stage",
                    "60-year-old human stage",
                ],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame(
            {"feature_name": ["CGAS"]},
            index=["gene_a"],
        ),
    )
    module = GeneModule(
        module_id="NASP_DNA_SENSING",
        positive_genes=("CGAS",),
        inverse_genes=(),
        context_dependent_genes=(),
        gene_id_output="symbols",
    )
    plot_calls: list[dict[str, object]] = []
    score_heatmap_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        tabula_sapiens,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens,
        "_plot_tabula_sapiens_metadata_umaps",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.GeneModules,
        "sensors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: None,
    )

    def capture_score_plot(*args, **kwargs) -> None:
        plot_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_obs_umap_panel",
        capture_score_plot,
    )

    def capture_score_heatmap(*args, **kwargs) -> None:
        score_heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_grouped_obs_score_heatmap",
        capture_score_heatmap,
    )

    def fake_scanpy_scores(adata_arg, *args, **kwargs):
        adata_arg.obs["NASP_DNA_SENSING_score"] = [-2.0, 1.0]
        return [module]

    def fake_aucell_scores(adata_arg, *args, **kwargs):
        adata_auc = adata_arg.copy()
        adata_auc.obs["NASP_DNA_SENSING_auc"] = [-0.25, 0.5]
        auc_df = adata_auc.obs[["NASP_DNA_SENSING_auc"]].copy()
        return adata_auc, auc_df, [module]

    monkeypatch.setattr(
        tabula_sapiens,
        "score_scanpy_modules",
        fake_scanpy_scores,
    )
    monkeypatch.setattr(
        tabula_sapiens,
        "score_aucell_modules",
        fake_aucell_scores,
    )

    tabula_sapiens.run_tabula_sapiens_scoring_analysis(
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
    assert scores.index.tolist() == ["cell_a", "cell_b"]
    assert scores.columns.tolist() == [
        "assay",
        "sex",
        "development_stage",
        "NASP_DNA_SENSING_score",
        "NASP_DNA_SENSING_auc",
    ]
    assert scores.loc["cell_a"].tolist() == [
        "10x 3' v3",
        "female",
        "50-year-old human stage",
        -2.0,
        -0.25,
    ]
    assert [(call["obs_keys"], call["filename"]) for call in plot_calls] == [
        (["NASP_DNA_SENSING_score"], "tabula_sapiens_scanpy_module_umaps"),
        (["NASP_DNA_SENSING_auc"], "tabula_sapiens_aucell_module_umaps"),
    ]
    assert [
        (call["score_keys"], call["groupby"], call["filename"])
        for call in score_heatmap_calls
    ] == [
        (
            ["NASP_DNA_SENSING_score"],
            "cell_type",
            "tabula_sapiens_scanpy_module_score_heatmap_by_cell_type",
        ),
        (
            ["NASP_DNA_SENSING_auc"],
            "cell_type",
            "tabula_sapiens_aucell_module_score_heatmap_by_cell_type",
        ),
    ]


def test_tabula_sapiens_heatmap_groupby_controls_all_heatmaps(
    tmp_path,
    monkeypatch,
) -> None:
    """A single heatmap_groupby value controls expression and score heatmaps."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {
                "tissue_in_publication": ["lung", "blood"],
                "cell_type": ["T cell", "B cell"],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame({"feature_name": ["CGAS"]}, index=["gene_a"]),
    )
    module = GeneModule(
        module_id="NASP_DNA_SENSING",
        positive_genes=("CGAS",),
        inverse_genes=(),
        context_dependent_genes=(),
        gene_id_output="symbols",
    )
    expression_heatmap_calls: list[dict[str, object]] = []
    score_heatmap_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        tabula_sapiens,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens,
        "_plot_tabula_sapiens_metadata_umaps",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.GeneModules,
        "sensors",
        lambda *args, **kwargs: ["CGAS"],
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_obs_umap_panel",
        lambda *args, **kwargs: None,
    )

    def capture_expression_heatmap(*args, **kwargs) -> None:
        expression_heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_expression_heatmap",
        capture_expression_heatmap,
    )

    def fake_scanpy_scores(adata_arg, *args, **kwargs):
        adata_arg.obs["NASP_DNA_SENSING_score"] = [1.0, 2.0]
        return [module]

    monkeypatch.setattr(
        tabula_sapiens,
        "score_scanpy_modules",
        fake_scanpy_scores,
    )

    def capture_score_heatmap(*args, **kwargs) -> None:
        score_heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_grouped_obs_score_heatmap",
        capture_score_heatmap,
    )

    tabula_sapiens.run_tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
        module_ids=["NASP_DNA_SENSING"],
        heatmap_groupby="tissue_in_publication",
        score_scanpy=True,
    )

    assert [
        (call["groupby"], call["filename"]) for call in expression_heatmap_calls
    ] == [
        (
            "tissue_in_publication",
            "NA_SENSORS_gene_expression_heatmap_by_tissue_in_publication",
        ),
    ]
    assert [
        (call["groupby"], call["filename"]) for call in score_heatmap_calls
    ] == [
        (
            "tissue_in_publication",
            "tabula_sapiens_scanpy_module_score_heatmap_by_"
            "tissue_in_publication",
        ),
    ]


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
        tabula_sapiens,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens,
        "_plot_tabula_sapiens_metadata_umaps",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.GeneModules,
        "sensors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_obs_umap_panel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_grouped_obs_score_heatmap",
        lambda *args, **kwargs: None,
    )

    def fake_scanpy_scores(adata_arg, *args, **kwargs):
        adata_arg.obs["NASP_DNA_SENSING_score"] = [1.0, 2.0]
        return [module]

    def fail_aucell(*args, **kwargs):
        raise RuntimeError("AUCell failed")

    monkeypatch.setattr(
        tabula_sapiens,
        "score_scanpy_modules",
        fake_scanpy_scores,
    )
    monkeypatch.setattr(
        tabula_sapiens,
        "score_aucell_modules",
        fail_aucell,
    )

    with pytest.raises(RuntimeError, match="AUCell failed"):
        tabula_sapiens.run_tabula_sapiens_scoring_analysis(
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
    assert scores.columns.tolist() == ["NASP_DNA_SENSING_score"]
    assert scores["NASP_DNA_SENSING_score"].tolist() == [1.0, 2.0]


def test_tabula_sapiens_sensor_heatmaps_use_tissue_and_cell_type(
    tmp_path,
    monkeypatch,
) -> None:
    """The sensor heatmaps group expression by tissue and cell type."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {
                "tissue_in_publication": ["lung", "lung"],
                "cell_type": ["T cell", "B cell"],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame({"feature_name": ["CGAS"]}, index=["gene_a"]),
    )
    heatmap_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        tabula_sapiens,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens,
        "_plot_tabula_sapiens_metadata_umaps",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.GeneModules,
        "sensors",
        lambda *args, **kwargs: ["CGAS"],
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: None,
    )

    def capture_heatmap(*args, **kwargs) -> None:
        heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_expression_heatmap",
        capture_heatmap,
    )

    tabula_sapiens.run_tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
    )

    assert heatmap_calls == [
        {
            "adata": adata,
            "genes": ["CGAS"],
            "groupby": "tissue_in_publication",
            "filename": (
                "NA_SENSORS_gene_expression_heatmap_by_tissue_in_publication"
            ),
            "gene_symbol_column": "feature_name",
            "expression_layer": None,
        },
        {
            "adata": adata,
            "genes": ["CGAS"],
            "groupby": "cell_type",
            "filename": "NA_SENSORS_gene_expression_heatmap_by_cell_type",
            "gene_symbol_column": "feature_name",
            "expression_layer": None,
        },
    ]


def test_tabula_sapiens_module_heatmaps_follow_marker_umap_modules(
    tmp_path,
    monkeypatch,
) -> None:
    """Module marker heatmaps use the same selected modules as marker UMAPs."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {
                "tissue_in_publication": ["lung", "blood"],
                "cell_type": ["T cell", "B cell"],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame({"feature_name": ["CGAS"]}, index=["gene_a"]),
    )
    heatmap_calls: list[dict[str, object]] = []
    umap_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        tabula_sapiens,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens,
        "_plot_tabula_sapiens_metadata_umaps",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabula_sapiens.GeneModules,
        "sensors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        tabula_sapiens.GeneModules,
        "genes",
        lambda *args, **kwargs: ["CGAS"],
    )
    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: None,
    )

    def capture_module_umaps(*args, **kwargs) -> None:
        umap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens,
        "plot_module_gene_umaps",
        capture_module_umaps,
    )

    def capture_heatmap(*args, **kwargs) -> None:
        heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens.SCVisualizer,
        "plot_multi_gene_expression_heatmap",
        capture_heatmap,
    )

    tabula_sapiens.run_tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
        module_ids=["NASP_DNA_SENSING"],
        plot_modules=True,
    )

    assert umap_calls[0]["module_ids"] == ["NASP_DNA_SENSING"]
    assert heatmap_calls == [
        {
            "adata": adata,
            "genes": ["CGAS"],
            "groupby": "tissue_in_publication",
            "filename": (
                "NASP_DNA_SENSING_gene_expression_heatmap_by_"
                "tissue_in_publication"
            ),
            "gene_symbol_column": "feature_name",
            "expression_layer": None,
        },
        {
            "adata": adata,
            "genes": ["CGAS"],
            "groupby": "cell_type",
            "filename": "NASP_DNA_SENSING_gene_expression_heatmap_by_cell_type",
            "gene_symbol_column": "feature_name",
            "expression_layer": None,
        },
    ]
