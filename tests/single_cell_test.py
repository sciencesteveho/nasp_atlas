"""Tests for single-cell utilities."""

from __future__ import annotations

import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import anndata as ad  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nasp_compendium.types import GeneModule

from nasp_atlas.cellxgene import add_development_stage_age_obs
from nasp_atlas.single_cell import EmbeddingConfig
from nasp_atlas.single_cell import SCUtils
from nasp_atlas.single_cell import SCVisualizer
from nasp_atlas.single_cell import combine_module_scores
from nasp_atlas.single_cell import inverse_module_score_name
from nasp_atlas.single_cell import module_score_name
from nasp_atlas.single_cell import positive_module_score_name
from nasp_atlas.single_cell import score_scanpy_module
from nasp_atlas.single_cell import split_anndata_by_obs


def test_embedding_config_roundtrip() -> None:
    """EmbeddingConfig round-trips through JSON."""
    config = EmbeddingConfig(
        name="standard_test",
        harmony_key="batch",
        n_top_genes=2000,
        regress_out=["pct_counts_mt"],
        hvg_kwargs={"flavor": "seurat"},
    )

    restored = EmbeddingConfig.from_json(config.to_json())

    assert restored == config
    assert restored.to_dict()["harmony_key"] == "batch"


def test_scutils_filter_obs_doublets() -> None:
    """SCUtils removes rows flagged as doublets."""
    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {"doublet": ["False", "True", "False"]},
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )

    filtered = SCUtils.filter_obs_doublets(adata)

    assert filtered.n_obs == 2
    assert filtered.obs_names.tolist() == ["cell_a", "cell_c"]


def test_scutils_map_categorical_column() -> None:
    """SCUtils maps obs columns into categorical labels."""
    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame({"sample": [0, 1, 1]}, index=["a", "b", "c"]),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )

    SCUtils.map_categorical_column(
        adata,
        source_col="sample",
        mapping={0: "baseline", 1: "stimulated"},
        destination_col="condition",
    )

    assert adata.obs["condition"].cat.categories.tolist() == [
        "baseline",
        "stimulated",
    ]
    assert adata.obs["condition"].tolist() == [
        "baseline",
        "stimulated",
        "stimulated",
    ]


def test_split_anndata_by_obs_writes_snake_case_files(tmp_path) -> None:
    """Tabula Sapiens helper writes one h5ad per obs value."""
    obs_index = pd.Index(
        ["cell_a", "cell_b", "cell_c", "cell_d"],
        dtype=object,
    )
    var_index = pd.Index(["gene_a", "gene_b"], dtype=object)
    adata = ad.AnnData(
        X=np.ones((4, 2)),
        obs=pd.DataFrame(
            {
                "tissue_type": pd.Series(
                    [
                        "Liver",
                        "Bone Marrow",
                        "Liver",
                        "Blood & Immune",
                    ],
                    index=obs_index,
                    dtype=object,
                ),
            },
            index=obs_index,
        ),
        var=pd.DataFrame(index=var_index),
    )
    h5ad_path = tmp_path / "tabula_sapiens.h5ad"
    output_dir = tmp_path / "split"
    adata.write_h5ad(h5ad_path)

    written = split_anndata_by_obs(
        h5ad_path,
        output_dir=output_dir,
        obs_key="tissue_type",
        output_name="tabula sapiens",
    )

    assert written == {
        "Blood & Immune": output_dir / "blood_immune_tabula_sapiens.h5ad",
        "Bone Marrow": output_dir / "bone_marrow_tabula_sapiens.h5ad",
        "Liver": output_dir / "liver_tabula_sapiens.h5ad",
    }
    liver = ad.read_h5ad(written["Liver"])
    assert liver.obs_names.tolist() == ["cell_a", "cell_c"]


def test_module_score_names_live_in_single_cell_utils() -> None:
    """Module score naming is owned by single-cell scoring utilities."""
    module = GeneModule(
        module_id="NASP_DNA_SENSING",
        positive_genes=("CGAS",),
        inverse_genes=("LMNB1",),
        context_dependent_genes=(),
        gene_id_output="symbols",
    )

    assert (
        positive_module_score_name(module, scorer="scanpy")
        == "NASP_DNA_SENSING_pos"
    )
    assert (
        inverse_module_score_name(module, scorer="scanpy")
        == "NASP_DNA_SENSING_inv"
    )
    assert module_score_name(module, scorer="scanpy") == (
        "NASP_DNA_SENSING_score"
    )
    assert (
        positive_module_score_name(module, scorer="aucell")
        == "NASP_DNA_SENSING_pos_auc"
    )
    assert module_score_name(module, scorer="aucell") == (
        "NASP_DNA_SENSING_auc"
    )


def test_combine_module_scores_subtracts_inverse_scores() -> None:
    """Single-cell utilities combine signed sub-scores."""
    module = GeneModule(
        module_id="NASP_DNA_SENSING",
        positive_genes=("CGAS",),
        inverse_genes=("LMNB1",),
        context_dependent_genes=(),
        gene_id_output="symbols",
    )
    scores = pd.DataFrame(
        {
            "NASP_DNA_SENSING_pos": [2.0, 4.0],
            "NASP_DNA_SENSING_inv": [0.5, 3.0],
        },
        index=["cell_a", "cell_b"],
    )

    combined = combine_module_scores(module, scores, scorer="scanpy")

    assert combined.name == "NASP_DNA_SENSING_score"
    assert combined.tolist() == [1.5, 1.0]


def test_score_scanpy_module_calls_scanpy_directly(monkeypatch) -> None:
    """Atlas scanpy scoring forwards explicit score_genes arguments."""
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(index=["CGAS", "LMNB1"]),
    )
    module = GeneModule(
        module_id="NASP_DNA_SENSING",
        positive_genes=("CGAS",),
        inverse_genes=("LMNB1",),
        context_dependent_genes=(),
        gene_id_output="var_names",
    )
    calls = []

    def fake_score_genes(adata_arg, **kwargs):
        calls.append(kwargs)
        if kwargs["score_name"].endswith("_pos"):
            adata_arg.obs[kwargs["score_name"]] = [2.0, 4.0]
        else:
            adata_arg.obs[kwargs["score_name"]] = [0.5, 3.0]

    monkeypatch.setattr(
        "nasp_atlas.single_cell.module_scoring.sc.tl.score_genes",
        fake_score_genes,
    )

    score_name = score_scanpy_module(adata, module, random_state=7)

    assert score_name == "NASP_DNA_SENSING_score"
    assert calls == [
        {
            "gene_list": ["CGAS"],
            "score_name": "NASP_DNA_SENSING_pos",
            "random_state": 7,
            "use_raw": False,
        },
        {
            "gene_list": ["LMNB1"],
            "score_name": "NASP_DNA_SENSING_inv",
            "random_state": 7,
            "use_raw": False,
        },
    ]
    assert adata.obs["NASP_DNA_SENSING_score"].tolist() == [1.5, 1.0]


def test_visualizer_resolves_feature_name_symbols(tmp_path) -> None:
    """SCVisualizer resolves display symbols to adata.var_names."""
    adata = ad.AnnData(
        X=np.ones((2, 3)),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(
            {"feature_name": ["AIM2", "CGAS", "ZBP1"]},
            index=["ENSG_A", "ENSG_B", "ENSG_C"],
        ),
    )
    viz = SCVisualizer(output_dir=tmp_path)

    resolved = viz._resolve_genes(
        adata,
        ["CGAS", "missing", "AIM2"],
        gene_symbol_column="feature_name",
    )

    assert resolved.var_names == ["ENSG_B", "ENSG_A"]
    assert resolved.labels == ["CGAS", "AIM2"]


def test_visualizer_styles_embedding_axes_square(tmp_path) -> None:
    """SCVisualizer keeps embedding panels square."""
    viz = SCVisualizer(output_dir=tmp_path)
    fig, ax = plt.subplots()
    ax.scatter([0, 1], [0, 2])

    viz._style_embedding_axes(ax)

    assert ax.get_box_aspect() == 1
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_visualizer_viridis_zero_is_gray() -> None:
    """SCVisualizer can make a Viridis map with gray zero."""
    cmap = SCVisualizer.umap_expression_cmap("viridis")

    assert cmap(0) == (
        0.9333333333333333,
        0.9333333333333333,
        0.9333333333333333,
        1.0,
    )


def test_visualizer_plots_obs_umap_panel(tmp_path) -> None:
    """SCVisualizer plots categorical and numeric obs UMAP panels."""
    adata = ad.AnnData(
        X=np.ones((4, 2)),
        obs=pd.DataFrame(
            {
                "group": ["a", "b", "a", "b"],
                "score": [0.0, 0.5, 1.0, 1.5],
            },
            index=["cell_a", "cell_b", "cell_c", "cell_d"],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    viz = SCVisualizer(output_dir=tmp_path)

    viz.plot_obs_umap_panel(
        adata,
        panels=[
            {
                "obs_key": "group",
                "title": "Group",
                "kind": "categorical",
                "color_map": {"a": "#111111", "b": "#eeeeee"},
                "legend_loc": "bottom",
                "legend_ncol": 2,
            },
            {
                "obs_key": "score",
                "title": "Score",
                "kind": "numeric",
                "cmap": "viridis",
            },
        ],
        filename="obs_panel",
        ncols=2,
        size=20,
    )

    assert (tmp_path / "obs_panel.png").exists()


def test_visualizer_resolves_ordered_umap_panel_inputs(tmp_path) -> None:
    """SCVisualizer accepts ordered mixed UMAP panel inputs."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {
                "group": ["a", "b"],
                "score": [0.0, 1.0],
                "prediction": [0.2, 0.8],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    viz = SCVisualizer(output_dir=tmp_path)

    panels = viz._resolve_umap_panel_specs(
        adata,
        [
            "score",
            {"obs_key": "group", "title": "Group"},
            {"obs_key": "prediction", "cbar_ticks": [0.2, 0.5, 0.8]},
        ],
    )

    assert [panel.obs_key for panel in panels] == [
        "score",
        "group",
        "prediction",
    ]
    assert [panel.kind for panel in panels] == [
        "numeric",
        "categorical",
        "numeric",
    ]
    assert panels[0].title == "score"
    assert panels[1].title == "Group"
    assert panels[2].cbar_ticks == [0.2, 0.5, 0.8]


def test_visualizer_umap_panel_does_not_call_scanpy_embedding(
    tmp_path,
    monkeypatch,
) -> None:
    """Generic obs UMAP panels use direct metadata rendering."""
    adata = ad.AnnData(
        X=np.ones((3, 1)),
        obs=pd.DataFrame(
            {
                "group": ["a", "b", "a"],
                "score": [0.0, 1.0, 2.0],
            },
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    def fail_embedding(*args, **kwargs):
        raise AssertionError("metadata UMAP should not call scanpy embedding")

    monkeypatch.setattr(
        "nasp_atlas.single_cell.visualization.sc.pl.embedding",
        fail_embedding,
    )
    viz = SCVisualizer(output_dir=tmp_path)

    viz.plot_umap_panel(
        adata,
        panels=["group", "score"],
        filename="direct_obs_panel",
        ncols=2,
        size=20,
    )

    assert (tmp_path / "direct_obs_panel.png").exists()


def test_visualizer_gene_umap_uses_x_not_raw_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    """Gene UMAP panels do not fall back to raw counts by default."""
    adata = ad.AnnData(
        X=np.array([[1.0], [2.0]]),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.raw = ad.AnnData(
        X=np.array([[1000.0], [2000.0]]),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    obs_df_kwargs = {}
    embedding_kwargs = {}

    def fake_obs_df(adata_arg, keys, **kwargs):
        obs_df_kwargs.update(kwargs)
        return pd.DataFrame({"gene_a": [1.0, 2.0]}, index=adata_arg.obs_names)

    def fake_embedding(*args, **kwargs):
        embedding_kwargs.update(kwargs)
        ax = kwargs["ax"]
        ax.scatter([0.0, 1.0], [0.0, 1.0], c=[1.0, 2.0])

    monkeypatch.setattr(
        "nasp_atlas.single_cell.visualization.sc.get.obs_df",
        fake_obs_df,
    )
    monkeypatch.setattr(
        "nasp_atlas.single_cell.visualization.sc.pl.embedding",
        fake_embedding,
    )
    viz = SCVisualizer(output_dir=tmp_path)

    viz.plot_multi_gene_umap_panel(
        adata,
        genes=["gene_a"],
        filename="gene_panel",
        expression_layer=None,
    )

    assert obs_df_kwargs["use_raw"] is False
    assert obs_df_kwargs["layer"] is None
    assert embedding_kwargs["use_raw"] is False
    assert embedding_kwargs["layer"] is None


def test_add_development_stage_age_obs() -> None:
    """CELLxGENE metadata helper adds numeric age values."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"development_stage": ["22-year-old stage", "unknown"]},
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )

    add_development_stage_age_obs(adata)

    assert adata.obs["age_years"].tolist()[0] == 22.0
    assert np.isnan(adata.obs["age_years"].tolist()[1])
