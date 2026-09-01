"""Tests for single-cell utilities."""

from __future__ import annotations

import dataclasses
from collections import Counter

import anndata as ad  # type: ignore[import]
import h5py  # type: ignore[import]
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from nasp_compendium.types import GeneModule

import nasp_atlas.single_cell.utils as single_cell_utils
import nasp_atlas.single_cell.visualization.umap as umap_visualization
from nasp_atlas.cellxgene import add_development_stage_age_obs
from nasp_atlas.single_cell import EmbeddingConfig
from nasp_atlas.single_cell import SCProcessor
from nasp_atlas.single_cell import SCUtils
from nasp_atlas.single_cell import combine_module_scores
from nasp_atlas.single_cell import expression_matrix
from nasp_atlas.single_cell import inverse_module_score_name
from nasp_atlas.single_cell import module_score_name
from nasp_atlas.single_cell import normalize_h5ad_string_storage
from nasp_atlas.single_cell import positive_module_score_name
from nasp_atlas.single_cell import read_h5ad
from nasp_atlas.single_cell import read_h5ad_rows
from nasp_atlas.single_cell import score_scanpy_module
from nasp_atlas.single_cell import split_anndata_by_obs
from nasp_atlas.single_cell.umap import resolve_umap_panel_specs
from nasp_atlas.single_cell.visualization import AssociationPlotter
from nasp_atlas.single_cell.visualization import ColorbarStyle
from nasp_atlas.single_cell.visualization import HeatmapPlotter
from nasp_atlas.single_cell.visualization import SummaryPlotter
from nasp_atlas.single_cell.visualization import UmapPlotter


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


def test_expression_matrix_validates_requested_raw_source() -> None:
    """Raw expression requests fail where source selection is resolved."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(index=["gene_a"]),
    )

    with pytest.raises(
        ValueError,
        match=r"use_raw=True requires adata.raw to be set",
    ):
        expression_matrix(
            adata,
            expression_layer=None,
            use_raw=True,
        )


def test_expression_matrix_selects_genes_from_raw() -> None:
    """Raw expression selection can recover genes absent from current var."""
    source = ad.AnnData(
        X=np.array([[1.0, 10.0], [2.0, 20.0]]),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    source.raw = source.copy()
    adata = source[:, ["gene_a"]].copy()

    present, matrix = expression_matrix(
        adata,
        ["gene_b", "missing"],
        expression_layer=None,
        use_raw=True,
    )

    assert present == ["gene_b"]
    np.testing.assert_array_equal(matrix, np.array([[10.0], [20.0]]))


def test_embedding_config_is_immutable() -> None:
    """EmbeddingConfig blocks attribute mutation after construction."""
    config = EmbeddingConfig(name="standard_test")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_neighbors = 30


def test_scprocessor_recompute_umap_writes_coordinates() -> None:
    """UMAP recomputation writes finite coordinates from the selected basis."""
    adata = ad.AnnData(
        X=np.ones((6, 1)),
        obs=pd.DataFrame(index=[f"cell_{index}" for index in range(6)]),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.obsm["X_latent"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    SCProcessor.recompute_umap(
        adata,
        use_rep="X_latent",
        n_neighbors=2,
        random_state=7,
        min_dist=0.2,
    )

    assert adata.obsm["X_umap"].shape == (6, 2)
    assert np.isfinite(adata.obsm["X_umap"]).all()


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


def test_split_anndata_by_obs_names_files_from_obs_values(tmp_path) -> None:
    """Split filenames use obs values before the configured output name."""
    obs_index = pd.Index(["cell_a", "cell_b", "cell_c"], dtype=object)
    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {
                "tissue_type": pd.Categorical(
                    ["Liver", "Lung", "Liver"],
                    categories=["Liver", "Lung", "Unused Tissue"],
                ),
            },
            index=obs_index,
        ),
        var=pd.DataFrame(index=pd.Index(["gene_a", "gene_b"], dtype=object)),
    )
    output_dir = tmp_path / "split"

    written = split_anndata_by_obs(
        adata,
        output_dir=output_dir,
        obs_key="tissue_type",
        output_name="tabula sapiens",
    )

    assert written == {
        "Liver": output_dir / "liver_tabula_sapiens.h5ad",
        "Lung": output_dir / "lung_tabula_sapiens.h5ad",
    }


def test_split_anndata_by_obs_reads_path_in_memory_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    """Path-based splitting reads the source h5ad into memory by default."""
    obs_index = pd.Index(["cell_a", "cell_b", "cell_c"], dtype=object)
    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {
                "tissue_type": pd.Series(
                    ["Liver", "Bone Marrow", "Liver"],
                    index=obs_index,
                    dtype=object,
                ),
            },
            index=obs_index,
        ),
        var=pd.DataFrame(index=pd.Index(["gene_a", "gene_b"], dtype=object)),
    )
    h5ad_path = tmp_path / "tabula_sapiens.h5ad"
    adata.write_h5ad(h5ad_path)

    original_read_h5ad = single_cell_utils.ad.read_h5ad
    reads: list[tuple[dict, ad.AnnData]] = []

    def read_h5ad_spy(*args, **kwargs):
        loaded = original_read_h5ad(*args, **kwargs)
        reads.append((kwargs, loaded))
        return loaded

    monkeypatch.setattr(single_cell_utils.ad, "read_h5ad", read_h5ad_spy)

    split_anndata_by_obs(
        h5ad_path,
        output_dir=tmp_path / "split",
        obs_key="tissue_type",
        output_name="tabula sapiens",
    )

    assert len(reads) == 1
    kwargs, loaded = reads[0]
    assert "backed" not in kwargs
    assert loaded.isbacked is False


def test_split_anndata_by_obs_reads_path_backed_when_requested(
    tmp_path,
    monkeypatch,
) -> None:
    """Path-based splitting can keep the source h5ad backed and closes it."""
    obs_index = pd.Index(["cell_a", "cell_b", "cell_c"], dtype=object)
    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {
                "tissue_type": pd.Series(
                    ["Liver", "Bone Marrow", "Liver"],
                    index=obs_index,
                    dtype=object,
                ),
            },
            index=obs_index,
        ),
        var=pd.DataFrame(index=pd.Index(["gene_a", "gene_b"], dtype=object)),
    )
    h5ad_path = tmp_path / "tabula_sapiens.h5ad"
    adata.write_h5ad(h5ad_path)

    original_read_h5ad = single_cell_utils.ad.read_h5ad
    backed_reads: list[ad.AnnData] = []

    def read_h5ad_spy(*args, **kwargs):
        loaded = original_read_h5ad(*args, **kwargs)
        if kwargs.get("backed") == "r":
            backed_reads.append(loaded)
        return loaded

    monkeypatch.setattr(single_cell_utils.ad, "read_h5ad", read_h5ad_spy)

    split_anndata_by_obs(
        h5ad_path,
        output_dir=tmp_path / "split",
        obs_key="tissue_type",
        output_name="tabula sapiens",
        backed=True,
    )

    assert len(backed_reads) == 1
    assert backed_reads[0].file.is_open is False


def test_split_anndata_by_obs_preserves_source_compression(tmp_path) -> None:
    """Path-based splitting keeps source matrix compression by default."""
    obs_index = pd.Index(["cell_a", "cell_b", "cell_c"], dtype=object)
    adata = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {
                "tissue_type": pd.Series(
                    ["Liver", "Bone Marrow", "Liver"],
                    index=obs_index,
                    dtype=object,
                ),
            },
            index=obs_index,
        ),
        var=pd.DataFrame(index=pd.Index(["gene_a", "gene_b"], dtype=object)),
    )
    h5ad_path = tmp_path / "tabula_sapiens.h5ad"
    output_dir = tmp_path / "split"
    adata.write_h5ad(h5ad_path, compression="gzip")

    written = split_anndata_by_obs(
        h5ad_path,
        output_dir=output_dir,
        obs_key="tissue_type",
        output_name="tabula sapiens",
    )

    with h5py.File(written["Liver"], "r") as h5:
        assert h5["X"].compression == "gzip"


def test_split_anndata_by_obs_reads_backed_sparse_raw(tmp_path) -> None:
    """Path-based splitting handles backed CSR raw matrices."""
    obs_index = pd.Index(["cell_a", "cell_b", "cell_c"], dtype=object)
    var_index = pd.Index(["gene_a", "gene_b"], dtype=object)
    adata = ad.AnnData(
        X=sp.csr_matrix(np.arange(6).reshape(3, 2)),
        obs=pd.DataFrame(
            {
                "tissue_type": pd.Series(
                    ["Liver", "Bone Marrow", "Liver"],
                    index=obs_index,
                    dtype=object,
                ),
            },
            index=obs_index,
        ),
        var=pd.DataFrame(index=var_index),
    )
    adata.raw = adata
    h5ad_path = tmp_path / "tabula_sapiens.h5ad"
    output_dir = tmp_path / "split"
    adata.write_h5ad(h5ad_path)

    written = split_anndata_by_obs(
        h5ad_path,
        output_dir=output_dir,
        obs_key="tissue_type",
        output_name="tabula sapiens",
        backed=True,
    )

    liver = ad.read_h5ad(written["Liver"])
    assert liver.obs_names.tolist() == ["cell_a", "cell_c"]
    assert liver.raw is not None
    assert liver.raw.var_names.tolist() == ["gene_a", "gene_b"]
    np.testing.assert_array_equal(
        liver.raw.X.toarray(),
        np.array([[0, 1], [4, 5]]),
    )


def test_normalize_h5ad_string_storage_converts_arrow_categories(
    tmp_path,
) -> None:
    """Anndata string categoricals write after normalization."""
    obs_index = pd.Index(["cell_a", "cell_b"], dtype=object)
    tissue_categories = pd.Index(
        ["Bone Marrow", "Liver"],
        dtype="string[pyarrow]",
    )
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            {
                "tissue_type": pd.Categorical.from_codes(
                    [1, 0],
                    categories=tissue_categories,
                ),
            },
            index=obs_index,
        ),
        var=pd.DataFrame(index=pd.Index(["gene_a", "gene_b"], dtype=object)),
    )
    normalized = tmp_path / "normalized.h5ad"

    normalize_h5ad_string_storage(adata)
    adata.write_h5ad(normalized)

    reread = ad.read_h5ad(normalized)
    assert reread.obs["tissue_type"].tolist() == ["Liver", "Bone Marrow"]


def test_module_score_names_identify_scorer_and_module_arm() -> None:
    """Module score names identify their scorer and signed module arm."""
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

    combined = combine_module_scores(
        module,
        scores,
        scorer="scanpy",
        standardize=False,
    )

    assert combined.name == "NASP_DNA_SENSING_score"
    assert combined.tolist() == [1.5, 1.0]


def test_score_scanpy_module_combines_signed_scores(monkeypatch) -> None:
    """Scanpy module scoring combines positive and inverse arm scores."""
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
    assert Counter(
        (tuple(call["gene_list"]), call["score_name"]) for call in calls
    ) == Counter(
        [
            (("CGAS",), "NASP_DNA_SENSING_pos"),
            (("LMNB1",), "NASP_DNA_SENSING_inv"),
        ]
    )
    assert all(call["random_state"] == 7 for call in calls)
    assert adata.obs["NASP_DNA_SENSING_score"].tolist() == [0.0, 0.0]


def test_read_h5ad_subset_preserves_layers_and_raw(tmp_path) -> None:
    """Random-subset reader keeps expression sources used downstream."""
    obs = pd.DataFrame(
        index=pd.Index(["cell_a", "cell_b", "cell_c"], dtype=object)
    )
    var = pd.DataFrame(index=pd.Index(["gene_a", "gene_b"], dtype=object))
    adata = ad.AnnData(
        X=sp.csr_matrix(np.arange(6).reshape(3, 2)),
        obs=obs,
        var=var,
    )
    adata.layers["decontXcounts"] = sp.csr_matrix(
        np.arange(6).reshape(3, 2) + 10
    )
    adata.raw = ad.AnnData(
        X=sp.csr_matrix(np.arange(6).reshape(3, 2) + 20),
        obs=obs.copy(),
        var=var.copy(),
    )
    h5ad_path = tmp_path / "layered.h5ad"
    adata.write_h5ad(h5ad_path)

    loaded, total = read_h5ad(h5ad_path, subset_fraction=1.0)

    assert total == 3
    assert "decontXcounts" in loaded.layers
    np.testing.assert_array_equal(
        loaded.layers["decontXcounts"].toarray(),
        np.arange(6).reshape(3, 2) + 10,
    )
    assert loaded.raw is not None
    np.testing.assert_array_equal(
        loaded.raw.X.toarray(),
        np.arange(6).reshape(3, 2) + 20,
    )


def test_read_h5ad_rows_loads_only_requested_sources(tmp_path) -> None:
    """Named-row loading preserves order and omits unrequested matrices."""
    obs = pd.DataFrame(index=pd.Index(["a", "b", "c"], dtype=object))
    var = pd.DataFrame(index=pd.Index(["g1", "g2"], dtype=object))
    adata = ad.AnnData(
        X=sp.csr_matrix(np.arange(6).reshape(3, 2)),
        obs=obs,
        var=var,
    )
    adata.layers["wanted"] = sp.csr_matrix(np.arange(6).reshape(3, 2) + 10)
    adata.layers["unused"] = sp.csr_matrix(np.arange(6).reshape(3, 2) + 20)
    adata.raw = ad.AnnData(
        X=sp.csr_matrix(np.arange(6).reshape(3, 2) + 30),
        obs=obs.copy(),
        var=var.copy(),
    )
    path = tmp_path / "named_rows.h5ad"
    adata.write_h5ad(path)

    loaded = read_h5ad_rows(
        path,
        ["c", "a"],
        layer_keys=["wanted"],
        read_x=False,
        read_raw=True,
    )

    assert loaded.obs_names.tolist() == ["c", "a"]
    assert list(loaded.layers) == ["wanted"]
    np.testing.assert_array_equal(
        loaded.layers["wanted"].toarray(),
        np.array([[14, 15], [10, 11]]),
    )
    assert loaded.raw is not None
    np.testing.assert_array_equal(
        loaded.raw.X.toarray(),
        np.array([[34, 35], [30, 31]]),
    )
    assert loaded.X.nnz == 0


def test_plotter_plots_obs_umap_panel(tmp_path) -> None:
    """UmapPlotter plots categorical and numeric obs UMAP panels."""
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
    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_umap_panel(
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
            },
        ],
        filename="obs_panel",
        ncols=2,
        size=20,
    )

    assert (tmp_path / "obs_panel.png").stat().st_size > 0


def test_resolve_umap_panel_specs_preserves_order() -> None:
    """Mixed UMAP panel inputs resolve in requested order."""
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
    panels = resolve_umap_panel_specs(
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


def test_plotter_umap_panel_writes_mixed_metadata_plot(tmp_path) -> None:
    """A mixed categorical and numeric UMAP panel is written to disk."""
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

    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_umap_panel(
        adata,
        panels=["group", "score"],
        filename="direct_obs_panel",
        ncols=2,
        size=20,
    )

    assert (tmp_path / "direct_obs_panel.png").stat().st_size > 0


def test_multi_obs_umap_panel_writes_available_scores(tmp_path) -> None:
    """The numeric UMAP API plots available scores and skips missing ones."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"signed_score": [-1.0, 2.0]},
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_multi_obs_umap_panel(
        adata,
        obs_keys=["signed_score", "missing"],
        filename="signed_scores",
        vmin=-2.0,
        vmax=2.0,
        colorbar_style=ColorbarStyle(
            height="20%",
            width="5%",
            pad=0.04,
        ),
        cbar_height="25%",
    )

    assert (tmp_path / "signed_scores.png").stat().st_size > 0


def test_plotter_gene_umap_uses_x_not_raw_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    """Gene UMAP panels do not fall back to raw counts by default."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1.0], [2.0]]),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.raw = ad.AnnData(
        X=np.array([[1000.0], [2000.0]]),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    plotted_values = []
    scatter = Axes.scatter

    def capture_scatter(self, *args, **kwargs):
        plotted_values.append(np.asarray(kwargs["c"]).copy())
        return scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_multi_gene_umap_panel(
        adata,
        genes=["gene_a"],
        filename="gene_panel",
        expression_layer=None,
    )

    assert len(plotted_values) == 1
    np.testing.assert_allclose(plotted_values[0], [1.0, 2.0])
    assert (tmp_path / "gene_panel.png").stat().st_size > 0


def test_plotter_gene_umap_extracts_once_and_renders_bounded_batches(
    tmp_path,
    monkeypatch,
) -> None:
    """Gene UMAP batches bound live panels and retain one output contract."""
    genes = [f"gene_{index}" for index in range(5)]
    adata = ad.AnnData(
        X=sp.csr_matrix(np.arange(15, dtype=float).reshape(3, 5)),
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c"]),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    expression_matrix_calls = 0
    scatter_figures = []
    get_expression_matrix = umap_visualization.expression_matrix
    scatter = Axes.scatter

    def capture_expression_matrix(*args, **kwargs):
        nonlocal expression_matrix_calls
        expression_matrix_calls += 1
        return get_expression_matrix(*args, **kwargs)

    def capture_scatter(self, *args, **kwargs):
        scatter_figures.append(self.figure)
        return scatter(self, *args, **kwargs)

    monkeypatch.setattr(
        umap_visualization,
        "expression_matrix",
        capture_expression_matrix,
    )
    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_multi_gene_umap_panel(
        adata,
        genes=genes,
        filename="batched_gene_panel",
        ncols=2,
        max_rows_per_batch=1,
    )

    panels_per_figure = Counter(scatter_figures)
    assert expression_matrix_calls == 1
    assert sorted(panels_per_figure.values()) == [1, 2, 2]
    assert list(tmp_path.iterdir()) == [tmp_path / "batched_gene_panel.png"]


def test_plotter_gene_umap_shared_colorbar_uses_global_expression_range(
    tmp_path,
    monkeypatch,
) -> None:
    """Shared gene scaling stays global across bounded render batches."""
    genes = [f"gene_{index}" for index in range(5)]
    matrix = np.arange(15, dtype=float).reshape(3, 5)
    adata = ad.AnnData(
        X=sp.csr_matrix(matrix),
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c"]),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    collections = []
    colorbar_calls: list[dict[str, object]] = []
    colorbar_mappables = []
    scatter = Axes.scatter
    colorbar = Figure.colorbar

    def capture_scatter(self, *args, **kwargs):
        collection = scatter(self, *args, **kwargs)
        collections.append(collection)
        return collection

    def capture_colorbar(self, *args, **kwargs):
        colorbar_calls.append(kwargs.copy())
        colorbar_mappables.append(args[0])
        return colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Figure, "colorbar", capture_colorbar)
    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_multi_gene_umap_panel(
        adata,
        genes=genes,
        filename="shared_gene_panel",
        ncols=2,
        max_rows_per_batch=1,
        shared_colorbar=True,
    )

    assert len(collections) == len(genes)
    assert {collection.get_clim() for collection in collections} == {
        (0.0, float(matrix.max()))
    }
    assert len(colorbar_calls) == 1
    assert colorbar_mappables[0].get_clim() == (0.0, float(matrix.max()))
    assert (tmp_path / "shared_gene_panel.png").stat().st_size > 0


def test_plotter_shared_gene_colorbar_matches_panel_size_across_rows(
    tmp_path,
    monkeypatch,
) -> None:
    """A shared bar retains the former per-panel colorbar dimensions."""
    obs = pd.DataFrame(index=["cell_a", "cell_b", "cell_c"])
    coordinates = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    one_row = ad.AnnData(
        X=sp.csr_matrix(np.arange(1, 7, dtype=float).reshape(3, 2)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["gene_0", "gene_1"]),
    )
    one_row.obsm["X_umap"] = coordinates
    three_rows = ad.AnnData(
        X=sp.csr_matrix(np.arange(1, 19, dtype=float).reshape(3, 6)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"gene_{index}" for index in range(6)]),
    )
    three_rows.obsm["X_umap"] = coordinates
    colorbar_axes: list[Axes] = []
    colorbar = Figure.colorbar

    def capture_colorbar(self, *args, **kwargs):
        rendered_colorbar = colorbar(self, *args, **kwargs)
        colorbar_axes.append(rendered_colorbar.ax)
        return rendered_colorbar

    monkeypatch.setattr(Figure, "colorbar", capture_colorbar)
    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_multi_gene_umap_panel(
        one_row,
        genes=one_row.var_names.tolist(),
        filename="one_row_unshared_genes",
        ncols=2,
        max_rows_per_batch=1,
        shared_colorbar=False,
    )
    reference_bounds = colorbar_axes[-1].get_window_extent()
    plotter.plot_multi_gene_umap_panel(
        three_rows,
        genes=three_rows.var_names.tolist(),
        filename="three_row_shared_genes",
        ncols=2,
        max_rows_per_batch=1,
        shared_colorbar=True,
    )
    shared_bounds = colorbar_axes[-1].get_window_extent()

    assert shared_bounds.width == pytest.approx(
        reference_bounds.width, rel=0.05
    )
    assert shared_bounds.height == pytest.approx(
        reference_bounds.height,
        rel=0.05,
    )


def test_plotter_gene_umap_rejects_undefined_shared_zero_range(
    tmp_path,
) -> None:
    """An all-zero gene panel does not invent a non-observed shared range."""
    adata = ad.AnnData(
        X=sp.csr_matrix(np.zeros((2, 1))),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(
        ValueError,
        match="requested 0-to-0 range is undefined",
    ):
        UmapPlotter(output_dir=tmp_path).plot_multi_gene_umap_panel(
            adata,
            genes=["gene_a"],
            filename="all_zero_gene",
            shared_colorbar=True,
        )


def test_multi_obs_umap_zscores_without_mutating_scores(
    tmp_path,
    monkeypatch,
) -> None:
    """Shared score panels use comparable SD units without changing obs."""
    adata = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "small_score": [1.0, 2.0, 3.0, 4.0],
                "large_score": [10.0, 20.0, 30.0, 40.0],
                "tiny_score": [0.0, 1e-9, 2e-9, 3e-9],
                "partial_score": [1.0, 2.0, np.nan, np.inf],
            },
            index=["cell_a", "cell_b", "cell_c", "cell_d"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.obsm["X_umap"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    original_obs = adata.obs.copy(deep=True)
    collections = []
    colorbar_calls: list[dict[str, object]] = []
    scatter = Axes.scatter
    colorbar = Figure.colorbar

    def capture_scatter(self, *args, **kwargs):
        collection = scatter(self, *args, **kwargs)
        collections.append(collection)
        return collection

    def capture_colorbar(self, *args, **kwargs):
        colorbar_calls.append(kwargs.copy())
        return colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Figure, "colorbar", capture_colorbar)
    plotter = UmapPlotter(output_dir=tmp_path)

    plotter.plot_multi_obs_umap_panel(
        adata,
        obs_keys=[
            "small_score",
            "large_score",
            "tiny_score",
            "partial_score",
        ],
        filename="standardized_scores",
        shared_colorbar=True,
        standardization="zscore",
        center_zero=True,
        vmin=-3.0,
        vmax=3.0,
        cbar_extend="both",
    )

    assert len(collections) == 4
    plotted_values = [
        np.ma.asarray(collection.get_array(), dtype=float)
        for collection in collections
    ]
    for values in plotted_values:
        finite_values = values.compressed()
        assert np.isclose(finite_values.mean(), 0.0)
        assert np.isclose(finite_values.std(ddof=0), 1.0)
    np.testing.assert_allclose(plotted_values[0], plotted_values[1])
    np.testing.assert_allclose(plotted_values[0], plotted_values[2])
    assert np.ma.getmaskarray(plotted_values[3]).tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert {collection.get_clim() for collection in collections} == {
        (-3.0, 3.0)
    }
    assert len(colorbar_calls) == 1
    assert colorbar_calls[0]["extend"] == "both"
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    assert (tmp_path / "standardized_scores.png").stat().st_size > 0


def test_multi_obs_umap_zscore_requires_symmetric_bound_pair(tmp_path) -> None:
    """A one-sided standardized limit cannot imply an asymmetric scale."""
    adata = ad.AnnData(
        X=np.ones((3, 1)),
        obs=pd.DataFrame(
            {"score": [-1.0, 0.0, 1.0]},
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(
        ValueError,
        match="require both vmin and vmax or neither",
    ):
        UmapPlotter(output_dir=tmp_path).plot_multi_obs_umap_panel(
            adata,
            obs_keys=["score"],
            filename="one_sided_zscore",
            standardization="zscore",
            center_zero=True,
            vmin=-3.0,
            vmax=None,
        )


def test_multi_obs_umap_marks_constant_zscore_unavailable(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    """A constant score is not fabricated as a zero z-score panel."""
    adata = ad.AnnData(
        X=np.ones((3, 1)),
        obs=pd.DataFrame(
            {"constant_score": [5.0, 5.0, 5.0]},
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    collections = []
    rendered_text: list[str] = []
    scatter = Axes.scatter
    add_text = Axes.text

    def capture_scatter(self, *args, **kwargs):
        collection = scatter(self, *args, **kwargs)
        collections.append(collection)
        return collection

    def capture_text(self, *args, **kwargs):
        rendered_text.append(str(args[2]))
        return add_text(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Axes, "text", capture_text)
    plotter = UmapPlotter(output_dir=tmp_path)

    with caplog.at_level("WARNING"):
        plotter.plot_multi_obs_umap_panel(
            adata,
            obs_keys=["constant_score"],
            filename="constant_score",
            shared_colorbar=True,
            standardization="zscore",
            center_zero=True,
            vmin=-3.0,
            vmax=3.0,
        )

    assert len(collections) == 1
    assert np.ma.getmaskarray(collections[0].get_array()).all()
    assert "Unavailable" in rendered_text
    assert "constant; z-score UMAP renders it as unavailable" in caplog.text
    assert (tmp_path / "constant_score.png").stat().st_size > 0


def test_plotter_gene_expression_heatmap_groups_obs(
    tmp_path,
    monkeypatch,
) -> None:
    """Gene heatmaps render group means in categorical and gene order."""
    adata = ad.AnnData(
        X=sp.csr_matrix(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 8.0],
            ]
        ),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(
                    ["b_cell", "t_cell", "b_cell"],
                    categories=["t_cell", "b_cell"],
                )
            },
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame(
            {"feature_name": ["AIM2", "CGAS"]},
            index=["gene_a", "gene_b"],
        ),
    )
    plotter = HeatmapPlotter(output_dir=tmp_path)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        plotter.plot_multi_gene_expression_heatmap(
            adata,
            genes=["CGAS", "AIM2"],
            groupby="cell_type",
            filename="gene_expression_heatmap",
            gene_symbol_column="feature_name",
            expression_layer=None,
        )

        heatmap_ax = figures[0].axes[0]
        np.testing.assert_allclose(
            heatmap_ax.images[0].get_array(),
            np.array([[4.0, 3.0], [5.0, 3.0]]),
        )
        assert [tick.get_text() for tick in heatmap_ax.get_xticklabels()] == [
            "CGAS",
            "AIM2",
        ]
        assert [tick.get_text() for tick in heatmap_ax.get_yticklabels()] == [
            "t_cell",
            "b_cell",
        ]
        assert (tmp_path / "gene_expression_heatmap.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_plotter_score_heatmap_groups_obs_scores(
    tmp_path,
    monkeypatch,
) -> None:
    """Score heatmaps render group means in categorical and score order."""
    adata = ad.AnnData(
        X=np.ones((3, 1)),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(
                    ["b_cell", "t_cell", "b_cell"],
                    categories=["t_cell", "b_cell"],
                ),
                "module_a": [1.0, -2.0, 3.0],
                "module_b": [0.0, 4.0, 2.0],
            },
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    plotter = HeatmapPlotter(output_dir=tmp_path)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        plotter.plot_grouped_obs_score_heatmap(
            adata,
            score_keys=["module_a", "module_b"],
            groupby="cell_type",
            filename="score_heatmap",
            score_labels=["Module A", "Module B"],
        )

        heatmap_ax = figures[0].axes[0]
        np.testing.assert_allclose(
            heatmap_ax.images[0].get_array(),
            np.array([[-2.0, 4.0], [2.0, 1.0]]),
        )
        assert [tick.get_text() for tick in heatmap_ax.get_xticklabels()] == [
            "Module A",
            "Module B",
        ]
        assert [tick.get_text() for tick in heatmap_ax.get_yticklabels()] == [
            "t_cell",
            "b_cell",
        ]
        assert (tmp_path / "score_heatmap.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_plotter_score_barplot_orders_groups_by_mean(tmp_path) -> None:
    """Score barplots order obs groups from largest to smallest mean score."""
    adata = ad.AnnData(
        X=np.ones((6, 1)),
        obs=pd.DataFrame(
            {
                "cell_type": ["B", "B", "T", "T", "Mono", "Mono"],
                "module_score": [1.0, 3.0, 5.0, 7.0, -1.0, 1.0],
            },
            index=[f"cell_{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )
    plotter = SummaryPlotter(output_dir=tmp_path)

    summary = plotter.plot_grouped_obs_score_barplot(
        adata,
        score_key="module_score",
        groupby="cell_type",
        filename="score_barplot",
    )

    assert summary.index.tolist() == ["T", "B", "Mono"]
    assert summary["mean"].tolist() == [6.0, 2.0, 0.0]
    assert (tmp_path / "score_barplot.png").stat().st_size > 0


def test_feature_regression_uses_readable_axes_and_full_width(
    tmp_path,
    monkeypatch,
) -> None:
    """Regression axes identify the predictor, scorer, and complete fit."""
    unit_frame = pd.DataFrame(
        {
            "feature_id": ["NASP_DNA_SENSING_auc"] * 3,
            "feature_value": [0.2, 0.4, 0.6],
            "age_years": [20.0, 40.0, 60.0],
            "statistical_unit": ["donor"] * 3,
            "sex": ["male"] * 3,
        }
    )
    result_row = pd.Series(
        {
            "slope": 0.01,
            "intercept": 0.0,
            "pearson_r": 1.0,
            "pearson_pvalue": 0.001,
            "feature_type": "module_score",
            "analysis_scope": "within_sex_donor",
            "stratify_key": "sex",
            "stratum": "male",
        }
    )
    plotter = AssociationPlotter(output_dir=tmp_path)
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        plotter.plot_feature_regression(
            unit_frame,
            feature_id="NASP_DNA_SENSING_auc",
            predictor_key="age_years",
            filename="age_regression",
            result_row=result_row,
            feature_label="NASP DNA sensing",
            color_key="sex",
            figsize=(1.2, 1.2),
        )

        ax = figures[0].axes[0]
        assert ax.get_xlabel() == "Age (years)"
        assert ax.get_ylabel() == "NASP DNA sensing score\n(AUCell)"
        assert ax.get_xlim()[0] < 20.0
        assert ax.get_xlim()[1] > 60.0
        np.testing.assert_allclose(
            ax.lines[0].get_xdata()[[0, -1]], ax.get_xlim()
        )
        assert (tmp_path / "age_regression.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_feature_regression_uses_requested_category_colors(
    tmp_path,
    monkeypatch,
) -> None:
    """Categorical regression points use the caller-supplied palette."""
    unit_frame = pd.DataFrame(
        {
            "feature_id": ["NASP_DNA_SENSING_auc"] * 4,
            "feature_value": [0.2, 0.3, 0.5, 0.6],
            "age_years": [20.0, 40.0, 20.0, 40.0],
            "sex": ["male", "male", "female", "female"],
            "statistical_unit": ["donor"] * 4,
        }
    )
    requested_colors = {
        "male": "#d2e7ef",
        "female": "#f9bebc",
    }
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        AssociationPlotter(tmp_path).plot_feature_regression(
            unit_frame,
            feature_id="NASP_DNA_SENSING_auc",
            predictor_key="age_years",
            filename="sex_colored_regression",
            color_key="sex",
            point_color=None,
            category_colors=requested_colors,
        )

        rendered_colors = {
            collection.get_label(): collection.get_facecolors()[0]
            for collection in figures[0].axes[0].collections
        }
        for category, color in requested_colors.items():
            np.testing.assert_allclose(
                rendered_colors[category],
                mcolors.to_rgba(color),
            )
    finally:
        for figure in figures:
            close_figure(figure)


def test_feature_regression_grid_combines_saved_strata(
    tmp_path,
    monkeypatch,
) -> None:
    """A regression grid renders each stratum's observations."""
    unit_frame = pd.DataFrame(
        {
            "feature_id": ["NASP_DNA_SENSING_score"] * 6,
            "feature_label": ["NASP_DNA_SENSING"] * 6,
            "feature_value": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8],
            "age_years": [20.0, 40.0, 60.0] * 2,
            "tissue_in_publication": ["liver"] * 3 + ["lung"] * 3,
            "statistical_unit": ["donor_tissue"] * 6,
        }
    )
    results = pd.DataFrame(
        {
            "feature_id": ["NASP_DNA_SENSING_score"] * 2,
            "feature_type": ["module_score"] * 2,
            "stratify_key": ["tissue_in_publication"] * 2,
            "stratum": ["liver", "lung"],
            "slope": [0.005, 0.005],
            "intercept": [0.0, 0.5],
            "pearson_r": [1.0, 1.0],
            "pearson_pvalue": [0.001, 0.001],
            "skipped": [False, False],
        }
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        AssociationPlotter(tmp_path).plot_feature_regression_grid(
            unit_frame,
            feature_id="NASP_DNA_SENSING_score",
            predictor_key="age_years",
            filename="atlas_tissue_regressions",
            results=results,
            stratify_key="tissue_in_publication",
        )

        axes = figures[0].axes
        assert [axis.get_title().splitlines()[0] for axis in axes] == [
            "Liver",
            "Lung",
        ]
        np.testing.assert_allclose(
            axes[0].collections[0].get_offsets()[:, 1],
            [0.1, 0.2, 0.3],
        )
        np.testing.assert_allclose(
            axes[1].collections[0].get_offsets()[:, 1],
            [0.6, 0.7, 0.8],
        )
        assert all(len(axis.lines) == 1 for axis in axes)
        text_sizes = {
            text.get_fontsize()
            for text in figures[0].findobj(Text)
            if text.get_text()
        }
        assert text_sizes == {5.0}
        assert (tmp_path / "atlas_tissue_regressions.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_feature_regression_grid_marks_non_estimable_strata(
    tmp_path,
    monkeypatch,
) -> None:
    """A non-estimable stratum keeps its points without drawing a fit line."""
    unit_frame = pd.DataFrame(
        {
            "feature_id": ["CGAS"] * 2,
            "feature_value": [0.1, 0.1],
            "age_years": [20.0, 40.0],
            "tissue": ["liver"] * 2,
        }
    )
    results = pd.DataFrame(
        {
            "feature_id": ["CGAS"],
            "stratify_key": ["tissue"],
            "stratum": ["liver"],
            "skipped": [True],
        }
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        AssociationPlotter(tmp_path).plot_feature_regression_grid(
            unit_frame,
            feature_id="CGAS",
            predictor_key="age_years",
            filename="non_estimable_regression",
            results=results,
            stratify_key="tissue",
        )

        axis = figures[0].axes[0]
        assert "Not estimable" in axis.get_title()
        assert len(axis.collections[0].get_offsets()) == 2
        assert len(axis.lines) == 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_feature_group_boxplot_stratifies_units_with_readable_context(
    tmp_path,
    monkeypatch,
) -> None:
    """Grouped scores render sex-stratified boxes with unit-aware labels."""
    unit_frame = pd.DataFrame(
        {
            "feature_id": ["NASP_DNA_SENSING_auc"] * 12,
            "feature_label": ["NASP_DNA_SENSING"] * 12,
            "feature_value": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
                1.10,
                1.20,
            ],
            "cell_type": ["B_cell"] * 6 + ["T_cell"] * 6,
            "sex": (["male"] * 3 + ["female"] * 3) * 2,
            "statistical_unit": ["donor_tissue_cell_type"] * 12,
        }
    )
    result_row = pd.Series(
        {
            "feature_type": "module_score",
            "parametric_test": "anova",
            "pvalue": 0.012,
        }
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        AssociationPlotter(output_dir=tmp_path).plot_feature_group_boxplot(
            unit_frame,
            feature_id="NASP_DNA_SENSING_auc",
            group_key="cell_type",
            filename="cell_type_boxplot",
            result_row=result_row,
            stratify_key="sex",
        )

        ax = figures[0].axes[0]
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "T cell",
            "B cell",
        ]
        assert ax.get_xlabel() == "Cell type"
        assert ax.get_ylabel() == "NASP DNA sensing score\n(AUCell)"
        assert {text.get_text() for text in ax.get_legend().get_texts()} == {
            "Male",
            "Female",
        }
        assert (tmp_path / "cell_type_boxplot.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


def test_feature_group_boxplot_default_width_scales_with_group_count(
    tmp_path,
    monkeypatch,
) -> None:
    """Categorical plots retain their tuned height and per-group width."""
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    def unit_frame(group_count: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature_id": ["NASP_DNA_SENSING_auc"] * group_count,
                "feature_value": np.linspace(0.1, 1.0, group_count),
                "cell_type": [
                    f"cell_type_{index:02d}" for index in range(group_count)
                ],
                "statistical_unit": ["donor_tissue_cell_type"] * group_count,
            }
        )

    try:
        for group_count in (2, 24):
            AssociationPlotter(output_dir=tmp_path).plot_feature_group_boxplot(
                unit_frame(group_count),
                feature_id="NASP_DNA_SENSING_auc",
                group_key="cell_type",
                filename=f"cell_type_boxplot_{group_count}",
            )

        np.testing.assert_allclose(figures[0].get_size_inches(), (1.8, 1.5))
        np.testing.assert_allclose(
            figures[1].get_size_inches(),
            (24 * 0.145, 1.5),
        )
    finally:
        for figure in figures:
            close_figure(figure)


def test_plotter_maps_cross_scorer_module_pairs_to_heatmap(
    tmp_path,
    monkeypatch,
) -> None:
    """Cross-scorer module pairs map to the requested matrix coordinates."""
    concordance = pd.DataFrame(
        {
            "scanpy_module_id": [
                "NASP_DNA_SENSING",
                "NASP_DNA_SENSING",
                "IFN_I_OUTPUT",
                "IFN_I_OUTPUT",
            ],
            "aucell_module_id": [
                "NASP_DNA_SENSING",
                "IFN_I_OUTPUT",
                "NASP_DNA_SENSING",
                "IFN_I_OUTPUT",
            ],
            "spearman_r": [0.92, 0.35, 0.28, 0.64],
        }
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(plt, "close", figures.append)

    try:
        SummaryPlotter(output_dir=tmp_path).plot_scorer_concordance_heatmap(
            concordance,
            filename="scorer_concordance",
            module_order=["NASP_DNA_SENSING", "IFN_I_OUTPUT"],
        )

        ax = figures[0].axes[0]
        np.testing.assert_allclose(
            ax.images[0].get_array(),
            [[0.92, 0.28], [0.35, 0.64]],
        )
        assert ax.get_xlabel() == "Scanpy"
        assert ax.get_ylabel() == "AUCell"
        assert not ax.lines
        assert (tmp_path / "scorer_concordance.png").stat().st_size > 0
    finally:
        for figure in figures:
            close_figure(figure)


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
