"""Tests for single-cell utilities."""

from __future__ import annotations

from collections import Counter

import anndata as ad  # type: ignore[import]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from matplotlib.axes import Axes
from nasp_compendium.types import GeneModule

from nasp_atlas.cellxgene import add_development_stage_age_obs
from nasp_atlas.single_cell import combine_module_scores
from nasp_atlas.single_cell import expression_matrix
from nasp_atlas.single_cell import normalize_h5ad_string_storage
from nasp_atlas.single_cell import read_h5ad
from nasp_atlas.single_cell import read_h5ad_rows
from nasp_atlas.single_cell import split_anndata_by_obs
from nasp_atlas.single_cell.visualization import HeatmapPlotter
from nasp_atlas.single_cell.visualization import UmapPlotter


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


def test_split_anndata_by_obs_preserves_cells_and_expression(tmp_path) -> None:
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

    observed_cells = []
    for tissue, path in written.items():
        subset = ad.read_h5ad(path)
        expected = adata[adata.obs.tissue_type.eq(tissue)]
        assert subset.obs_names.tolist() == expected.obs_names.tolist()
        np.testing.assert_allclose(subset.X, expected.X)
        observed_cells.extend(subset.obs_names)
    assert Counter(observed_cells) == Counter(adata.obs_names)


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

    np.testing.assert_allclose(plotted_values[0], [1.0, 2.0])
    assert (tmp_path / "gene_panel.png").stat().st_size > 0


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
    scatter = Axes.scatter

    def capture_scatter(self, *args, **kwargs):
        collection = scatter(self, *args, **kwargs)
        collections.append(collection)
        return collection

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
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
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    assert (tmp_path / "standardized_scores.png").stat().st_size > 0


def test_multi_obs_umap_marks_constant_zscore_unavailable(
    tmp_path,
    monkeypatch,
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
    scatter = Axes.scatter

    def capture_scatter(self, *args, **kwargs):
        collection = scatter(self, *args, **kwargs)
        collections.append(collection)
        return collection

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    plotter = UmapPlotter(output_dir=tmp_path)

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

    assert np.ma.getmaskarray(collections[0].get_array()).all()
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
