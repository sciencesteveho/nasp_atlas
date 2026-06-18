"""Tests for single-cell utilities."""

from __future__ import annotations

import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import anndata as ad  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nasp_atlas.single_cell import EmbeddingConfig
from nasp_atlas.single_cell import SCUtils
from nasp_atlas.single_cell import SCVisualizer


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
