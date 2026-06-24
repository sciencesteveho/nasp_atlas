"""Tests for the Tabula Sapiens analysis workflow."""

from __future__ import annotations

import importlib
import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
from nasp_compendium.types import GeneModule


tabula_sapiens = importlib.import_module("nasp_atlas.analysis.tabula_sapiens")


def test_symmetric_score_limit_handles_signed_and_zero_scores() -> None:
    """Signed score limits are symmetric and never collapse to zero."""
    scores = pd.DataFrame({"score": [-2.5, 0.0, 1.0, np.nan]})

    assert tabula_sapiens._symmetric_score_limit(scores) == 2.5
    assert (
        tabula_sapiens._symmetric_score_limit(
            pd.DataFrame({"score": [0.0, 0.0]})
        )
        == 1.0
    )


def test_tabula_sapiens_saves_combined_scores_and_uses_signed_umaps(
    tmp_path,
    monkeypatch,
) -> None:
    """The batch workflow persists final scores and plots signed ranges."""
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
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
        "NASP_DNA_SENSING_score",
        "NASP_DNA_SENSING_auc",
    ]
    assert scores.loc["cell_a"].tolist() == [-2.0, -0.25]
    assert plot_calls == [
        {
            "obs_keys": ["NASP_DNA_SENSING_score"],
            "filename": "tabula_sapiens_scanpy_module_umaps",
            "cmap": "RdBu_r",
            "ncols": 5,
            "size": 37500.0,
            "vmin": -2.0,
            "vmax": 2.0,
        },
        {
            "obs_keys": ["NASP_DNA_SENSING_auc"],
            "filename": "tabula_sapiens_aucell_module_umaps",
            "cmap": "RdBu_r",
            "ncols": 5,
            "size": 37500.0,
            "vmin": -0.5,
            "vmax": 0.5,
        },
    ]
