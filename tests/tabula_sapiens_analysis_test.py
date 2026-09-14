"""Tests for the Tabula Sapiens analysis workflow."""

from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import pytest
from nasp_compendium.types import GeneModule  # type: ignore[import]


tabula_sapiens = importlib.import_module("nasp_atlas.analysis.tabula_sapiens")
tabula_sapiens_workflows = importlib.import_module(
    "nasp_atlas.analysis.tabula_sapiens.workflows"
)


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
