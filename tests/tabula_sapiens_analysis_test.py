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


def test_single_tissue_subsets_before_recomputing_umap(
    tmp_path,
    monkeypatch,
) -> None:
    """The single-tissue option removes other tissues before UMAP work."""
    adata = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "tissue_in_publication": ["lung", "liver", "lung", "liver"],
                "cell_type": ["A", "B", "A", "B"],
            },
            index=["a", "b", "c", "d"],
        ),
        var=pd.DataFrame({"feature_name": ["CGAS"]}, index=["CGAS"]),
    )
    observed_sizes: dict[str, int] = {}

    def capture_recompute(adata_arg, **kwargs) -> None:
        observed_sizes["recompute"] = adata_arg.n_obs

    def capture_metadata_plot(adata_arg, **kwargs) -> None:
        observed_sizes["metadata_plot"] = adata_arg.n_obs

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "read_h5ad",
        lambda *args, **kwargs: (adata, adata.n_obs),
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.SCProcessor,
        "recompute_umap",
        capture_recompute,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "plot_tabula_sapiens_metadata_umaps",
        capture_metadata_plot,
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

    tabula_sapiens.tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
        module_ids=[],
        single_tissue="lung",
    )

    assert observed_sizes == {"recompute": 2, "metadata_plot": 2}


def test_tabula_sapiens_saves_combined_scores_and_plots_score_umaps(
    tmp_path,
    monkeypatch,
) -> None:
    """The workflow saves scores and plots only native-scale score UMAPs."""
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
    retired_score_umaps = [
        tmp_path / "tabula_sapiens_scanpy_module_zscore_umaps.png",
        tmp_path / "tabula_sapiens_aucell_module_zscore_umaps.png",
    ]
    for retired_score_umap in retired_score_umaps:
        retired_score_umap.write_bytes(b"stale figure")

    plot_calls: list[dict[str, object]] = []
    score_heatmap_calls: list[dict[str, object]] = []
    concordance_calls: list[pd.DataFrame] = []

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

    def capture_score_plot(*args, **kwargs) -> None:
        plot_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.UmapPlotter,
        "plot_multi_obs_umap_panel",
        capture_score_plot,
    )

    def capture_score_heatmap(*args, **kwargs) -> None:
        score_heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "plot_grouped_obs_score_heatmap",
        capture_score_heatmap,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.SummaryPlotter,
        "plot_scorer_concordance_heatmap",
        lambda _, concordance, **kwargs: concordance_calls.append(concordance),
    )

    def fake_scanpy_scores(adata_arg, *args, **kwargs):
        adata_arg.obs["NASP_DNA_SENSING_pos"] = [0.1, 0.8]
        adata_arg.obs["NASP_DNA_SENSING_score"] = [-2.0, 1.0]
        return [module]

    def fake_aucell_scores(adata_arg, *args, **kwargs):
        adata_auc = adata_arg.copy()
        adata_auc.obs["NASP_DNA_SENSING_pos_auc"] = [0.2, 0.9]
        adata_auc.obs["NASP_DNA_SENSING_auc"] = [-0.25, 0.5]
        auc_df = adata_auc.obs[
            ["NASP_DNA_SENSING_pos_auc", "NASP_DNA_SENSING_auc"]
        ].copy()
        return adata_auc, auc_df, [module]

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "score_scanpy_modules",
        fake_scanpy_scores,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "score_aucell_modules",
        fake_aucell_scores,
    )

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
    assert scores.index.tolist() == ["cell_a", "cell_b"]
    assert {
        "cell_type",
        "sex",
        "assay",
        "development_stage",
        "scoring_scorers",
        "scoring_module_ids",
        "scoring_n_modules",
        "NASP_DNA_SENSING_pos",
        "NASP_DNA_SENSING_score",
        "NASP_DNA_SENSING_pos_auc",
        "NASP_DNA_SENSING_auc",
    } <= set(scores.columns)
    assert scores.loc[
        "cell_a", ["cell_type", "sex", "assay", "development_stage"]
    ].tolist() == [
        "T cell",
        "female",
        "10x 3' v3",
        "50-year-old human stage",
    ]
    assert scores["scoring_scorers"].unique().tolist() == ["scanpy,aucell"]
    assert scores["scoring_requested_scorers"].unique().tolist() == [
        "scanpy,aucell"
    ]
    assert scores["scoring_module_ids"].unique().tolist() == [
        "NASP_DNA_SENSING"
    ]
    assert scores["scoring_n_modules"].unique().tolist() == [1]
    assert scores.loc["cell_a", "NASP_DNA_SENSING_score"] == -2.0
    assert scores.loc["cell_a", "NASP_DNA_SENSING_auc"] == -0.25
    calls_by_filename = {str(call["filename"]): call for call in plot_calls}
    assert set(calls_by_filename) == {
        "tabula_sapiens_scanpy_module_umaps",
        "tabula_sapiens_aucell_module_umaps",
    }
    for scorer, score_key in (
        ("scanpy", "NASP_DNA_SENSING_score"),
        ("aucell", "NASP_DNA_SENSING_auc"),
    ):
        raw_call = calls_by_filename[f"tabula_sapiens_{scorer}_module_umaps"]
        assert raw_call["obs_keys"] == [score_key]
        assert raw_call["standardization"] == "none"
        assert raw_call["shared_colorbar"] is False
    assert all(not path.exists() for path in retired_score_umaps)
    assert Counter(
        (tuple(call["score_keys"]), call["groupby"])
        for call in score_heatmap_calls
    ) == Counter(
        [
            (("NASP_DNA_SENSING_score",), "cell_type"),
            (("NASP_DNA_SENSING_auc",), "cell_type"),
        ]
    )
    assert len(concordance_calls) == 1
    assert concordance_calls[0]["scanpy_module_id"].tolist() == [
        "NASP_DNA_SENSING"
    ]
    assert concordance_calls[0]["aucell_module_id"].tolist() == [
        "NASP_DNA_SENSING"
    ]
    assert (
        tmp_path / "tabula_sapiens_cross_scorer_module_correlations.csv"
    ).is_file()


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
        lambda *args, **kwargs: ["CGAS"],
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

    def capture_expression_heatmap(*args, **kwargs) -> None:
        expression_heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "plot_multi_gene_expression_heatmap",
        capture_expression_heatmap,
    )

    def fake_scanpy_scores(adata_arg, *args, **kwargs):
        adata_arg.obs["NASP_DNA_SENSING_score"] = [1.0, 2.0]
        return [module]

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "score_scanpy_modules",
        fake_scanpy_scores,
    )

    def capture_score_heatmap(*args, **kwargs) -> None:
        score_heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "plot_grouped_obs_score_heatmap",
        capture_score_heatmap,
    )

    tabula_sapiens.tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
        module_ids=["NASP_DNA_SENSING"],
        heatmap_groupby="tissue_in_publication",
        score_scanpy=True,
    )

    assert {call["groupby"] for call in expression_heatmap_calls} == {
        "tissue_in_publication"
    }
    assert {call["groupby"] for call in score_heatmap_calls} == {
        "tissue_in_publication"
    }


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
        lambda *args, **kwargs: ["CGAS"],
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.UmapPlotter,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: None,
    )

    def capture_heatmap(*args, **kwargs) -> None:
        heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "plot_multi_gene_expression_heatmap",
        capture_heatmap,
    )

    tabula_sapiens.tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
    )

    assert Counter(
        (tuple(call["genes"]), call["groupby"]) for call in heatmap_calls
    ) == Counter(
        [
            (("CGAS",), "tissue_in_publication"),
            (("CGAS",), "cell_type"),
        ]
    )


def test_tabula_sapiens_module_heatmaps_follow_marker_umap_modules(
    tmp_path,
    monkeypatch,
) -> None:
    """Module marker heatmaps use the same selected modules as marker UMAPs."""
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            {
                "tissue_in_publication": ["lung", "blood"],
                "cell_type": ["T cell", "B cell"],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame(
            {"feature_name": ["CGAS", "IFIH1"]},
            index=["gene_a", "gene_b"],
        ),
    )
    heatmap_calls: list[dict[str, object]] = []
    summary_calls: list[dict[str, object]] = []
    umap_calls: list[dict[str, object]] = []
    module_gene_calls: list[str] = []

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
        lambda *args, **kwargs: ["CGAS"],
    )

    def capture_module_genes(module_id, *args, **kwargs):
        module_gene_calls.append(module_id)
        return ["IFIH1"]

    monkeypatch.setattr(
        tabula_sapiens_workflows.GeneModules,
        "genes",
        capture_module_genes,
    )

    summarize_expression = (
        tabula_sapiens_workflows.HeatmapPlotter.summarize_gene_expression_by_obs
    )

    def capture_summary(self, adata_arg, genes, **kwargs):
        summary_calls.append({"genes": tuple(genes), **kwargs})
        return summarize_expression(self, adata_arg, genes, **kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "summarize_gene_expression_by_obs",
        capture_summary,
    )

    def capture_module_umaps(*args, **kwargs) -> None:
        umap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.UmapPlotter,
        "plot_multi_gene_umap_panel",
        capture_module_umaps,
    )

    def capture_heatmap(*args, **kwargs) -> None:
        heatmap_calls.append(kwargs)

    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "plot_multi_gene_expression_heatmap",
        capture_heatmap,
    )

    tabula_sapiens.tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
        module_ids=["NASP_DNA_SENSING"],
        plot_modules=True,
    )

    module_umap_calls = [
        call
        for call in umap_calls
        if call.get("filename") != "NA_SENSORS_gene_expression_umaps"
    ]
    assert Counter(
        tuple(call["genes"]) for call in module_umap_calls
    ) == Counter([("IFIH1",)])
    assert Counter(
        (tuple(call["genes"]), call["groupby"]) for call in heatmap_calls
    ) == Counter(
        [
            (("CGAS",), "tissue_in_publication"),
            (("CGAS",), "cell_type"),
            (("IFIH1",), "tissue_in_publication"),
            (("IFIH1",), "cell_type"),
        ]
    )
    assert module_gene_calls == ["NASP_DNA_SENSING"]
    assert Counter(
        (call["genes"], call["groupby"]) for call in summary_calls
    ) == Counter(
        [
            (("CGAS", "IFIH1"), "tissue_in_publication"),
            (("CGAS", "IFIH1"), "cell_type"),
        ]
    )
    assert all(call["grouped_expression"] is not None for call in heatmap_calls)


def test_tabula_sapiens_expression_umaps_use_shared_gene_scale(
    tmp_path,
    monkeypatch,
) -> None:
    """Sensor and module expression UMAPs request comparable gene colors."""
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            {
                "tissue_in_publication": ["lung", "blood"],
                "cell_type": ["T cell", "B cell"],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame(
            {"feature_name": ["CGAS", "IFIH1"]},
            index=["gene_a", "gene_b"],
        ),
    )
    umap_calls: list[dict[str, object]] = []

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
        lambda *args, **kwargs: ["CGAS"],
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.GeneModules,
        "genes",
        lambda *args, **kwargs: ["IFIH1"],
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "summarize_gene_expression_by_obs",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.UmapPlotter,
        "plot_multi_gene_umap_panel",
        lambda *args, **kwargs: umap_calls.append(kwargs),
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.HeatmapPlotter,
        "plot_multi_gene_expression_heatmap",
        lambda *args, **kwargs: None,
    )

    tabula_sapiens.tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "input.h5ad",
        output_dir=tmp_path,
        module_ids=["NASP_DNA_SENSING"],
        plot_modules=True,
    )

    assert Counter(
        (tuple(call["genes"]), call["shared_colorbar"]) for call in umap_calls
    ) == Counter(
        [
            (("CGAS",), True),
            (("IFIH1",), True),
        ]
    )


def test_tissue_analysis_scores_once_then_analyzes_each_scorer(
    tmp_path,
    monkeypatch,
) -> None:
    """One shared scoring run feeds separate Scanpy and AUCell analyses."""
    h5ad_path = tmp_path / "liver.h5ad"
    h5ad_path.touch()
    scoring_calls: list[dict[str, object]] = []
    association_calls: list[dict[str, object]] = []

    def capture_scoring(**kwargs) -> None:
        scoring_calls.append(kwargs)
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
        lambda **kwargs: association_calls.append(kwargs),
    )

    outputs = tabula_sapiens.tabula_sapiens_tissue_analysis(
        h5ad_path=h5ad_path,
        output_dir=tmp_path / "results",
        tissue_label="liver",
        run_name="liver_run",
        module_ids=["NASP_DNA_SENSING"],
        detection_threshold=0.25,
    )

    assert len(scoring_calls) == 1
    assert scoring_calls[0]["score_scanpy"]
    assert scoring_calls[0]["score_aucell"]
    assert scoring_calls[0]["heatmap_groupby"] == "cell_type"
    assert scoring_calls[0]["single_tissue"] == "liver"
    assert Counter(call["scorer"] for call in association_calls) == Counter(
        ["scanpy", "aucell"]
    )
    assert all(
        call["score_csv_path"] == outputs["score_table"]
        for call in association_calls
    )
    assert all(
        call["detection_threshold"] == 0.25 for call in association_calls
    )
    assert outputs["association_scanpy"].name == "scanpy"
    assert outputs["association_aucell"].name == "aucell"


def test_tabula_sapiens_analysis_omits_subsetting_for_complete_atlas(
    tmp_path,
    monkeypatch,
) -> None:
    """An omitted tissue label passes the complete h5ad to scoring."""
    h5ad_path = tmp_path / "tabula_sapiens.h5ad"
    h5ad_path.touch()
    scoring_calls: list[dict[str, object]] = []

    def capture_scoring(**kwargs) -> None:
        scoring_calls.append(kwargs)
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

    outputs = tabula_sapiens.tabula_sapiens_tissue_analysis(
        h5ad_path=h5ad_path,
        output_dir=tmp_path / "results",
        run_name="full_atlas",
    )

    assert {call["single_tissue"] for call in scoring_calls} == {None}
    assert outputs["run_dir"] == tmp_path / "results" / "full_atlas"


def test_tissue_analysis_resume_reuses_complete_score_table(
    tmp_path,
    monkeypatch,
) -> None:
    """Resume skips scoring only when every requested scorer completed."""
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
        lambda **kwargs: pytest.fail("complete scores should be reused"),
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
    scoring_calls = 0

    def capture_scoring(**kwargs) -> None:
        nonlocal scoring_calls
        scoring_calls += 1
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

    assert scoring_calls == 1


def test_single_tissue_loader_keeps_umap_representation(
    tmp_path,
    monkeypatch,
) -> None:
    """Tissue UMAP recomputation loads its requested representation."""
    adata = ad.AnnData(
        X=np.ones((1, 1)),
        obs=pd.DataFrame(
            {"tissue_in_publication": ["liver"]},
            index=["cell_a"],
        ),
        var=pd.DataFrame({"feature_name": ["CGAS"]}, index=["CGAS"]),
    )
    loaded_obsm: list[tuple[str, ...]] = []

    def capture_read(*args, **kwargs):
        if "obsm_keys" in kwargs:
            loaded_obsm.append(kwargs["obsm_keys"])
        return adata, adata.n_obs

    monkeypatch.setattr(
        tabula_sapiens_workflows,
        "read_h5ad",
        capture_read,
    )
    monkeypatch.setattr(
        tabula_sapiens_workflows.SCProcessor,
        "recompute_umap",
        lambda *args, **kwargs: None,
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

    tabula_sapiens.tabula_sapiens_scoring_analysis(
        h5ad_path=tmp_path / "liver.h5ad",
        output_dir=tmp_path / "results",
        module_ids=[],
        single_tissue="liver",
        single_tissue_use_rep="X_scvi",
    )

    assert any(set(keys) == {"X_umap", "X_scvi"} for keys in loaded_obsm)
