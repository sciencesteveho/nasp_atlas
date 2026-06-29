"""Tabula Sapiens NASP scoring workflow."""

from __future__ import annotations

import gc
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore

from nasp_atlas.cellxgene.metadata import add_development_stage_age_obs
from nasp_atlas.cellxgene.metadata import category_color_map_from_uns
from nasp_atlas.single_cell.io import read_h5ad
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import plot_module_gene_umaps
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_modules
from nasp_atlas.single_cell.scprocessor import SCProcessor
from nasp_atlas.single_cell.visualization import SCVisualizer
from nasp_atlas.single_cell.visualization import UmapPanelSpec


logger = logging.getLogger(__name__)


def run_tabula_sapiens_scoring_analysis(
    *,
    h5ad_path: str | Path,
    output_dir: str | Path,
    subset_fraction: float | None = None,
    random_state: int = 0,
    tissue_key: str = "tissue_in_publication",
    cell_type_key: str = "cell_type",
    sex_key: str = "sex",
    development_stage_key: str = "development_stage",
    age_key: str = "age_years",
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    module_ids: Sequence[str] | None = None,
    metadata_panels: Sequence[str | UmapPanelSpec] | None = None,
    sensor_group: str = "nucleic_acid_sensors",
    plot_modules: bool = False,
    score_scanpy: bool = False,
    score_aucell: bool = False,
    score_table_filename: str = "tabula_sapiens_module_scores.csv.gz",
    heatmap_obs_keys: Sequence[str] | None = None,
    score_heatmap_obs_key: str | None = None,
    single_tissue: str | None = None,
    single_tissue_use_rep: str | None = "X_scvi",
) -> None:
    """Run Tabula Sapiens metadata plots and NASP module scoring.

    Args:
      h5ad_path: Input h5ad path.
      output_dir: Directory where plots and score tables are written.
      subset_fraction: Optional row fraction to load for exploratory runs.
      random_state: Random seed used for subsetting and recomputed UMAPs.
      tissue_key: Obs column identifying tissues.
      cell_type_key: Obs column identifying cell types.
      sex_key: Obs column identifying donor sex.
      development_stage_key: Obs column with CELLxGENE development stage.
      age_key: Obs column to hold numeric age in years.
      gene_symbol_column: Var column containing gene symbols.
      expression_layer: Expression layer used for gene-expression plots.
      module_ids: Optional module IDs to score and plot. Defaults to all
        modules from `GeneModules`.
      metadata_panels: Optional metadata UMAP panel specification.
      sensor_group: Sensor group name used to select sensor genes.
      plot_modules: Whether to plot per-module marker-gene UMAPs and heatmaps.
      score_scanpy: Whether to score modules with scanpy.
      score_aucell: Whether to score modules with AUCell.
      score_table_filename: Output score table filename under `output_dir`.
      heatmap_obs_keys: Obs columns used for sensor/module gene-expression
        heatmaps. Defaults to tissue and cell type.
      score_heatmap_obs_key: Obs column used for scanpy/AUCell score heatmaps.
        Defaults to `cell_type_key`; pass `tissue_key` for atlas-wide tissue
        summaries.
      single_tissue: If set, recompute neighbors/UMAP for one tissue.
      single_tissue_use_rep: Representation used for single-tissue UMAP
        recomputation.
    """
    # Load data
    adata, _ = read_h5ad(
        h5ad_path,
        subset_fraction=subset_fraction,
        random_state=random_state,
    )

    # If single tissue, recompute neighbors and UMAP
    point_size = 75000 / adata.n_obs
    if single_tissue is not None:
        SCProcessor._recompute_umap(
            adata,
            use_rep=single_tissue_use_rep,
            random_state=random_state,
            min_dist=0.425,
        )
        point_size /= 4

    # Init visualizer and plot metadata
    viz = SCVisualizer(output_dir=output_dir)

    _plot_tabula_sapiens_metadata_umaps(
        adata,
        viz=viz,
        tissue_key=tissue_key,
        sex_key=sex_key,
        development_stage_key=development_stage_key,
        age_key=age_key,
        filename="tabula_sapiens_metadata_umaps",
        panels=metadata_panels,
        size=point_size,
    )

    heatmap_groupby_keys = _resolve_heatmap_obs_keys(
        tissue_key=tissue_key,
        cell_type_key=cell_type_key,
        heatmap_obs_keys=heatmap_obs_keys,
    )
    score_heatmap_groupby_key = score_heatmap_obs_key or cell_type_key

    # Sensor-specific UMAP
    sensors = GeneModules.sensors(sensor_group)
    viz.plot_multi_gene_umap_panel(
        adata=adata,
        genes=sensors,
        filename="NA_SENSORS_gene_expression_umaps",
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
        ncols=6,
        size=point_size,
    )
    _plot_gene_expression_heatmaps_by_obs(
        adata=adata,
        genes=sensors,
        viz=viz,
        filename_prefix="NA_SENSORS",
        groupby_keys=heatmap_groupby_keys,
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
    )

    if module_ids is None:
        selected_module_ids = GeneModules().module_ids()
    else:
        selected_module_ids = list(module_ids)

    # Per-module marker gene umaps
    if plot_modules:
        plot_module_gene_umaps(
            adata=adata,
            module_ids=selected_module_ids,
            viz=viz,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            ncols=6,
            size=point_size,
        )
        _plot_module_gene_heatmaps_by_obs(
            adata=adata,
            module_ids=selected_module_ids,
            viz=viz,
            groupby_keys=heatmap_groupby_keys,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
        )

    # Scoring functions and viz
    score_tables: list[pd.DataFrame] = []
    scanpy_scores: pd.DataFrame | None = None
    if score_scanpy:
        scanpy_modules = score_scanpy_modules(
            adata,
            selected_module_ids,
            gene_symbol_column=gene_symbol_column,
            random_state=random_state,
        )
        scanpy_score_keys = [
            module_score_name(module, scorer="scanpy")
            for module in scanpy_modules
        ]
        scanpy_obs = cast(pd.DataFrame, adata.obs)
        scanpy_scores = scanpy_obs.loc[:, scanpy_score_keys].copy()
        score_tables.append(scanpy_scores)
        _write_score_tables(
            score_tables,
            output_dir=output_dir,
            filename=score_table_filename,
        )
        viz.plot_multi_obs_umap_panel(
            adata,
            obs_keys=scanpy_score_keys,
            filename="tabula_sapiens_scanpy_module_umaps",
            cmap="RdYlBu_r",
            ncols=5,
            size=point_size,
            vmin=None,
            vmax=None,
            center_zero=True,
        )
        viz.plot_grouped_obs_score_heatmap(
            adata,
            score_keys=scanpy_score_keys,
            groupby=score_heatmap_groupby_key,
            filename=(
                "tabula_sapiens_scanpy_module_score_heatmap_by_"
                f"{_safe_filename_token(score_heatmap_groupby_key)}"
            ),
            score_labels=[str(module.module_id) for module in scanpy_modules],
            cmap="RdYlBu_r",
        )
        del scanpy_obs, scanpy_modules, scanpy_score_keys
        if not score_aucell:
            score_tables.clear()
            scanpy_scores = None
            gc.collect()

    if score_aucell:
        adata_auc, auc_df, auc_modules = score_aucell_modules(
            adata,
            selected_module_ids,
            gene_symbol_column=gene_symbol_column,
        )
        auc_score_keys: list[str] | None = None
        auc_obs: pd.DataFrame | None = None
        auc_scores: pd.DataFrame | None = None
        try:
            auc_score_keys = [
                module_score_name(module, scorer="aucell")
                for module in auc_modules
            ]
            auc_obs = cast(pd.DataFrame, adata_auc.obs)
            auc_scores = auc_obs.loc[:, auc_score_keys].copy()
            score_tables.append(auc_scores)
            _write_score_tables(
                score_tables,
                output_dir=output_dir,
                filename=score_table_filename,
            )
            viz.plot_multi_obs_umap_panel(
                adata_auc,
                obs_keys=auc_score_keys,
                filename="tabula_sapiens_aucell_module_umaps",
                cmap="RdYlBu_r",
                ncols=5,
                size=point_size,
                vmin=None,
                vmax=None,
                center_zero=True,
            )
            viz.plot_grouped_obs_score_heatmap(
                adata_auc,
                score_keys=auc_score_keys,
                groupby=score_heatmap_groupby_key,
                filename=(
                    "tabula_sapiens_aucell_module_score_heatmap_by_"
                    f"{_safe_filename_token(score_heatmap_groupby_key)}"
                ),
                score_labels=[str(module.module_id) for module in auc_modules],
                cmap="RdYlBu_r",
            )
        finally:
            del adata_auc, auc_df, auc_modules
            score_tables.clear()
            auc_score_keys = None
            auc_obs = None
            auc_scores = None
            scanpy_scores = None
            gc.collect()


def _write_score_tables(
    score_tables: Sequence[pd.DataFrame],
    *,
    output_dir: str | Path,
    filename: str,
) -> None:
    """Persist the currently available score columns."""
    if not score_tables:
        return

    scores = pd.concat(score_tables, axis="columns")
    score_path = Path(output_dir) / filename
    scores.to_csv(
        score_path,
        index=True,
        index_label="obs_name",
        compression="infer",
    )
    logger.info("[tabula_sapiens] module scores -> %s", score_path)


def _resolve_heatmap_obs_keys(
    *,
    tissue_key: str,
    cell_type_key: str,
    heatmap_obs_keys: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return deduplicated obs keys used for expression heatmaps."""
    requested_keys = (
        (tissue_key, cell_type_key)
        if heatmap_obs_keys is None
        else tuple(heatmap_obs_keys)
    )
    return tuple(dict.fromkeys(requested_keys))


def _plot_gene_expression_heatmaps_by_obs(
    *,
    adata: ad.AnnData,
    genes: Sequence[str],
    viz: SCVisualizer,
    filename_prefix: str,
    groupby_keys: Sequence[str],
    gene_symbol_column: str,
    expression_layer: str | None,
) -> None:
    """Plot one gene-expression heatmap for each requested obs key."""
    gene_list = list(genes)
    if not gene_list:
        return

    for groupby_key in groupby_keys:
        viz.plot_multi_gene_expression_heatmap(
            adata=adata,
            genes=gene_list,
            groupby=groupby_key,
            filename=(
                f"{filename_prefix}_gene_expression_heatmap_by_"
                f"{_safe_filename_token(groupby_key)}"
            ),
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
        )


def _plot_module_gene_heatmaps_by_obs(
    *,
    adata: ad.AnnData,
    module_ids: Sequence[str],
    viz: SCVisualizer,
    groupby_keys: Sequence[str],
    gene_symbol_column: str,
    expression_layer: str | None,
) -> None:
    """Plot module marker-gene heatmaps for each requested obs key."""
    for module_id in module_ids:
        module_genes = GeneModules.genes(
            module_id,
            adata=adata,
            gene_symbol_column=gene_symbol_column,
            output="symbols",
        )
        if not module_genes:
            logger.info("%s: no matched genes; skipping heatmaps", module_id)
            continue

        logger.info(
            "%s: plotting %s marker gene heatmaps",
            module_id,
            len(module_genes),
        )
        _plot_gene_expression_heatmaps_by_obs(
            adata=adata,
            genes=module_genes,
            viz=viz,
            filename_prefix=module_id,
            groupby_keys=groupby_keys,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
        )


def _safe_filename_token(value: str) -> str:
    """Return a filesystem-safe token for generated plot filenames."""
    token = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )
    return token.strip("_") or "obs"


def _plot_tabula_sapiens_metadata_umaps(
    adata: ad.AnnData,
    *,
    viz: SCVisualizer,
    tissue_key: str = "tissue_in_publication",
    sex_key: str = "sex",
    development_stage_key: str = "development_stage",
    age_key: str = "age_years",
    filename: str = "tabula_sapiens_metadata_umaps",
    sex_color_map: Mapping[str, str] | None = None,
    tissue_color_map: Mapping[str, str] | None = None,
    panels: Sequence[str | UmapPanelSpec] | None = None,
    size: float | None = None,
) -> None:
    """Plot selected Tabula Sapiens metadata UMAPs in requested order."""
    add_development_stage_age_obs(
        adata,
        stage_column=development_stage_key,
        age_column=age_key,
    )
    point_size = size if size is not None else 25000 / adata.n_obs
    sex_colors = dict(
        sex_color_map
        if sex_color_map is not None
        else {
            "female": "#f4cae4",
            "male": "#cbd5e8",
        }
    )
    tissue_colors = _resolve_tissue_color_map(
        adata=adata,
        tissue_key=tissue_key,
        tissue_color_map=tissue_color_map,
    )

    tissue_panel: UmapPanelSpec = {
        "obs_key": tissue_key,
        "title": "Tissues",
        "kind": "categorical",
        "legend_loc": "bottom",
        "legend_ncol": 3,
    }
    if tissue_colors is not None:
        tissue_panel["color_map"] = tissue_colors

    panel_defaults: dict[str, UmapPanelSpec] = {
        tissue_key: tissue_panel,
        sex_key: {
            "obs_key": sex_key,
            "title": "Sex",
            "kind": "categorical",
            "color_map": sex_colors,
            "legend_loc": "bottom",
            "legend_ncol": 2,
        },
        age_key: {
            "obs_key": age_key,
            "title": "Age",
            "kind": "numeric",
            "cmap": "viridis",
            "cbar_ticks": [30, 40, 50, 60, 70],
        },
    }

    selected_panels = (
        panels if panels is not None else (tissue_key, sex_key, age_key)
    )

    requested_panels: list[str | UmapPanelSpec] = []
    for panel in selected_panels:
        if isinstance(panel, str):
            requested_panels.append(panel_defaults.get(panel, panel))
            continue

        default = panel_defaults.get(panel["obs_key"], {})  # type: ignore
        requested_panels.append(cast(UmapPanelSpec, {**default, **panel}))

    viz.plot_umap_panel(
        adata,
        panels=requested_panels,
        filename=filename,
        ncols=len(requested_panels),
        size=point_size,
        col_wspace=0.005,
    )


def _resolve_tissue_color_map(
    *,
    adata: ad.AnnData,
    tissue_key: str,
    tissue_color_map: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Return tissue colors from input mapping or Scanpy.uns."""
    if tissue_color_map is not None:
        return dict(tissue_color_map)

    if f"{tissue_key}_colors" not in adata.uns:
        return None

    return {
        str(category): color
        for category, color in category_color_map_from_uns(
            adata,
            tissue_key,
        ).items()
    }
