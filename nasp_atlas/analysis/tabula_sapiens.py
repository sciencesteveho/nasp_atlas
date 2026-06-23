"""Tabula Sapiens NASP scoring workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore
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


def run_tabula_sapiens_scoring_analysis(
    *,
    h5ad_path: str | Path,
    output_dir: str | Path,
    subset_fraction: float | None = None,
    random_state: int = 0,
    tissue_key: str = "tissue_in_publication",
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
    single_tissue: str | None = None,
    single_tissue_use_rep: str | None = "X_scvi",
) -> None:
    """Run Tabula Sapiens metadata plots and NASP module scoring."""
    # Load data
    adata, _ = read_h5ad(
        h5ad_path,
        subset_fraction=subset_fraction,
        random_state=random_state,
    )
    point_size = 75000 / adata.n_obs

    # If single tissue, recompute neighbors and UMAP
    if single_tissue is not None:
        SCProcessor._recompute_umap(
            adata,
            use_rep=single_tissue_use_rep,
            random_state=random_state,
        )

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

    # Scoring functions and viz
    score_cmap = SCVisualizer.umap_expression_cmap("viridis")
    if score_scanpy:
        scanpy_modules = score_scanpy_modules(
            adata,
            selected_module_ids,
            gene_symbol_column=gene_symbol_column,
            random_state=random_state,
        )
        viz.plot_multi_obs_umap_panel(
            adata,
            obs_keys=[
                module_score_name(module, scorer="scanpy")
                for module in scanpy_modules
            ],
            filename="tabula_sapiens_scanpy_module_umaps",
            cmap=score_cmap,
            ncols=5,
            size=point_size,
            vmin=0,
        )

    if score_aucell:
        adata_auc, _, auc_modules = score_aucell_modules(
            adata,
            selected_module_ids,
            gene_symbol_column=gene_symbol_column,
        )
        viz.plot_multi_obs_umap_panel(
            adata_auc,
            obs_keys=[
                module_score_name(module, scorer="aucell")
                for module in auc_modules
            ],
            filename="tabula_sapiens_aucell_module_umaps",
            cmap=score_cmap,
            ncols=5,
            size=point_size,
            vmin=0,
        )


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
