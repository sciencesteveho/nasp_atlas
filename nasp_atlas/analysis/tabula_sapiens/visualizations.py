"""Tabula Sapiens metadata and NASP visualization workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import pandas as pd

from nasp_atlas.cellxgene.metadata import add_development_stage_age_obs
from nasp_atlas.cellxgene.metadata import category_color_map_from_uns
from nasp_atlas.single_cell.umap import UmapPanelSpec
from nasp_atlas.single_cell.visualization import SCVisualizer


__all__ = [
    "metadata_umap_panels",
    "plot_global_nasp_visualizations",
    "plot_nasp_association_visualizations",
    "plot_tabula_sapiens_metadata_umaps",
    "resolve_tissue_color_map",
]


def plot_nasp_association_visualizations(
    *,
    output_dir: str | Path,
    module_coupling: pd.DataFrame,
    context_summary: pd.DataFrame,
    hypothesis_priorities: pd.DataFrame,
    sensor_output_coupling: pd.DataFrame,
    regression_results: pd.DataFrame,
    age_stability: pd.DataFrame,
    mechanistic_edges: pd.DataFrame,
    cell_type_key: str = "cell_type",
) -> None:
    """Plot immediately valid summaries for one tissue analysis.

    Correlation figures are descriptive donor-cell-type summaries. Age-effect
    plots use regressions stratified by tissue and cell type, where donors are
    the independent observations within each stratum.

    Args:
      output_dir: Directory receiving PNG figures.
      module_coupling: Pairwise module-correlation table.
      context_summary: Donor-supported relative NASP context table.
      hypothesis_priorities: Ranked context-hypothesis table.
      sensor_output_coupling: Sensor-gene/output-module coupling table.
      regression_results: Continuous association result table.
      age_stability: Cross-stratum age-effect stability table.
      mechanistic_edges: Expected mechanistic-edge coupling table.
      cell_type_key: Context column identifying cell type.

    Example Usage:
      >>> plot_nasp_association_visualizations(
      ...     output_dir="results/association_plots/nasp",
      ...     module_coupling=module_coupling,
      ...     context_summary=context_summary,
      ...     hypothesis_priorities=hypothesis_priorities,
      ...     sensor_output_coupling=sensor_output_coupling,
      ...     regression_results=regression_results,
      ...     age_stability=age_stability,
      ...     mechanistic_edges=mechanistic_edges,
      ... )
    """
    visualizer = SCVisualizer(output_dir=output_dir)
    if not module_coupling.empty:
        visualizer.plot_module_coupling_heatmap(
            module_coupling,
            filename="nasp_module_coupling_heatmap",
            show_fdr=False,
        )
    if not context_summary.empty:
        visualizer.plot_competence_output_state_map(
            context_summary,
            filename="nasp_competence_output_state_map",
            label_columns=[cell_type_key],
        )
    if not hypothesis_priorities.empty:
        visualizer.plot_ranked_nasp_hypotheses(
            hypothesis_priorities,
            filename="nasp_ranked_hypotheses",
            label_columns=[cell_type_key],
        )
    if not sensor_output_coupling.empty:
        visualizer.plot_sensor_output_mismatch(
            sensor_output_coupling,
            filename="nasp_sensor_output_mismatch",
            show_fdr=False,
        )
    if not regression_results.empty and "analysis_scope" in regression_results:
        visualizer.plot_age_effect_dotplot(
            regression_results,
            filename="nasp_age_effects_by_cell_type",
        )
    if not age_stability.empty:
        visualizer.plot_age_effect_consistency(
            age_stability,
            filename="nasp_age_effect_consistency_across_cell_types",
            analysis_scope="within_tissue_cell_type",
        )
    if not mechanistic_edges.empty:
        visualizer.plot_mechanistic_edge_network(
            mechanistic_edges,
            filename="nasp_mechanistic_edge_network",
            show_fdr=False,
        )


def plot_global_nasp_visualizations(
    *,
    output_dir: str | Path,
    module_coupling: pd.DataFrame,
    context_summary: pd.DataFrame,
    hypothesis_priorities: pd.DataFrame,
    sensor_output_coupling: pd.DataFrame,
    regression_results: pd.DataFrame,
    age_stability: pd.DataFrame,
    mechanistic_edges: pd.DataFrame,
    tissue_key: str = "tissue_in_publication",
    cell_type_key: str = "cell_type",
) -> None:
    """Plot atlas-wide NASP summaries after tissue results are aggregated.

    Duplicate coupling and edge rows are summarized by median correlation to
    produce consensus heatmaps and networks. State maps are faceted by tissue,
    while hypothesis and age plots retain explicit tissue-cell-type labels.
    The caller remains responsible for recomputing global ranks and stability
    tables rather than concatenating per-tissue percentiles as absolute values.

    Args:
      output_dir: Directory receiving atlas-wide PNG figures.
      module_coupling: Concatenated or meta-analyzed module correlations.
      context_summary: Globally harmonized context evidence table.
      hypothesis_priorities: Globally recomputed hypothesis priorities.
      sensor_output_coupling: Concatenated sensor/output coupling table.
      regression_results: Concatenated context-specific age regressions.
      age_stability: Stability recomputed across tissue strata.
      mechanistic_edges: Concatenated or meta-analyzed mechanistic edges.
      tissue_key: Column identifying tissue.
      cell_type_key: Column identifying cell type.

    Example Usage:
      >>> plot_global_nasp_visualizations(
      ...     output_dir="results/global_nasp_plots",
      ...     module_coupling=module_coupling,
      ...     context_summary=context_summary,
      ...     hypothesis_priorities=hypothesis_priorities,
      ...     sensor_output_coupling=sensor_output_coupling,
      ...     regression_results=regression_results,
      ...     age_stability=age_stability,
      ...     mechanistic_edges=mechanistic_edges,
      ... )
    """
    visualizer = SCVisualizer(output_dir=output_dir)
    if not module_coupling.empty:
        visualizer.plot_module_coupling_heatmap(
            module_coupling,
            filename="global_nasp_module_coupling_consensus",
            show_fdr=False,
        )
    if not context_summary.empty:
        visualizer.plot_competence_output_state_map(
            context_summary,
            filename="global_nasp_competence_output_states",
            label_columns=[cell_type_key],
            facet_column=tissue_key,
        )
    if not hypothesis_priorities.empty:
        visualizer.plot_ranked_nasp_hypotheses(
            hypothesis_priorities,
            filename="global_nasp_ranked_hypotheses",
            label_columns=[tissue_key, cell_type_key],
        )
    if not sensor_output_coupling.empty:
        visualizer.plot_sensor_output_mismatch(
            sensor_output_coupling,
            filename="global_nasp_sensor_output_consensus",
            show_fdr=False,
        )
    if not regression_results.empty and "analysis_scope" in regression_results:
        visualizer.plot_age_effect_dotplot(
            regression_results,
            filename="global_nasp_age_effects",
        )
    if not age_stability.empty:
        visualizer.plot_age_effect_consistency(
            age_stability,
            filename="global_nasp_age_effect_consistency",
            analysis_scope="within_tissue",
        )
    if not mechanistic_edges.empty:
        visualizer.plot_mechanistic_edge_network(
            mechanistic_edges,
            filename="global_nasp_mechanistic_edge_consensus",
            show_fdr=False,
        )


def plot_tabula_sapiens_metadata_umaps(
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
    """Plot selected Tabula Sapiens metadata UMAPs in requested order.

    Example Usage:
      >>> plot_tabula_sapiens_metadata_umaps(
      ...     adata,
      ...     viz=viz,
      ...     filename="tabula_sapiens_metadata_umaps",
      ... )
    """
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
    tissue_colors = resolve_tissue_color_map(
        adata=adata,
        tissue_key=tissue_key,
        tissue_color_map=tissue_color_map,
    )

    requested_panels = metadata_umap_panels(
        tissue_key=tissue_key,
        sex_key=sex_key,
        age_key=age_key,
        sex_colors=sex_colors,
        tissue_colors=tissue_colors,
        panels=panels,
    )

    viz.plot_umap_panel(
        adata,
        panels=requested_panels,
        filename=filename,
        ncols=len(requested_panels),
        size=point_size,
        col_wspace=0.005,
    )


def metadata_umap_panels(
    *,
    tissue_key: str,
    sex_key: str,
    age_key: str,
    sex_colors: Mapping[str, str],
    tissue_colors: Mapping[str, str] | None,
    panels: Sequence[str | UmapPanelSpec] | None,
) -> list[str | UmapPanelSpec]:
    """Resolve metadata panel specs with Tabula Sapiens defaults."""
    tissue_panel: UmapPanelSpec = {
        "obs_key": tissue_key,
        "title": "Tissues",
        "kind": "categorical",
        "legend_loc": "bottom",
        "legend_ncol": 3,
    }
    if tissue_colors is not None:
        tissue_panel["color_map"] = dict(tissue_colors)

    panel_defaults: dict[str, UmapPanelSpec] = {
        tissue_key: tissue_panel,
        sex_key: {
            "obs_key": sex_key,
            "title": "Sex",
            "kind": "categorical",
            "color_map": dict(sex_colors),
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

        default = panel_defaults.get(panel["obs_key"], {})
        requested_panels.append(cast(UmapPanelSpec, {**default, **panel}))
    return requested_panels


def resolve_tissue_color_map(
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
