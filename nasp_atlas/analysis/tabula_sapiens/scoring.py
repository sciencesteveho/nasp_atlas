"""Tabula Sapiens NASP scoring outputs."""

from __future__ import annotations

import gc
import hashlib
import logging
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]
from nasp_compendium.types import GeneModule  # type: ignore[import]

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import metadata_columns
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import inverse_module_score_name
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import positive_module_score_name
from nasp_atlas.single_cell.reference_sets import reference_bundle_hash
from nasp_atlas.single_cell.reference_sets import reference_gene_sets
from nasp_atlas.single_cell.reference_sets import reference_metadata
from nasp_atlas.single_cell.score_diagnostics import compare_module_scorers
from nasp_atlas.single_cell.score_diagnostics import (
    cross_scorer_module_correlations,
)
from nasp_atlas.single_cell.umap import UmapPanelSpec
from nasp_atlas.single_cell.visualization import ColorbarStyle
from nasp_atlas.single_cell.visualization import GroupedGeneExpression
from nasp_atlas.single_cell.visualization import HeatmapPlotter
from nasp_atlas.single_cell.visualization import SummaryPlotter
from nasp_atlas.single_cell.visualization import UmapPlotter


logger = logging.getLogger(__name__)

__all__ = [
    "module_scoring_outputs",
    "plot_gene_expression_heatmaps_by_obs",
    "plot_reference_score_umaps",
    "plot_scorer_concordance_by_source",
    "safe_filename_token",
    "score_table_obs_metadata",
    "score_table_provenance",
    "write_score_tables",
]


class _ScanpyModuleScorer(Protocol):
    """Callable protocol for scanpy module scoring."""

    def __call__(
        self,
        adata: ad.AnnData,
        module_ids: Sequence[str],
        *,
        gene_symbol_column: str,
        random_state: int,
        expression_layer: str | None,
        gene_modules: Sequence[GeneModule] | None = None,
    ) -> Sequence[GeneModule]:
        """Score modules in place and return the scored modules."""
        ...


class _AucellModuleScorer(Protocol):
    """Callable protocol for AUCell module scoring."""

    def __call__(
        self,
        adata: ad.AnnData,
        module_ids: Sequence[str],
        *,
        gene_symbol_column: str,
        expression_layer: str | None,
        chunk_size: int,
        random_state: int,
        num_workers: int,
        gene_modules: Sequence[GeneModule] | None = None,
    ) -> tuple[ad.AnnData, pd.DataFrame, Sequence[GeneModule]]:
        """Return AUCell-scored AnnData, raw AUC table, and modules."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class _ModuleScoringPlan:
    """Settings shared by both module-scoring stages."""

    gene_symbol_column: str
    expression_layer: str | None
    random_state: int
    score_heatmap_groupby_key: str
    point_size: float
    gene_modules: tuple[GeneModule, ...] | None = None


def module_scoring_outputs(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    umap_plotter: UmapPlotter,
    heatmap_plotter: HeatmapPlotter,
    summary_plotter: SummaryPlotter,
    output_dir: str | Path,
    score_table_filename: str,
    score_scanpy: bool,
    score_aucell: bool,
    scanpy_scorer: _ScanpyModuleScorer,
    aucell_scorer: _AucellModuleScorer,
    donor_key: str,
    tissue_key: str,
    cell_type_key: str,
    sex_key: str,
    assay_key: str,
    development_stage_key: str,
    age_key: str,
    gene_symbol_column: str,
    expression_layer: str | None,
    score_heatmap_groupby_key: str,
    subset_fraction: float | None,
    random_state: int,
    aucell_chunk_size: int,
    aucell_num_workers: int,
    point_size: float,
    include_reference_sets: bool = False,
    reference_symbol_renames: Mapping[str, str] | None = None,
) -> None:
    """Score modules and write score tables, UMAPs, and heatmaps.

    Args:
      adata: AnnData to score.
      module_ids: Module ids selected for scoring.
      umap_plotter: Plotter used for module-score UMAPs.
      heatmap_plotter: Plotter used for module-score heatmaps.
      summary_plotter: Plotter used for scorer-concordance summaries.
      output_dir: Directory where score tables are written.
      score_table_filename: Score CSV filename under `output_dir`.
      score_scanpy: Whether to run scanpy scoring.
      score_aucell: Whether to run AUCell scoring.
      scanpy_scorer: Callable implementing scanpy scoring.
      aucell_scorer: Callable implementing AUCell scoring.
      donor_key: Obs column identifying donors.
      tissue_key: Obs column identifying tissues.
      cell_type_key: Obs column identifying cell types.
      sex_key: Obs column identifying donor sex.
      assay_key: Obs column identifying sequencing assay.
      development_stage_key: Obs column with development stage labels.
      age_key: Obs column with numeric age.
      gene_symbol_column: Var column containing gene symbols.
      expression_layer: Expression layer used by scoring.
      score_heatmap_groupby_key: Obs column used for score heatmaps.
      subset_fraction: Optional loaded-cell fraction recorded as provenance.
      random_state: Random seed used for scoring provenance and scanpy.
      aucell_chunk_size: Maximum cells densified in one AUCell block.
      aucell_num_workers: Worker processes used by each AUCell block.
      point_size: UMAP point size for score plots.
      include_reference_sets: Score bundled Reactome and Hallmark signatures.
      reference_symbol_renames: Dataset symbols relabelled in
        `gene_symbol_column`, mapped to their new labels (for example RIGI to
        the curated DDX58), so reference sets still find those features.
    """
    score_tables: list[pd.DataFrame] = []
    requested_scorers = [
        scorer
        for scorer, enabled in (
            ("scanpy", score_scanpy),
            ("aucell", score_aucell),
        )
        if enabled
    ]
    for scorer in ("scanpy", "aucell"):
        retired_umap = (
            Path(output_dir)
            / f"tabula_sapiens_{scorer}_module_zscore_umaps.png"
        )
        if retired_umap.is_file():
            retired_umap.unlink()
            logger.info(
                "[tabula_sapiens] removed retired score UMAP -> %s",
                retired_umap,
            )

    schema = ObsSchema(
        donor_key=donor_key,
        tissue_key=tissue_key,
        cell_type_key=cell_type_key,
        sex_key=sex_key,
        assay_key=assay_key,
        development_stage_key=development_stage_key,
        age_key=age_key,
    )
    definitions = None
    if include_reference_sets:
        definitions = _reference_scoring_modules(
            adata,
            module_ids,
            gene_symbol_column=gene_symbol_column,
            output_dir=Path(output_dir),
            symbol_renames=reference_symbol_renames or {},
        )
        module_ids = [module.module_id for module in definitions]

    scoring_plan = _ModuleScoringPlan(
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
        random_state=random_state,
        score_heatmap_groupby_key=score_heatmap_groupby_key,
        point_size=point_size,
        gene_modules=definitions,
    )

    score_metadata = score_table_obs_metadata(
        cast(pd.DataFrame, adata.obs),
        obs_keys=metadata_columns(schema),
    )
    score_provenance = score_table_provenance(
        n_obs=adata.n_obs,
        scorers=(),
        requested_scorers=requested_scorers,
        subset_fraction=subset_fraction,
        random_state=random_state,
        module_ids=module_ids,
        expression_layer=expression_layer,
        aucell_chunk_size=aucell_chunk_size,
        aucell_num_workers=aucell_num_workers,
    )
    score_provenance.index = score_metadata.index
    if include_reference_sets:
        score_provenance["scoring_reference_bundle_sha256"] = (
            reference_bundle_hash()
        )
    score_metadata = pd.concat([score_metadata, score_provenance], axis=1)

    completed_scorers: list[str] = []
    if score_scanpy:
        score_metadata["scoring_scorers"] = "scanpy"

        _score_scanpy_outputs(
            adata,
            module_ids,
            umap_plotter=umap_plotter,
            heatmap_plotter=heatmap_plotter,
            score_tables=score_tables,
            score_metadata=score_metadata,
            output_dir=output_dir,
            score_table_filename=score_table_filename,
            scanpy_scorer=scanpy_scorer,
            scoring_plan=scoring_plan,
        )

        completed_scorers.append("scanpy")
        if not score_aucell:
            score_tables.clear()
            gc.collect()

    if score_aucell:
        score_metadata["scoring_scorers"] = ",".join(
            [*completed_scorers, "aucell"]
        )

        _score_aucell_outputs(
            adata,
            module_ids,
            umap_plotter=umap_plotter,
            heatmap_plotter=heatmap_plotter,
            summary_plotter=summary_plotter,
            score_tables=score_tables,
            score_metadata=score_metadata,
            output_dir=output_dir,
            score_table_filename=score_table_filename,
            aucell_scorer=aucell_scorer,
            scoring_plan=scoring_plan,
            chunk_size=aucell_chunk_size,
            num_workers=aucell_num_workers,
        )


def write_score_tables(
    score_tables: Sequence[pd.DataFrame],
    *,
    obs_metadata: pd.DataFrame | None = None,
    output_dir: str | Path,
    filename: str,
) -> None:
    """Persist the currently available score columns."""
    if not score_tables:
        return

    scores = pd.concat(score_tables, axis="columns")
    if obs_metadata is not None and not obs_metadata.empty:
        scores = pd.concat([obs_metadata.reindex(scores.index), scores], axis=1)
    score_path = Path(output_dir) / filename
    score_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[tabula_sapiens] writing module scores: cells=%d, columns=%d -> %s",
        len(scores),
        len(scores.columns),
        score_path,
    )
    scores.to_csv(
        score_path,
        index=True,
        index_label="obs_name",
        compression="infer",
    )
    logger.info("[tabula_sapiens] module scores -> %s", score_path)


def score_table_obs_metadata(
    obs: pd.DataFrame,
    *,
    obs_keys: Sequence[str],
) -> pd.DataFrame:
    """Return obs metadata columns to include with score outputs."""
    available_keys = [key for key in dict.fromkeys(obs_keys) if key in obs]
    if missing_keys := [
        key for key in dict.fromkeys(obs_keys) if key not in obs
    ]:
        logger.warning(
            "[tabula_sapiens] score metadata columns missing: %s",
            ", ".join(missing_keys),
        )
    if not available_keys:
        return pd.DataFrame(index=obs.index)
    return obs.loc[:, available_keys].copy()


def score_table_provenance(
    *,
    n_obs: int,
    scorers: Sequence[str],
    requested_scorers: Sequence[str] | None = None,
    subset_fraction: float | None,
    random_state: int,
    module_ids: Sequence[str],
    expression_layer: str | None = None,
    aucell_chunk_size: int | None = None,
    aucell_num_workers: int | None = None,
) -> pd.DataFrame:
    """Build a broadcast provenance frame recording scoring parameters.

    The provenance columns are repeated for every cell so that the score CSV
    is self-describing once written to disk and detached from the AnnData.

    Args:
      n_obs: Number of cells (rows) the provenance is broadcast across.
      scorers: Scorer names that completed and produced score columns.
      requested_scorers: Scorers requested at workflow start. Defaults to
        `scorers` for direct callers.
      subset_fraction: Fraction of cells sampled prior to scoring, or None.
      random_state: Random seed used for sampling and scanpy scoring.
      module_ids: Module identifiers that were scored.
      expression_layer: Expression layer used, or None for `adata.X`.
      aucell_chunk_size: Maximum AUCell block size when AUCell was requested.
      aucell_num_workers: AUCell workers when AUCell was requested.

    Returns:
      DataFrame with a single set of provenance values repeated per cell.
    """
    provenance = {
        "scoring_scorers": ",".join(scorers),
        "scoring_requested_scorers": ",".join(
            scorers if requested_scorers is None else requested_scorers
        ),
        "scoring_expression_source": expression_layer or "X",
        "scoring_subset_fraction": (
            "" if subset_fraction is None else float(subset_fraction)
        ),
        "scoring_random_state": random_state,
        "scoring_module_ids": ",".join(module_ids),
        "scoring_n_modules": len(module_ids),
        "scoring_aucell_chunk_size": (
            "" if aucell_chunk_size is None else aucell_chunk_size
        ),
        "scoring_aucell_num_workers": (
            "" if aucell_num_workers is None else aucell_num_workers
        ),
        "scoring_marker_panel_sha256": hashlib.sha256(
            GeneModules.default_panel_path().read_bytes()
        ).hexdigest(),
    }
    return pd.DataFrame([provenance] * n_obs)


def plot_gene_expression_heatmaps_by_obs(
    *,
    adata: ad.AnnData,
    genes: Sequence[str],
    plotter: HeatmapPlotter,
    filename_prefix: str,
    groupby_keys: Sequence[str],
    gene_symbol_column: str,
    expression_layer: str | None,
    grouped_expression_by_obs: (
        Mapping[str, GroupedGeneExpression] | None
    ) = None,
) -> None:
    """Plot one gene-expression heatmap for each requested obs key.

    Example Usage:
      >>> plot_gene_expression_heatmaps_by_obs(
      ...     adata=adata,
      ...     genes=["CD3D", "MS4A1"],
      ...     plotter=plotter,
      ...     filename_prefix="tabula_sapiens",
      ...     groupby_keys=["tissue", "cell_type"],
      ...     gene_symbol_column="feature_name",
      ...     expression_layer="log1p",
      ... )
    """
    gene_list = list(genes)
    if not gene_list:
        return

    for groupby_key in groupby_keys:
        plotter.plot_multi_gene_expression_heatmap(
            adata=adata,
            genes=gene_list,
            groupby=groupby_key,
            filename=(
                f"{filename_prefix}_gene_expression_heatmap_by_"
                f"{safe_filename_token(groupby_key)}"
            ),
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            grouped_expression=(
                grouped_expression_by_obs.get(groupby_key)
                if grouped_expression_by_obs is not None
                else None
            ),
        )


def safe_filename_token(value: str) -> str:
    """Return a filesystem-safe token for generated plot filenames."""
    token = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )
    return token.strip("_") or "obs"


def plot_reference_score_umaps(
    adata: ad.AnnData,
    score_keys: Sequence[str],
    *,
    scorer: ScorerName,
    plotter: UmapPlotter,
    point_size: float = 2.0,
    ncols: int = 4,
    panel_w: float = 2.5,
    panel_h: float = 2.5,
    row_hspace: float = 0.15,
    col_wspace: float = 0.15,
    cmap: str | None = None,
    colorbar_style: ColorbarStyle | None = None,
    cbar_height: str | float | None = None,
    cbar_width: str | float | None = None,
    cbar_pad: float | None = None,
) -> None:
    """Plot one native-score UMAP grid per external scoring database.

    Each gene set retains its own color scale. Panel dimensions are in inches;
    row spacing is a fraction of panel height. Only requested scores are shown.
    `cmap` replaces the scorer default ("RdBu_r" on a zero-centered Scanpy
    scale, "Blues" from zero for AUCell) without changing the limits.
    `colorbar_style` sets geometry and tick styling. `cbar_height` and
    `cbar_width` override its dimensions in inches (floats) or percentages of
    the panel (strings). `cbar_pad` adds a gap in panel-width units to the
    renderer's 0.02 offset. None retains the renderer's existing defaults.

    Example Usage:
      >>> plot_reference_score_umaps(
      ...     adata, score_keys, scorer="scanpy", plotter=plotter,
      ...     ncols=4, panel_w=2.5, panel_h=2.5,
      ...     cbar_height=0.3, cbar_width=0.05, cbar_pad=0.02,
      ... )
    """
    suffix = "_score" if scorer == "scanpy" else "_auc"
    title_wrap_width = min(35, max(18, round(panel_w * 18)))
    for source, metadata in reference_metadata().groupby("source", sort=False):
        panels: list[UmapPanelSpec] = []
        for entry in metadata.itertuples(index=False):
            key = f"{entry.module_id}{suffix}"
            if key not in score_keys:
                continue
            values = np.asarray(adata.obs[key], dtype=float)
            finite = values[np.isfinite(values)]
            bound = (
                max(float(np.abs(finite).max()), 1e-12) if finite.size else 1.0
            )
            panels.append(
                {
                    "obs_key": key,
                    "title": textwrap.fill(str(entry.name), title_wrap_width),
                    "kind": "numeric",
                    "cmap": cmap
                    or ("RdBu_r" if scorer == "scanpy" else "Blues"),
                    "vmin": -bound if scorer == "scanpy" else 0.0,
                    "vmax": bound,
                }
            )
        if not panels:
            continue

        database = (
            "hallmark"
            if source == "MSigDB Hallmark"
            else safe_filename_token(str(source)).lower()
        )
        plotter.plot_umap_panel(
            adata,
            panels=panels,
            filename=f"tabula_sapiens_{database}_{scorer}_module_umaps",
            ncols=ncols,
            panel_w=panel_w,
            panel_h=panel_h,
            row_hspace=row_hspace,
            col_wspace=col_wspace,
            size=min(point_size, 8.0),
            colorbar_style=colorbar_style,
            cbar_height=cbar_height,
            cbar_width=cbar_width,
            cbar_pad=cbar_pad,
        )


def plot_scorer_concordance_by_source(
    correlations: pd.DataFrame,
    module_ids: Sequence[str],
    *,
    plotter: SummaryPlotter,
    cell_size: float = 0.0975,
) -> None:
    """Plot separate curated and external Scanpy/AUCell correlation matrices.

    `cell_size` controls heatmap cell dimensions in inches. Both axes of each
    matrix contain only modules from that source group.

    Example Usage:
      >>> plot_scorer_concordance_by_source(
      ...     correlations, module_ids, plotter=plotter, cell_size=0.12,
      ... )
    """
    reference_ids = set(reference_metadata().module_id)
    for group, order in (
        ("curated", [key for key in module_ids if key not in reference_ids]),
        ("external", [key for key in module_ids if key in reference_ids]),
    ):
        if not order:
            continue
        selected = correlations.loc[
            correlations.scanpy_module_id.isin(order)
            & correlations.aucell_module_id.isin(order)
        ]
        plotter.plot_scorer_concordance_heatmap(
            selected,
            filename=f"tabula_sapiens_scorer_concordance_{group}",
            module_order=order,
            cell_size=cell_size,
        )


def _module_export_score_keys(
    modules: Sequence[GeneModule],
    *,
    scorer: ScorerName,
) -> list[str]:
    """Return positive, inverse, and composite columns for export."""
    keys: list[str] = []
    for module in modules:
        if module.positive_genes:
            keys.append(positive_module_score_name(module, scorer=scorer))
        inverse_name = inverse_module_score_name(module, scorer=scorer)
        if inverse_name is not None:
            keys.append(inverse_name)
        keys.append(module_score_name(module, scorer=scorer))
    return keys


def _plot_module_score_umaps(
    adata: ad.AnnData,
    score_keys: list[str],
    *,
    scorer: ScorerName,
    plotter: UmapPlotter,
    point_size: float,
) -> None:
    """Plot final module scores with their separate native-scale colorbars."""
    reference_ids = set(reference_metadata().module_id)
    suffix = "_score" if scorer == "scanpy" else "_auc"
    reference_keys = [
        key for key in score_keys if key.removesuffix(suffix) in reference_ids
    ]
    plot_reference_score_umaps(
        adata,
        reference_keys,
        scorer=scorer,
        plotter=plotter,
        point_size=point_size,
        ncols=4,
        panel_w=1.25,
        panel_h=1.25,
        row_hspace=0.15,
        col_wspace=0.35,
        cbar_height="28.5%",
        cbar_width="4%",
        cbar_pad=0.02,
    )
    if score_keys := [key for key in score_keys if key not in reference_keys]:
        plotter.plot_multi_obs_umap_panel(
            adata,
            obs_keys=score_keys,
            filename=f"tabula_sapiens_{scorer}_module_umaps",
            cmap="RdBu_r",
            ncols=6,
            panel_w=0.6,
            panel_h=0.6,
            row_hspace=0.75,
            col_wspace=0.5,
            title_wrap_width=16,
            cbar_width=0.05,
            cbar_height=0.3,
            size=point_size,
            vmin=None,
            vmax=None,
            center_zero=True,
            shared_colorbar=False,
            standardization="none",
        )
    else:
        return


def _score_scanpy_outputs(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    umap_plotter: UmapPlotter,
    heatmap_plotter: HeatmapPlotter,
    score_tables: list[pd.DataFrame],
    score_metadata: pd.DataFrame,
    output_dir: str | Path,
    score_table_filename: str,
    scanpy_scorer: _ScanpyModuleScorer,
    scoring_plan: _ModuleScoringPlan,
) -> None:
    """Run scanpy scoring and emit scanpy score outputs."""
    references = [
        module
        for module in scoring_plan.gene_modules or ()
        if module.module_id.startswith(("REACTOME_", "HALLMARK_"))
    ]
    reference_ids = {module.module_id for module in references}
    scanpy_modules = list(
        scanpy_scorer(
            adata,
            [
                identifier
                for identifier in module_ids
                if identifier not in reference_ids
            ],
            gene_symbol_column=scoring_plan.gene_symbol_column,
            random_state=scoring_plan.random_state,
            expression_layer=scoring_plan.expression_layer,
        )
    )
    if references:
        scanpy_modules.extend(
            scanpy_scorer(
                adata,
                [module.module_id for module in references],
                gene_modules=references,
                gene_symbol_column=scoring_plan.gene_symbol_column,
                random_state=scoring_plan.random_state,
                expression_layer=scoring_plan.expression_layer,
            )
        )
    scanpy_score_keys = [
        module_score_name(module, scorer="scanpy") for module in scanpy_modules
    ]
    scanpy_obs = cast(pd.DataFrame, adata.obs)
    scanpy_export_keys = [
        key
        for key in _module_export_score_keys(
            scanpy_modules,
            scorer="scanpy",
        )
        if key in scanpy_obs.columns
    ]
    scanpy_scores = scanpy_obs.loc[:, scanpy_export_keys].copy()
    score_tables.append(scanpy_scores)

    write_score_tables(
        score_tables,
        obs_metadata=score_metadata,
        output_dir=output_dir,
        filename=score_table_filename,
    )

    _plot_module_score_umaps(
        adata,
        scanpy_score_keys,
        scorer="scanpy",
        plotter=umap_plotter,
        point_size=scoring_plan.point_size,
    )

    heatmap_plotter.plot_grouped_obs_score_heatmap(
        adata,
        score_keys=scanpy_score_keys,
        groupby=scoring_plan.score_heatmap_groupby_key,
        filename=(
            "tabula_sapiens_scanpy_module_score_heatmap_by_"
            f"{safe_filename_token(scoring_plan.score_heatmap_groupby_key)}"
        ),
        score_labels=[str(module.module_id) for module in scanpy_modules],
        cmap="RdBu_r",
    )

    del scanpy_obs, scanpy_modules, scanpy_score_keys, scanpy_scores


def _score_aucell_outputs(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    umap_plotter: UmapPlotter,
    heatmap_plotter: HeatmapPlotter,
    summary_plotter: SummaryPlotter,
    score_tables: list[pd.DataFrame],
    score_metadata: pd.DataFrame,
    output_dir: str | Path,
    score_table_filename: str,
    aucell_scorer: _AucellModuleScorer,
    scoring_plan: _ModuleScoringPlan,
    chunk_size: int,
    num_workers: int,
) -> None:
    """Run AUCell scoring and emit AUCell score outputs."""
    adata_auc, auc_df, auc_modules = aucell_scorer(
        adata,
        module_ids,
        gene_symbol_column=scoring_plan.gene_symbol_column,
        expression_layer=scoring_plan.expression_layer,
        chunk_size=chunk_size,
        random_state=scoring_plan.random_state,
        num_workers=num_workers,
        gene_modules=scoring_plan.gene_modules,
    )
    auc_score_keys: list[str] | None = None
    auc_scores: pd.DataFrame | None = None
    try:
        auc_score_keys = [
            module_score_name(module, scorer="aucell") for module in auc_modules
        ]
        auc_export_keys = [
            key
            for key in _module_export_score_keys(
                auc_modules,
                scorer="aucell",
            )
            if key in auc_df.columns
        ]
        auc_scores = auc_df.loc[:, auc_export_keys].copy()
        score_tables.append(auc_scores)

        write_score_tables(
            score_tables,
            obs_metadata=score_metadata,
            output_dir=output_dir,
            filename=score_table_filename,
        )
        if len(score_tables) >= 2:
            _write_cross_scorer_comparison_outputs(
                score_tables,
                auc_modules,
                output_dir=output_dir,
                plotter=summary_plotter,
            )
        _plot_module_score_umaps(
            adata_auc,
            auc_score_keys,
            scorer="aucell",
            plotter=umap_plotter,
            point_size=scoring_plan.point_size,
        )

        heatmap_plotter.plot_grouped_obs_score_heatmap(
            adata_auc,
            score_keys=auc_score_keys,
            groupby=scoring_plan.score_heatmap_groupby_key,
            filename=(
                "tabula_sapiens_aucell_module_score_heatmap_by_"
                f"{safe_filename_token(scoring_plan.score_heatmap_groupby_key)}"
            ),
            score_labels=[str(module.module_id) for module in auc_modules],
            cmap="RdBu_r",
        )
    finally:
        del adata_auc, auc_df, auc_modules
        score_tables.clear()
        auc_score_keys = None
        auc_scores = None
        gc.collect()


def _write_cross_scorer_comparison_outputs(
    score_tables: Sequence[pd.DataFrame],
    auc_modules: Sequence[GeneModule],
    *,
    output_dir: str | Path,
    plotter: SummaryPlotter,
) -> None:
    """Write complete scorer comparison tables and source-group heatmaps."""
    combined_scores = pd.concat(score_tables, axis="columns")
    scored_module_ids = [str(module.module_id) for module in auc_modules]
    concordance = compare_module_scorers(
        combined_scores,
        scored_module_ids,
    )
    _write_scorer_comparison_table(
        concordance,
        output_dir=output_dir,
        filename="tabula_sapiens_scorer_concordance.csv",
        log_message="[tabula_sapiens] scorer concordance -> %s",
    )

    cross_module_correlations = cross_scorer_module_correlations(
        combined_scores,
        scored_module_ids,
    )
    _write_scorer_comparison_table(
        cross_module_correlations,
        output_dir=output_dir,
        filename="tabula_sapiens_cross_scorer_module_correlations.csv",
        log_message="[tabula_sapiens] cross-scorer module correlations -> %s",
    )
    plot_scorer_concordance_by_source(
        cross_module_correlations,
        scored_module_ids,
        plotter=plotter,
    )


def _write_scorer_comparison_table(
    table: pd.DataFrame,
    *,
    output_dir: str | Path,
    filename: str,
    log_message: str,
) -> None:
    """Write a scorer comparison table to CSV and log its path."""
    output_path = Path(output_dir) / filename
    table.to_csv(output_path, index=False)
    logger.info(log_message, output_path)


def _reference_scoring_modules(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    gene_symbol_column: str,
    output_dir: Path,
    symbol_renames: Mapping[str, str],
) -> tuple[GeneModule, ...]:
    """Resolve requested signatures and publish reference-gene coverage.

    `symbol_renames` relabels reference genes whose features carry another
    label in `gene_symbol_column`; it changes the join key, not membership.
    """
    reference_modules = tuple(
        replace(
            module,
            positive_genes=tuple(
                symbol_renames.get(gene, gene) for gene in module.positive_genes
            ),
            inverse_genes=tuple(
                symbol_renames.get(gene, gene) for gene in module.inverse_genes
            ),
        )
        for module in reference_gene_sets()
    )
    symbols = (
        adata.var[gene_symbol_column]
        .astype(object)
        .fillna(pd.Series(adata.var_names, index=adata.var_names))
        .astype(str)
        .str.strip()
    )
    available = set(symbols)

    coverage = reference_metadata().set_index("module_id")
    for module in reference_modules:
        found = available.intersection(module.positive_genes)
        coverage.loc[module.module_id, "n_measured_genes"] = len(found)
        coverage.loc[module.module_id, "missing_genes"] = ";".join(
            sorted(set(module.positive_genes) - found)
        )
        coverage.loc[module.module_id, "status"] = (
            "ok" if found else "unavailable_no_genes"
        )
    coverage.to_csv(Path(output_dir) / "reference_gene_coverage.csv")

    return tuple(
        GeneModules.modules(
            identifier,
            adata=adata,  # type: ignore[arg-type]
            gene_symbol_column=gene_symbol_column,
            output="symbols",
        )
        for identifier in module_ids
    ) + tuple(
        module
        for module in reference_modules
        if available.intersection(module.positive_genes)
    )
