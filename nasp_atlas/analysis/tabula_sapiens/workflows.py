"""Tabula Sapiens NASP analysis workflows."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.tabula_sapiens import associations
from nasp_atlas.analysis.tabula_sapiens import scoring
from nasp_atlas.analysis.tabula_sapiens.visualizations import (
    plot_nasp_association_visualizations,
)
from nasp_atlas.analysis.tabula_sapiens.visualizations import (
    plot_tabula_sapiens_metadata_umaps,
)
from nasp_atlas.single_cell.associations import Aggregation
from nasp_atlas.single_cell.associations import EqtlMergeMode
from nasp_atlas.single_cell.associations import FeatureSpec
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import StatisticalUnit
from nasp_atlas.single_cell.associations import aggregate_feature_frame
from nasp_atlas.single_cell.associations import build_cell_feature_frame
from nasp_atlas.single_cell.associations import metadata_columns
from nasp_atlas.single_cell.associations import (
    partial_correlation_controlling_tissue,
)
from nasp_atlas.single_cell.associations import regress_features_on_continuous
from nasp_atlas.single_cell.associations import resolve_feature_specs
from nasp_atlas.single_cell.associations import (
    summarize_continuous_association_stability,
)
from nasp_atlas.single_cell.context_summary import summarize_module_contexts
from nasp_atlas.single_cell.hypothesis_priorities import (
    expected_module_coupling_report,
)
from nasp_atlas.single_cell.hypothesis_priorities import rank_nasp_hypotheses
from nasp_atlas.single_cell.io import read_h5ad
from nasp_atlas.single_cell.io import read_h5ad_rows
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_modules
from nasp_atlas.single_cell.scprocessor import SCProcessor
from nasp_atlas.single_cell.umap import UmapPanelSpec
from nasp_atlas.single_cell.visualization import SCVisualizer


logger = logging.getLogger(__name__)

__all__ = [
    "association_analysis",
    "tabula_sapiens_scoring_analysis",
    "tabula_sapiens_tissue_analysis",
]


@dataclass(slots=True)
class _NaspProfileResults:
    """Results derived from donor-tissue-cell-type NASP profiles."""

    profile_unit_frame: pd.DataFrame | None
    profiles: pd.DataFrame
    context_summary: pd.DataFrame
    hypothesis_priorities: pd.DataFrame
    mechanistic_edges: pd.DataFrame
    module_context_ranking: pd.DataFrame
    module_coupling: pd.DataFrame
    sensor_output_coupling: pd.DataFrame


@dataclass(slots=True)
class _AssociationWorkflowState:
    """Mutable result state shared across association analysis stages."""

    plot_count: int = 0
    manifest: list[dict[str, object]] = field(default_factory=list)
    regression_tables: list[pd.DataFrame] = field(default_factory=list)
    group_test_tables: list[pd.DataFrame] = field(default_factory=list)
    group_summary_tables: list[pd.DataFrame] = field(default_factory=list)
    eqtl_annotation_tables: list[pd.DataFrame] = field(default_factory=list)


def association_analysis(
    *,
    h5ad_path: str | Path,
    score_csv_path: str | Path,
    output_dir: str | Path,
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    expression_use_raw: bool = False,
    donor_key: str = "donor_id",
    tissue_key: str = "tissue_in_publication",
    cell_type_key: str = "cell_type",
    sex_key: str = "sex",
    assay_key: str = "assay",
    development_stage_key: str = "development_stage",
    age_key: str = "age_years",
    module_ids: Sequence[str] | None = None,
    sensor_group: str | None = "nucleic_acid_sensors",
    gene_symbols: Sequence[str] | None = None,
    statistical_unit: str = "donor",
    aggregation: str = "mean",
    scorer: str | None = None,
    eqtl_table_path: str | Path | None = None,
    eqtl_merge_mode: str = "gene",
    run_cell_level_descriptive_plots: bool = False,
    run_donor_aware_tests: bool = True,
    detection_threshold: float = 0.0,
    max_plots: int | None = 200,
    plot_nasp_visualizations: bool = True,
) -> None:
    """Run donor-aware NASP associations over precomputed module scores.

    Module scores come only from score_csv_path; gene features come from the
    aligned AnnData expression source. Cell-level plots remain descriptive.
    Donor-aware regression stages state their independent unit explicitly.
    """
    _validate_association_options(
        statistical_unit=statistical_unit,
        aggregation=aggregation,
        scorer=scorer,
        eqtl_merge_mode=eqtl_merge_mode,
    )

    output_path = Path(output_dir)
    layout = associations._association_output_dirs(output_path)
    schema = ObsSchema(
        donor_key=donor_key,
        tissue_key=tissue_key,
        cell_type_key=cell_type_key,
        sex_key=sex_key,
        assay_key=assay_key,
        development_stage_key=development_stage_key,
        age_key=age_key,
    )
    adata, scores = _load_association_inputs(
        h5ad_path=h5ad_path,
        score_csv_path=score_csv_path,
        schema=schema,
        expression_layer=expression_layer,
        expression_use_raw=expression_use_raw,
        age_key=age_key,
    )
    feature_specs = _resolve_association_features(
        adata,
        scores,
        layout=layout,
        module_ids=module_ids,
        sensor_group=sensor_group,
        gene_symbols=gene_symbols,
        scorer=scorer,
        gene_symbol_column=gene_symbol_column,
        expression_use_raw=expression_use_raw,
    )
    if not feature_specs:
        _write_empty_association_tables(layout)
        return

    cell_frame = build_cell_feature_frame(
        adata,
        scores,
        feature_specs,
        schema=schema,
        expression_layer=expression_layer,
        use_raw=expression_use_raw,
    )
    unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit=cast(StatisticalUnit, statistical_unit),
        aggregation=cast(Aggregation, aggregation),
        schema=schema,
        detection_threshold=detection_threshold,
    )
    nasp_results = _build_nasp_profile_results(
        cell_frame,
        schema=schema,
        aggregation=cast(Aggregation, aggregation),
        detection_threshold=detection_threshold,
        donor_key=donor_key,
        tissue_key=tissue_key,
        cell_type_key=cell_type_key,
    )
    for filename, table in (
        ("nasp_evidence_profiles.csv", nasp_results.profiles),
        ("nasp_context_summary.csv", nasp_results.context_summary),
        ("nasp_hypothesis_priorities.csv", nasp_results.hypothesis_priorities),
        ("nasp_mechanistic_edges.csv", nasp_results.mechanistic_edges),
        (
            "nasp_module_context_ranking.csv",
            nasp_results.module_context_ranking,
        ),
        ("nasp_module_coupling.csv", nasp_results.module_coupling),
        (
            "nasp_sensor_output_coupling.csv",
            nasp_results.sensor_output_coupling,
        ),
    ):
        associations._write_association_table(
            table,
            layout["tables"] / filename,
        )

    workflow_state = _AssociationWorkflowState()
    if run_donor_aware_tests and age_key in unit_frame.columns:
        _run_age_associations(
            cell_frame,
            unit_frame,
            nasp_results.profile_unit_frame,
            workflow_state=workflow_state,
            layout=layout,
            schema=schema,
            statistical_unit=statistical_unit,
            aggregation=cast(Aggregation, aggregation),
            detection_threshold=detection_threshold,
            donor_key=donor_key,
            tissue_key=tissue_key,
            cell_type_key=cell_type_key,
            sex_key=sex_key,
            age_key=age_key,
            max_plots=max_plots,
        )
    if run_donor_aware_tests:
        _run_group_associations(
            cell_frame,
            unit_frame,
            workflow_state=workflow_state,
            layout=layout,
            schema=schema,
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            tissue_key=tissue_key,
            cell_type_key=cell_type_key,
            sex_key=sex_key,
            max_plots=max_plots,
        )
    if eqtl_table_path is not None:
        associations._run_eqtl_associations(
            unit_frame,
            eqtl_table_path=eqtl_table_path,
            eqtl_merge_mode=cast(EqtlMergeMode, eqtl_merge_mode),
            schema=schema,
            regression_tables=workflow_state.regression_tables,
            annotation_tables=workflow_state.eqtl_annotation_tables,
        )
    if run_cell_level_descriptive_plots and age_key in cell_frame.columns:
        workflow_state.plot_count = (
            associations._run_cell_level_descriptive_plots(
                cell_frame,
                predictor_key=age_key,
                visualizer=SCVisualizer(output_dir=str(layout["regressions"])),
                aggregation=aggregation,
                manifest=workflow_state.manifest,
                plot_count=workflow_state.plot_count,
                max_plots=max_plots,
            )
        )

    _write_association_results(
        workflow_state,
        nasp_results=nasp_results,
        layout=layout,
        plot_nasp_visualizations=plot_nasp_visualizations,
        cell_type_key=cell_type_key,
    )
    logger.info(
        "[tabula_sapiens] association analysis complete -> %s",
        output_path,
    )


def _validate_association_options(
    *,
    statistical_unit: str,
    aggregation: str,
    scorer: str | None,
    eqtl_merge_mode: str,
) -> None:
    """Reject unsupported workflow options before reading large inputs."""
    valid_units = {
        "cell",
        "metacell",
        "donor",
        "donor_tissue",
        "donor_tissue_cell_type",
        "donor_tissue_sex",
    }
    if statistical_unit not in valid_units:
        raise ValueError(f"unsupported statistical_unit: {statistical_unit}")

    valid_aggregations = {
        "mean",
        "median",
        "sum",
        "fraction_expressing",
        "percent_expressing",
    }
    if aggregation not in valid_aggregations:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    if scorer not in (None, "scanpy", "aucell"):
        raise ValueError(f"unsupported scorer: {scorer}")
    if eqtl_merge_mode not in ("gene", "tissue", "module", "donor"):
        raise ValueError(f"unsupported eqtl_merge_mode: {eqtl_merge_mode}")


def _load_association_inputs(
    *,
    h5ad_path: str | Path,
    score_csv_path: str | Path,
    schema: ObsSchema,
    expression_layer: str | None,
    expression_use_raw: bool,
    age_key: str,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Load only scored observations and the expression sources they require."""
    scores = pd.read_csv(score_csv_path, index_col="obs_name")
    scores.index = scores.index.astype(str)

    layer_keys = (
        (expression_layer,)
        if expression_layer is not None and not expression_use_raw
        else ()
    )
    read_x = expression_layer is None and not expression_use_raw
    adata = read_h5ad_rows(
        h5ad_path,
        scores.index.astype(str).tolist(),
        layer_keys=layer_keys,
        read_x=read_x,
        read_raw=expression_use_raw,
    )
    adata = associations._align_anndata_to_scores(adata, scores)
    associations._backfill_obs_metadata(
        adata,
        scores,
        metadata_columns(schema),
    )

    if associations._required_score_metadata_present(
        scores,
        adata,
        [age_key],
    ):
        logger.warning(
            "[tabula_sapiens] age column %r absent from scores and obs; "
            "age-based association will be skipped",
            age_key,
        )

    return adata, scores


def _resolve_association_features(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    *,
    layout: dict[str, Path],
    module_ids: Sequence[str] | None,
    sensor_group: str | None,
    gene_symbols: Sequence[str] | None,
    scorer: str | None,
    gene_symbol_column: str,
    expression_use_raw: bool,
) -> list[FeatureSpec]:
    """Resolve scored modules and expression genes with explicit diagnostics."""
    has_scanpy = any(str(column).endswith("_score") for column in scores)
    has_aucell = any(str(column).endswith("_auc") for column in scores)
    if scorer is None and has_scanpy and has_aucell:
        logger.warning(
            "[tabula_sapiens] score table contains Scanpy and AUCell scores; "
            "association defaults to Scanpy. Use scorer='aucell' for a "
            "parallel sensitivity analysis."
        )

    resolved_scorer = scorer or associations._detect_scorer_from_scores(scores)
    scored_module_ids = associations._scored_module_ids_from_scores(
        scores,
        scorer=resolved_scorer,
    )
    target_module_ids = (
        list(module_ids) if module_ids is not None else scored_module_ids
    )
    feature_specs, skipped = resolve_feature_specs(
        adata,
        scores,
        module_ids=target_module_ids,
        gene_symbols=gene_symbols,
        sensor_group=sensor_group,
        scorer=cast(ScorerName, resolved_scorer),
        gene_symbol_column=gene_symbol_column,
        use_raw=expression_use_raw,
    )
    associations._write_association_table(
        pd.DataFrame(
            skipped,
            columns=["feature_type", "requested", "skip_reason"],
        ),
        layout["tables"] / "association_skipped_features.csv",
    )
    if not feature_specs:
        logger.warning(
            "[tabula_sapiens] no features resolved for association; "
            "nothing to test"
        )

    return feature_specs


def _write_empty_association_tables(layout: dict[str, Path]) -> None:
    """Write stable empty outputs when no requested feature can be analyzed."""
    for filename in (
        "nasp_evidence_profiles.csv",
        "nasp_context_summary.csv",
        "nasp_hypothesis_priorities.csv",
        "nasp_mechanistic_edges.csv",
        "nasp_module_context_ranking.csv",
        "nasp_module_coupling.csv",
        "nasp_sensor_output_coupling.csv",
        "association_age_stability.csv",
        "association_regression_results.csv",
        "association_group_test_results.csv",
        "association_group_summary.csv",
        "association_eqtl_annotations.csv",
        "association_plot_manifest.csv",
    ):
        associations._write_association_table(
            pd.DataFrame(),
            layout["tables"] / filename,
        )


def _build_nasp_profile_results(
    cell_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    aggregation: Aggregation,
    detection_threshold: float,
    donor_key: str,
    tissue_key: str,
    cell_type_key: str,
) -> _NaspProfileResults:
    """Build donor-context profile, coupling, and prioritization results."""
    profile_keys = [donor_key, tissue_key, cell_type_key]
    if any(key not in cell_frame.columns for key in profile_keys):
        missing = [key for key in profile_keys if key not in cell_frame]
        logger.warning(
            "[tabula_sapiens] NASP profiles skipped; missing metadata: %s",
            ", ".join(missing),
        )
        return _NaspProfileResults(
            profile_unit_frame=None,
            profiles=pd.DataFrame(),
            context_summary=pd.DataFrame(),
            hypothesis_priorities=pd.DataFrame(),
            mechanistic_edges=pd.DataFrame(),
            module_context_ranking=pd.DataFrame(),
            module_coupling=pd.DataFrame(),
            sensor_output_coupling=pd.DataFrame(),
        )

    profile_unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit="donor_tissue_cell_type",
        aggregation=aggregation,
        schema=schema,
        detection_threshold=detection_threshold,
    )
    profiles, context_summary, module_coupling = (
        associations._nasp_profile_tables(
            profile_unit_frame,
            schema=schema,
            unit_columns=profile_keys,
            strata_columns=[tissue_key, cell_type_key],
        )
    )
    sensor_output_coupling = associations._sensor_output_coupling(
        profile_unit_frame,
        unit_columns=profile_keys,
        strata_columns=[tissue_key, cell_type_key],
    )
    module_context_ranking = summarize_module_contexts(
        profile_unit_frame,
        context_columns=[tissue_key, cell_type_key],
        unit_columns=profile_keys,
        donor_column=donor_key,
    )
    mechanistic_edges = (
        pd.DataFrame()
        if module_coupling.empty
        else expected_module_coupling_report(
            module_coupling,
            edge_specs=associations._EXPECTED_NASP_EDGES,
        )
    )
    hypothesis_priorities = (
        pd.DataFrame()
        if context_summary.empty
        else rank_nasp_hypotheses(
            context_summary,
            context_columns=[tissue_key, cell_type_key],
        )
    )

    return _NaspProfileResults(
        profile_unit_frame=profile_unit_frame,
        profiles=profiles,
        context_summary=context_summary,
        hypothesis_priorities=hypothesis_priorities,
        mechanistic_edges=mechanistic_edges,
        module_context_ranking=module_context_ranking,
        module_coupling=module_coupling,
        sensor_output_coupling=sensor_output_coupling,
    )


def _run_age_associations(
    cell_frame: pd.DataFrame,
    unit_frame: pd.DataFrame,
    profile_unit_frame: pd.DataFrame | None,
    *,
    workflow_state: _AssociationWorkflowState,
    layout: dict[str, Path],
    schema: ObsSchema,
    statistical_unit: str,
    aggregation: Aggregation,
    detection_threshold: float,
    donor_key: str,
    tissue_key: str,
    cell_type_key: str,
    sex_key: str,
    age_key: str,
    max_plots: int | None,
) -> None:
    """Run global and context-specific donor-aware age analyses."""
    visualizer = SCVisualizer(output_dir=str(layout["regressions"]))
    scope = (
        "global_donor"
        if statistical_unit == "donor"
        else f"pooled_{statistical_unit}"
    )
    workflow_state.plot_count = associations._run_continuous_associations(
        unit_frame,
        predictor_key=age_key,
        visualizer=visualizer,
        layout=layout,
        statistical_unit=statistical_unit,
        aggregation=aggregation,
        manifest=workflow_state.manifest,
        regression_tables=workflow_state.regression_tables,
        plot_count=workflow_state.plot_count,
        max_plots=max_plots,
        stratify_key=None,
        analysis_scope=scope,
    )

    if (
        sex_key in unit_frame.columns
        and unit_frame[sex_key].dropna().nunique() >= 2
    ):
        workflow_state.plot_count = associations._run_continuous_associations(
            unit_frame,
            predictor_key=age_key,
            visualizer=visualizer,
            layout=layout,
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            manifest=workflow_state.manifest,
            regression_tables=workflow_state.regression_tables,
            plot_count=workflow_state.plot_count,
            max_plots=max_plots,
            stratify_key=sex_key,
            analysis_scope=f"within_sex_{statistical_unit}",
        )

    if donor_key in cell_frame.columns and tissue_key in cell_frame.columns:
        donor_tissue_frame = aggregate_feature_frame(
            cell_frame,
            statistical_unit="donor_tissue",
            aggregation=aggregation,
            schema=schema,
            detection_threshold=detection_threshold,
        )
        age_by_tissue = regress_features_on_continuous(
            donor_tissue_frame,
            predictor_key=age_key,
            stratify_key=tissue_key,
        )
        age_by_tissue["analysis_scope"] = "within_tissue"
        workflow_state.regression_tables.append(age_by_tissue)

    if profile_unit_frame is not None:
        context_key = "tissue_cell_type_context"
        context_frame = profile_unit_frame.copy()
        context_frame[context_key] = (
            context_frame[tissue_key].astype(str)
            + "::"
            + context_frame[cell_type_key].astype(str)
        )
        age_by_context = regress_features_on_continuous(
            context_frame,
            predictor_key=age_key,
            stratify_key=context_key,
        )
        age_by_context["analysis_scope"] = "within_tissue_cell_type"
        workflow_state.regression_tables.append(age_by_context)

    if (
        tissue_key in unit_frame.columns
        and unit_frame[tissue_key].dropna().nunique() >= 2
    ):
        partial = partial_correlation_controlling_tissue(
            unit_frame,
            predictor_key=age_key,
            schema=schema,
        )
        associations._write_association_table(
            partial,
            layout["tables"] / "association_partial_correlation_age.csv",
        )


def _run_group_associations(
    cell_frame: pd.DataFrame,
    unit_frame: pd.DataFrame,
    *,
    workflow_state: _AssociationWorkflowState,
    layout: dict[str, Path],
    schema: ObsSchema,
    statistical_unit: str,
    aggregation: str,
    tissue_key: str,
    cell_type_key: str,
    sex_key: str,
    max_plots: int | None,
) -> None:
    """Run categorical tests supported by the selected statistical unit."""
    if (
        cell_type_key in unit_frame.columns
        and unit_frame[cell_type_key].dropna().nunique() < 2
        and cell_type_key in cell_frame.columns
        and cell_frame[cell_type_key].dropna().nunique() >= 2
    ):
        logger.warning(
            "[tabula_sapiens] skipping cell-type tests because %s "
            "does not preserve cell type; use "
            "statistical_unit='donor_tissue_cell_type'",
            statistical_unit,
        )

    group_keys = [
        key
        for key in (sex_key, tissue_key, cell_type_key)
        if key in unit_frame.columns and unit_frame[key].dropna().nunique() >= 2
    ]
    for group_key in group_keys:
        workflow_state.plot_count = associations._run_group_associations(
            unit_frame,
            group_key=group_key,
            schema=schema,
            visualizer=SCVisualizer(output_dir=str(layout["boxplots"])),
            barplot_visualizer=SCVisualizer(output_dir=str(layout["barplots"])),
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            tissue_key=tissue_key,
            manifest=workflow_state.manifest,
            group_test_tables=workflow_state.group_test_tables,
            group_summary_tables=workflow_state.group_summary_tables,
            plot_count=workflow_state.plot_count,
            max_plots=max_plots,
        )


def _age_stability_table(regression_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize age-effect direction stability within analysis scopes."""
    tables: list[pd.DataFrame] = []
    if "analysis_scope" not in regression_results.columns:
        return associations._concat_or_empty(tables)

    for analysis_scope in ("within_tissue", "within_tissue_cell_type"):
        scoped = regression_results.loc[
            regression_results["analysis_scope"] == analysis_scope
        ]
        if scoped.empty:
            continue

        stability = summarize_continuous_association_stability(scoped)
        stability["analysis_scope"] = analysis_scope
        tables.append(stability)

    return associations._concat_or_empty(tables)


def _write_association_results(
    workflow_state: _AssociationWorkflowState,
    *,
    nasp_results: _NaspProfileResults,
    layout: dict[str, Path],
    plot_nasp_visualizations: bool,
    cell_type_key: str,
) -> None:
    """Finalize inferential tables, NASP figures, and the plot manifest."""
    regression_results = associations._concat_or_empty(
        workflow_state.regression_tables
    )
    age_stability = _age_stability_table(regression_results)

    for filename, table in (
        ("association_regression_results.csv", regression_results),
        ("association_age_stability.csv", age_stability),
        (
            "association_group_test_results.csv",
            associations._concat_or_empty(workflow_state.group_test_tables),
        ),
        (
            "association_group_summary.csv",
            associations._concat_or_empty(workflow_state.group_summary_tables),
        ),
        (
            "association_eqtl_annotations.csv",
            associations._concat_or_empty(
                workflow_state.eqtl_annotation_tables
            ),
        ),
    ):
        associations._write_association_table(
            table,
            layout["tables"] / filename,
        )

    if plot_nasp_visualizations:
        plot_nasp_association_visualizations(
            output_dir=layout["nasp"],
            module_coupling=nasp_results.module_coupling,
            context_summary=nasp_results.context_summary,
            hypothesis_priorities=nasp_results.hypothesis_priorities,
            sensor_output_coupling=nasp_results.sensor_output_coupling,
            regression_results=regression_results,
            age_stability=age_stability,
            mechanistic_edges=nasp_results.mechanistic_edges,
            cell_type_key=cell_type_key,
        )

    associations._write_association_table(
        pd.DataFrame(workflow_state.manifest),
        layout["tables"] / "association_plot_manifest.csv",
    )


def tabula_sapiens_tissue_analysis(
    *,
    h5ad_path: str | Path,
    output_dir: str | Path,
    tissue_label: str | None = None,
    run_name: str | None = None,
    scorers: Sequence[ScorerName] = ("scanpy", "aucell"),
    resume_from_scores: bool = False,
    subset_fraction: float | None = None,
    random_state: int = 42,
    tissue_key: str = "tissue_in_publication",
    cell_type_key: str = "cell_type",
    sex_key: str = "sex",
    development_stage_key: str = "development_stage",
    assay_key: str = "assay",
    age_key: str = "age_years",
    donor_key: str = "donor_id",
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    module_ids: Sequence[str] | None = None,
    sensor_group: str = "nucleic_acid_sensors",
    plot_modules: bool = True,
    aucell_chunk_size: int = 1_000,
    aucell_num_workers: int = 1,
    score_table_filename: str = "tabula_sapiens_module_scores.csv.gz",
    single_tissue_use_rep: str | None = "X_scvi",
    statistical_unit: StatisticalUnit = "donor",
    aggregation: Aggregation = "mean",
    max_plots: int | None = 200,
    plot_nasp_visualizations: bool = True,
) -> dict[str, Path]:
    """Score and analyze one tissue h5ad with each requested scorer.

    The workflow writes one isolated run directory per tissue input. Scoring is
    completed first into a shared score table, then the full donor-aware
    association workflow runs once per scorer so Scanpy and AUCell remain
    separate sensitivity analyses. Setting `resume_from_scores=True` reuses a
    score table only when its provenance records every requested scorer.

    Args:
      h5ad_path: Per-tissue h5ad input path.
      output_dir: Root directory containing tissue run directories.
      tissue_label: Exact tissue value used to subset and recompute UMAP. Leave
        unset when the input is already tissue-specific and its embedding
        should be retained.
      run_name: Output directory name. Defaults to `tissue_label`, then the
        input filename stem.
      scorers: Scorers to run and analyze ("scanpy", "aucell", or both).
      resume_from_scores: Reuse a complete existing score table when possible.
      subset_fraction: Optional cell fraction for exploratory runs.
      random_state: Random seed for loading, UMAP, and scoring.
      tissue_key: Obs column identifying tissue.
      cell_type_key: Obs column identifying cell type.
      sex_key: Obs column identifying sex.
      development_stage_key: Obs column identifying development stage.
      assay_key: Obs column identifying assay.
      age_key: Obs column containing or receiving numeric age.
      donor_key: Obs column identifying donors.
      gene_symbol_column: Var column containing gene symbols.
      expression_layer: Expression layer used for scoring and associations.
      module_ids: Module IDs to score. Defaults to all compendium modules.
      sensor_group: Compendium sensor group used for gene-level associations.
      plot_modules: Whether to generate per-module marker plots.
      aucell_chunk_size: Cells processed in one AUCell block.
      aucell_num_workers: AUCell worker processes per block.
      score_table_filename: Shared score-table filename under `scoring`.
      single_tissue_use_rep: Representation for tissue UMAP recomputation.
      statistical_unit: Primary association statistical unit.
      aggregation: Cell-to-unit score aggregation.
      max_plots: Maximum association plots written per scorer.
      plot_nasp_visualizations: Whether to render mechanistic NASP summaries.

    Returns:
      Paths for the run directory, score table, and each scorer's association
      output directory.

    Raises:
      FileNotFoundError: If the input h5ad or completed score table is absent.
      ValueError: If no scorer or an unsupported scorer is requested.
    """
    input_path = Path(h5ad_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"tissue h5ad does not exist: {input_path}")
    selected_scorers = tuple(dict.fromkeys(scorers))
    if not selected_scorers:
        raise ValueError("scorers must contain at least one scorer")
    if invalid_scorers := set(selected_scorers).difference(
        {"scanpy", "aucell"}
    ):
        invalid_text = ", ".join(sorted(invalid_scorers))
        raise ValueError(f"unsupported scorers: {invalid_text}")

    output_name = run_name or tissue_label or input_path.stem
    run_dir = Path(output_dir) / scoring.safe_filename_token(output_name)
    scoring_dir = run_dir / "scoring"
    score_path = scoring_dir / score_table_filename
    if resume_from_scores and _score_table_has_scorers(
        score_path,
        selected_scorers,
    ):
        logger.info(
            "[tabula_sapiens] reusing complete score table -> %s",
            score_path,
        )
    else:
        tabula_sapiens_scoring_analysis(
            h5ad_path=input_path,
            output_dir=scoring_dir,
            subset_fraction=subset_fraction,
            random_state=random_state,
            tissue_key=tissue_key,
            cell_type_key=cell_type_key,
            sex_key=sex_key,
            development_stage_key=development_stage_key,
            assay_key=assay_key,
            age_key=age_key,
            donor_key=donor_key,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            module_ids=module_ids,
            sensor_group=sensor_group,
            plot_modules=plot_modules,
            score_scanpy="scanpy" in selected_scorers,
            score_aucell="aucell" in selected_scorers,
            aucell_chunk_size=aucell_chunk_size,
            aucell_num_workers=aucell_num_workers,
            score_table_filename=score_table_filename,
            heatmap_groupby=cell_type_key,
            single_tissue=tissue_label,
            single_tissue_use_rep=single_tissue_use_rep,
        )
    if not score_path.is_file():
        raise FileNotFoundError(
            f"scoring completed without the expected score table: {score_path}"
        )

    outputs = {
        "run_dir": run_dir,
        "score_table": score_path,
    }
    for scorer_name in selected_scorers:
        association_dir = run_dir / "associations" / scorer_name
        association_analysis(
            h5ad_path=input_path,
            score_csv_path=score_path,
            output_dir=association_dir,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            donor_key=donor_key,
            tissue_key=tissue_key,
            cell_type_key=cell_type_key,
            sex_key=sex_key,
            assay_key=assay_key,
            development_stage_key=development_stage_key,
            age_key=age_key,
            module_ids=module_ids,
            sensor_group=sensor_group,
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            scorer=scorer_name,
            max_plots=max_plots,
            plot_nasp_visualizations=plot_nasp_visualizations,
        )
        outputs[f"association_{scorer_name}"] = association_dir
    return outputs


def tabula_sapiens_scoring_analysis(
    *,
    h5ad_path: str | Path,
    output_dir: str | Path,
    subset_fraction: float | None = None,
    random_state: int = 42,
    tissue_key: str = "tissue_in_publication",
    cell_type_key: str = "cell_type",
    sex_key: str = "sex",
    development_stage_key: str = "development_stage",
    assay_key: str = "assay",
    age_key: str = "age_years",
    donor_key: str = "donor_id",
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    module_ids: Sequence[str] | None = None,
    metadata_panels: Sequence[str | UmapPanelSpec] | None = None,
    sensor_group: str = "nucleic_acid_sensors",
    plot_modules: bool = False,
    score_scanpy: bool = False,
    score_aucell: bool = False,
    aucell_chunk_size: int = 1_000,
    aucell_num_workers: int = 1,
    score_table_filename: str = "tabula_sapiens_module_scores.csv.gz",
    heatmap_groupby: str | None = None,
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
      assay_key: Obs column identifying sequencing assay.
      age_key: Obs column to hold numeric age in years.
      gene_symbol_column: Var column containing gene symbols.
      expression_layer: Expression layer used for gene-expression plots.
      module_ids: Optional module IDs to score and plot. Defaults to all
        modules from `GeneModules`.
      metadata_panels: Optional metadata UMAP panel specification.
      donor_key: Obs column identifying donors.
      sensor_group: Sensor group name used to select sensor genes.
      plot_modules: Whether to plot per-module marker-gene UMAPs and heatmaps.
      score_scanpy: Whether to score modules with scanpy.
      score_aucell: Whether to score modules with AUCell.
      aucell_chunk_size: Maximum cells densified in one AUCell block.
      aucell_num_workers: Worker processes used by each AUCell block.
      score_table_filename: Output score table filename under `output_dir`.
      heatmap_groupby: Obs column used for all gene-expression and module-score
        heatmaps. Use this for the common case where every heatmap should be
        summarized at the same level, such as tissue or cell type.
      heatmap_obs_keys: Obs columns used for sensor/module gene-expression
        heatmaps. Defaults to `heatmap_groupby` when set, otherwise tissue and
        cell type.
      score_heatmap_obs_key: Obs column used for scanpy/AUCell score heatmaps.
        Defaults to `heatmap_groupby` when set, otherwise `cell_type_key`.
      single_tissue: If set, recompute neighbors/UMAP for one tissue.
      single_tissue_use_rep: Representation used for single-tissue UMAP
        recomputation.
    """
    adata, point_size = _load_scoring_adata(
        h5ad_path=h5ad_path,
        subset_fraction=subset_fraction,
        random_state=random_state,
        tissue_key=tissue_key,
        expression_layer=expression_layer,
        single_tissue=single_tissue,
        single_tissue_use_rep=single_tissue_use_rep,
    )
    viz = SCVisualizer(output_dir=output_dir)
    plot_tabula_sapiens_metadata_umaps(
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

    if heatmap_groupby is not None:
        requested_heatmap_keys = heatmap_obs_keys or (heatmap_groupby,)
    elif heatmap_obs_keys is None:
        requested_heatmap_keys = (tissue_key, cell_type_key)
    else:
        requested_heatmap_keys = heatmap_obs_keys
    heatmap_groupby_keys = tuple(dict.fromkeys(requested_heatmap_keys))
    score_heatmap_groupby_key = (
        score_heatmap_obs_key or heatmap_groupby or cell_type_key
    )

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
    scoring.plot_gene_expression_heatmaps_by_obs(
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

    if plot_modules:
        _plot_module_gene_umaps(
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

    scoring.module_scoring_outputs(
        adata,
        selected_module_ids,
        viz=viz,
        output_dir=output_dir,
        score_table_filename=score_table_filename,
        score_scanpy=score_scanpy,
        score_aucell=score_aucell,
        scanpy_scorer=score_scanpy_modules,
        aucell_scorer=score_aucell_modules,
        donor_key=donor_key,
        tissue_key=tissue_key,
        cell_type_key=cell_type_key,
        sex_key=sex_key,
        assay_key=assay_key,
        development_stage_key=development_stage_key,
        age_key=age_key,
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
        score_heatmap_groupby_key=score_heatmap_groupby_key,
        subset_fraction=subset_fraction,
        random_state=random_state,
        aucell_chunk_size=aucell_chunk_size,
        aucell_num_workers=aucell_num_workers,
        point_size=point_size,
    )


def _load_scoring_adata(
    *,
    h5ad_path: str | Path,
    subset_fraction: float | None,
    random_state: int,
    tissue_key: str,
    expression_layer: str | None,
    single_tissue: str | None,
    single_tissue_use_rep: str | None,
) -> tuple[ad.AnnData, float]:
    """Load scoring inputs and optionally recompute one tissue embedding."""
    requested_obsm = ["X_umap"]
    if single_tissue is not None and single_tissue_use_rep is not None:
        requested_obsm.append(single_tissue_use_rep)

    adata, _ = read_h5ad(
        h5ad_path,
        subset_fraction=subset_fraction,
        random_state=random_state,
        read_x=expression_layer is None,
        layer_keys=(() if expression_layer is None else (expression_layer,)),
        read_raw=False,
        obsm_keys=tuple(dict.fromkeys(requested_obsm)),
    )
    if single_tissue is not None:
        if tissue_key not in adata.obs.columns:
            raise KeyError(f"tissue column not found: {tissue_key}")

        tissue_mask = adata.obs[tissue_key].astype(str) == single_tissue
        if not bool(tissue_mask.any()):
            raise ValueError(
                f"single_tissue {single_tissue!r} is absent from "
                f"adata.obs[{tissue_key!r}]"
            )

        adata = adata[tissue_mask.to_numpy(), :].copy()
        SCProcessor.recompute_umap(
            adata,
            use_rep=single_tissue_use_rep,
            random_state=random_state,
            min_dist=0.425,
        )

    point_size = 75000 / adata.n_obs
    if single_tissue is not None:
        point_size /= 4

    return adata, point_size


def _score_table_has_scorers(
    score_path: Path,
    scorers: Sequence[ScorerName],
) -> bool:
    """Return whether a score table records all requested scorers."""
    if not score_path.is_file():
        return False
    try:
        provenance = pd.read_csv(
            score_path,
            usecols=["scoring_scorers"],
            nrows=1,
        )
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    if provenance.empty:
        return False
    completed = {
        scorer.strip()
        for scorer in str(provenance.iloc[0, 0]).split(",")
        if scorer.strip()
    }
    return set(scorers).issubset(completed)


def _plot_module_gene_umaps(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    viz: SCVisualizer,
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    ncols: int = 6,
    size: float | None = None,
) -> None:
    """Plot one multi-gene UMAP panel per Tabula Sapiens NASP module."""
    point_size = size if size is not None else 120000 / adata.n_obs
    for module_id in module_ids:
        module_genes = GeneModules.genes(
            module_id,
            adata=adata,  # type: ignore[arg-type]  # upstream protocol mismatch
            gene_symbol_column=gene_symbol_column,
            output="symbols",
        )
        if not module_genes:
            logger.info("%s: no matched genes; skipping UMAPs", module_id)
            continue

        logger.info(
            "%s: plotting %s marker genes",
            module_id,
            len(module_genes),
        )
        viz.plot_multi_gene_umap_panel(
            adata=adata,
            genes=module_genes,
            filename=f"{module_id}_gene_expression_umaps",
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            ncols=ncols,
            size=point_size,
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
            adata=adata,  # type: ignore[arg-type]  # upstream protocol mismatch
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
        scoring.plot_gene_expression_heatmaps_by_obs(
            adata=adata,
            genes=module_genes,
            viz=viz,
            filename_prefix=module_id,
            groupby_keys=groupby_keys,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
        )
