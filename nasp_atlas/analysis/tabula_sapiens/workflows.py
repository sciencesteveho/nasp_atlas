"""Tabula Sapiens NASP analysis workflows."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.tabula_sapiens import associations
from nasp_atlas.analysis.tabula_sapiens import scoring
from nasp_atlas.analysis.tabula_sapiens.mixed_models import (
    TabulaMixedModelResults,
)
from nasp_atlas.analysis.tabula_sapiens.mixed_models import (
    tabula_sapiens_mixed_model_inference,
)
from nasp_atlas.analysis.tabula_sapiens.visualizations import (
    plot_nasp_association_visualizations,
)
from nasp_atlas.analysis.tabula_sapiens.visualizations import (
    plot_tabula_sapiens_metadata_umaps,
)
from nasp_atlas.analysis.tabula_sapiens.visualizations import (
    plot_tabula_sapiens_mixed_model_inference,
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
from nasp_atlas.single_cell.visualization import AssociationPlotter
from nasp_atlas.single_cell.visualization import GroupedGeneExpression
from nasp_atlas.single_cell.visualization import HeatmapPlotter
from nasp_atlas.single_cell.visualization import SummaryPlotter
from nasp_atlas.single_cell.visualization import UmapPlotter


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

    def regression_results(self) -> pd.DataFrame:
        """Return retained regression results with their stable schema."""
        return _ensure_declared_columns(
            associations._concat_or_empty(self.regression_tables),
            declared=(
                "feature_type",
                "feature_id",
                "feature_label",
                "predictor",
                "statistical_unit",
                "aggregation",
                "stratify_key",
                "stratum",
                "analysis_role",
                "n_units",
                "fdr_method",
                "n",
                "pearson_r",
                "pearson_pvalue",
                "spearman_r",
                "spearman_pvalue",
                "slope",
                "intercept",
                "ols_pvalue",
                "skipped",
                "skip_reason",
                "pearson_pvalue_fdr",
                "spearman_pvalue_fdr",
                "ols_pvalue_fdr",
                "analysis_scope",
                "response_estimand",
                "eqtl_fdr_family",
                "eqtl_source_schema",
                "eqtl_count_unit",
                "eqtl_predictor_transform",
                "eqtl_table_path",
            ),
        )

    def eqtl_annotations(self) -> pd.DataFrame:
        """Return retained eQTL annotations with their stable schema."""
        return _ensure_declared_columns(
            associations._concat_or_empty(self.eqtl_annotation_tables),
            declared=(
                "feature_type",
                "feature_id",
                "feature_label",
                "feature_value",
                "statistical_unit",
                "aggregation",
                "unit_id",
                "n_cells",
                "n_cells_total",
                "eqtl_matched",
                "eqtl_merge_mode",
                "eqtl_source_schema",
                "eqtl_count_unit",
                "eqtl_predictor_transform",
                "eqtl_table_path",
            ),
        )

    def plot_manifest(self) -> pd.DataFrame:
        """Return produced plot records with their stable public schema."""
        return pd.DataFrame(
            self.manifest,
            columns=(
                "kind",
                "feature_id",
                "predictor",
                "statistical_unit",
                "aggregation",
                "stratum",
                "analysis_scope",
                "path",
            ),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _AssociationPlan:
    """Metadata schema and inferential policy shared by association stages."""

    obs_schema: ObsSchema
    statistical_unit: StatisticalUnit
    aggregation: Aggregation
    detection_threshold: float
    condition_reference: str | None
    mixed_model_min_cells: int
    mixed_model_min_donors: int
    mixed_model_min_studies: int
    mixed_model_min_repeated_contexts: int


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
    condition_key: str = "disease",
    study_key: str = "dataset_id",
    condition_reference: str | None = "normal",
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
    mixed_model_min_cells: int = 10,
    mixed_model_min_donors: int = 3,
    mixed_model_min_studies: int = 3,
    mixed_model_min_repeated_contexts: int = 3,
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
        detection_threshold=detection_threshold,
        condition_reference=condition_reference,
        mixed_model_min_cells=mixed_model_min_cells,
        mixed_model_min_donors=mixed_model_min_donors,
        mixed_model_min_studies=mixed_model_min_studies,
        mixed_model_min_repeated_contexts=mixed_model_min_repeated_contexts,
    )

    output_path = Path(output_dir)
    layout = associations._association_output_dirs(output_path)
    _reconcile_association_outputs(layout)
    schema = ObsSchema(
        donor_key=donor_key,
        tissue_key=tissue_key,
        cell_type_key=cell_type_key,
        sex_key=sex_key,
        assay_key=assay_key,
        development_stage_key=development_stage_key,
        age_key=age_key,
        condition_key=condition_key,
        study_key=study_key,
    )
    association_plan = _AssociationPlan(
        obs_schema=schema,
        statistical_unit=cast(StatisticalUnit, statistical_unit),
        aggregation=cast(Aggregation, aggregation),
        detection_threshold=detection_threshold,
        condition_reference=condition_reference,
        mixed_model_min_cells=mixed_model_min_cells,
        mixed_model_min_donors=mixed_model_min_donors,
        mixed_model_min_studies=mixed_model_min_studies,
        mixed_model_min_repeated_contexts=mixed_model_min_repeated_contexts,
    )
    adata, scores = _load_association_inputs(
        h5ad_path=h5ad_path,
        score_csv_path=score_csv_path,
        schema=schema,
        expression_layer=expression_layer,
        expression_use_raw=expression_use_raw,
    )
    feature_specs = _resolve_association_features(
        adata,
        scores,
        tables_dir=layout.tables_dir,
        module_ids=module_ids,
        sensor_group=sensor_group,
        gene_symbols=gene_symbols,
        scorer=scorer,
        gene_symbol_column=gene_symbol_column,
        expression_use_raw=expression_use_raw,
    )
    provenance = _association_provenance(
        h5ad_path=h5ad_path,
        score_csv_path=score_csv_path,
        eqtl_table_path=eqtl_table_path,
        scores=scores,
        feature_specs=feature_specs,
        module_ids=module_ids,
        gene_symbols=gene_symbols,
        sensor_group=sensor_group,
        scorer=scorer or associations._detect_scorer_from_scores(scores),
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
        expression_use_raw=expression_use_raw,
        eqtl_merge_mode=eqtl_merge_mode,
        run_cell_level_descriptive_plots=run_cell_level_descriptive_plots,
        run_donor_aware_tests=run_donor_aware_tests,
        max_plots=max_plots,
        plot_nasp_visualizations=plot_nasp_visualizations,
        association_plan=association_plan,
    )
    associations._write_association_table(
        provenance,
        layout.tables_dir / "association_provenance.csv",
    )
    if not feature_specs:
        _write_empty_association_tables(layout.tables_dir)
        return

    cell_frame = build_cell_feature_frame(
        adata,
        scores,
        feature_specs,
        schema=association_plan.obs_schema,
        expression_layer=expression_layer,
        use_raw=expression_use_raw,
    )
    unit_frame = aggregate_feature_frame(
        cell_frame,
        statistical_unit=association_plan.statistical_unit,
        aggregation=association_plan.aggregation,
        schema=association_plan.obs_schema,
        detection_threshold=association_plan.detection_threshold,
    )
    nasp_results = _build_nasp_profile_results(
        cell_frame,
        association_plan=association_plan,
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
            layout.tables_dir / filename,
        )

    workflow_state = _AssociationWorkflowState()
    mixed_model_results: TabulaMixedModelResults | None = None
    if run_donor_aware_tests:
        mixed_model_results = tabula_sapiens_mixed_model_inference(
            cell_frame,
            schema=association_plan.obs_schema,
            aggregation=association_plan.aggregation,
            detection_threshold=association_plan.detection_threshold,
            condition_reference=association_plan.condition_reference,
            minimum_cells=association_plan.mixed_model_min_cells,
            minimum_donors=association_plan.mixed_model_min_donors,
            minimum_studies=association_plan.mixed_model_min_studies,
            minimum_repeated_contexts=(
                association_plan.mixed_model_min_repeated_contexts
            ),
        )
    if eqtl_table_path is not None:
        eqtl_unit_frame = unit_frame
        if eqtl_merge_mode in {"gene_tissue", "tissue"}:
            eqtl_unit_frame = aggregate_feature_frame(
                cell_frame,
                statistical_unit="donor_tissue",
                aggregation=association_plan.aggregation,
                schema=association_plan.obs_schema,
                detection_threshold=association_plan.detection_threshold,
            )
        associations._run_eqtl_associations(
            eqtl_unit_frame,
            eqtl_table_path=eqtl_table_path,
            eqtl_merge_mode=cast(EqtlMergeMode, eqtl_merge_mode),
            schema=association_plan.obs_schema,
            regression_tables=workflow_state.regression_tables,
            annotation_tables=workflow_state.eqtl_annotation_tables,
        )
    if (
        run_cell_level_descriptive_plots
        and association_plan.obs_schema.age_key in cell_frame.columns
    ):
        workflow_state.plot_count = (
            associations._run_cell_level_descriptive_plots(
                cell_frame,
                predictor_key=association_plan.obs_schema.age_key,
                plotter=AssociationPlotter(
                    output_dir=str(layout.regression_plots_dir)
                ),
                aggregation=association_plan.aggregation,
                manifest=workflow_state.manifest,
                plot_count=workflow_state.plot_count,
                max_plots=max_plots,
            )
        )

    _write_association_results(
        workflow_state,
        nasp_results=nasp_results,
        mixed_model_results=mixed_model_results,
        layout=layout,
        association_plan=association_plan,
        plot_nasp_visualizations=plot_nasp_visualizations,
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
    detection_threshold: float,
    condition_reference: str | None,
    mixed_model_min_cells: int,
    mixed_model_min_donors: int,
    mixed_model_min_studies: int,
    mixed_model_min_repeated_contexts: int,
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
    if eqtl_merge_mode not in (
        "gene",
        "gene_tissue",
        "tissue",
        "module",
        "donor",
    ):
        raise ValueError(f"unsupported eqtl_merge_mode: {eqtl_merge_mode}")
    _validate_detection_threshold(detection_threshold)
    if condition_reference is not None and not condition_reference.strip():
        raise ValueError("condition_reference must not be empty")
    if mixed_model_min_cells < 1:
        raise ValueError("mixed_model_min_cells must be at least 1")
    if mixed_model_min_donors < 3:
        raise ValueError("mixed_model_min_donors must be at least 3")
    if mixed_model_min_studies < 3:
        raise ValueError("mixed_model_min_studies must be at least 3")
    if mixed_model_min_repeated_contexts < 3:
        raise ValueError("mixed_model_min_repeated_contexts must be at least 3")


def _validate_detection_threshold(detection_threshold: float) -> None:
    """Reject thresholds that cannot define a finite expression boundary."""
    if (
        isinstance(detection_threshold, bool)
        or not isinstance(detection_threshold, Real)
        or not isfinite(detection_threshold)
    ):
        raise ValueError("detection_threshold must be a finite real number")


def _association_provenance(
    *,
    h5ad_path: str | Path,
    score_csv_path: str | Path,
    eqtl_table_path: str | Path | None,
    scores: pd.DataFrame,
    feature_specs: Sequence[FeatureSpec],
    module_ids: Sequence[str] | None,
    gene_symbols: Sequence[str] | None,
    sensor_group: str | None,
    scorer: str,
    gene_symbol_column: str,
    expression_layer: str | None,
    expression_use_raw: bool,
    eqtl_merge_mode: str,
    run_cell_level_descriptive_plots: bool,
    run_donor_aware_tests: bool,
    max_plots: int | None,
    plot_nasp_visualizations: bool,
    association_plan: _AssociationPlan,
) -> pd.DataFrame:
    """Return one record tying results to inputs and feature sources."""
    h5ad_identity = _file_identity("h5ad", h5ad_path)
    score_identity = _file_identity("score_csv", score_csv_path)
    eqtl_identity = _optional_file_identity("eqtl", eqtl_table_path)
    marker_path = GeneModules.default_panel_path().resolve()
    marker_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    modules = [
        spec for spec in feature_specs if spec.feature_type == "module_score"
    ]
    genes = [
        spec for spec in feature_specs if spec.feature_type == "gene_expression"
    ]
    record: dict[str, object] = {
        **h5ad_identity,
        **score_identity,
        **eqtl_identity,
        "scorer": scorer,
        "gene_symbol_column": gene_symbol_column,
        "expression_source": (
            "raw" if expression_use_raw else expression_layer or "X"
        ),
        "expression_use_raw": expression_use_raw,
        "sensor_group": sensor_group,
        "requested_module_ids": _json_sequence(module_ids),
        "requested_gene_symbols": _json_sequence(gene_symbols),
        "resolved_module_feature_ids": _json_sequence(
            [spec.feature_id for spec in modules]
        ),
        "resolved_module_labels": _json_sequence(
            [spec.feature_label for spec in modules]
        ),
        "resolved_gene_feature_ids": _json_sequence(
            [spec.feature_id for spec in genes]
        ),
        "resolved_gene_labels": _json_sequence(
            [spec.feature_label for spec in genes]
        ),
        "compendium_marker_panel_path": str(marker_path),
        "compendium_marker_panel_sha256": marker_hash,
        "statistical_unit": association_plan.statistical_unit,
        "aggregation": association_plan.aggregation,
        "detection_threshold": association_plan.detection_threshold,
        "condition_key": association_plan.obs_schema.condition_key,
        "configured_condition_reference": (
            association_plan.condition_reference
        ),
        "study_key": association_plan.obs_schema.study_key,
        "mixed_model_min_cells": association_plan.mixed_model_min_cells,
        "mixed_model_min_donors": association_plan.mixed_model_min_donors,
        "mixed_model_min_studies": association_plan.mixed_model_min_studies,
        "mixed_model_min_repeated_contexts": (
            association_plan.mixed_model_min_repeated_contexts
        ),
        "eqtl_merge_mode": eqtl_merge_mode,
        "run_cell_level_descriptive_plots": (run_cell_level_descriptive_plots),
        "run_donor_aware_tests": run_donor_aware_tests,
        "max_plots": max_plots,
        "plot_nasp_visualizations": plot_nasp_visualizations,
    }
    score_provenance = _constant_score_provenance(scores)
    record |= score_provenance
    scoring_hash = score_provenance.get("scoring_marker_panel_sha256")
    record["marker_panel_matches_score"] = (
        str(scoring_hash) == marker_hash
        if isinstance(scoring_hash, str) and scoring_hash
        else pd.NA
    )
    return pd.DataFrame.from_records([record])


def _file_identity(prefix: str, path: str | Path) -> dict[str, object]:
    """Return stable path, size, and modification identity for one input."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        f"{prefix}_path": str(resolved),
        f"{prefix}_size_bytes": stat.st_size,
        f"{prefix}_mtime_ns": stat.st_mtime_ns,
    }


def _optional_file_identity(
    prefix: str,
    path: str | Path | None,
) -> dict[str, object]:
    """Return nullable file identity fields for an optional source."""
    if path is None:
        return {
            f"{prefix}_path": pd.NA,
            f"{prefix}_size_bytes": pd.NA,
            f"{prefix}_mtime_ns": pd.NA,
        }
    return _file_identity(prefix, path)


def _constant_score_provenance(scores: pd.DataFrame) -> dict[str, object]:
    """Extract constant score-table provenance without hiding conflicts."""
    record: dict[str, object] = {}
    for column in (
        "scoring_scorers",
        "scoring_requested_scorers",
        "scoring_expression_source",
        "scoring_subset_fraction",
        "scoring_random_state",
        "scoring_module_ids",
        "scoring_n_modules",
        "scoring_aucell_chunk_size",
        "scoring_aucell_num_workers",
        "scoring_marker_panel_sha256",
    ):
        if column not in scores:
            record[column] = pd.NA
            continue
        values = scores[column].dropna().drop_duplicates()
        if len(values) > 1:
            raise ValueError(
                f"score-table provenance column {column!r} is inconsistent"
            )
        record[column] = values.iloc[0] if len(values) == 1 else pd.NA
    return record


def _json_sequence(values: Sequence[str] | None) -> str:
    """Serialize an optional ordered selection without delimiter ambiguity."""
    return json.dumps(list(values or ()), separators=(",", ":"))


def _reconcile_association_outputs(
    layout: associations._AssociationOutputLayout,
) -> None:
    """Remove workflow-owned stale artifacts before a deterministic rerun."""
    for filename in (
        "association_age_stability.csv",
        "association_partial_correlation_age.csv",
        "association_group_test_results.csv",
        "association_group_summary.csv",
        "association_provenance.csv",
        "association_skipped_features.csv",
        "association_mixed_model_contrasts.csv",
        "association_mixed_model_fixed_effects.csv",
        "association_mixed_model_term_tests.csv",
        "association_mixed_model_variance_components.csv",
        "association_mixed_model_diagnostics.csv",
        "association_mixed_model_availability.csv",
        "association_regression_results.csv",
        "association_eqtl_annotations.csv",
        "association_plot_manifest.csv",
        "nasp_evidence_profiles.csv",
        "nasp_context_summary.csv",
        "nasp_hypothesis_priorities.csv",
        "nasp_mechanistic_edges.csv",
        "nasp_module_context_ranking.csv",
        "nasp_module_coupling.csv",
        "nasp_sensor_output_coupling.csv",
    ):
        (layout.tables_dir / filename).unlink(missing_ok=True)
    for directory in (
        layout.regression_plots_dir,
        layout.boxplots_dir,
        layout.barplots_dir,
    ):
        for path in directory.glob("*.png"):
            path.unlink()
    for stem in (
        "nasp_mixed_adjusted_cell_type_effects",
        "nasp_mixed_condition_effects_by_cell_type",
        "nasp_mixed_age_slopes_by_cell_type",
        "nasp_mixed_paired_tissue_effects",
        "nasp_mixed_assay_batch_effects",
        "nasp_mixed_variance_decomposition",
    ):
        (layout.mixed_model_plots_dir / f"{stem}.png").unlink(missing_ok=True)
    for stem in (
        "nasp_module_coupling_heatmap",
        "nasp_competence_output_state_map",
        "nasp_ranked_hypotheses",
        "nasp_sensor_output_mismatch",
        "nasp_mechanistic_edge_network",
        "nasp_age_effects_by_cell_type",
        "nasp_sensor_age_effects_by_cell_type",
        "nasp_age_effect_consistency_across_cell_types",
        "nasp_sensor_age_effect_consistency_across_cell_types",
    ):
        (layout.nasp_plots_dir / f"{stem}.png").unlink(missing_ok=True)


def _load_association_inputs(
    *,
    h5ad_path: str | Path,
    score_csv_path: str | Path,
    schema: ObsSchema,
    expression_layer: str | None,
    expression_use_raw: bool,
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
        [schema.age_key],
    ):
        logger.warning(
            "[tabula_sapiens] age column %r absent from scores and obs; "
            "age-based association will be skipped",
            schema.age_key,
        )

    return adata, scores


def _resolve_association_features(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    *,
    tables_dir: Path,
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
        tables_dir / "association_skipped_features.csv",
    )
    if not feature_specs:
        logger.warning(
            "[tabula_sapiens] no features resolved for association; "
            "nothing to test"
        )

    return feature_specs


def _write_empty_association_tables(tables_dir: Path) -> None:
    """Write stable empty outputs when no requested feature can be analyzed."""
    empty_state = _AssociationWorkflowState()
    for filename in (
        "nasp_evidence_profiles.csv",
        "nasp_context_summary.csv",
        "nasp_hypothesis_priorities.csv",
        "nasp_mechanistic_edges.csv",
        "nasp_module_context_ranking.csv",
        "nasp_module_coupling.csv",
        "nasp_sensor_output_coupling.csv",
    ):
        associations._write_association_table(
            pd.DataFrame(),
            tables_dir / filename,
        )
    associations._write_association_table(
        empty_state.regression_results(),
        tables_dir / "association_regression_results.csv",
    )
    associations._write_association_table(
        empty_state.eqtl_annotations(),
        tables_dir / "association_eqtl_annotations.csv",
    )
    associations._write_association_table(
        empty_state.plot_manifest(),
        tables_dir / "association_plot_manifest.csv",
    )
    empty = TabulaMixedModelResults.empty()
    for filename, table in (
        ("association_mixed_model_contrasts.csv", empty.contrasts),
        ("association_mixed_model_fixed_effects.csv", empty.fixed_effects),
        ("association_mixed_model_term_tests.csv", empty.term_tests),
        (
            "association_mixed_model_variance_components.csv",
            empty.variance_components,
        ),
        ("association_mixed_model_diagnostics.csv", empty.diagnostics),
        ("association_mixed_model_availability.csv", empty.availability),
    ):
        associations._write_association_table(table, tables_dir / filename)


def _build_nasp_profile_results(
    cell_frame: pd.DataFrame,
    *,
    association_plan: _AssociationPlan,
) -> _NaspProfileResults:
    """Build donor-context profile, coupling, and prioritization results."""
    schema = association_plan.obs_schema
    donor_key = schema.donor_key
    tissue_key = schema.tissue_key
    cell_type_key = schema.cell_type_key
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
        aggregation=association_plan.aggregation,
        schema=schema,
        detection_threshold=association_plan.detection_threshold,
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
    edge_specs = (
        (
            "NASP_DNA_SENSING",
            "SIGNALING_CONTEXT_TBK1_IRF",
            "dna_sensing_to_proximal_signaling",
        ),
        (
            "NASP_RNA_SENSING",
            "SIGNALING_CONTEXT_TBK1_IRF",
            "rna_sensing_to_proximal_signaling",
        ),
        (
            "NASP_DNA_SENSING",
            "IFN_I_OUTPUT",
            "dna_sensing_to_ifn_output",
        ),
        (
            "NASP_RNA_SENSING",
            "IFN_I_OUTPUT",
            "rna_sensing_to_ifn_output",
        ),
        (
            "NASP_DNA_SENSING",
            "NFKB_CYTOKINE_OUTPUT",
            "dna_sensing_to_nfkb_output",
        ),
        (
            "NASP_RNA_SENSING",
            "NFKB_CYTOKINE_OUTPUT",
            "rna_sensing_to_nfkb_output",
        ),
        (
            "SIGNALING_CONTEXT_TBK1_IRF",
            "IFN_I_OUTPUT",
            "proximal_signaling_to_ifn_output",
        ),
        (
            "SIGNALING_CONTEXT_TLR",
            "NFKB_CYTOKINE_OUTPUT",
            "tlr_context_to_nfkb_output",
        ),
        (
            "SIGNALING_CONTEXT_NFKB",
            "NFKB_CYTOKINE_OUTPUT",
            "nfkb_context_to_nfkb_output",
        ),
        (
            "SIGNALING_CONTEXT_IFN_JAK_STAT",
            "IFN_I_OUTPUT",
            "ifn_response_context_to_ifn_output",
        ),
        ("NASP_RNA_SENSING", "ISR", "rna_sensing_to_isr"),
        (
            "NASP_DNA_SENSING",
            "INFLAMMASOME",
            "dna_sensing_to_inflammasome",
        ),
        (
            "MITOCHONDRIAL_NA_SENSING",
            "NASP_DNA_SENSING",
            "mitochondrial_na_to_dna_sensing",
        ),
        (
            "MITOCHONDRIAL_NA_SENSING",
            "INFLAMMASOME",
            "mitochondrial_na_to_inflammasome",
        ),
        ("TE_DEREPRESSION", "NASP_DNA_SENSING", "te_to_dna_sensing"),
        ("TE_DEREPRESSION", "NASP_RNA_SENSING", "te_to_rna_sensing"),
        (
            "CGAMP_TRANSPORT",
            "IFN_I_OUTPUT",
            "cgamp_transport_to_ifn_output",
        ),
        ("IFN_I_OUTPUT", "NASP_FEEDBACK", "ifn_output_to_feedback"),
        (
            "NFKB_CYTOKINE_OUTPUT",
            "NASP_FEEDBACK",
            "nfkb_output_to_feedback",
        ),
        ("IFN_I_OUTPUT", "INFLAMMAGING", "ifn_output_to_inflammaging"),
        ("NFKB_CYTOKINE_OUTPUT", "SASP", "nfkb_output_to_sasp"),
    )
    mechanistic_edges = (
        pd.DataFrame()
        if module_coupling.empty
        else expected_module_coupling_report(
            module_coupling,
            edge_specs=edge_specs,
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
    layout: associations._AssociationOutputLayout,
    association_plan: _AssociationPlan,
    max_plots: int | None,
) -> None:
    """Run global and context-specific donor-aware age analyses."""
    schema = association_plan.obs_schema
    donor_key = schema.donor_key
    tissue_key = schema.tissue_key
    cell_type_key = schema.cell_type_key
    sex_key = schema.sex_key
    age_key = schema.age_key
    plotter = AssociationPlotter(output_dir=str(layout.regression_plots_dir))
    scope = (
        "global_donor"
        if association_plan.statistical_unit == "donor"
        else f"pooled_{association_plan.statistical_unit}"
    )
    workflow_state.plot_count = associations._run_continuous_associations(
        unit_frame,
        predictor_key=age_key,
        plotter=plotter,
        statistical_unit=association_plan.statistical_unit,
        aggregation=association_plan.aggregation,
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
            plotter=plotter,
            statistical_unit=association_plan.statistical_unit,
            aggregation=association_plan.aggregation,
            manifest=workflow_state.manifest,
            regression_tables=workflow_state.regression_tables,
            plot_count=workflow_state.plot_count,
            max_plots=max_plots,
            stratify_key=sex_key,
            analysis_scope=f"within_sex_{association_plan.statistical_unit}",
        )

    if donor_key in cell_frame.columns and tissue_key in cell_frame.columns:
        donor_tissue_frame = aggregate_feature_frame(
            cell_frame,
            statistical_unit="donor_tissue",
            aggregation=association_plan.aggregation,
            schema=schema,
            detection_threshold=association_plan.detection_threshold,
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
            layout.tables_dir / "association_partial_correlation_age.csv",
        )


def _run_group_associations(
    cell_frame: pd.DataFrame,
    unit_frame: pd.DataFrame,
    *,
    workflow_state: _AssociationWorkflowState,
    layout: associations._AssociationOutputLayout,
    association_plan: _AssociationPlan,
    max_plots: int | None,
) -> None:
    """Run categorical tests supported by the selected statistical unit."""
    schema = association_plan.obs_schema
    tissue_key = schema.tissue_key
    cell_type_key = schema.cell_type_key
    sex_key = schema.sex_key
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
            association_plan.statistical_unit,
        )

    group_keys = [
        key
        for key in (sex_key, tissue_key, cell_type_key)
        if key in unit_frame.columns and unit_frame[key].dropna().nunique() >= 2
    ]
    for group_key in group_keys:
        plot_results = group_key != sex_key
        stratify_key = (
            sex_key
            if plot_results
            and sex_key in unit_frame.columns
            and unit_frame[sex_key].dropna().nunique() > 0
            else None
        )
        workflow_state.plot_count = associations._run_group_associations(
            unit_frame,
            group_key=group_key,
            schema=schema,
            plotter=AssociationPlotter(output_dir=str(layout.boxplots_dir)),
            barplot_plotter=AssociationPlotter(
                output_dir=str(layout.barplots_dir)
            ),
            statistical_unit=association_plan.statistical_unit,
            aggregation=association_plan.aggregation,
            tissue_key=tissue_key,
            stratify_key=stratify_key,
            stratify_colors=(
                {"male": "#d2e7ef", "female": "#f9bebc"}
                if stratify_key is not None
                else None
            ),
            plot_results=plot_results,
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
    mixed_model_results: TabulaMixedModelResults | None,
    layout: associations._AssociationOutputLayout,
    association_plan: _AssociationPlan,
    plot_nasp_visualizations: bool,
) -> None:
    """Finalize mixed-model tables, retained analyses, and figures."""
    schema = association_plan.obs_schema
    regression_results = workflow_state.regression_results()
    eqtl_annotations = workflow_state.eqtl_annotations()
    empty_mixed = TabulaMixedModelResults.empty()
    mixed_tables = (
        {
            "association_mixed_model_contrasts.csv": (
                mixed_model_results.contrasts
            ),
            "association_mixed_model_fixed_effects.csv": (
                mixed_model_results.fixed_effects
            ),
            "association_mixed_model_term_tests.csv": (
                mixed_model_results.term_tests
            ),
            "association_mixed_model_variance_components.csv": (
                mixed_model_results.variance_components
            ),
            "association_mixed_model_diagnostics.csv": (
                mixed_model_results.diagnostics
            ),
            "association_mixed_model_availability.csv": (
                mixed_model_results.availability
            ),
        }
        if mixed_model_results is not None
        else {
            "association_mixed_model_contrasts.csv": empty_mixed.contrasts,
            "association_mixed_model_fixed_effects.csv": (
                empty_mixed.fixed_effects
            ),
            "association_mixed_model_term_tests.csv": empty_mixed.term_tests,
            "association_mixed_model_variance_components.csv": (
                empty_mixed.variance_components
            ),
            "association_mixed_model_diagnostics.csv": (
                empty_mixed.diagnostics
            ),
            "association_mixed_model_availability.csv": (
                empty_mixed.availability
            ),
        }
    )

    for filename, table in (
        ("association_regression_results.csv", regression_results),
        (
            "association_eqtl_annotations.csv",
            eqtl_annotations,
        ),
    ):
        associations._write_association_table(
            table,
            layout.tables_dir / filename,
        )
    for filename, table in mixed_tables.items():
        associations._write_association_table(
            table,
            layout.tables_dir / filename,
        )

    if plot_nasp_visualizations:
        plot_nasp_association_visualizations(
            output_dir=layout.nasp_plots_dir,
            module_coupling=nasp_results.module_coupling,
            context_summary=nasp_results.context_summary,
            hypothesis_priorities=nasp_results.hypothesis_priorities,
            sensor_output_coupling=nasp_results.sensor_output_coupling,
            regression_results=pd.DataFrame(),
            age_stability=pd.DataFrame(),
            mechanistic_edges=nasp_results.mechanistic_edges,
            tissue_key=schema.tissue_key,
            cell_type_key=schema.cell_type_key,
        )
        workflow_state.manifest.extend(
            {
                "kind": "nasp_summary",
                "feature_id": None,
                "predictor": None,
                "statistical_unit": "donor_tissue_cell_type",
                "aggregation": association_plan.aggregation,
                "stratum": None,
                "analysis_scope": path.stem,
                "path": str(path),
            }
            for path in sorted(layout.nasp_plots_dir.glob("*.png"))
        )
        if mixed_model_results is not None:
            mixed_paths = plot_tabula_sapiens_mixed_model_inference(
                output_dir=layout.mixed_model_plots_dir,
                contrasts=mixed_model_results.contrasts,
                variance_components=(mixed_model_results.variance_components),
            )
            workflow_state.manifest.extend(
                {
                    "kind": "mixed_model_inference",
                    "feature_id": None,
                    "predictor": None,
                    "statistical_unit": _mixed_model_observational_unit(
                        mixed_model_results
                    ),
                    "aggregation": _mixed_model_aggregation(
                        mixed_model_results
                    ),
                    "stratum": None,
                    "analysis_scope": path.stem,
                    "path": str(path),
                }
                for path in mixed_paths
            )

    associations._write_association_table(
        workflow_state.plot_manifest(),
        layout.tables_dir / "association_plot_manifest.csv",
    )


def _ensure_declared_columns(
    table: pd.DataFrame,
    *,
    declared: Sequence[str],
) -> pd.DataFrame:
    """Return a stable base schema while preserving analysis-specific fields."""
    normalized = table.copy()
    for column in declared:
        if column not in normalized:
            normalized[column] = pd.Series(index=normalized.index, dtype=object)
    extras = [column for column in normalized if column not in declared]
    return normalized.reindex(columns=[*declared, *extras])


def _mixed_model_aggregation(results: TabulaMixedModelResults) -> str:
    """Return the recorded aggregation used by mixed-model outputs."""
    for table in (
        results.contrasts,
        results.variance_components,
        results.availability,
    ):
        if "aggregation" not in table:
            continue
        if (
            values := table["aggregation"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ):
            return values[0]
    return ""


def _mixed_model_observational_unit(
    results: TabulaMixedModelResults,
) -> str:
    """Return the recorded observational unit used by mixed models."""
    for table in (
        results.contrasts,
        results.variance_components,
        results.availability,
    ):
        if "observational_unit" not in table:
            continue
        if values := (
            table["observational_unit"].dropna().astype(str).unique().tolist()
        ):
            return values[0]
    return ""


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
    condition_key: str = "disease",
    study_key: str = "dataset_id",
    condition_reference: str | None = "normal",
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
    detection_threshold: float = 0.0,
    mixed_model_min_cells: int = 10,
    mixed_model_min_donors: int = 3,
    mixed_model_min_studies: int = 3,
    mixed_model_min_repeated_contexts: int = 3,
    max_plots: int | None = 200,
    plot_nasp_visualizations: bool = True,
) -> dict[str, Path]:
    """Score and analyze a complete or tissue-subset h5ad per scorer.

    Omitting `tissue_label` analyzes every cell in the input h5ad and retains
    its existing embedding. Supplying a label subsets that tissue and
    recomputes its embedding. Scoring is completed first into a shared score
    table, then the donor-aware association workflow runs once per scorer so
    Scanpy and AUCell remain separate sensitivity analyses. Setting
    `resume_from_scores=True` reuses a score table only when its provenance
    records every requested scorer.

    Args:
      h5ad_path: Complete-atlas or tissue-specific h5ad input path.
      output_dir: Root directory containing isolated run directories.
      tissue_label: Exact tissue value used to subset and recompute UMAP. Leave
        unset to analyze the complete input h5ad and retain its embedding.
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
      condition_key: Obs column identifying biological condition.
      study_key: Obs column identifying source studies in combined inputs.
      condition_reference: Reference condition for within-cell-type effects.
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
      detection_threshold: Finite per-cell floor used by expressing-fraction
        aggregations.
      mixed_model_min_cells: Minimum cells supporting each modeled aggregate.
      mixed_model_min_donors: Minimum donors supporting a modeled level.
      mixed_model_min_studies: Minimum studies supporting a study random
        intercept.
      mixed_model_min_repeated_contexts: Minimum repeated assay contexts
        supporting a context variance component.
      max_plots: Maximum association plots written per scorer.
      plot_nasp_visualizations: Whether to render mechanistic NASP summaries.

    Returns:
      Paths for the run directory, score table, and each scorer's association
      output directory.

    Raises:
      FileNotFoundError: If the input h5ad or completed score table is absent.
      ValueError: If no scorer or an unsupported scorer is requested.
    """
    _validate_detection_threshold(detection_threshold)
    input_path = Path(h5ad_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input h5ad does not exist: {input_path}")
    selected_scorers = tuple(dict.fromkeys(scorers))
    if not selected_scorers:
        raise ValueError("scorers must contain at least one scorer")
    if invalid_scorers := set(selected_scorers).difference(
        {"scanpy", "aucell"}
    ):
        invalid_text = ", ".join(sorted(invalid_scorers))
        raise ValueError(f"unsupported scorers: {invalid_text}")

    schema = ObsSchema(
        donor_key=donor_key,
        tissue_key=tissue_key,
        cell_type_key=cell_type_key,
        sex_key=sex_key,
        assay_key=assay_key,
        development_stage_key=development_stage_key,
        age_key=age_key,
        condition_key=condition_key,
        study_key=study_key,
    )
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
            tissue_key=schema.tissue_key,
            cell_type_key=schema.cell_type_key,
            sex_key=schema.sex_key,
            development_stage_key=schema.development_stage_key,
            assay_key=schema.assay_key,
            age_key=schema.age_key,
            donor_key=schema.donor_key,
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
            heatmap_groupby=schema.cell_type_key,
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
            donor_key=schema.donor_key,
            tissue_key=schema.tissue_key,
            cell_type_key=schema.cell_type_key,
            sex_key=schema.sex_key,
            assay_key=schema.assay_key,
            development_stage_key=schema.development_stage_key,
            age_key=schema.age_key,
            condition_key=schema.condition_key,
            study_key=schema.study_key,
            condition_reference=condition_reference,
            module_ids=module_ids,
            sensor_group=sensor_group,
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            detection_threshold=detection_threshold,
            scorer=scorer_name,
            mixed_model_min_cells=mixed_model_min_cells,
            mixed_model_min_donors=mixed_model_min_donors,
            mixed_model_min_studies=mixed_model_min_studies,
            mixed_model_min_repeated_contexts=(
                mixed_model_min_repeated_contexts
            ),
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
    umap_plotter = UmapPlotter(output_dir=output_dir)
    heatmap_plotter = HeatmapPlotter(output_dir=output_dir)
    summary_plotter = SummaryPlotter(output_dir=output_dir)
    plot_tabula_sapiens_metadata_umaps(
        adata,
        plotter=umap_plotter,
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

    if module_ids is None:
        selected_module_ids = GeneModules().module_ids()
    else:
        selected_module_ids = list(module_ids)

    sensors = GeneModules.sensors(sensor_group)
    module_genes_by_id = (
        _module_genes_by_id(
            adata=adata,
            module_ids=selected_module_ids,
            gene_symbol_column=gene_symbol_column,
        )
        if plot_modules
        else {}
    )
    heatmap_genes = list(
        dict.fromkeys(
            [
                *sensors,
                *(
                    gene
                    for genes in module_genes_by_id.values()
                    for gene in genes
                ),
            ]
        )
    )
    grouped_expression_by_obs = {}
    if heatmap_genes:
        for groupby_key in heatmap_groupby_keys:
            logger.info(
                "Precomputing heatmap means for %d genes by %s",
                len(heatmap_genes),
                groupby_key,
            )
            grouped_expression_by_obs[groupby_key] = (
                heatmap_plotter.summarize_gene_expression_by_obs(
                    adata,
                    heatmap_genes,
                    groupby=groupby_key,
                    gene_symbol_column=gene_symbol_column,
                    expression_layer=expression_layer,
                )
            )

    umap_plotter.plot_multi_gene_umap_panel(
        adata=adata,
        genes=sensors,
        filename="NA_SENSORS_gene_expression_umaps",
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
        ncols=6,
        shared_colorbar=True,
        size=point_size,
    )
    scoring.plot_gene_expression_heatmaps_by_obs(
        adata=adata,
        genes=sensors,
        plotter=heatmap_plotter,
        filename_prefix="NA_SENSORS",
        groupby_keys=heatmap_groupby_keys,
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
        grouped_expression_by_obs=grouped_expression_by_obs,
    )

    if plot_modules:
        _plot_module_gene_umaps(
            adata=adata,
            module_genes_by_id=module_genes_by_id,
            plotter=umap_plotter,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            ncols=6,
            point_size=point_size,
        )
        _plot_module_gene_heatmaps_by_obs(
            adata=adata,
            module_genes_by_id=module_genes_by_id,
            plotter=heatmap_plotter,
            groupby_keys=heatmap_groupby_keys,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            grouped_expression_by_obs=grouped_expression_by_obs,
        )

    scoring.module_scoring_outputs(
        adata,
        selected_module_ids,
        umap_plotter=umap_plotter,
        heatmap_plotter=heatmap_plotter,
        summary_plotter=summary_plotter,
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

    point_size = 120000 / adata.n_obs
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


def _module_genes_by_id(
    *,
    adata: ad.AnnData,
    module_ids: Sequence[str],
    gene_symbol_column: str,
) -> dict[str, list[str]]:
    """Resolve each module's available marker symbols exactly once."""
    return {
        module_id: GeneModules.genes(
            module_id,
            adata=adata,  # type: ignore[arg-type]  # upstream protocol mismatch
            gene_symbol_column=gene_symbol_column,
            output="symbols",
        )
        for module_id in module_ids
    }


def _plot_module_gene_umaps(
    adata: ad.AnnData,
    module_genes_by_id: Mapping[str, Sequence[str]],
    *,
    plotter: UmapPlotter,
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    ncols: int = 6,
    point_size: float,
) -> None:
    """Plot one multi-gene UMAP panel per Tabula Sapiens NASP module."""
    for module_id, module_genes in module_genes_by_id.items():
        if not module_genes:
            logger.info("%s: no matched genes; skipping UMAPs", module_id)
            continue

        logger.info(
            "%s: plotting %s marker genes",
            module_id,
            len(module_genes),
        )
        plotter.plot_multi_gene_umap_panel(
            adata=adata,
            genes=list(module_genes),
            filename=f"{module_id}_gene_expression_umaps",
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            ncols=ncols,
            shared_colorbar=True,
            size=point_size,
        )


def _plot_module_gene_heatmaps_by_obs(
    *,
    adata: ad.AnnData,
    module_genes_by_id: Mapping[str, Sequence[str]],
    plotter: HeatmapPlotter,
    groupby_keys: Sequence[str],
    gene_symbol_column: str,
    expression_layer: str | None,
    grouped_expression_by_obs: Mapping[str, GroupedGeneExpression],
) -> None:
    """Plot module marker-gene heatmaps for each requested obs key."""
    for module_id, module_genes in module_genes_by_id.items():
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
            plotter=plotter,
            filename_prefix=module_id,
            groupby_keys=groupby_keys,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            grouped_expression_by_obs=grouped_expression_by_obs,
        )
