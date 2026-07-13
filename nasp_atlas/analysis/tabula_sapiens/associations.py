"""Tabula Sapiens donor-aware association workflow."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.tabula_sapiens import scoring
from nasp_atlas.single_cell.associations import EqtlMergeMode
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import merge_eqtl_counts
from nasp_atlas.single_cell.associations import metadata_columns
from nasp_atlas.single_cell.associations import regress_features_on_continuous
from nasp_atlas.single_cell.associations import summarize_feature_groups
from nasp_atlas.single_cell.associations import test_feature_groups
from nasp_atlas.single_cell.associations import validate_eqtl_table
from nasp_atlas.single_cell.module_profiles import RoleAssignment
from nasp_atlas.single_cell.module_profiles import module_gene_overlap
from nasp_atlas.single_cell.module_profiles import pairwise_module_correlations
from nasp_atlas.single_cell.module_profiles import (
    relative_nasp_evidence_profiles,
)
from nasp_atlas.single_cell.visualization import SCVisualizer


logger = logging.getLogger(__name__)

_EXPECTED_NASP_EDGES: tuple[tuple[str, str, str], ...] = (
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
    ("NASP_DNA_SENSING", "INFLAMMASOME", "dna_sensing_to_inflammasome"),
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
    ("CGAMP_TRANSPORT", "IFN_I_OUTPUT", "cgamp_transport_to_ifn_output"),
    ("IFN_I_OUTPUT", "NASP_FEEDBACK", "ifn_output_to_feedback"),
    (
        "NFKB_CYTOKINE_OUTPUT",
        "NASP_FEEDBACK",
        "nfkb_output_to_feedback",
    ),
    ("IFN_I_OUTPUT", "INFLAMMAGING", "ifn_output_to_inflammaging"),
    ("NFKB_CYTOKINE_OUTPUT", "SASP", "nfkb_output_to_sasp"),
)


def _plot_allowed(plot_count: int, max_plots: int | None) -> bool:
    """Return whether another plot may be emitted."""
    return max_plots is None or plot_count < max_plots


def _detect_scorer_from_scores(scores: pd.DataFrame) -> str:
    """Infer which scorer produced the score columns in a score table.

    Args:
      scores: Score table whose column names follow the module_score_name
        convention (`*_score` for scanpy, `*_auc` for AUCell).

    Returns:
      "scanpy" when `*_score` columns dominate, otherwise "aucell".
    """
    columns = [str(column) for column in scores.columns]
    n_scanpy = sum(bool(column.endswith("_score")) for column in columns)
    n_aucell = sum(
        column.endswith("_auc")
        and not column.endswith(("_pos_auc", "_inv_auc"))
        for column in columns
    )
    return "aucell" if n_aucell > n_scanpy else "scanpy"


def _backfill_obs_metadata(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    metadata_keys: Sequence[str],
) -> None:
    """Copy metadata columns present only in the score table into obs.

    Metadata such as `age_years` is derived during scoring and may be absent
    from the raw AnnData. Any requested key missing from `adata.obs` but
    present in the aligned score table is backfilled so association can read a
    consistent metadata schema from obs. AnnData is modified in place but no
    scoring or expression values are altered.

    Args:
      adata: AnnData whose obs receives the backfilled columns.
      scores: Score table indexed by obs name.
      metadata_keys: Metadata column names to backfill when possible.
    """
    obs_index = pd.Index(adata.obs_names).astype(str)
    aligned = scores.reindex(obs_index)
    for key in metadata_keys:
        if key in adata.obs.columns:
            continue
        if key in aligned.columns:
            adata.obs[key] = aligned[key].to_numpy()


def _validate_score_alignment(
    adata: ad.AnnData,
    scores: pd.DataFrame,
) -> None:
    """Validate that a score table aligns to AnnData by obs name.

    Args:
      adata: AnnData providing the reference obs names.
      scores: Score table expected to be indexed by obs name.

    Raises:
      ValueError: If the score index is non-unique or shares no obs names with
        the AnnData, i.e. the tables cannot be joined by name.
    """
    if not scores.index.is_unique:
        n_dup = int(scores.index.duplicated().sum())
        raise ValueError(
            f"score table obs_name index is not unique ({n_dup} duplicates); "
            "cannot join scores to AnnData by name"
        )
    obs_index = pd.Index(adata.obs_names).astype(str)
    score_index = pd.Index(scores.index).astype(str)
    overlap = obs_index.intersection(score_index)
    if len(overlap) == 0:
        raise ValueError(
            "score table shares no obs_name values with the AnnData; "
            "the score CSV does not correspond to this h5ad"
        )
    if extra_scores := score_index.difference(obs_index).tolist():
        preview = ", ".join(extra_scores[:3])
        raise ValueError(
            f"score table contains {len(extra_scores)} obs_name values absent "
            f"from the AnnData (for example: {preview})"
        )


def _align_anndata_to_scores(
    adata: ad.AnnData,
    scores: pd.DataFrame,
) -> ad.AnnData:
    """Restrict AnnData to scored cells so every feature uses one population.

    A score table may intentionally represent a reproducible cell subset of a
    larger AnnData. Keeping unscored AnnData rows would use different cells for
    module scores and gene-expression features, and would overstate module
    sample sizes. Score rows absent from AnnData remain an error; AnnData rows
    absent from scores are explicitly dropped here.

    Args:
      adata: AnnData providing the reference observations.
      scores: Validated score table indexed by observation name.

    Returns:
      AnnData containing exactly the score-table observations, in score order.
    """
    _validate_score_alignment(adata, scores)
    obs_index = pd.Index(adata.obs_names).astype(str)
    score_index = pd.Index(scores.index).astype(str)
    if obs_index.equals(score_index):
        return adata

    positions = obs_index.get_indexer(score_index)
    if bool((positions < 0).any()):
        raise ValueError("score table could not be aligned to AnnData obs_name")
    logger.info(
        "[tabula_sapiens] restricting AnnData from %d to %d scored cells",
        adata.n_obs,
        len(score_index),
    )
    return adata[positions, :].copy()


def _required_score_metadata_present(
    scores: pd.DataFrame,
    adata: ad.AnnData,
    metadata_keys: Sequence[str],
) -> list[str]:
    """Return requested metadata keys missing from both score table and obs.

    Args:
      scores: Score table indexed by obs name.
      adata: AnnData providing obs metadata.
      metadata_keys: Metadata column names required downstream.

    Returns:
      Sorted list of keys absent from both sources.
    """
    available = set(scores.columns) | set(adata.obs.columns)
    return sorted(key for key in metadata_keys if key not in available)


def _association_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return the association output directory layout.

    Args:
      output_dir: Root association output directory.

    Returns:
      Mapping of layout keys to created directories: `tables`, `regressions`,
      `boxplots`, `barplots`, and `nasp`.
    """
    tables = output_dir / "association_tables"
    plots = output_dir / "association_plots"
    layout = {
        "tables": tables,
        "regressions": plots / "regressions",
        "boxplots": plots / "boxplots",
        "barplots": plots / "barplots",
        "nasp": plots / "nasp",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _nasp_profile_tables(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    unit_columns: Sequence[str],
    strata_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build relative NASP profiles, context summaries, and module coupling.

    Args:
      unit_frame: Donor-aware long feature frame containing module scores.
      schema: Metadata column-name schema.
      unit_columns: Columns jointly identifying each independent unit.
      strata_columns: Biological context columns used to center correlations.

    Returns:
      Relative unit profiles, tissue/cell-type summaries, and pairwise module
      correlations annotated with signed marker-gene overlap.
    """
    wide = _wide_module_scores(unit_frame, schema=schema)
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    role_assignments: dict[str, RoleAssignment] = {
        "NASP_DNA_SENSING": "competence",
        "NASP_RNA_SENSING": "competence",
        "IFN_I_OUTPUT": "output",
        "NFKB_CYTOKINE_OUTPUT": "output",
        "ISR": "output",
        "INFLAMMASOME": "output",
        "NASP_RESTRICTION": "restriction",
        "NASP_FEEDBACK": "feedback",
        "SASP": "post",
        "INFLAMMAGING": "post",
        "IMMUNOSENESCENCE": "post",
        "SENESCENCE": "post",
        "AGING_HALLMARKS": "post",
    }
    available_roles: dict[str, RoleAssignment] = {
        module_id: role
        for module_id, role in role_assignments.items()
        if module_id in wide.columns
    }
    profiles = wide.copy()
    if available_roles:
        evidence = relative_nasp_evidence_profiles(
            wide,
            module_roles=available_roles,
        )
        profiles = pd.concat([wide, evidence], axis="columns")
    profiles.index.name = "unit_id"
    profiles = profiles.reset_index()

    context_columns = [
        key
        for key in (schema.tissue_key, schema.cell_type_key)
        if key in profiles.columns
    ]
    context_summary = _summarize_nasp_profiles(
        profiles,
        context_columns=context_columns,
        donor_key=schema.donor_key,
    )

    raw_correlations = pairwise_module_correlations(
        unit_frame,
        unit_columns=unit_columns,
    ).assign(analysis="across_contexts")
    usable_strata = [
        key
        for key in strata_columns
        if key in unit_frame.columns and unit_frame[key].notna().any()
    ]
    correlation_tables = [raw_correlations]
    if usable_strata:
        centered = pairwise_module_correlations(
            unit_frame,
            unit_columns=unit_columns,
            strata_columns=usable_strata,
            center_within_strata=True,
        ).assign(analysis="within_context_centered")
        correlation_tables.append(centered)
    correlations = _concat_or_empty(correlation_tables)

    module_ids = sorted(
        unit_frame.loc[
            unit_frame["feature_type"] == "module_score",
            "feature_label",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    modules = [GeneModules.modules(module_id) for module_id in module_ids]
    overlap = module_gene_overlap(modules)
    if not correlations.empty and not overlap.empty:
        correlations = correlations.merge(
            overlap,
            on=["module_a", "module_b"],
            how="left",
        )
    return profiles, context_summary, correlations


def _wide_module_scores(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
) -> pd.DataFrame:
    """Pivot donor-aware module scores while retaining unit metadata."""
    modules = unit_frame[unit_frame["feature_type"] == "module_score"].copy()
    if modules.empty:
        return pd.DataFrame()
    if "unit_id" not in modules.columns:
        raise KeyError("unit frame missing required column: unit_id")
    scores = modules.pivot(
        index="unit_id",
        columns="feature_label",
        values="feature_value",
    )
    metadata_keys = [
        key for key in metadata_columns(schema) if key in modules.columns
    ]
    metadata = modules.groupby("unit_id", observed=True, dropna=False)[
        metadata_keys
    ].first()
    counts = modules.groupby("unit_id", observed=True, dropna=False).agg(
        n_cells=("n_cells", "min"),
        n_cells_total=("n_cells_total", "max"),
    )
    return metadata.join(counts).join(scores)


def _sensor_output_coupling(
    unit_frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str],
    strata_columns: Sequence[str],
) -> pd.DataFrame:
    """Correlate individual sensor genes with canonical output modules."""
    output_modules = {
        "IFN_I_OUTPUT",
        "NFKB_CYTOKINE_OUTPUT",
        "ISR",
        "INFLAMMASOME",
    }
    module_rows = unit_frame[
        (unit_frame["feature_type"] == "module_score")
        & unit_frame["feature_label"].isin(output_modules)
    ].copy()
    gene_rows = unit_frame[
        unit_frame["feature_type"] == "gene_expression"
    ].copy()
    if module_rows.empty or gene_rows.empty:
        return pd.DataFrame()

    module_rows["relationship_id"] = "module::" + module_rows[
        "feature_label"
    ].astype(str)
    gene_rows["relationship_id"] = "gene::" + gene_rows["feature_label"].astype(
        str
    )
    relationship_frame = pd.concat(
        [module_rows, gene_rows],
        axis="index",
        ignore_index=True,
    )
    module_ids = sorted(module_rows["relationship_id"].unique().tolist())
    gene_ids = sorted(gene_rows["relationship_id"].unique().tolist())
    tested_pairs = [
        (gene_id, module_id) for gene_id in gene_ids for module_id in module_ids
    ]
    raw = pairwise_module_correlations(
        relationship_frame,
        unit_columns=unit_columns,
        module_column="relationship_id",
        feature_type_column=None,
        module_pairs=tested_pairs,
    ).assign(analysis="across_contexts")
    usable_strata = [
        key
        for key in strata_columns
        if key in relationship_frame.columns
        and relationship_frame[key].notna().any()
    ]
    tables = [raw]
    if usable_strata:
        centered = pairwise_module_correlations(
            relationship_frame,
            unit_columns=unit_columns,
            strata_columns=usable_strata,
            center_within_strata=True,
            module_column="relationship_id",
            feature_type_column=None,
            module_pairs=tested_pairs,
        ).assign(analysis="within_context_centered")
        tables.append(centered)
    result = _concat_or_empty(tables)
    result["gene"] = result["module_a"].str.removeprefix("gene::")
    result["output_module"] = result["module_b"].str.removeprefix("module::")
    output_genes = {
        module_id: set(GeneModules.genes(module_id))
        for module_id in output_modules
    }
    result["gene_in_output_module"] = [
        gene in output_genes[module_id]
        for gene, module_id in zip(
            result["gene"],
            result["output_module"],
            strict=True,
        )
    ]
    return result.drop(columns=["module_a", "module_b"])


def _summarize_nasp_profiles(
    profiles: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    donor_key: str,
) -> pd.DataFrame:
    """Summarize relative NASP axes and mismatch prevalence by context."""
    if not context_columns or "relative_competence" not in profiles.columns:
        return pd.DataFrame()
    value_columns = [
        column
        for column in profiles.columns
        if column.startswith("relative_") or column.endswith("_gap")
    ]
    grouped = profiles.groupby(
        list(context_columns),
        observed=True,
        dropna=False,
    )
    summary = grouped[value_columns].mean(numeric_only=True).reset_index()
    summary["n_units"] = grouped.size().to_numpy()
    if donor_key in profiles.columns:
        summary["n_donors"] = grouped[donor_key].nunique().to_numpy()
    return summary


def _association_plot_filename(
    *,
    kind: str,
    feature_id: str,
    key: str,
    statistical_unit: str,
    aggregation: str,
    stratum: str | None = None,
) -> str:
    """Build a safe snake_case filename for an association plot.

    Args:
      kind: Plot kind token, e.g. "regression" or "boxplot".
      feature_id: Feature identifier (score column or var name).
      key: Predictor or grouping obs key.
      statistical_unit: Statistical unit of the plotted frame.
      aggregation: Aggregation used to build the unit frame.
      stratum: Optional stratum label appended to the filename.

    Returns:
      A filename ending in `.png` with safe filename tokens.
    """
    parts = [
        kind,
        scoring.safe_filename_token(feature_id),
        scoring.safe_filename_token(key),
        scoring.safe_filename_token(statistical_unit),
        scoring.safe_filename_token(aggregation),
    ]
    if stratum is not None:
        parts.append(scoring.safe_filename_token(stratum))
    return "_".join(parts) + ".png"


def _scored_module_ids_from_scores(
    scores: pd.DataFrame,
    *,
    scorer: str,
) -> list[str]:
    """Infer scored module ids from score-table column names.

    Args:
      scores: Score table whose columns follow the module_score_name
        convention.
      scorer: Scorer naming convention ("scanpy" or "aucell").

    Returns:
      Module ids parsed from the signed score columns for this scorer.
    """
    suffix = "_score" if scorer == "scanpy" else "_auc"
    ignore = ("_pos", "_inv", "_pos_auc", "_inv_auc")
    module_ids: list[str] = []
    for column in scores.columns:
        name = str(column)
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        if any(stem.endswith(token) for token in ignore):
            continue
        module_ids.append(stem)
    return module_ids


def _concat_or_empty(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate result tables, returning an empty frame when none exist.

    Args:
      tables: Result frames to concatenate.

    Returns:
      A single concatenated frame, or an empty frame if the input is empty.
    """
    if non_empty := [table for table in tables if not table.empty]:
        return pd.concat(non_empty, axis="index", ignore_index=True)
    else:
        return pd.DataFrame()


def _write_association_table(table: pd.DataFrame, path: Path) -> None:
    """Write an association result table to CSV.

    Args:
      table: Result table to persist; empty frames are still written so the
        expected output file always exists.
      path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("[tabula_sapiens] association table -> %s", path)


def _run_continuous_associations(
    unit_frame: pd.DataFrame,
    *,
    predictor_key: str,
    visualizer: SCVisualizer,
    layout: Mapping[str, Path],
    statistical_unit: str,
    aggregation: str,
    manifest: list[dict[str, object]],
    regression_tables: list[pd.DataFrame],
    plot_count: int,
    max_plots: int | None,
    stratify_key: str | None,
    analysis_scope: str = "requested_unit",
) -> int:
    """Run continuous regressions and emit per-feature regression plots.

    Args:
      unit_frame: Aggregated unit-level feature frame.
      predictor_key: Continuous predictor column (e.g. age).
      visualizer: Visualizer writing into the regressions directory.
      layout: Association output directory layout.
      statistical_unit: Statistical unit label recorded in filenames.
      aggregation: Aggregation label recorded in filenames.
      manifest: Mutable plot-manifest accumulator.
      regression_tables: Mutable list of regression result frames.
      plot_count: Number of plots already emitted.
      max_plots: Optional cap on plots emitted across association outputs.
      stratify_key: Optional stratifying column for per-stratum regressions.
      analysis_scope: Label identifying the biological scope of the result.

    Returns:
      Updated number of emitted plots.
    """
    result = regress_features_on_continuous(
        unit_frame,
        predictor_key=predictor_key,
        stratify_key=stratify_key,
    )
    result["analysis_scope"] = analysis_scope
    regression_tables.append(result)
    for _, row in result.iterrows():
        if bool(row.get("skipped", False)):
            continue
        stratum = row.get("stratum")
        stratum_label = None if pd.isna(stratum) else str(stratum)
        subframe = unit_frame[unit_frame["feature_id"] == row["feature_id"]]
        if stratify_key is not None and stratum_label is not None:
            subframe = subframe[
                subframe[stratify_key].astype(str) == stratum_label
            ]
        if not _plot_allowed(plot_count, max_plots):
            continue
        filename = _association_plot_filename(
            kind="regression",
            feature_id=str(row["feature_id"]),
            key=predictor_key,
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            stratum=stratum_label,
        )
        visualizer.plot_feature_regression(
            subframe,
            feature_id=str(row["feature_id"]),
            predictor_key=predictor_key,
            filename=filename,
            result_row=row,
            feature_label=str(row.get("feature_label", row["feature_id"])),
        )
        plot_count += 1
        manifest.append(
            {
                "kind": "regression",
                "feature_id": row["feature_id"],
                "predictor": predictor_key,
                "statistical_unit": statistical_unit,
                "aggregation": aggregation,
                "stratum": stratum_label,
                "analysis_scope": analysis_scope,
                "path": str(layout["regressions"] / filename),
            }
        )
    return plot_count


def _run_group_associations(
    unit_frame: pd.DataFrame,
    *,
    group_key: str,
    schema: ObsSchema,
    visualizer: SCVisualizer,
    barplot_visualizer: SCVisualizer,
    statistical_unit: str,
    aggregation: str,
    tissue_key: str,
    manifest: list[dict[str, object]],
    group_test_tables: list[pd.DataFrame],
    group_summary_tables: list[pd.DataFrame],
    plot_count: int,
    max_plots: int | None,
) -> int:
    """Run categorical group tests and emit boxplots and tissue barplots.

    Args:
      unit_frame: Aggregated unit-level feature frame.
      group_key: Categorical grouping column (sex/tissue/cell type/donor).
      schema: Column-name schema.
      visualizer: Visualizer writing boxplots.
      barplot_visualizer: Visualizer writing tissue-activity barplots.
      statistical_unit: Statistical unit label recorded in filenames.
      aggregation: Aggregation label recorded in filenames.
      tissue_key: Tissue column, used to select barplot outputs.
      manifest: Mutable plot-manifest accumulator.
      group_test_tables: Mutable list of group-test result frames.
      group_summary_tables: Mutable list of group-summary frames.
      plot_count: Number of plots already emitted.
      max_plots: Optional cap on plots emitted across association outputs.

    Returns:
      Updated number of emitted plots.
    """
    summary = summarize_feature_groups(
        unit_frame, group_key=group_key, schema=schema
    )
    group_summary_tables.append(summary)
    result = test_feature_groups(unit_frame, group_key=group_key)
    group_test_tables.append(result)

    is_tissue = group_key == tissue_key
    for feature_id in unit_frame["feature_id"].drop_duplicates():
        subframe = unit_frame[unit_frame["feature_id"] == feature_id]
        row = _first_omnibus_row(result, feature_id)
        if not _plot_allowed(plot_count, max_plots):
            continue
        target = barplot_visualizer if is_tissue else visualizer
        kind = "barplot" if is_tissue else "boxplot"
        filename = _association_plot_filename(
            kind=kind,
            feature_id=str(feature_id),
            key=group_key,
            statistical_unit=statistical_unit,
            aggregation=aggregation,
        )
        target.plot_feature_group_boxplot(
            subframe,
            feature_id=str(feature_id),
            group_key=group_key,
            filename=filename,
            result_row=row,
        )
        plot_count += 1
        manifest.append(
            {
                "kind": kind,
                "feature_id": feature_id,
                "predictor": group_key,
                "statistical_unit": statistical_unit,
                "aggregation": aggregation,
                "stratum": None,
                "path": str(target.output_dir / filename),
            }
        )
    return plot_count


def _first_omnibus_row(
    result: pd.DataFrame,
    feature_id: str,
) -> pd.Series | None:
    """Return the omnibus test row for a feature, if present.

    Args:
      result: Group-test result frame.
      feature_id: Feature whose omnibus row is requested.

    Returns:
      The omnibus row (where comparison equals "omnibus" when available, else
      the first matching row), or None when the feature has no rows.
    """
    subset = result[result["feature_id"] == feature_id]
    if subset.empty:
        return None
    if "comparison" in subset.columns:
        omnibus = subset[subset["comparison"] == "omnibus"]
        if not omnibus.empty:
            return omnibus.iloc[0]
    return subset.iloc[0]


def _run_eqtl_associations(
    unit_frame: pd.DataFrame,
    *,
    eqtl_table_path: str | Path,
    eqtl_merge_mode: str,
    schema: ObsSchema,
    regression_tables: list[pd.DataFrame],
    annotation_tables: list[pd.DataFrame],
) -> None:
    """Merge eQTL counts and test burden only when it varies by unit.

    Args:
      unit_frame: Aggregated unit-level feature frame.
      eqtl_table_path: Path to the eQTL count table.
      eqtl_merge_mode: How eQTL counts join the units.
      schema: Column-name schema.
      regression_tables: Mutable list of regression result frames.
      annotation_tables: Mutable list receiving the merged annotation frame.
    """
    eqtl_table = pd.read_csv(eqtl_table_path)
    value_columns = validate_eqtl_table(
        eqtl_table, merge_mode=cast("EqtlMergeMode", eqtl_merge_mode)
    )
    merged = merge_eqtl_counts(
        unit_frame,
        eqtl_table,
        merge_mode=cast("EqtlMergeMode", eqtl_merge_mode),
        schema=schema,
    )
    annotation_tables.append(merged)
    if eqtl_merge_mode in {"gene", "module"}:
        logger.info(
            "[tabula_sapiens] eQTL %s burden is feature-level annotation; "
            "skipping invalid within-feature regressions",
            eqtl_merge_mode,
        )
        return
    for predictor_key in value_columns:
        result = regress_features_on_continuous(
            merged, predictor_key=predictor_key
        )
        regression_tables.append(result)


def _run_cell_level_descriptive_plots(
    cell_frame: pd.DataFrame,
    *,
    predictor_key: str,
    visualizer: SCVisualizer,
    aggregation: str,
    manifest: list[dict[str, object]],
    plot_count: int,
    max_plots: int | None,
) -> int:
    """Emit cell-level descriptive regression plots (never inferential).

    Args:
      cell_frame: Long cell-level feature frame.
      predictor_key: Continuous predictor column (e.g. age).
      visualizer: Visualizer writing into the regressions directory.
      aggregation: Aggregation label recorded in filenames.
      manifest: Mutable plot-manifest accumulator.
      plot_count: Number of plots already emitted.
      max_plots: Optional cap on plots emitted across association outputs.

    Returns:
      Updated number of emitted plots.
    """
    for feature_id in cell_frame["feature_id"].drop_duplicates():
        subframe = cell_frame[cell_frame["feature_id"] == feature_id].assign(
            statistical_unit="cell"
        )
        if not _plot_allowed(plot_count, max_plots):
            continue
        filename = _association_plot_filename(
            kind="regression_celldescriptive",
            feature_id=str(feature_id),
            key=predictor_key,
            statistical_unit="cell",
            aggregation=aggregation,
        )
        visualizer.plot_feature_regression(
            subframe,
            feature_id=str(feature_id),
            predictor_key=predictor_key,
            filename=filename,
            feature_label=str(feature_id),
        )
        plot_count += 1
        manifest.append(
            {
                "kind": "regression_cell_descriptive",
                "feature_id": feature_id,
                "predictor": predictor_key,
                "statistical_unit": "cell",
                "aggregation": aggregation,
                "stratum": None,
                "path": str(visualizer.output_dir / filename),
            }
        )
    return plot_count
