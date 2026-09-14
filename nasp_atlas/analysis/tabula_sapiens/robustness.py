"""Publish donor and gene sensitivity analyses alongside atlas associations."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]
from nasp_compendium.types import GeneModule  # type: ignore[import]

from nasp_atlas.analysis.tabula_sapiens.mixed_model_scopes import _atomic_table
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.donor_sensitivity import donor_sensitivity
from nasp_atlas.single_cell.gene_diagnostics import GeneDiagnosticResults
from nasp_atlas.single_cell.gene_diagnostics import module_gene_diagnostics
from nasp_atlas.single_cell.gene_sensitivity import gene_removal_sensitivity
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.visualization.robustness import RobustnessPlotter


logger = logging.getLogger(__name__)

__all__ = ["robustness_analysis"]


def robustness_analysis(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    cell_frame: pd.DataFrame,
    *,
    output_dir: Path,
    provenance: pd.DataFrame,
    module_ids: Sequence[str],
    module_pairs: Sequence[tuple[str, str]],
    schema: ObsSchema,
    scorer: ScorerName,
    run_donor_sensitivity: bool = False,
    run_gene_diagnostics: bool = False,
    run_gene_removal_sensitivity: bool = False,
    plot_visualizations: bool = True,
    dominant_genes: int = 1,
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    use_raw: bool = False,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    detection_threshold: float = 0.0,
) -> GeneDiagnosticResults | None:  # sourcery skip: low-code-quality
    """Write requested diagnostics with explicit stage and input provenance.

    Gene removal also writes its required gene diagnostics. The manifest is
    the authority for current outputs; disabled or failed stages do not make
    previous files current. Optional figures use the shared publication style.
    No external datasets or module definitions change.

    Returns:
      Gene diagnostics when computed, for reuse by mechanism companions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, object]] = [
        {"stage": "donor_sensitivity", "enabled": run_donor_sensitivity},
        {
            "stage": "gene_diagnostics",
            "enabled": run_gene_diagnostics or run_gene_removal_sensitivity,
        },
        {
            "stage": "gene_removal_sensitivity",
            "enabled": run_gene_removal_sensitivity,
        },
    ]
    for stage in stages:
        stage["status"] = "pending" if stage["enabled"] else "disabled"
    manifest = output_dir / "robustness_manifest.csv"
    _atomic_table(pd.DataFrame(stages), manifest)

    if not any(stage["enabled"] for stage in stages):
        return
    if not module_ids:
        for stage in stages:
            if stage["enabled"]:
                stage["status"] = "unavailable_no_module_scores"
        _atomic_table(pd.DataFrame(stages), manifest)
        return

    _write_robustness_provenance(
        provenance,
        output_dir=output_dir,
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
        dominant_genes=dominant_genes,
        plot_visualizations=plot_visualizations,
    )

    modules = [GeneModules.modules(module_id) for module_id in module_ids]
    diagnostics = None

    for stage in stages:
        if not stage["enabled"]:
            continue

        stage["status"] = "running"
        _atomic_table(pd.DataFrame(stages), manifest)
        logger.info("Computing robustness stage: %s", stage["stage"])

        completed = False
        try:
            if stage["stage"] == "donor_sensitivity":
                donors = donor_sensitivity(
                    cell_frame,
                    schema=schema,
                    minimum_cells=minimum_cells,
                    minimum_donors=minimum_donors,
                )
                tables = {
                    "nasp_donor_scores": donors.donor_scores,
                    "nasp_context_ranks": donors.context_ranks,
                    "nasp_leave_one_donor_out": donors.leave_one_donor_out,
                    "nasp_matched_tissue_differences": donors.paired_tissues,
                }
            elif stage["stage"] == "gene_diagnostics":
                _validate_marker_provenance(scores)
                diagnostics = module_gene_diagnostics(
                    adata,
                    scores,
                    modules,
                    schema=schema,
                    scorer=scorer,
                    gene_symbol_column=gene_symbol_column,
                    expression_layer=expression_layer,
                    use_raw=use_raw,
                    detection_threshold=detection_threshold,
                    minimum_cells=minimum_cells,
                    minimum_donors=minimum_donors,
                )
                tables = {
                    "nasp_gene_donor_expression": diagnostics.donor_expression,
                    "nasp_gene_context_diagnostics": (
                        diagnostics.context_summary
                    ),
                }
            elif diagnostics is None:
                raise RuntimeError(
                    "Gene removal requires completed gene diagnostics"
                )
            else:
                tables = _gene_removal_tables(
                    adata,
                    scores,
                    modules,
                    diagnostics,
                    module_pairs=module_pairs,
                    schema=schema,
                    scorer=scorer,
                    gene_symbol_column=gene_symbol_column,
                    expression_layer=expression_layer,
                    use_raw=use_raw,
                    dominant_genes=dominant_genes,
                    minimum_cells=minimum_cells,
                    minimum_donors=minimum_donors,
                )

            _publish_robustness_tables(
                tables,
                stage,
                output_dir=output_dir,
                schema=schema,
                plot_visualizations=plot_visualizations,
                minimum_donors=minimum_donors,
            )
            stage["status"] = "completed"
            completed = True
        finally:
            if not completed:
                stage["status"] = "failed"
            _atomic_table(pd.DataFrame(stages), manifest)

    return diagnostics


def _gene_removal_tables(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    modules: Sequence[GeneModule],
    diagnostics: GeneDiagnosticResults,
    *,
    module_pairs: Sequence[tuple[str, str]],
    schema: ObsSchema,
    scorer: ScorerName,
    gene_symbol_column: str,
    expression_layer: str | None,
    use_raw: bool,
    dominant_genes: int,
    minimum_cells: int,
    minimum_donors: int,
) -> dict[str, pd.DataFrame]:
    """Recover original scoring settings and calculate gene-removal tables."""
    random_state = int(_score_setting(scores, "scoring_random_state"))
    source = "raw" if use_raw else expression_layer or "X"
    if _score_setting(scores, "scoring_expression_source") != source:
        raise ValueError("Gene removal expression source differs from scoring")

    chunk_size, workers = 1000, 1
    if scorer == "aucell":
        chunk_size = int(_score_setting(scores, "scoring_aucell_chunk_size"))
        workers = int(_score_setting(scores, "scoring_aucell_num_workers"))

    sensitivity = gene_removal_sensitivity(
        adata,
        scores,
        modules,
        diagnostics,
        module_pairs=module_pairs,
        schema=schema,
        scorer=scorer,
        gene_symbol_column=gene_symbol_column,
        expression_layer=expression_layer,
        use_raw=use_raw,
        dominant_genes=dominant_genes,
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
        random_state=random_state,
        aucell_chunk_size=chunk_size,
        aucell_num_workers=workers,
    )

    return {
        "nasp_gene_removal_variants": sensitivity.variants,
        "nasp_gene_removal_donor_scores": sensitivity.donor_scores,
        "nasp_gene_removal_context_changes": (sensitivity.context_changes),
        "nasp_overlap_removed_coupling": (sensitivity.overlap_coupling),
    }


def _publish_robustness_tables(
    tables: dict[str, pd.DataFrame],
    stage: dict[str, object],
    *,
    output_dir: Path,
    schema: ObsSchema,
    plot_visualizations: bool,
    minimum_donors: int,
) -> None:
    """Publish a stage's tables and figures and register produced artifacts."""
    for name, table in tables.items():
        _atomic_table(table, output_dir / f"{name}.csv")

    artifacts = [output_dir / f"{name}.csv" for name in tables]
    if plot_visualizations:
        figures = RobustnessPlotter(output_dir / "figures").plot_tables(
            tables, schema=schema, minimum_donors=minimum_donors
        )
        artifacts.extend(figures)
        stage["visualizations"] = (
            "completed" if figures else "unavailable_no_estimable_rows"
        )
    else:
        stage["visualizations"] = "disabled"

    stage["artifacts"] = ";".join(str(path) for path in artifacts)


def _write_robustness_provenance(
    provenance: pd.DataFrame,
    *,
    output_dir: Path,
    minimum_cells: int,
    minimum_donors: int,
    dominant_genes: int,
    plot_visualizations: bool,
) -> None:
    """Record aggregation, support and selection rules for all stages."""
    _atomic_table(
        provenance.assign(
            robustness_aggregation="mean",
            robustness_minimum_cells=minimum_cells,
            robustness_minimum_donors=minimum_donors,
            dominant_genes=dominant_genes,
            donor_rank_statistic="median_of_donor_means",
            driver_selection="maximum_context_median_absolute_expression_per_arm_gene",
            sensitivity_interpretation="descriptive_no_pvalues",
            robustness_plot_visualizations=plot_visualizations,
        ),
        output_dir / "robustness_provenance.csv",
    )


def _score_setting(scores: pd.DataFrame, key: str) -> str:
    """Require one unambiguous score-generation setting for a rerun."""
    if (
        key not in scores
        or scores[key].isna().any()
        or scores[key].nunique() != 1
    ):
        raise ValueError(
            f"Missing or inconsistent {key}; regenerate the score table"
        )

    return str(scores[key].iloc[0])


def _validate_marker_provenance(scores: pd.DataFrame) -> None:
    """Prevent interpreting old scores using a different curated gene panel."""
    expected = hashlib.sha256(
        GeneModules.default_panel_path().read_bytes()
    ).hexdigest()
    if _score_setting(scores, "scoring_marker_panel_sha256") != expected:
        raise ValueError(
            "Marker panel differs from saved scores; regenerate scores"
        )
