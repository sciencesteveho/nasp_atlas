"""Publish mechanistic diagnostics using the shared scored cell population."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.tabula_sapiens.mixed_model_scopes import _atomic_table
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.gene_diagnostics import GeneDiagnosticResults
from nasp_atlas.single_cell.gene_diagnostics import module_gene_diagnostics
from nasp_atlas.single_cell.mechanisms import donor_score_units
from nasp_atlas.single_cell.mechanisms import mechanism_tables
from nasp_atlas.single_cell.mechanisms import reference_comparisons
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.reference_sets import reference_bundle_hash
from nasp_atlas.single_cell.visualization.mechanisms import MechanismPlotter


def mechanism_analysis(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    *,
    output_dir: Path,
    provenance: pd.DataFrame,
    module_ids: Sequence[str],
    schema: ObsSchema,
    scorer: ScorerName,
    diagnostics: GeneDiagnosticResults | None = None,
    run_mechanism_diagnostics: bool = False,
    include_reference_sets: bool = False,
    plot_visualizations: bool = True,
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    use_raw: bool = False,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    detection_threshold: float = 0.0,
) -> None:
    """Write requested mechanism/reference tables and optional figures.

    Gene definitions and scores must match. Disabled/failed manifests never
    advertise outputs from previous runs as current. No network access occurs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "mechanism_manifest.csv"
    record: dict[str, object] = {
        "mechanisms_enabled": run_mechanism_diagnostics,
        "references_enabled": include_reference_sets,
        "status": "running",
    }
    if not (run_mechanism_diagnostics or include_reference_sets):
        record["status"] = "disabled"
        _atomic_table(pd.DataFrame([record]), manifest)
        return
    _atomic_table(pd.DataFrame([record]), manifest)

    try:
        _validate_mechanism_provenance(
            scores,
            expression_layer=expression_layer,
            use_raw=use_raw,
            run_mechanism_diagnostics=run_mechanism_diagnostics,
            include_reference_sets=include_reference_sets,
        )
        units = _mechanism_score_units(
            adata,
            scores,
            schema=schema,
            scorer=scorer,
            minimum_cells=minimum_cells,
        )

        tables = {}
        if run_mechanism_diagnostics:
            if diagnostics is None:
                diagnostics = _mechanism_gene_diagnostics(
                    adata,
                    scores,
                    module_ids,
                    schema=schema,
                    scorer=scorer,
                    gene_symbol_column=gene_symbol_column,
                    expression_layer=expression_layer,
                    use_raw=use_raw,
                    minimum_cells=minimum_cells,
                    minimum_donors=minimum_donors,
                    detection_threshold=detection_threshold,
                )

            if diagnostics is not None:
                tables |= mechanism_tables(
                    diagnostics,
                    units,
                    schema=schema,
                    scorer=scorer,
                    minimum_cells=minimum_cells,
                    minimum_donors=minimum_donors,
                )

        if include_reference_sets:
            tables["reference_comparisons"] = reference_comparisons(
                units,
                module_ids,
                schema=schema,
                scorer=scorer,
                minimum_donors=minimum_donors,
            )

        artifacts = _publish_mechanism_outputs(
            tables,
            output_dir=output_dir,
            provenance=provenance,
            schema=schema,
            plot_visualizations=plot_visualizations,
            minimum_cells=minimum_cells,
            minimum_donors=minimum_donors,
        )
        record.update(
            status="completed" if tables else "unavailable_no_modules",
            artifacts=";".join(str(path) for path in artifacts),
        )
    finally:
        if record["status"] == "running":
            record["status"] = "failed"
        _atomic_table(pd.DataFrame([record]), manifest)


def _validate_mechanism_provenance(
    scores: pd.DataFrame,
    *,
    expression_layer: str | None,
    use_raw: bool,
    run_mechanism_diagnostics: bool,
    include_reference_sets: bool,
) -> None:
    """Require matching curated definitions, expression and reference bundle."""
    panel_hash = hashlib.sha256(
        GeneModules.default_panel_path().read_bytes()
    ).hexdigest()
    if (
        "scoring_marker_panel_sha256" not in scores
        or not scores.scoring_marker_panel_sha256.eq(panel_hash).all()
    ):
        raise ValueError(
            "Mechanism diagnostics require scores from the current marker panel"
        )

    source = "raw" if use_raw else expression_layer or "X"
    if run_mechanism_diagnostics and (
        "scoring_expression_source" not in scores
        or not scores.scoring_expression_source.eq(source).all()
    ):
        raise ValueError("Mechanism expression source differs from scoring")

    if include_reference_sets and (
        "scoring_reference_bundle_sha256" not in scores
        or not scores.scoring_reference_bundle_sha256.eq(
            reference_bundle_hash()
        ).all()
    ):
        raise ValueError(
            "Reference gene sets differ from scoring; regenerate scores"
        )


def _mechanism_score_units(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    *,
    schema: ObsSchema,
    scorer: ScorerName,
    minimum_cells: int,
) -> pd.DataFrame:
    """Select final scorer columns and aggregate the aligned population."""
    suffix = "_score" if scorer == "scanpy" else "_auc"
    columns = [
        column
        for column in scores
        if isinstance(column, str)
        and column.endswith(suffix)
        and not column.endswith(("_pos_auc", "_inv_auc"))
    ]

    obs = adata.obs
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Mechanism diagnostics require in-memory obs")

    return donor_score_units(
        obs,
        scores,
        columns,
        schema=schema,
        minimum_cells=minimum_cells,
    )


def _mechanism_gene_diagnostics(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    module_ids: Sequence[str],
    *,
    schema: ObsSchema,
    scorer: ScorerName,
    gene_symbol_column: str,
    expression_layer: str | None,
    use_raw: bool,
    minimum_cells: int,
    minimum_donors: int,
    detection_threshold: float,
) -> GeneDiagnosticResults | None:
    """Inspect only scored modules needed by the mechanism companions."""
    needed = {
        "NASP_FEEDBACK",
        "NASP_RESTRICTION",
        "NASP_RNA_SENSING",
        "IFN_I_OUTPUT",
        "SIGNALING_CONTEXT_IFN_JAK_STAT",
    }
    modules = [
        GeneModules.modules(identifier)
        for identifier in module_ids
        if identifier in needed
    ]
    if modules:
        return module_gene_diagnostics(
            adata,
            scores,
            modules,
            schema=schema,
            scorer=scorer,
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            use_raw=use_raw,
            minimum_cells=minimum_cells,
            minimum_donors=minimum_donors,
            detection_threshold=detection_threshold,
        )

    return None


def _publish_mechanism_outputs(
    tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    provenance: pd.DataFrame,
    schema: ObsSchema,
    plot_visualizations: bool,
    minimum_cells: int,
    minimum_donors: int,
) -> list[Path]:
    """Publish companion tables, optional figures and scientific provenance."""
    artifacts = []
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        _atomic_table(table, path)
        artifacts.append(path)

    if plot_visualizations:
        artifacts.extend(
            MechanismPlotter(output_dir / "figures").plot_tables(
                tables, schema=schema, minimum_donors=minimum_donors
            )
        )

    _atomic_table(
        provenance.assign(
            mechanism_minimum_cells=minimum_cells,
            mechanism_minimum_donors=minimum_donors,
            mechanism_aggregation="mean",
            mechanism_interpretation="descriptive_no_pvalues",
        ),
        output_dir / "mechanism_provenance.csv",
    )

    return artifacts
