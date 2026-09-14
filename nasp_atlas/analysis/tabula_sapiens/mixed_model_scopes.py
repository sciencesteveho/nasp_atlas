"""Fit and publish independently selected combined and tissue model scopes."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import fields
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from nasp_atlas.analysis.tabula_sapiens.mixed_models import (
    TabulaMixedModelResults,
)
from nasp_atlas.analysis.tabula_sapiens.mixed_models import (
    tabula_sapiens_mixed_model_inference,
)
from nasp_atlas.analysis.tabula_sapiens.scoring import safe_filename_token
from nasp_atlas.analysis.tabula_sapiens.visualizations import (
    plot_tabula_sapiens_mixed_model_inference,
)
from nasp_atlas.single_cell.associations import Aggregation
from nasp_atlas.single_cell.associations import ObsSchema


logger = logging.getLogger(__name__)

__all__ = ["mixed_model_scope_analysis", "write_mixed_model_tables"]


def write_mixed_model_tables(
    results: TabulaMixedModelResults,
    output_dir: Path,
) -> list[Path]:
    """Publish the six model tables atomically, returning their actual paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for field in fields(results):
        table = getattr(results, field.name)
        path = output_dir / f"association_mixed_model_{field.name}.csv"
        _atomic_table(table, path)
        paths.append(path)
    return paths


def mixed_model_scope_analysis(
    cell_frame: pd.DataFrame,
    *,
    output_dir: Path,
    provenance: pd.DataFrame,
    schema: ObsSchema,
    combined: bool = True,
    per_tissue: bool = False,
    aggregation: Aggregation = "mean",
    detection_threshold: float = 0.0,
    condition_reference: str | None = "normal",
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    minimum_studies: int = 3,
    minimum_repeated_contexts: int = 3,
    plot_visualizations: bool = True,
) -> TabulaMixedModelResults | None:
    """Fit selected scopes from shared scores without reloading or rescoring.

    Combined means all scored cells in this input, which can itself be a
    subset. Tissue models use every scored cell belonging to that tissue.
    Each tissue refits its own covariates, age center and random structure.
    FDR families are separate by scope and scorer. The scope manifest is
    replaced before execution and after each scope, so stale directories are
    never advertised as current results. Failures propagate and leave pending
    scopes explicit; resume reruns inference from validated scores.
    """
    scopes = _model_scopes(
        cell_frame,
        output_dir=output_dir,
        schema=schema,
        combined=combined,
        per_tissue=per_tissue,
    )
    manifest_path = (
        output_dir / "association_tables" / "association_mixed_model_scopes.csv"
    )
    _atomic_table(pd.DataFrame(scopes), manifest_path)

    combined_result = None
    for scope in scopes:
        if not scope["enabled"]:
            continue

        tissue = str(scope["tissue"])
        frame = (
            cell_frame
            if scope["scope"] == "combined_input"
            else cell_frame.loc[
                cell_frame[schema.tissue_key].astype(str).eq(tissue)
            ]
        )
        destination = Path(str(scope["output_dir"]))
        scope.update(
            status="running",
            n_cells=frame.obs_name.nunique(),
            n_tissues=frame[schema.tissue_key].nunique(),
        )
        _atomic_table(pd.DataFrame(scopes), manifest_path)
        logger.info(
            "Fitting mixed models: %s %s (%d cells)",
            scope["scope"],
            scope["tissue"],
            scope["n_cells"],
        )

        completed = False
        try:
            result = tabula_sapiens_mixed_model_inference(
                frame,
                schema=schema,
                aggregation=aggregation,
                detection_threshold=detection_threshold,
                condition_reference=condition_reference,
                minimum_cells=minimum_cells,
                minimum_donors=minimum_donors,
                minimum_studies=minimum_studies,
                minimum_repeated_contexts=minimum_repeated_contexts,
            )
            _publish_scope_results(
                result,
                frame,
                scope,
                destination=destination,
                provenance=provenance,
                plot_visualizations=plot_visualizations,
            )
            if scope["scope"] == "combined_input":
                combined_result = result
            completed = True
        finally:
            if not completed:
                scope["status"] = "failed"
            _atomic_table(pd.DataFrame(scopes), manifest_path)

    return combined_result


def _model_scopes(
    cell_frame: pd.DataFrame,
    *,
    output_dir: Path,
    schema: ObsSchema,
    combined: bool,
    per_tissue: bool,
) -> list[dict[str, object]]:
    """Resolve independent model destinations and explicit missing tissues."""
    if schema.tissue_key not in cell_frame:
        raise KeyError(f"Missing tissue column: {schema.tissue_key}")
    tissues = sorted(
        cell_frame[schema.tissue_key].dropna().astype(str).unique()
    )
    scopes: list[dict[str, object]] = [
        {
            "scope": "combined_input",
            "tissue": "",
            "enabled": combined,
            "output_dir": str(output_dir),
            "status": "pending" if combined else "disabled",
        }
    ]
    for tissue in tissues:
        digest = hashlib.sha256(tissue.encode()).hexdigest()[:10]
        tissue_dir = (
            output_dir
            / "mixed_models_by_tissue"
            / f"{safe_filename_token(tissue)}_{digest}"
        )
        scopes.append(
            {
                "scope": "per_tissue",
                "tissue": tissue,
                "enabled": per_tissue,
                "output_dir": str(tissue_dir),
                "status": "pending" if per_tissue else "disabled",
            }
        )

    missing = cell_frame[schema.tissue_key].isna()
    if missing.any():
        scopes.append(
            {
                "scope": "per_tissue",
                "tissue": "",
                "enabled": False,
                "output_dir": "",
                "status": "unavailable_missing_tissue",
                "n_cells": cell_frame.loc[missing, "obs_name"].nunique(),
            }
        )

    return scopes


def _publish_scope_results(
    result: TabulaMixedModelResults,
    frame: pd.DataFrame,
    scope: dict[str, object],
    *,
    destination: Path,
    provenance: pd.DataFrame,
    plot_visualizations: bool,
) -> None:
    """Publish model tables and figures with scope-specific provenance."""
    tables = write_mixed_model_tables(
        result, destination / "association_tables"
    )

    scoped_provenance = provenance.assign(
        mixed_model_scope=str(scope["scope"]),
        mixed_model_tissue=str(scope["tissue"]),
        mixed_model_scored_cells=frame.obs_name.nunique(),
    )
    provenance_path = (
        destination / "association_tables" / "association_provenance.csv"
    )
    _atomic_table(scoped_provenance, provenance_path)

    paths = []
    if plot_visualizations and scope["scope"] == "per_tissue":
        paths = plot_tabula_sapiens_mixed_model_inference(
            output_dir=destination / "association_plots" / "mixed_models",
            contrasts=result.contrasts,
            variance_components=result.variance_components,
        )

    scope.update(
        status="completed",
        n_estimable_contrasts=int(result.contrasts.estimable.sum()),
        artifacts=";".join(
            str(path) for path in [*tables, provenance_path, *paths]
        ),
    )


def _atomic_table(table: pd.DataFrame, path: Path) -> None:
    """Prevent interrupted writes from publishing a partial table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, suffix=".csv", delete=False
    ) as handle:
        temporary = Path(handle.name)

    try:
        table.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
