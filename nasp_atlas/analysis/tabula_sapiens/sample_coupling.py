"""Publish sample-shared and within-sample sensor-output coupling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.tabula_sapiens.mixed_model_scopes import _atomic_table
from nasp_atlas.analysis.tabula_sapiens.mixed_models import (
    _INDEPENDENT_UNIT_KEY,
)
from nasp_atlas.analysis.tabula_sapiens.mixed_models import (
    _factorized_identifier,
)
from nasp_atlas.analysis.tabula_sapiens.mixed_models import (
    _mixed_model_unit_frame,
)
from nasp_atlas.single_cell.associations import Aggregation
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.sample_coupling import SampleCouplingResults
from nasp_atlas.single_cell.sample_coupling import sample_coupling
from nasp_atlas.single_cell.sample_coupling import sensor_output_pairs
from nasp_atlas.single_cell.visualization.sample_coupling import (
    SampleCouplingPlotter,
)


def sample_coupling_analysis(
    cell_frame: pd.DataFrame,
    *,
    output_dir: Path,
    provenance: pd.DataFrame,
    schema: ObsSchema,
    run_sample_coupling: bool = False,
    plot_visualizations: bool = True,
    aggregation: Aggregation = "mean",
    detection_threshold: float = 0.0,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    output_modules: Sequence[str] = (
        "IFN_I_OUTPUT",
        "NFKB_CYTOKINE_OUTPUT",
        "ISR",
        "INFLAMMASOME",
    ),
) -> None:
    """Write donor-tissue sample sharing and within/between-sample coupling.

    Units are the mixed-model donor, tissue, cell-type and assay aggregates
    with at least `minimum_cells` cells. A sample is one donor's tissue, so
    within-sample slopes compare cell types that share the donor, tissue,
    processing and any paracrine or circulating signal. As in
    `nasp_sensor_output_coupling.csv`, every gene-expression feature (the
    resolved sensors and any requested genes) is paired with each scored
    `output_modules` entry; a gene that belongs to the output module is
    reported as excluded rather than fitted. With `plot_visualizations`, the
    slopes and variance partition are also drawn under `figures/`. The
    disabled or failed manifest never advertises earlier outputs as current.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "sample_coupling_manifest.csv"
    record: dict[str, object] = {
        "sample_coupling_enabled": run_sample_coupling,
        "status": "running",
    }

    if not run_sample_coupling:
        record["status"] = "disabled"
        _atomic_table(pd.DataFrame([record]), manifest)
        return
    _atomic_table(pd.DataFrame([record]), manifest)

    try:
        units, context_keys = _sample_coupling_units(
            cell_frame,
            schema=schema,
            aggregation=aggregation,
            detection_threshold=detection_threshold,
            minimum_cells=minimum_cells,
        )
        sensors, responses, excluded = sensor_output_pairs(
            units, output_modules
        )

        results = sample_coupling(
            units,
            predictors=sensors,
            responses=responses,
            independent_unit_key="independent_donor",
            sample_key="donor_tissue_sample",
            context_keys=context_keys,
            excluded_pairs=excluded,
            minimum_independent_units=minimum_donors,
        )

        artifacts = _publish_sample_coupling(
            results,
            output_dir=output_dir,
            plot_visualizations=plot_visualizations,
            provenance=provenance.assign(
                sample_coupling_observational_unit=units.attrs[
                    "observational_unit"
                ],
                sample_coupling_sample="independent donor x tissue",
                sample_coupling_centering_contexts=";".join(context_keys),
                sample_coupling_aggregation=aggregation,
                sample_coupling_detection_threshold=detection_threshold,
                sample_coupling_minimum_cells=minimum_cells,
                sample_coupling_minimum_donors=minimum_donors,
                sample_coupling_output_modules=";".join(responses),
                sample_coupling_n_excluded_pairs=len(excluded),
                sample_coupling_interpretation=(
                    "exploratory_observational_wald_mixed_models"
                ),
            ),
        )

        record.update(
            status="completed",
            artifacts=";".join(str(path) for path in artifacts),
        )
    finally:
        if record["status"] == "running":
            record["status"] = "failed"
        _atomic_table(pd.DataFrame([record]), manifest)


def _sample_coupling_units(
    cell_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    aggregation: Aggregation,
    detection_threshold: float,
    minimum_cells: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Reuse mixed-model units and label their donor-tissue samples."""
    units, _ = _mixed_model_unit_frame(
        cell_frame,
        schema=schema,
        aggregation=aggregation,
        detection_threshold=detection_threshold,
        minimum_cells=minimum_cells,
    )

    independent_keys = [
        key for key in (schema.study_key, schema.donor_key) if key in units
    ]
    units["independent_donor"] = units[_INDEPENDENT_UNIT_KEY]
    units["donor_tissue_sample"] = _factorized_identifier(
        units,
        keys=[*independent_keys, schema.tissue_key],
        prefix="sample_",
    )

    context_keys = [
        key
        for key in (
            schema.study_key,
            schema.tissue_key,
            schema.cell_type_key,
            schema.assay_key,
        )
        if key in units
    ]
    return units, context_keys


def _publish_sample_coupling(
    results: SampleCouplingResults,
    *,
    output_dir: Path,
    plot_visualizations: bool,
    provenance: pd.DataFrame,
) -> list[Path]:
    """Publish result tables, optional figures, then their provenance."""
    artifacts = []
    for name, table in (
        ("sample_variance_partition", results.variance_partition),
        ("sensor_output_sample_coupling", results.coupling),
        ("sample_coupling_diagnostics", results.diagnostics),
    ):
        path = output_dir / f"{name}.csv"
        _atomic_table(table, path)
        artifacts.append(path)

    if plot_visualizations:
        plotter = SampleCouplingPlotter(output_dir / "figures")
        artifacts.extend(plotter.plot_coupling(results.coupling))
        artifacts.extend(
            plotter.plot_variance_partition(results.variance_partition)
        )

    path = output_dir / "sample_coupling_provenance.csv"
    _atomic_table(provenance, path)
    artifacts.append(path)

    return artifacts
