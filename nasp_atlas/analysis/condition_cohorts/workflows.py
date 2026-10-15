"""Flat orchestration of one condition-cohort analysis population."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.condition_cohorts.atlas_calibration import (
    atlas_sensor_profiles,
)
from nasp_atlas.analysis.condition_cohorts.atlas_calibration import (
    calibrate_reference_arm,
)
from nasp_atlas.analysis.condition_cohorts.atlas_calibration import (
    read_atlas_calibration_spec,
)
from nasp_atlas.analysis.condition_cohorts.cell_states import (
    contrast_cell_states,
)
from nasp_atlas.analysis.condition_cohorts.cell_states import (
    contrast_state_differences,
)
from nasp_atlas.analysis.condition_cohorts.composition import (
    ConditionComposition,
)
from nasp_atlas.analysis.condition_cohorts.composition import (
    estimate_condition_composition,
)
from nasp_atlas.analysis.condition_cohorts.condition_coupling import (
    ConditionCoupling,
)
from nasp_atlas.analysis.condition_cohorts.condition_coupling import (
    estimate_condition_coupling,
)
from nasp_atlas.analysis.condition_cohorts.condition_effects import (
    estimate_condition_effects,
)
from nasp_atlas.analysis.condition_cohorts.condition_effects import (
    estimate_continuous_effects,
)
from nasp_atlas.analysis.condition_cohorts.design import ConditionDesign
from nasp_atlas.analysis.condition_cohorts.design import audit_condition_design
from nasp_atlas.analysis.condition_cohorts.design import contrast_availability
from nasp_atlas.analysis.condition_cohorts.figures import CohortResults
from nasp_atlas.analysis.condition_cohorts.figures import plot_condition_figures
from nasp_atlas.analysis.condition_cohorts.ifn_adjustment import (
    leave_self_out_ifn_scores,
)
from nasp_atlas.analysis.condition_cohorts.ifn_adjustment import (
    sensors_in_module,
)
from nasp_atlas.analysis.condition_cohorts.ifn_strata import IfnStrataResult
from nasp_atlas.analysis.condition_cohorts.ifn_strata import stratify_by_ifn
from nasp_atlas.analysis.condition_cohorts.intake import ConditionCohortIntake
from nasp_atlas.analysis.condition_cohorts.intake import PreparedConditionCohort
from nasp_atlas.analysis.condition_cohorts.intake import (
    prepare_condition_cohort,
)
from nasp_atlas.analysis.condition_cohorts.intake import select_condition_cohort
from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.analysis.condition_cohorts.specification import (
    read_condition_cohort_spec,
)
from nasp_atlas.analysis.condition_cohorts.units import UNIT_KEYS
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits
from nasp_atlas.analysis.condition_cohorts.units import build_condition_units
from nasp_atlas.analysis.score_checkpoint import score_checkpoint_matches
from nasp_atlas.analysis.score_checkpoint import score_checkpoint_path
from nasp_atlas.analysis.score_checkpoint import write_score_checkpoint
from nasp_atlas.analysis.stage_artifacts import file_identity
from nasp_atlas.analysis.stage_artifacts import read_published_table
from nasp_atlas.analysis.stage_artifacts import read_stage_manifest
from nasp_atlas.analysis.stage_artifacts import reuse_or_publish_tables
from nasp_atlas.analysis.tabula_sapiens import tabula_sapiens_scoring_analysis
from nasp_atlas.single_cell.io import read_h5ad_metadata
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.visualization.conditions import ConditionPlotter


logger = logging.getLogger(__name__)


# Recorded in every model stage's request, so a rerun never reuses results
# from a different estimator.
DONOR_MODEL: str = "donor_mean_ols_hc3"


def condition_cohort_analysis(
    *,
    spec_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    minimum_state_pairs: int = 6,
    target_sum: float = 10_000.0,
    scorers: Sequence[ScorerName] = ("scanpy", "aucell"),
    random_state: int = 42,
    n_bootstrap: int = 1_000,
    aucell_chunk_size: int = 1_000,
    aucell_num_workers: int = 1,
    development_donors_per_arm: int | None = None,
    plot_visualizations: bool = True,
) -> dict[str, Path]:
    """Prepare one declared population and publish its analysis stages.

    Outputs go to `<output_dir>/<cohort_id>/<population_id>/`, one
    subdirectory per stage; a development subset adds a `_dev<N>` suffix to
    the population directory. Each stage publishes its manifest last and is
    reused on a rerun only when its request and artifacts match. All arms
    are scored together, so signed-score standardization and Scanpy control
    genes are shared rather than absorbing the arm difference.

    Args:
      spec_path: Condition-cohort YAML request.
      data_dir: Directory holding the spec's input file by name.
      output_dir: Root for cohort runs.
      minimum_cells: Cells needed for a donor-sample-cell-type unit.
      minimum_donors: Supported donors needed in each compared arm.
      minimum_state_pairs: Donors with both states needed per cell type for
        a within-donor state contrast.
      target_sum: Library size used for log1p normalization.
      scorers: Module scorers to run ("scanpy", "aucell", or both).
      random_state: Seed for Scanpy controls, bootstraps and plotting.
      n_bootstrap: Donor resamples for the composition split.
      aucell_chunk_size: Cells ranked in one AUCell block.
      aucell_num_workers: AUCell worker processes per block.
      development_donors_per_arm: Keep this many whole donors per arm, drawn
        with `random_state`, for fast development runs only.
      plot_visualizations: Render every stage's figures.

    Returns:
      The run directory and each completed stage directory or score table.
    """
    spec = read_condition_cohort_spec(spec_path)
    run_dir = _run_directory(spec, output_dir, development_donors_per_arm)
    ifn_scorer: ScorerName = "scanpy" if "scanpy" in scorers else scorers[0]

    # Prepare
    prepared = prepare_condition_cohort(
        spec,
        input_path=Path(data_dir) / spec.input_filename,
        output_dir=run_dir / "intake",
        target_sum=target_sum,
        development_donors_per_arm=development_donors_per_arm,
        random_state=random_state,
    )

    design = _publish_design(
        prepared,
        spec,
        output_dir=run_dir / "design",
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
    )

    # Score and aggregate
    score_path = _score_condition_cohort(
        prepared,
        output_dir=run_dir / "scoring",
        scorers=scorers,
        random_state=random_state,
        aucell_chunk_size=aucell_chunk_size,
        aucell_num_workers=aucell_num_workers,
    )

    units = _publish_units(
        prepared,
        spec,
        score_path=score_path,
        output_dir=run_dir / "units",
        scorers=scorers,
        minimum_cells=minimum_cells,
    )

    # Analyze
    effects = _publish_condition_effects(
        units,
        spec,
        design=design,
        run_dir=run_dir,
        minimum_donors=minimum_donors,
    )

    adjusted = _publish_ifn_adjusted_effects(
        units,
        spec,
        design=design,
        run_dir=run_dir,
        prepared_h5ad=prepared.h5ad_path,
        minimum_donors=minimum_donors,
        ifn_scorer=ifn_scorer,
        random_state=random_state,
    )

    strata = (
        _publish_ifn_strata(
            units, spec, run_dir=run_dir, minimum_donors=minimum_donors
        )
        if spec.ifn_strata is not None
        else None
    )

    slopes = (
        _publish_continuous_effects(
            units, spec, run_dir=run_dir, minimum_donors=minimum_donors
        )
        if spec.continuous_contrasts
        else None
    )

    coupling = _publish_condition_coupling(
        units,
        spec,
        run_dir=run_dir,
        scorer=ifn_scorer,
        minimum_donors=minimum_donors,
    )

    composition = _publish_composition(
        units,
        spec,
        run_dir=run_dir,
        scorer=ifn_scorer,
        minimum_donors=minimum_donors,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    states = (
        _publish_cell_states(
            prepared,
            spec,
            score_path=score_path,
            run_dir=run_dir,
            scorers=scorers,
            minimum_cells=minimum_cells,
            minimum_pairs=minimum_state_pairs,
        )
        if spec.cell_state is not None
        else None
    )

    state_differences = (
        _publish_state_contrasts(
            spec, run_dir=run_dir, minimum_pairs=minimum_state_pairs
        )
        if spec.state_contrasts
        else None
    )

    # Present
    if plot_visualizations:
        plot_condition_figures(
            CohortResults(
                design=design,
                units=units,
                effects=effects,
                adjusted=adjusted,
                strata=strata,
                slopes=slopes,
                coupling=coupling,
                composition=composition,
                states=states,
                state_differences=state_differences,
            ),
            spec,
            prepared_h5ad=prepared.h5ad_path,
            score_path=score_path,
            output_dir=run_dir / "figures",
            ifn_scorer=ifn_scorer,
            minimum_donors=minimum_donors,
            random_state=random_state,
        )

    stages = [
        "intake",
        "design",
        "units",
        "condition_effects",
        "ifn_adjusted_effects",
        "ifn_strata",
        "continuous_effects",
        "condition_coupling",
        "composition",
        "cell_states",
        "state_contrasts",
        "figures",
    ]

    return {
        "run": run_dir,
        "scores": score_path,
        **{
            stage: run_dir / stage
            for stage in stages
            if (run_dir / stage).exists()
        },
    }


def atlas_calibration_analysis(
    *,
    spec_path: str | Path,
    calibration_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    development_donors_per_arm: int | None = None,
    plot_visualizations: bool = True,
) -> Path:
    """Compare a completed run's reference arm with a healthy atlas (AB8).

    The run's published units supply the reference arm; the atlas named in
    `calibration_path` is read from `data_dir` and must match its declared
    SHA-256. Results go to `<run>/atlas_calibration/` and are reused when
    the request and artifacts match. Descriptive only.

    Returns:
      The calibration stage directory.
    """
    spec = read_condition_cohort_spec(spec_path)
    calibration = read_atlas_calibration_spec(calibration_path)
    run_dir = _run_directory(spec, output_dir, development_donors_per_arm)
    input_path = Path(data_dir) / calibration.input_filename

    request = {
        **_upstream(run_dir, "units"),
        **asdict(calibration),
        "reference": spec.condition.reference,
        "minimum_cells": minimum_cells,
        "minimum_donors": minimum_donors,
    }

    def build() -> dict[str, pd.DataFrame]:
        if file_identity(input_path)["sha256"] != calibration.input_sha256:
            raise ValueError(
                f"{input_path} does not match its declared SHA-256"
            )

        result = calibrate_reference_arm(
            _read_units(run_dir / "units", stage="units"),
            atlas_sensor_profiles(
                calibration, input_path=input_path, minimum_cells=minimum_cells
            ),
            reference=spec.condition.reference,
            minimum_donors=minimum_donors,
        )

        return {
            "calibration_profiles.csv": result.profiles,
            "calibration_agreement.csv": result.agreement,
        }

    stage_dir = run_dir / "atlas_calibration"
    tables = reuse_or_publish_tables(
        stage_dir, request, build, stage="atlas_calibration"
    )

    if plot_visualizations:
        ConditionPlotter(run_dir / "figures").plot_atlas_calibration(
            tables["calibration_profiles.csv"],
            tables["calibration_agreement.csv"],
            atlas_id=calibration.atlas_id,
        )
    return stage_dir


def audit_condition_cohort(
    spec_path: str | Path,
    *,
    data_dir: str | Path,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
) -> tuple[ConditionCohortIntake, ConditionDesign]:
    """Select cells and audit the design from metadata alone; write nothing.

    Use this to check arm sizes, confounding and estimand availability
    before preparing expression. The input checksum is not verified.
    """
    spec = read_condition_cohort_spec(spec_path)
    obs, _ = read_h5ad_metadata(Path(data_dir) / spec.input_filename)
    intake = select_condition_cohort(obs, spec)
    design = audit_condition_design(
        intake,
        spec,
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
    )

    return intake, design


def _run_directory(
    spec: ConditionCohortSpec,
    output_dir: str | Path,
    development_donors_per_arm: int | None,
) -> Path:
    """Return `<output_dir>/<cohort>/<population>[_dev<N>]`."""
    suffix = (
        f"_dev{development_donors_per_arm}"
        if development_donors_per_arm is not None
        else ""
    )
    return Path(output_dir) / spec.cohort_id / f"{spec.population_id}{suffix}"


def _publish_design(
    prepared: PreparedConditionCohort,
    spec: ConditionCohortSpec,
    *,
    output_dir: Path,
    minimum_cells: int,
    minimum_donors: int,
) -> ConditionDesign:
    """Publish arm support, covariate balance and estimand availability."""
    request = {
        "intake_identity": prepared.manifest["identity"],
        "contrasts": [asdict(contrast) for contrast in spec.contrasts],
        "minimum_cells": minimum_cells,
        "minimum_donors": minimum_donors,
    }

    def build() -> dict[str, pd.DataFrame]:
        design = audit_condition_design(
            prepared.intake,
            spec,
            minimum_cells=minimum_cells,
            minimum_donors=minimum_donors,
        )

        return {
            "arm_support.csv": design.arm_support,
            "covariate_balance.csv": design.covariate_balance,
            "estimand_availability.csv": design.estimand_availability,
        }

    tables = reuse_or_publish_tables(output_dir, request, build, stage="design")
    return ConditionDesign(
        arm_support=tables["arm_support.csv"],
        covariate_balance=tables["covariate_balance.csv"],
        estimand_availability=tables["estimand_availability.csv"],
    )


def _score_condition_cohort(
    prepared: PreparedConditionCohort,
    *,
    output_dir: Path,
    scorers: Sequence[ScorerName],
    random_state: int,
    aucell_chunk_size: int,
    aucell_num_workers: int,
) -> Path:
    """Score every module over all arms with the Tabula Sapiens workflow.

    The completed score table is reused only when its checkpoint matches the
    intake identity, curated panel and scoring settings. A completed table
    from another request, such as an earlier compendium panel, is never
    overwritten, because later stages of that run were built from it; an
    incomplete one is recomputed. Ages shown in the
    metadata UMAP come from development-stage labels and are display-only;
    models use the spec's parsed covariates.
    """
    score_path = output_dir / "condition_cohort_module_scores.csv.gz"
    request = {
        "intake_identity": prepared.manifest["identity"],
        "scorers": list(scorers),
        "random_state": random_state,
        "marker_panel_sha256": _marker_panel_sha256(),
        "aucell_chunk_size": aucell_chunk_size,
        "aucell_num_workers": aucell_num_workers,
    }

    if score_checkpoint_matches(score_path, request):
        logger.info("Reusing complete score table %s", score_path)
        return score_path
    if score_checkpoint_path(score_path).exists():
        raise ValueError(
            f"{score_path} was completed for a different request (input, "
            "compendium panel or scoring settings); write this run to a new "
            "output path instead of overwriting it"
        )

    tabula_sapiens_scoring_analysis(
        h5ad_path=prepared.h5ad_path,
        output_dir=output_dir,
        random_state=random_state,
        tissue_key="tissue",
        cell_type_key="cell_type",
        sex_key="sex",
        development_stage_key="development_stage",
        assay_key="assay",
        age_key="stage_age_years",
        donor_key="donor_id",
        gene_symbol_column="feature_name",
        metadata_panels=(
            {
                "obs_key": "cell_type",
                "title": "Cell type",
                "kind": "categorical",
                "legend_loc": "bottom",
                "legend_ncol": 3,
            },
            {
                "obs_key": "condition",
                "title": "Condition",
                "kind": "categorical",
                "legend_loc": "bottom",
                "legend_ncol": 2,
            },
            {
                "obs_key": "assay",
                "title": "Assay",
                "kind": "categorical",
                "legend_loc": "bottom",
                "legend_ncol": 1,
            },
        ),
        score_scanpy="scanpy" in scorers,
        score_aucell="aucell" in scorers,
        aucell_chunk_size=aucell_chunk_size,
        aucell_num_workers=aucell_num_workers,
        score_table_filename=score_path.name,
        heatmap_groupby="cell_type",
        single_tissue=None,
    )

    write_score_checkpoint(score_path, request)
    return score_path


def _publish_units(
    prepared: PreparedConditionCohort,
    spec: ConditionCohortSpec,
    *,
    score_path: Path,
    output_dir: Path,
    scorers: Sequence[ScorerName],
    minimum_cells: int,
    unit_keys: Sequence[str] = UNIT_KEYS,
    stage: str = "units",
) -> ConditionUnits:
    """Publish aggregates of scores and sensors per unit.

    Units are donor-sample-cell-type aggregates by default; the cell-state
    stage passes finer `unit_keys` and its own `stage` name.
    """
    request = {
        "intake_identity": prepared.manifest["identity"],
        "scores_sha256": file_identity(score_path)["sha256"],
        "scorers": list(scorers),
        "minimum_cells": minimum_cells,
        "unit_keys": list(unit_keys),
        "marker_panel_sha256": _marker_panel_sha256(),
    }

    def build() -> dict[str, pd.DataFrame]:
        units = build_condition_units(
            prepared.h5ad_path,
            pd.read_csv(score_path, index_col="obs_name"),
            spec,
            scorers=scorers,
            minimum_cells=minimum_cells,
            unit_keys=unit_keys,
        )

        return {
            "unit_metadata.csv": units.metadata,
            "unit_features.csv.gz": units.features,
            "sensor_mapping.csv": units.sensor_mapping,
        }

    tables = reuse_or_publish_tables(output_dir, request, build, stage=stage)
    return ConditionUnits(
        metadata=tables["unit_metadata.csv"],
        features=tables["unit_features.csv.gz"],
        sensor_mapping=tables["sensor_mapping.csv"],
    )


def _publish_condition_effects(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    design: ConditionDesign,
    run_dir: Path,
    minimum_donors: int,
    adjust_for_depth: bool = True,
) -> pd.DataFrame:
    """Publish planned condition contrasts within cell type (AB1)."""
    return _publish_table(
        run_dir,
        stage="condition_effects",
        table="condition_contrasts.csv",
        request={
            **_upstream(run_dir, "units", "design"),
            "method": DONOR_MODEL,
            "contrasts": [asdict(contrast) for contrast in spec.contrasts],
            "covariates": [asdict(covariate) for covariate in spec.covariates],
            "minimum_donors": minimum_donors,
            "adjust_for_depth": adjust_for_depth,
        },
        build=lambda: estimate_condition_effects(
            units,
            contrasts=spec.contrasts,
            covariates=spec.covariates,
            availability=design.estimand_availability,
            minimum_donors=minimum_donors,
            adjust_for_depth=adjust_for_depth,
        ),
    )


def _publish_ifn_adjusted_effects(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    design: ConditionDesign,
    run_dir: Path,
    prepared_h5ad: Path,
    minimum_donors: int,
    ifn_scorer: ScorerName,
    random_state: int,
) -> pd.DataFrame:
    """Publish planned contrasts adjusted for unit IFN-I output (AB3).

    A sensor that is also an IFN_I_OUTPUT gene, such as IFIT1, is adjusted
    for the module scored without it rather than for a score containing
    itself; those scores are published beside the contrasts.
    """
    sensors = sensors_in_module("IFN_I_OUTPUT")
    request: dict[str, object] = {
        **_upstream(run_dir, "units", "design"),
        "method": DONOR_MODEL,
        "contrasts": [asdict(contrast) for contrast in spec.contrasts],
        "covariates": [asdict(covariate) for covariate in spec.covariates],
        "minimum_donors": minimum_donors,
        "ifn_scorer": ifn_scorer,
    }

    if sensors:
        # Only then does the stage score anything; panels without such
        # sensors keep their earlier request and identity.
        request["ifn_leave_self_out"] = {
            "sensors": sensors,
            "random_state": random_state,
        }

    def build() -> dict[str, pd.DataFrame]:
        without_self = (
            leave_self_out_ifn_scores(
                prepared_h5ad,
                sensors=sensors,
                scorer=ifn_scorer,
                random_state=random_state,
            )
            if sensors
            else None
        )

        tables = {
            "ifn_adjusted_contrasts.csv": estimate_condition_effects(
                units,
                contrasts=spec.contrasts,
                covariates=spec.covariates,
                availability=design.estimand_availability,
                minimum_donors=minimum_donors,
                ifn_scorer=ifn_scorer,
                ifn_leave_self_out=without_self,
                analysis="ifn_adjusted_effects",
            )
        }

        if without_self is not None:
            tables["ifn_leave_self_out_scores.csv"] = without_self
        return tables

    tables = reuse_or_publish_tables(
        run_dir / "ifn_adjusted_effects",
        request,
        build,
        stage="ifn_adjusted_effects",
    )
    return tables["ifn_adjusted_contrasts.csv"]


def _publish_ifn_strata(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    run_dir: Path,
    minimum_donors: int,
) -> IfnStrataResult:
    """Publish IFN-low and IFN-high disease strata contrasts (AB2)."""
    if spec.ifn_strata is None:
        raise ValueError("The spec declares no ifn_strata")
    strata_spec = spec.ifn_strata

    request = {
        **_upstream(run_dir, "units"),
        "method": DONOR_MODEL,
        "ifn_strata": asdict(strata_spec),
        "reference": spec.condition.reference,
        "covariates": [asdict(covariate) for covariate in spec.covariates],
        "minimum_donors": minimum_donors,
    }

    def build() -> dict[str, pd.DataFrame]:
        strata = stratify_by_ifn(
            units, strata_spec, reference=spec.condition.reference
        )

        metadata = strata.units.metadata
        _, availability = contrast_availability(
            metadata.loc[metadata["supported"]],
            arms=sorted(metadata["condition"].unique()),
            contrasts=strata.contrasts,
            minimum_donors=minimum_donors,
        )

        return {
            "ifn_strata_donors.csv": strata.donors,
            "ifn_strata_availability.csv": availability,
            "ifn_strata_contrasts.csv": estimate_condition_effects(
                strata.units,
                contrasts=strata.contrasts,
                covariates=spec.covariates,
                availability=availability,
                minimum_donors=minimum_donors,
                analysis="ifn_strata_effects",
            ),
        }

    tables = reuse_or_publish_tables(
        run_dir / "ifn_strata", request, build, stage="ifn_strata"
    )

    return IfnStrataResult(
        donors=tables["ifn_strata_donors.csv"],
        contrasts=tables["ifn_strata_contrasts.csv"],
    )


def _publish_continuous_effects(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    run_dir: Path,
    minimum_donors: int,
) -> pd.DataFrame:
    """Publish planned slopes on donor measurements, such as eGFR."""
    return _publish_table(
        run_dir,
        stage="continuous_effects",
        table="continuous_contrasts.csv",
        request={
            **_upstream(run_dir, "units"),
            "method": DONOR_MODEL,
            "continuous_contrasts": [
                asdict(slope) for slope in spec.continuous_contrasts
            ],
            "covariates": [asdict(covariate) for covariate in spec.covariates],
            "minimum_donors": minimum_donors,
        },
        build=lambda: estimate_continuous_effects(
            units,
            slopes=spec.continuous_contrasts,
            covariates=spec.covariates,
            arm_columns=spec.arm_obs_columns,
            minimum_donors=minimum_donors,
        ),
    )


def _publish_condition_coupling(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    run_dir: Path,
    scorer: str,
    minimum_donors: int,
) -> ConditionCoupling:
    """Publish per-arm sample coupling and arm differences (AB4)."""
    request = {
        **_upstream(run_dir, "units"),
        "arms": list(spec.condition.arms),
        "contrasts": [asdict(contrast) for contrast in spec.contrasts],
        "scorer": scorer,
        "minimum_donors": minimum_donors,
    }

    def build() -> dict[str, pd.DataFrame]:
        coupling = estimate_condition_coupling(
            units, spec, scorer=scorer, minimum_donors=minimum_donors
        )

        return {
            "coupling_by_arm.csv": coupling.coupling,
            "variance_partition_by_arm.csv": coupling.variance_partition,
            "coupling_differences.csv": coupling.differences,
            "coupling_diagnostics.csv": coupling.diagnostics,
        }

    tables = reuse_or_publish_tables(
        run_dir / "condition_coupling",
        request,
        build,
        stage="condition_coupling",
    )

    return ConditionCoupling(
        coupling=tables["coupling_by_arm.csv"],
        variance_partition=tables["variance_partition_by_arm.csv"],
        differences=tables["coupling_differences.csv"],
        diagnostics=tables["coupling_diagnostics.csv"],
    )


def _publish_composition(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    run_dir: Path,
    scorer: str,
    minimum_donors: int,
    n_bootstrap: int,
    random_state: int,
) -> ConditionComposition:
    """Publish composition contrasts and composition-within splits (AB6)."""
    request = {
        **_upstream(run_dir, "units"),
        "method": DONOR_MODEL,
        "contrasts": [asdict(contrast) for contrast in spec.contrasts],
        "covariates": [asdict(covariate) for covariate in spec.covariates],
        "scorer": scorer,
        "minimum_donors": minimum_donors,
        "n_bootstrap": n_bootstrap,
        "random_state": random_state,
    }

    def build() -> dict[str, pd.DataFrame]:
        composition = estimate_condition_composition(
            units,
            spec,
            scorer=scorer,
            minimum_donors=minimum_donors,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        return {
            "sample_proportions.csv": composition.proportions,
            "composition_effects.csv": composition.effects,
            "composition_decomposition.csv": composition.decomposition,
            "composition_contributions.csv": composition.contributions,
        }

    tables = reuse_or_publish_tables(
        run_dir / "composition", request, build, stage="composition"
    )

    return ConditionComposition(
        proportions=tables["sample_proportions.csv"],
        effects=tables["composition_effects.csv"],
        decomposition=tables["composition_decomposition.csv"],
        contributions=tables["composition_contributions.csv"],
    )


def _publish_cell_states(
    prepared: PreparedConditionCohort,
    spec: ConditionCohortSpec,
    *,
    score_path: Path,
    run_dir: Path,
    scorers: Sequence[ScorerName],
    minimum_cells: int,
    minimum_pairs: int,
) -> pd.DataFrame:
    """Publish state units and within-donor state contrasts (AB5)."""
    if spec.cell_state is None:
        raise ValueError("The spec declares no cell_state")
    states = spec.cell_state

    state_units = _publish_units(
        prepared,
        spec,
        score_path=score_path,
        output_dir=run_dir / "state_units",
        scorers=scorers,
        minimum_cells=minimum_cells,
        unit_keys=(*UNIT_KEYS, states.column),
        stage="state_units",
    )

    return _publish_table(
        run_dir,
        stage="cell_states",
        table="cell_state_contrasts.csv",
        request={
            **_upstream(run_dir, "state_units"),
            "method": "paired_donor_ols_hc3",
            "cell_state": asdict(states),
            "arms": list(spec.condition.arms),
            "minimum_pairs": minimum_pairs,
        },
        build=lambda: contrast_cell_states(
            state_units,
            states,
            arms=spec.condition.arms,
            minimum_pairs=minimum_pairs,
        ),
    )


def _publish_state_contrasts(
    spec: ConditionCohortSpec,
    *,
    run_dir: Path,
    minimum_pairs: int,
) -> pd.DataFrame:
    """Publish arm differences in within-donor state differences.

    Reads the hash-checked state units published by the cell-state stage.
    """
    if spec.cell_state is None:
        raise ValueError("state_contrasts need a declared cell_state")
    states = spec.cell_state

    def build() -> pd.DataFrame:
        return contrast_state_differences(
            _read_units(run_dir / "state_units", stage="state_units"),
            states,
            contrasts=spec.state_contrasts,
            minimum_pairs=minimum_pairs,
        )

    return _publish_table(
        run_dir,
        stage="state_contrasts",
        table="state_arm_contrasts.csv",
        request={
            **_upstream(run_dir, "state_units"),
            "method": "paired_donor_difference_ols_hc3",
            "cell_state": asdict(states),
            "state_contrasts": [
                asdict(contrast) for contrast in spec.state_contrasts
            ],
            "minimum_pairs": minimum_pairs,
        },
        build=build,
    )


def _publish_table(
    run_dir: Path,
    *,
    stage: str,
    table: str,
    request: Mapping[str, object],
    build: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Publish a single-table stage, or reuse it when the request matches."""
    tables = reuse_or_publish_tables(
        run_dir / stage, request, lambda: {table: build()}, stage=stage
    )
    return tables[table]


def _read_units(stage_dir: Path, *, stage: str) -> ConditionUnits:
    """Read a published units stage, checking each table's bytes."""
    return ConditionUnits(
        metadata=read_published_table(
            stage_dir, "unit_metadata.csv", stage=stage
        ),
        features=read_published_table(
            stage_dir, "unit_features.csv.gz", stage=stage
        ),
        sensor_mapping=read_published_table(
            stage_dir, "sensor_mapping.csv", stage=stage
        ),
    )


def _upstream(run_dir: Path, *stages: str) -> dict[str, str]:
    """Return the request identities of completed upstream stages."""
    return {
        f"{stage}_identity": read_stage_manifest(
            run_dir / stage / f"{stage}_manifest.json", stage=stage
        )["identity"]
        for stage in stages
    }


def _marker_panel_sha256() -> str:
    """Identify the installed compendium panel used for modules and sensors."""
    return str(file_identity(GeneModules.default_panel_path())["sha256"])
