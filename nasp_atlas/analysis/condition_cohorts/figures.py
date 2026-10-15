"""Figures for every stage of one condition-cohort run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.condition_cohorts.composition import (
    ConditionComposition,
)
from nasp_atlas.analysis.condition_cohorts.condition_coupling import (
    ConditionCoupling,
)
from nasp_atlas.analysis.condition_cohorts.design import ConditionDesign
from nasp_atlas.analysis.condition_cohorts.ifn_strata import IfnStrataResult
from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.analysis.condition_cohorts.units import SENSOR_MEASURES
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.visualization.condition_composition import (
    CompositionPlotter,
)
from nasp_atlas.single_cell.visualization.conditions import ConditionPlotter
from nasp_atlas.single_cell.visualization.sample_coupling import (
    SampleCouplingPlotter,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CohortResults:
    """Published tables of one run, as consumed by its figures.

    Attributes:
      design: Arm support, covariate balance and estimand availability.
      units: Unit metadata and features.
      effects: AB1 contrasts.
      adjusted: AB3 IFN-adjusted contrasts.
      strata: AB2 donors and contrasts, when the spec declares strata.
      slopes: Continuous contrasts, when the spec declares them.
      coupling: AB4 per-arm coupling and arm differences.
      composition: AB6 composition contrasts and splits.
      states: AB5 within-donor state contrasts, when declared.
      state_differences: Arm differences in the AB5 state difference, when
        the spec declares state contrasts.
    """

    design: ConditionDesign
    units: ConditionUnits
    effects: pd.DataFrame
    adjusted: pd.DataFrame
    strata: IfnStrataResult | None
    slopes: pd.DataFrame | None
    coupling: ConditionCoupling
    composition: ConditionComposition
    states: pd.DataFrame | None
    state_differences: pd.DataFrame | None


def plot_condition_figures(
    results: CohortResults,
    spec: ConditionCohortSpec,
    *,
    prepared_h5ad: Path,
    score_path: Path,
    output_dir: Path,
    ifn_scorer: ScorerName,
    minimum_donors: int,
    random_state: int,
    umap_modules: tuple[str, ...] = ("IFN_I_OUTPUT", "NFKB_CYTOKINE_OUTPUT"),
) -> list[Path]:
    """Render one figure set per stage from its published tables.

    Figures are presentation only: they are redrawn on every run and never
    feed back into estimates. Each saves the rows it displays.
    """
    condition = ConditionPlotter(output_dir)
    paths = _design_figures(condition, results, minimum_donors=minimum_donors)
    paths += condition.plot_condition_dotplot(
        results.units.features,
        results.units.metadata,
        arms=spec.condition.arms,
        minimum_donors=minimum_donors,
    )
    paths += _split_umaps(
        condition,
        prepared_h5ad,
        score_path,
        arms=spec.condition.arms,
        modules=umap_modules,
        scorer=ifn_scorer,
        random_state=random_state,
    )
    paths += _effect_figures(condition, results, spec)
    paths += _coupling_figures(output_dir, results.coupling)
    paths += _composition_figures(output_dir, results.composition)

    logger.info("Wrote %d figure files to %s", len(paths), output_dir)
    return paths


def _design_figures(
    plotter: ConditionPlotter,
    results: CohortResults,
    *,
    minimum_donors: int,
) -> list[Path]:
    """Plot arm support and covariate balance."""
    return plotter.plot_arm_support(
        results.design.arm_support, minimum_donors=minimum_donors
    ) + plotter.plot_covariate_balance(results.design.covariate_balance)


def _split_umaps(
    plotter: ConditionPlotter,
    prepared_h5ad: Path,
    score_path: Path,
    *,
    arms: tuple[str, ...],
    modules: tuple[str, ...],
    scorer: ScorerName,
    random_state: int,
) -> list[Path]:
    """Plot output-module scores on the UMAP, one panel per arm."""
    prepared = ad.read_h5ad(prepared_h5ad, backed="r")
    try:
        if "X_umap" not in prepared.obsm:
            logger.info("No X_umap in %s; skipping split UMAPs", prepared_h5ad)
            return []

        coordinates = np.asarray(prepared.obsm["X_umap"])[:, :2]
        cells = pd.DataFrame(
            {
                "x": coordinates[:, 0],
                "y": coordinates[:, 1],
                "condition": np.asarray(prepared.obs["condition"]).astype(str),
            },
            index=prepared.obs_names.astype(str),
        )
    finally:
        prepared.file.close()

    catalog = GeneModules()
    columns = {
        module: module_score_name(catalog.get_module(module), scorer=scorer)
        for module in modules
    }

    scores = pd.read_csv(
        score_path,
        index_col="obs_name",
        usecols=["obs_name", *columns.values()],
    )

    paths = []
    for module, column in columns.items():
        paths += plotter.plot_split_umap(
            cells.assign(value=scores[column].reindex(cells.index)),
            arms=arms,
            value_label=f"{module} score ({scorer})",
            filename=f"umap_{module.lower()}_by_arm",
            random_state=random_state,
        )

    return paths


def _effect_figures(
    plotter: ConditionPlotter,
    results: CohortResults,
    spec: ConditionCohortSpec,
) -> list[Path]:
    """Plot effect matrices for every effect stage, and the IFN checks."""
    paths: list[Path] = []
    tables = [
        ("", results.effects),
        ("ifn_adjusted_", results.adjusted),
        ("", results.strata.contrasts if results.strata else None),
        ("", results.slopes),
        ("", results.states),
        ("", results.state_differences),
    ]

    for prefix, table in tables:
        if table is None or table.empty:
            continue
        paths += _effect_matrices(plotter, table, prefix=prefix)

    for contrast in spec.contrasts:
        for measure in SENSOR_MEASURES:
            paths += plotter.plot_ifn_adjustment(
                results.effects,
                results.adjusted,
                estimand=contrast.estimand,
                feature_type=measure,
            )

    if results.strata is not None:
        paths += plotter.plot_ifn_strata(results.strata.donors)

    return paths


def _effect_matrices(
    plotter: ConditionPlotter,
    table: pd.DataFrame,
    *,
    prefix: str,
) -> list[Path]:
    """Plot one matrix per estimand and feature family of an effect table."""
    paths: list[Path] = []
    fitted = table.loc[
        table["cell_type"].notna() & table["feature_type"].notna()
    ]

    for estimand in dict.fromkeys(fitted["estimand"]):
        rows = fitted.loc[fitted["estimand"].eq(estimand)]

        for measure in SENSOR_MEASURES:
            paths += plotter.plot_condition_effects(
                rows,
                estimand=estimand,
                feature_type=measure,
                filename=f"{prefix}{estimand}_{measure}",
            )

        for scorer in sorted(rows["scorer"].dropna().unique()):
            paths += plotter.plot_condition_effects(
                rows,
                estimand=estimand,
                feature_type="module_score",
                scorer=str(scorer),
                filename=f"{prefix}{estimand}_module_score_{scorer}",
            )

    return paths


def _coupling_figures(
    output_dir: Path,
    coupling: ConditionCoupling,
) -> list[Path]:
    """Plot coupling and variance partition per arm, and arm differences."""
    plotter = SampleCouplingPlotter(output_dir)
    paths: list[Path] = []

    for arm, arm_coupling in coupling.coupling.groupby("condition"):
        paths += plotter.plot_coupling(
            arm_coupling, filename=f"sensor_output_coupling_{arm}"
        )

    for arm, partition in coupling.variance_partition.groupby("condition"):
        paths += plotter.plot_variance_partition(
            partition, filename=f"variance_partition_{arm}"
        )

    differences = coupling.differences
    if differences.empty:
        return paths

    for contrast, rows in differences.groupby("contrast"):
        paths += plotter.plot_coupling(
            rows.rename(columns={"difference": "estimate"}),
            filename=f"coupling_difference_{contrast}",
            x_label="Slope difference (target - reference)",
            title=(
                f"{contrast}: change in sensor-output coupling between arms "
                "(independent arms; 95% CI)"
            ),
        )

    return paths


def _composition_figures(
    output_dir: Path,
    composition: ConditionComposition,
) -> list[Path]:
    """Plot composition contrasts and the sensor-level splits."""
    plotter = CompositionPlotter(output_dir)
    paths = plotter.plot_composition_effects(composition.effects)

    for estimand in dict.fromkeys(composition.decomposition["estimand"]):
        for measure in SENSOR_MEASURES:
            paths += plotter.plot_decomposition(
                composition.decomposition,
                estimand=estimand,
                feature_type=measure,
            )

    return paths
