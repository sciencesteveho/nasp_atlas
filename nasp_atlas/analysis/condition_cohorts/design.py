"""Support, balance and identifiability of a cohort's planned contrasts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nasp_atlas.analysis.condition_cohorts.intake import ConditionCohortIntake
from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.analysis.condition_cohorts.specification import ContrastSpec
from nasp_atlas.analysis.condition_cohorts.specification import CovariateSpec


@dataclass(frozen=True)
class ConditionDesign:
    """Design facts established before any expression value is read.

    Attributes:
      arm_support: Donors, samples and cells per cell type and arm, counting
        only donor-sample-cell-type units with at least `minimum_cells`.
      covariate_balance: Per covariate and arm, donor-level summaries; for
        numeric covariates the standardized mean difference from the
        reference arm, for categorical ones the donors at each level and
        whether that level occurs in only one arm.
      estimand_availability: For each planned contrast and cell type, the
        supported donors in each arm and whether inference is possible.
    """

    arm_support: pd.DataFrame
    covariate_balance: pd.DataFrame
    estimand_availability: pd.DataFrame


def audit_condition_design(
    intake: ConditionCohortIntake,
    spec: ConditionCohortSpec,
    *,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
) -> ConditionDesign:
    """Count support and balance for the declared arms and contrasts.

    A unit is one donor, sample and cell type; it supports a cell type when
    it has at least `minimum_cells` cells. A contrast is estimable in a cell
    type when each arm has at least `minimum_donors` supported donors;
    otherwise it is reported as insufficient, never dropped. A categorical
    level found in only one arm is flagged because adjusting for it removes
    those donors' information about the condition.
    """
    if minimum_cells < 1 or minimum_donors < 2:
        raise ValueError("Require minimum_cells >= 1 and minimum_donors >= 2")

    units = (
        intake.selected_obs.groupby(
            ["condition", "donor_id", "sample_id", "cell_type"],
            observed=True,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )

    arm_support, availability = contrast_availability(
        units.loc[units["n_cells"].ge(minimum_cells)],
        arms=spec.condition.arms,
        contrasts=spec.contrasts,
        minimum_donors=minimum_donors,
    )

    return ConditionDesign(
        arm_support=arm_support,
        covariate_balance=_covariate_balance(intake.samples, spec),
        estimand_availability=availability,
    )


def contrast_availability(
    supported_units: pd.DataFrame,
    *,
    arms: Sequence[str],
    contrasts: Sequence[ContrastSpec],
    minimum_donors: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count supported donors per cell type and arm; decide estimability.

    `supported_units` holds one row per supported donor-sample-cell-type unit
    with `condition`, `donor_id`, `sample_id`, `cell_type` and `n_cells`.
    Use it for derived arms, such as IFN strata, as well as declared ones.

    Returns:
      Arm support per cell type and arm, and availability per contrast and
      cell type.
    """
    arm_support = _arm_support(supported_units, arms)
    return arm_support, _estimand_availability(
        arm_support, contrasts, minimum_donors=minimum_donors
    )


def _arm_support(
    supported: pd.DataFrame,
    arms: Sequence[str],
) -> pd.DataFrame:
    """Count supported donors, samples and cells per cell type and arm."""
    counts = (
        supported.groupby(["cell_type", "condition"], observed=True)
        .agg(
            n_donors=("donor_id", "nunique"),
            n_samples=("sample_id", "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .reset_index()
    )

    index = pd.MultiIndex.from_product(
        [sorted(supported["cell_type"].unique()), list(arms)],
        names=["cell_type", "condition"],
    )

    return (
        counts.set_index(["cell_type", "condition"])
        .reindex(index, fill_value=0)
        .reset_index()
    )


def _covariate_balance(
    samples: pd.DataFrame,
    spec: ConditionCohortSpec,
) -> pd.DataFrame:
    """Summarize covariates per arm at the donor level."""
    rows: list[dict[str, object]] = []
    for covariate in spec.covariates:
        if covariate.kind == "categorical":
            rows.extend(_categorical_balance(samples, covariate.name, spec))
        else:
            rows.extend(_numeric_balance(samples, covariate, spec))
    return pd.DataFrame(rows)


def _numeric_balance(
    samples: pd.DataFrame,
    covariate: CovariateSpec,
    spec: ConditionCohortSpec,
) -> list[dict[str, object]]:
    """Summarize a numeric covariate's donor means per arm, with SMD."""
    reference = spec.condition.reference
    donors = samples.groupby(["condition", "donor_id"], observed=True)[
        covariate.name
    ].mean()
    by_arm = {
        str(arm): values.dropna().to_numpy(dtype=float)
        for arm, values in donors.groupby(level="condition", observed=True)
    }

    rows = []
    for arm in spec.condition.arms:
        values = by_arm.get(arm, np.array([]))
        rows.append(
            {
                "covariate": covariate.name,
                "kind": covariate.kind,
                "condition": arm,
                "level": pd.NA,
                "n_donors": values.size,
                "mean": values.mean() if values.size else np.nan,
                "std": values.std(ddof=1) if values.size > 1 else np.nan,
                "smd_vs_reference": (
                    _standardized_difference(values, by_arm[reference])
                    if arm != reference and reference in by_arm
                    else np.nan
                ),
                "level_in_single_arm": pd.NA,
            }
        )

    return rows


def _standardized_difference(
    values: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return the mean difference over the pooled donor-level SD."""
    if values.size < 2 or reference.size < 2:
        return np.nan
    pooled = np.sqrt((values.var(ddof=1) + reference.var(ddof=1)) / 2.0)
    if pooled == 0:
        return np.nan
    return float((values.mean() - reference.mean()) / pooled)


def _categorical_balance(
    samples: pd.DataFrame,
    name: str,
    spec: ConditionCohortSpec,
) -> list[dict[str, object]]:
    """Count donors per categorical level and arm; flag one-arm levels."""
    levels = (
        samples.groupby([name, "condition"], observed=True)["donor_id"]
        .nunique()
        .unstack("condition", fill_value=0)
        .reindex(columns=list(spec.condition.arms), fill_value=0)
    )
    totals = samples.groupby("condition", observed=True)["donor_id"].nunique()

    rows = []
    for level, counts in levels.iterrows():
        single_arm = int((counts > 0).sum()) == 1
        for arm, n_donors in counts.items():
            rows.append(
                {
                    "covariate": name,
                    "kind": "categorical",
                    "condition": arm,
                    "level": level,
                    "n_donors": int(n_donors),
                    "mean": n_donors / totals.get(arm, np.nan),
                    "std": np.nan,
                    "smd_vs_reference": np.nan,
                    "level_in_single_arm": single_arm,
                }
            )

    return rows


def _estimand_availability(
    arm_support: pd.DataFrame,
    contrasts: Sequence[ContrastSpec],
    *,
    minimum_donors: int,
) -> pd.DataFrame:
    """Decide, per contrast and cell type, whether each arm is supported."""
    donors = arm_support.pivot(
        index="cell_type", columns="condition", values="n_donors"
    )

    rows = []
    for contrast in contrasts:
        for cell_type, counts in donors.iterrows():
            n_target = int(counts[contrast.target])
            n_reference = int(counts[contrast.reference])
            estimable = min(n_target, n_reference) >= minimum_donors
            rows.append(
                {
                    "estimand": contrast.estimand,
                    "role": contrast.role,
                    "target": contrast.target,
                    "reference": contrast.reference,
                    "cell_type": cell_type,
                    "n_target_donors": n_target,
                    "n_reference_donors": n_reference,
                    "status": "estimable" if estimable else "not_estimable",
                    "reason": pd.NA if estimable else "insufficient_donors",
                }
            )

    return pd.DataFrame(rows)
