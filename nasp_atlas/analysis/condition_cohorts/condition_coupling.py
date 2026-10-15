"""Within- and between-sample sensor-output coupling compared across arms."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    adjust_fdr_within,
)
from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.analysis.condition_cohorts.specification import ContrastSpec
from nasp_atlas.analysis.condition_cohorts.units import UNIT_KEYS
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits
from nasp_atlas.single_cell.sample_coupling import sample_coupling
from nasp_atlas.single_cell.sample_coupling import sensor_output_pairs


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionCoupling:
    """Per-arm coupling, its variance partition, and arm differences.

    Attributes:
      coupling: `sample_coupling` estimates (within-sample, between-sample
        and contextual slopes) for every sensor-output pair, one block per
        arm, with a `condition` column.
      variance_partition: Donor, sample and residual variance per feature
        and arm.
      differences: Target-minus-reference arm differences of each slope,
        with a 95% interval, z test and BH-FDR within contrast and estimand.
      diagnostics: Per-arm fit diagnostics.
    """

    coupling: pd.DataFrame
    variance_partition: pd.DataFrame
    differences: pd.DataFrame
    diagnostics: pd.DataFrame


def estimate_condition_coupling(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    scorer: str = "scanpy",
    output_modules: Sequence[str] = (
        "IFN_I_OUTPUT",
        "NFKB_CYTOKINE_OUTPUT",
        "ISR",
        "INFLAMMASOME",
    ),
    minimum_donors: int = 3,
    alpha: float = 0.05,
) -> ConditionCoupling:
    """Fit sample coupling within each arm and compare arms per contrast.

    Each arm is fitted separately with `sample_coupling`: sensor mean log1p
    CP10k predicts each output module's unit mean score, a sample is one
    processed sample, and features are centered within cell type and assay.
    A sensor in the output module is excluded rather than fitted. Fitting
    arms separately lets every arm keep its own variance components.

    Arm differences assume the arms share no donor, so their estimates are
    independent and the difference has variance equal to the sum of the
    two. The spec must therefore not allow a condition to vary within a
    donor. A disease-specific gain in within-sample slope suggests coupling
    inside the cell types that change; a gain only between samples suggests
    a shared, sample-wide signal. Both are observational associations.
    """
    if spec.condition.varies_within_donor:
        raise ValueError(
            "Arm-wise coupling differences need donor-disjoint arms; this "
            "spec lets a donor contribute several arms"
        )

    frame = _coupling_frame(units, scorer=scorer)
    sensors, responses, excluded = sensor_output_pairs(frame, output_modules)

    coupling, partition, diagnostics = [], [], []
    for arm in spec.condition.arms:
        arm_frame = frame.loc[frame["condition"].eq(arm)]
        if arm_frame["donor_id"].nunique() < minimum_donors:
            logger.info("Skipping coupling for %s: too few donors", arm)
            continue

        result = sample_coupling(
            arm_frame,
            predictors=sensors,
            responses=responses,
            independent_unit_key="donor_id",
            sample_key="sample_id",
            context_keys=("cell_type", "assay"),
            excluded_pairs=excluded,
            minimum_independent_units=minimum_donors,
        )

        coupling.append(result.coupling.assign(condition=arm))
        partition.append(result.variance_partition.assign(condition=arm))
        diagnostics.append(result.diagnostics.assign(condition=arm))

    coupling_table = pd.concat(coupling, ignore_index=True)
    return ConditionCoupling(
        coupling=coupling_table,
        variance_partition=pd.concat(partition, ignore_index=True),
        differences=_arm_differences(coupling_table, spec, alpha=alpha),
        diagnostics=pd.concat(diagnostics, ignore_index=True),
    )


def _coupling_frame(units: ConditionUnits, *, scorer: str) -> pd.DataFrame:
    """Join supported units' sensor levels and module scores to design."""
    features = units.features
    selected = (
        features["feature_type"].eq("gene_expression")
        & features["status"].isin(["present", "undetected"])
    ) | (
        features["feature_type"].eq("module_score")
        & features["scorer"].eq(scorer)
    )

    metadata = units.metadata.loc[
        units.metadata["supported"], [*UNIT_KEYS, "condition", "assay"]
    ]
    return features.loc[
        selected,
        [
            *UNIT_KEYS,
            "feature_type",
            "feature_id",
            "feature_label",
            "feature_value",
        ],
    ].merge(metadata, on=list(UNIT_KEYS), validate="many_to_one")


def _arm_differences(
    coupling: pd.DataFrame,
    spec: ConditionCohortSpec,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Compare each slope between a contrast's arms, assuming independence."""
    keys = ["pair_id", "predictor", "response", "estimand"]
    columns = [
        *keys,
        "estimate",
        "standard_error",
        "estimable",
        "status",
        "n_independent_units",
    ]
    quantile = float(stats.norm.ppf(1.0 - alpha / 2.0))

    frames = []
    for contrast in spec.contrasts:
        target = coupling.loc[coupling["condition"].eq(contrast.target)]
        reference = coupling.loc[coupling["condition"].eq(contrast.reference)]
        if target.empty or reference.empty:
            continue

        merged = target[columns].merge(
            reference[columns],
            on=keys,
            suffixes=("_target", "_reference"),
            validate="one_to_one",
        )
        frames.append(_slope_differences(merged, contrast, quantile=quantile))

    if not frames:
        return pd.DataFrame()
    return adjust_fdr_within(
        pd.concat(frames, ignore_index=True), ("contrast", "estimand")
    )


def _slope_differences(
    merged: pd.DataFrame,
    contrast: ContrastSpec,
    *,
    quantile: float,
) -> pd.DataFrame:
    """Target minus reference slope with a normal interval, per pair.

    The arms share no donor, so the difference's variance is the sum of the
    two arms' variances. A difference is estimable only when both slopes
    are; a pair excluded in the target arm keeps that status.
    """
    estimable = merged["estimable_target"].astype(bool) & merged[
        "estimable_reference"
    ].astype(bool)

    # Non-estimable fits carry missing values, so force numeric dtype.
    numeric = merged[
        [
            "estimate_target",
            "estimate_reference",
            "standard_error_target",
            "standard_error_reference",
        ]
    ].apply(pd.to_numeric, errors="coerce")

    difference = numeric["estimate_target"] - numeric["estimate_reference"]
    error = (
        numeric["standard_error_target"] ** 2
        + numeric["standard_error_reference"] ** 2
    ) ** 0.5
    statistic = difference / error

    return merged.assign(
        contrast=contrast.estimand,
        role=contrast.role,
        target=contrast.target,
        reference=contrast.reference,
        difference=difference.where(estimable),
        standard_error=error.where(estimable),
        ci_low=(difference - quantile * error).where(estimable),
        ci_high=(difference + quantile * error).where(estimable),
        statistic=statistic.where(estimable),
        pvalue=pd.Series(
            2.0 * stats.norm.sf(np.abs(statistic)), index=merged.index
        ).where(estimable),
        estimable=estimable,
        status=np.select(
            [estimable, merged["status_target"].eq("excluded")],
            ["ok", "excluded"],
            default="not_estimable",
        ),
        n_independent_units=merged["n_independent_units_target"]
        + merged["n_independent_units_reference"],
    )
