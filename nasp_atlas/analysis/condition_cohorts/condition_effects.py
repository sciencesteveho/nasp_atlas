"""Condition effects on sensors and modules within each cell type."""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence

import pandas as pd

from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    adjust_fdr_within,
)
from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    donor_regression,
)
from nasp_atlas.analysis.condition_cohorts.specification import (
    ContinuousContrastSpec,
)
from nasp_atlas.analysis.condition_cohorts.specification import ContrastSpec
from nasp_atlas.analysis.condition_cohorts.specification import CovariateSpec
from nasp_atlas.analysis.condition_cohorts.units import SENSOR_MEASURES
from nasp_atlas.analysis.condition_cohorts.units import UNIT_KEYS
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits


logger = logging.getLogger(__name__)


FEATURE_KEYS: tuple[str, ...] = (
    "feature_type",
    "scorer",
    "feature_id",
    "feature_label",
)


def estimate_condition_effects(
    units: ConditionUnits,
    *,
    contrasts: Sequence[ContrastSpec],
    covariates: Sequence[CovariateSpec],
    availability: pd.DataFrame,
    minimum_donors: int = 3,
    adjust_for_depth: bool = True,
    ifn_scorer: str | None = None,
    ifn_leave_self_out: pd.DataFrame | None = None,
    analysis: str = "condition_effects",
) -> pd.DataFrame:
    """Estimate each planned contrast within each cell type, per feature.

    For one contrast, feature and cell type, each donor's supported units
    are averaged and `feature ~ arm + covariates [+ depth] [+ IFN]` is fitted
    by `donor_regression`: ordinary least squares on donor means with HC3
    errors. Every cell type therefore has its own residual variance and
    covariate slopes, which matters because sensor variances differ many-
    fold between cell types. Cell types where either arm lacks donors
    (`availability`) and sensors absent from the gene universe are kept as
    rows with reasons.

    Depth is the unit's log median genes detected, because disease often
    changes depth. With `ifn_scorer`, the unit mean IFN_I_OUTPUT score from
    that scorer is also adjusted for (AB3): the estimate is then the arm
    difference at a fixed IFN-I output level, and IFN_I_OUTPUT itself is not
    fitted. A sensor that is also an IFN_I_OUTPUT gene is instead adjusted
    for the module scored without it, from `ifn_leave_self_out` (see
    `leave_self_out_ifn_scores`), so it is never adjusted for itself; the
    `ifn_covariate` column names the score each row used. BH-FDR is
    computed within one contrast and feature family: module scores per
    scorer, sensor level, or sensor detection.

    Returns:
      One row per contrast, feature and cell type: target-minus-reference
      estimate, HC3 standard error, 95% t interval, raw and BH p-values,
      donor support, fit statistics, covariates used and dropped, status
      and reason.
    """
    numeric = [c.name for c in covariates if c.kind != "categorical"]
    categorical = [c.name for c in covariates if c.kind == "categorical"]
    if adjust_for_depth:
        numeric.append("log_median_genes_detected")
    if ifn_scorer is not None:
        numeric.append("ifn_output")

    tables = []
    for contrast in contrasts:
        cell_types = availability.loc[
            availability["estimand"].eq(contrast.estimand)
            & availability["status"].eq("estimable"),
            "cell_type",
        ].tolist()
        frame = _contrast_frame(
            units,
            contrast,
            cell_types=cell_types,
            ifn_scorer=ifn_scorer,
            ifn_leave_self_out=ifn_leave_self_out,
        )

        fitted = (
            donor_regression(
                frame,
                focal="is_target",
                focal_kind="indicator",
                numeric_covariates=numeric,
                categorical_covariates=categorical,
                feature_keys=FEATURE_KEYS,
                minimum_donors=minimum_donors,
            )
            if not frame.empty
            else pd.DataFrame()
        )

        tables.append(_label_contrast(fitted, contrast, analysis=analysis))
        tables.append(
            _unsupported_rows(
                availability, contrast, units.sensor_mapping, analysis=analysis
            )
        )

    effects = pd.concat(tables, ignore_index=True)
    if ifn_scorer is not None:
        effects = _label_ifn_covariate(effects, ifn_leave_self_out)

    return adjust_fdr_within(effects, ("estimand", "feature_type", "scorer"))


def estimate_continuous_effects(
    units: ConditionUnits,
    *,
    slopes: Sequence[ContinuousContrastSpec],
    covariates: Sequence[CovariateSpec],
    arm_columns: Collection[str] = (),
    minimum_donors: int = 3,
    adjust_for_depth: bool = True,
    analysis: str = "continuous_effects",
) -> pd.DataFrame:
    """Estimate each planned slope within each cell type, per feature.

    For one slope, only donors of its arms with a parsed measurement enter;
    donors without one are counted out of this slope alone. Within each
    cell type, `feature ~ measurement / step + arm + covariates [+ depth]`
    is fitted on donor means by `donor_regression`, so the estimate is the
    change per `step` within arms. A cell type needs `minimum_donors`
    donors with spread in the measurement. BH-FDR is computed within one
    slope and feature family. A covariate read from the slope's own column
    (such as age when the slope is per decade of age) is the measurement
    itself and is left out of that slope's model. Likewise, when the arms
    are defined from the slope's own column (`arm_columns`, such as age
    groups for an age slope), the arm is not adjusted for, because it would
    absorb the slope. The slope is an observational association, not a
    dose-response.
    """
    tables = []
    for slope in slopes:
        adjusted = [c for c in covariates if c.column != slope.column]
        by_arm = len(slope.arms) > 1 and slope.column not in arm_columns
        numeric = [c.name for c in adjusted if c.kind != "categorical"]
        categorical = [c.name for c in adjusted if c.kind == "categorical"]
        if adjust_for_depth:
            numeric.append("log_median_genes_detected")

        fitted = donor_regression(
            _slope_frame(units, slope),
            focal="focal",
            focal_kind="continuous",
            numeric_covariates=numeric,
            categorical_covariates=[
                *categorical,
                *(["condition"] if by_arm else []),
            ],
            feature_keys=FEATURE_KEYS,
            minimum_donors=minimum_donors,
        )

        tables.append(
            fitted.assign(
                analysis=analysis,
                estimand=slope.estimand,
                role=slope.role,
                target=";".join(slope.arms),
                effect_scale=fitted["feature_type"]
                + f"_per_{slope.step:g}_{slope.column}",
            )
        )

    return adjust_fdr_within(
        pd.concat(tables, ignore_index=True),
        ("estimand", "feature_type", "scorer"),
    )


def _slope_frame(
    units: ConditionUnits,
    slope: ContinuousContrastSpec,
) -> pd.DataFrame:
    """Join measured unit features to one slope's design.

    Only supported units of the slope's arms with a parsed measurement
    enter; `focal` is the measurement in units of `slope.step`.
    """
    measure = f"{slope.estimand}_measure"
    metadata = units.metadata
    selected = metadata.loc[
        metadata["supported"]
        & metadata["condition"].isin(slope.arms)
        & metadata[measure].notna()
    ]
    design = selected.assign(focal=selected[measure] / slope.step)

    return units.features.loc[
        units.features["status"].isin(["present", "undetected"]),
        [*UNIT_KEYS, *FEATURE_KEYS, "feature_value"],
    ].merge(design, on=list(UNIT_KEYS), how="inner", validate="many_to_one")


def _contrast_frame(
    units: ConditionUnits,
    contrast: ContrastSpec,
    *,
    cell_types: Sequence[str],
    ifn_scorer: str | None,
    ifn_leave_self_out: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join measured unit features to one contrast's supported design.

    With `ifn_scorer`, every row carries the unit's IFN_I_OUTPUT score as
    `ifn_output`, except sensors scored in `ifn_leave_self_out`, which carry
    the module scored without themselves.
    """
    metadata = units.metadata
    design = metadata.loc[
        metadata["supported"]
        & metadata["condition"].isin([contrast.target, contrast.reference])
        & metadata["cell_type"].isin(cell_types)
    ].assign(is_target=lambda frame: frame["condition"].eq(contrast.target))

    features = units.features
    measured = features["status"].isin(["present", "undetected"])
    if ifn_scorer is not None:
        ifn = features.loc[
            features["feature_label"].eq("IFN_I_OUTPUT")
            & features["scorer"].eq(ifn_scorer),
            [*UNIT_KEYS, "feature_value"],
        ].rename(columns={"feature_value": "ifn_output"})
        design = design.merge(
            ifn, on=list(UNIT_KEYS), how="inner", validate="one_to_one"
        )
        measured &= features["feature_label"].ne("IFN_I_OUTPUT")

    frame = features.loc[
        measured, [*UNIT_KEYS, *FEATURE_KEYS, "feature_value"]
    ].merge(design, on=list(UNIT_KEYS), how="inner", validate="many_to_one")
    if ifn_scorer is None or ifn_leave_self_out is None:
        return frame

    own = frame["feature_type"].isin(SENSOR_MEASURES) & frame[
        "feature_label"
    ].isin(ifn_leave_self_out["feature_label"])

    # Reused tables come back from CSV with inferred types, so numeric-looking
    # donor IDs are compared as text.
    keys = [*UNIT_KEYS, "feature_label"]
    without_self = (
        frame.loc[own, keys]
        .astype(str)
        .merge(
            ifn_leave_self_out[[*keys, "ifn_output"]].astype(
                dict.fromkeys(keys, str)
            ),
            on=keys,
            how="left",
            validate="many_to_one",
        )
    )
    if without_self["ifn_output"].isna().any():
        raise ValueError(
            "Leave-self-out IFN scores are missing for some units; rebuild "
            "them from the same prepared cells"
        )

    frame.loc[own, "ifn_output"] = without_self["ifn_output"].to_numpy()
    return frame


def _label_ifn_covariate(
    effects: pd.DataFrame,
    ifn_leave_self_out: pd.DataFrame | None,
) -> pd.DataFrame:
    """Name the IFN-I score each row was adjusted for.

    Sensors scored without themselves get their leave-self-out definition;
    every other row used the full IFN_I_OUTPUT module.
    """
    own = (
        ifn_leave_self_out.drop_duplicates("feature_label").set_index(
            "feature_label"
        )["score_definition"]
        if ifn_leave_self_out is not None
        else pd.Series(dtype=object)
    )

    is_sensor = effects["feature_type"].isin(SENSOR_MEASURES)
    return effects.assign(
        ifn_covariate=effects["feature_label"]
        .map(own)
        .where(is_sensor)
        .fillna("IFN_I_OUTPUT")
    )


def _label_contrast(
    fitted: pd.DataFrame,
    contrast: ContrastSpec,
    *,
    analysis: str,
) -> pd.DataFrame:
    """Attach the contrast identity and effect scale to fitted rows."""
    if fitted.empty:
        return fitted

    effect_scales = {
        "module_score": "score_difference",
        "gene_expression": "log1p_cp10k_difference",
        "gene_detection": "detected_fraction_difference",
    }
    return fitted.assign(
        analysis=analysis,
        estimand=contrast.estimand,
        role=contrast.role,
        target=contrast.target,
        reference=contrast.reference,
        effect_scale=fitted["feature_type"].map(effect_scales),
    )


def _unsupported_rows(
    availability: pd.DataFrame,
    contrast: ContrastSpec,
    sensor_mapping: pd.DataFrame,
    *,
    analysis: str,
) -> pd.DataFrame:
    """Report cell types and sensors that could not be fitted, with reasons."""
    common = {
        "analysis": analysis,
        "estimand": contrast.estimand,
        "role": contrast.role,
        "target": contrast.target,
        "reference": contrast.reference,
        "estimable": False,
    }

    cell_types = availability.loc[
        availability["estimand"].eq(contrast.estimand)
        & availability["status"].ne("estimable")
    ]
    rows = [
        {
            **common,
            "cell_type": row.cell_type,
            "status": "not_estimable",
            "reason": row.reason,
        }
        for row in cell_types.itertuples(index=False)
    ]

    unmapped = sensor_mapping.loc[
        sensor_mapping["mapping_status"].ne("present")
    ]
    rows.extend(
        {
            **common,
            "feature_type": feature_type,
            "feature_id": row.gene,
            "feature_label": row.gene,
            "status": "unavailable",
            "reason": f"sensor_{row.mapping_status}",
        }
        for row in unmapped.itertuples(index=False)
        for feature_type in ("gene_expression", "gene_detection")
    )

    return pd.DataFrame(rows)
