"""Altered versus reference cell states compared within donor (AB5)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from nasp_atlas.analysis.condition_cohorts.condition_effects import FEATURE_KEYS
from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    adjust_fdr_within,
)
from nasp_atlas.analysis.condition_cohorts.donor_regression import fit_table
from nasp_atlas.analysis.condition_cohorts.donor_regression import ols_hc3
from nasp_atlas.analysis.condition_cohorts.specification import CellStateSpec
from nasp_atlas.analysis.condition_cohorts.specification import ContrastSpec
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits


logger = logging.getLogger(__name__)


def contrast_cell_states(
    state_units: ConditionUnits,
    states: CellStateSpec,
    *,
    arms: tuple[str, ...],
    minimum_pairs: int = 6,
    adjust_for_depth: bool = True,
) -> pd.DataFrame:
    """Compare altered and reference states of a cell type within donors.

    `state_units` are units keyed by donor, sample, cell type and the state
    column. Within a donor and cell type, the supported units of each state
    group are pooled with cell weights; the donor's paired difference is
    altered minus reference. Per arm and cell type, the mean paired
    difference is estimated by OLS with HC3 errors. With
    `adjust_for_depth`, the paired difference in log median genes detected
    is a covariate and the estimate is the state difference at equal depth,
    because altered (for example degenerative) cells often have fewer genes
    detected. Each donor is its own control, so donor-level confounders
    cancel. A cell type needs `minimum_pairs` donors with both states. BH-FDR
    is computed within one arm and feature family.

    Returns:
      One row per arm, cell type and feature: estimate (altered minus
      reference), HC3 error, t interval, raw and BH p-values, pairs, fit
      statistics, status and reason.
    """
    pairs = donor_state_pairs(state_units, states)

    records = []
    for (arm, cell_type, *feature), group in pairs.groupby(
        ["condition", "cell_type", *FEATURE_KEYS], dropna=False, observed=True
    ):
        if arm not in arms:
            continue

        identity = {
            "arm": arm,
            "cell_type": cell_type,
            **dict(zip(FEATURE_KEYS, feature, strict=True)),
            "n_pairs": len(group),
        }

        if len(group) < minimum_pairs:
            records.append(
                {**identity, "estimable": False, "status": "insufficient_pairs"}
            )
            continue

        columns = [np.ones(len(group))]
        if adjust_for_depth:
            columns.append(group["depth_difference"].to_numpy())
        records.append(
            {
                **identity,
                **ols_hc3(
                    np.column_stack(columns),
                    group["difference"].to_numpy(),
                    coefficient=0,
                ),
            }
        )

    effects = fit_table(records).assign(
        analysis="cell_state_effects",
        estimand=lambda frame: (
            "state_" + frame["arm"] + "_altered_vs_reference"
        ),
        state_column=states.column,
        reference_state=states.reference,
        altered_states=";".join(states.altered),
        effect_scale=lambda frame: frame["feature_type"] + "_state_difference",
    )
    return adjust_fdr_within(effects, ("arm", "feature_type", "scorer"))


def contrast_state_differences(
    state_units: ConditionUnits,
    states: CellStateSpec,
    *,
    contrasts: Sequence[ContrastSpec],
    minimum_pairs: int = 6,
    adjust_for_depth: bool = True,
) -> pd.DataFrame:
    """Compare the within-donor state difference between two arms.

    Each donor's paired difference (altered minus reference, built as in
    `contrast_cell_states`) is the response. Per contrast, cell type and
    feature, `difference ~ target arm + depth difference` is fitted over
    the donors of both arms by OLS with HC3 errors. The estimate is the
    target arm's mean paired difference minus the reference arm's, a
    difference in differences, such as the change with weight loss in
    remitters minus that in non-remitters. Stable donor traits cancel
    within each donor, and anything shared by both arms' visits cancels
    between arms. Arms must not share donors; each needs `minimum_pairs`
    donors with both states. BH-FDR is computed within one contrast and
    feature family.

    Returns:
      One row per contrast, cell type and feature: estimate (target minus
      reference difference in state differences), HC3 error, t interval,
      raw and BH p-values, pairs per arm, fit statistics, status and reason.
    """
    pairs = donor_state_pairs(state_units, states)

    records = []
    for contrast in contrasts:
        compared = pairs.loc[
            pairs["condition"].isin([contrast.target, contrast.reference])
        ]

        for (cell_type, *feature), group in compared.groupby(
            ["cell_type", *FEATURE_KEYS], dropna=False, observed=True
        ):
            in_target = group["condition"].eq(contrast.target).to_numpy()
            identity = {
                "estimand": contrast.estimand,
                "role": contrast.role,
                "target": contrast.target,
                "reference": contrast.reference,
                "cell_type": cell_type,
                **dict(zip(FEATURE_KEYS, feature, strict=True)),
                "n_pairs": len(group),
                "n_level_donors": int(in_target.sum()),
                "n_reference_donors": int((~in_target).sum()),
            }

            if min(in_target.sum(), (~in_target).sum()) < minimum_pairs:
                records.append(
                    {
                        **identity,
                        "estimable": False,
                        "status": "insufficient_pairs",
                    }
                )
                continue

            columns = [np.ones(len(group)), in_target.astype(float)]
            if adjust_for_depth:
                columns.append(group["depth_difference"].to_numpy())
            records.append(
                {
                    **identity,
                    **ols_hc3(
                        np.column_stack(columns),
                        group["difference"].to_numpy(),
                        coefficient=1,
                    ),
                }
            )

    effects = fit_table(records).assign(
        analysis="state_arm_effects",
        state_column=states.column,
        reference_state=states.reference,
        altered_states=";".join(states.altered),
        effect_scale=lambda frame: (
            frame["feature_type"] + "_state_difference_target_minus_reference"
        ),
    )
    return adjust_fdr_within(effects, ("estimand", "feature_type", "scorer"))


def donor_state_pairs(
    state_units: ConditionUnits,
    states: CellStateSpec,
) -> pd.DataFrame:
    """Return each donor's altered-minus-reference difference per feature.

    Only supported units of a declared state with a measured feature enter.
    Within a donor and cell type, each state's units are pooled with cell
    weights; donors without both states are left out.

    Returns:
      One row per arm (`condition`), donor, cell type and feature with the
      paired `difference` and the paired `depth_difference` in log median
      genes detected.
    """
    groups = {
        states.reference: "reference",
        **dict.fromkeys(states.altered, "altered"),
    }

    metadata = state_units.metadata.assign(
        state_group=lambda frame: frame[states.column].map(groups)
    )
    metadata = metadata.loc[
        metadata["supported"] & metadata["state_group"].notna()
    ]

    keys = ["donor_id", "sample_id", "cell_type", states.column]
    measured = state_units.features.loc[
        state_units.features["status"].isin(["present", "undetected"]),
        [*keys, *FEATURE_KEYS, "feature_value"],
    ].merge(
        metadata[
            [
                *keys,
                "condition",
                "state_group",
                "n_cells",
                "log_median_genes_detected",
            ]
        ],
        on=keys,
        validate="many_to_one",
    )

    pairs = _paired_differences(measured)
    if pairs.empty:
        raise ValueError(
            f"No donor has supported {states.reference!r} and "
            f"{list(states.altered)!r} units in {states.column!r}; check the "
            "cell_state labels"
        )
    return pairs


def _paired_differences(measured: pd.DataFrame) -> pd.DataFrame:
    """Return each donor's altered-minus-reference difference per feature."""
    keys = ["condition", "donor_id", "cell_type", *FEATURE_KEYS]
    weighted = measured.assign(
        value_cells=measured["feature_value"] * measured["n_cells"],
        depth_cells=measured["log_median_genes_detected"] * measured["n_cells"],
    )

    pooled = weighted.groupby(
        [*keys, "state_group"], dropna=False, observed=True
    )[["value_cells", "depth_cells", "n_cells"]].sum()
    pooled = pooled.assign(
        value=pooled["value_cells"] / pooled["n_cells"],
        depth=pooled["depth_cells"] / pooled["n_cells"],
    )[["value", "depth"]].unstack("state_group")

    complete = pooled.dropna()
    return pd.DataFrame(
        {
            "difference": complete[("value", "altered")]
            - complete[("value", "reference")],
            "depth_difference": complete[("depth", "altered")]
            - complete[("depth", "reference")],
        }
    ).reset_index()
