"""Pre-specified IFN-I output strata within a disease arm."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nasp_atlas.analysis.condition_cohorts.specification import ContrastSpec
from nasp_atlas.analysis.condition_cohorts.specification import IfnStrataSpec
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IfnStrata:
    """Disease donors split by IFN-I output, with the derived contrasts.

    Attributes:
      units: Units whose disease-arm `condition` is replaced by
        "<arm>_ifn_high" or "<arm>_ifn_low"; donors without the stratifying
        cell type become "<arm>_unassigned" and enter no contrast.
      donors: One row per reference or disease donor: donor IFN-I output in
        the stratifying cell type, the reference threshold and the stratum.
      contrasts: Low stratum minus reference, and high minus low stratum.
    """

    units: ConditionUnits
    donors: pd.DataFrame
    contrasts: tuple[ContrastSpec, ...]


@dataclass(frozen=True)
class IfnStrataResult:
    """Published IFN-strata tables.

    Attributes:
      donors: One row per reference or disease donor with its IFN-I output,
        the reference threshold and its stratum.
      contrasts: Effect rows for IFN-low versus reference and IFN-high
        versus IFN-low.
    """

    donors: pd.DataFrame
    contrasts: pd.DataFrame


def stratify_by_ifn(
    units: ConditionUnits,
    strata: IfnStrataSpec,
    *,
    reference: str,
) -> IfnStrata:
    """Split a disease arm by IFN-I output, with a threshold from references.

    A donor's IFN-I output is the mean, over its supported units of
    `strata.cell_type`, of the unit mean IFN_I_OUTPUT score from
    `strata.scorer`. The threshold is the `strata.quantile` of reference
    donors only, so it is fixed before any disease value or sensor result
    is seen. Disease donors above it are IFN-high; the rest are IFN-low.
    Sensors shifted in IFN-low donors differ from references without a
    detectable IFN-I state.
    """
    metadata = units.metadata
    donor_ifn = _donor_ifn_output(units, strata)
    reference_values = donor_ifn.xs(reference, level="condition")
    if reference_values.size < 3:
        raise ValueError(
            f"Only {reference_values.size} reference donors have supported "
            f"{strata.cell_type} units; cannot set an IFN threshold"
        )
    threshold = float(np.quantile(reference_values, strata.quantile))

    donors = _assign_strata(
        metadata, donor_ifn, strata, reference=reference, threshold=threshold
    )

    stratum = donors.set_index("donor_id")["stratum"]
    relabeled = metadata.copy()
    in_arm = relabeled["condition"].eq(strata.arm)
    relabeled.loc[in_arm, "condition"] = relabeled.loc[in_arm, "donor_id"].map(
        stratum
    )

    logger.info(
        "IFN strata (threshold %.3f): %s",
        threshold,
        donors["stratum"].value_counts().to_dict(),
    )

    low, high = f"{strata.arm}_ifn_low", f"{strata.arm}_ifn_high"
    return IfnStrata(
        units=ConditionUnits(
            metadata=relabeled,
            features=units.features,
            sensor_mapping=units.sensor_mapping,
        ),
        donors=donors,
        contrasts=(
            ContrastSpec(
                estimand=f"{low}_vs_{reference}",
                target=low,
                reference=reference,
                role="primary",
            ),
            ContrastSpec(
                estimand=f"{high}_vs_{low}",
                target=high,
                reference=low,
                role="primary",
            ),
        ),
    )


def _donor_ifn_output(
    units: ConditionUnits, strata: IfnStrataSpec
) -> pd.Series:
    """Mean unit IFN_I_OUTPUT per donor and arm in the stratifying cell type."""
    metadata = units.metadata
    ifn = units.features.loc[
        units.features["feature_label"].eq("IFN_I_OUTPUT")
        & units.features["scorer"].eq(strata.scorer),
        ["donor_id", "sample_id", "cell_type", "feature_value"],
    ]

    stratifying = ifn.merge(
        metadata.loc[
            metadata["supported"] & metadata["cell_type"].eq(strata.cell_type),
            ["donor_id", "sample_id", "cell_type", "condition"],
        ],
        on=["donor_id", "sample_id", "cell_type"],
        validate="one_to_one",
    )

    return stratifying.groupby(["donor_id", "condition"], observed=True)[
        "feature_value"
    ].mean()


def _assign_strata(
    metadata: pd.DataFrame,
    donor_ifn: pd.Series,
    strata: IfnStrataSpec,
    *,
    reference: str,
    threshold: float,
) -> pd.DataFrame:
    """Label reference donors and IFN-high, IFN-low or unassigned patients.

    A disease donor without the stratifying cell type has no IFN value and
    is unassigned, so it enters no stratum contrast.
    """
    donors = (
        metadata.loc[
            metadata["condition"].isin([strata.arm, reference]),
            ["donor_id", "condition"],
        ]
        .drop_duplicates()
        .merge(
            donor_ifn.rename("ifn_output").reset_index(),
            on=["donor_id", "condition"],
            how="left",
            validate="one_to_one",
        )
    )

    donors["threshold"] = threshold
    donors["stratum"] = np.select(
        [
            donors["condition"].eq(reference),
            donors["ifn_output"].isna(),
            donors["ifn_output"].gt(threshold),
        ],
        [reference, f"{strata.arm}_unassigned", f"{strata.arm}_ifn_high"],
        default=f"{strata.arm}_ifn_low",
    )
    return donors
