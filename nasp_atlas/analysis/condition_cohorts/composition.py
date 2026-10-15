"""Cell-type composition and its share of tissue-level condition shifts."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nasp_atlas.analysis.condition_cohorts.condition_effects import FEATURE_KEYS
from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    adjust_fdr_within,
)
from nasp_atlas.analysis.condition_cohorts.donor_regression import (
    donor_regression,
)
from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.analysis.condition_cohorts.specification import ContrastSpec
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionComposition:
    """Composition shifts and the composition-within split of tissue shifts.

    Attributes:
      proportions: One row per sample and cell type: cells, sample share and
        centred log-ratio (CLR) share. Descriptive.
      effects: Target-minus-reference CLR difference per contrast and cell
        type from donor-level models, with BH-FDR within contrast.
      decomposition: Per contrast and feature, the tissue-level shift and its
        composition and within-cell-type parts, with donor-bootstrap
        percentile intervals. Descriptive.
      contributions: The same parts split by cell type (point estimates).
    """

    proportions: pd.DataFrame
    effects: pd.DataFrame
    decomposition: pd.DataFrame
    contributions: pd.DataFrame


def estimate_condition_composition(
    units: ConditionUnits,
    spec: ConditionCohortSpec,
    *,
    scorer: str = "scanpy",
    minimum_donors: int = 3,
    pseudocount: float = 0.5,
    n_bootstrap: int = 1_000,
    random_state: int = 42,
) -> ConditionComposition:
    """Compare cell-type composition between arms and split tissue shifts.

    Composition uses every selected cell, including units below the support
    minimum. A sample's share of cell type k is its cells of k over all its
    cells; the CLR share adds `pseudocount` cells to every type so absent
    types stay finite. CLR shares are compared per planned contrast and cell
    type with `donor_regression` (donor means, sample covariates, HC3).

    For each feature, the tissue-level value of a sample is the
    cell-weighted mean over all its cells, the sum over k of share times
    unit mean. With arm means P_k of the share and m_k of the unit mean
    (cell-weighted over donors), the arm difference splits exactly
    (Kitagawa) into a composition part, sum of dP_k (m_k^T + m_k^R) / 2,
    and a within part, sum of (P_k^T + P_k^R) / 2 dm_k. Donors are
    resampled within arms with `random_state` for percentile intervals.
    The split is descriptive: it says whether a tissue-level shift tracks a
    change in which cells are present or a change inside cell types.
    """
    if spec.condition.varies_within_donor:
        raise ValueError(
            "Composition contrasts need donor-disjoint arms; this spec lets "
            "a donor contribute several arms"
        )

    proportions = _sample_proportions(units.metadata, pseudocount=pseudocount)
    rng = np.random.default_rng(random_state)
    effects, decompositions, contributions = [], [], []

    for contrast in spec.contrasts:
        effects.append(
            _clr_effects(
                proportions, spec, contrast, minimum_donors=minimum_donors
            )
        )
        split, parts = _decompose(
            units,
            proportions,
            contrast,
            scorer=scorer,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        decompositions.append(split)
        contributions.append(parts)

    return ConditionComposition(
        proportions=proportions,
        effects=adjust_fdr_within(
            pd.concat(effects, ignore_index=True), ("estimand",)
        ),
        decomposition=pd.concat(decompositions, ignore_index=True).assign(
            n_bootstrap=n_bootstrap, random_state=random_state
        ),
        contributions=pd.concat(contributions, ignore_index=True),
    )


def _sample_proportions(
    metadata: pd.DataFrame,
    *,
    pseudocount: float,
) -> pd.DataFrame:
    """Return every sample's cell-type shares, absent types as zero."""
    design = [
        column
        for column in metadata.columns
        if column not in {"cell_type", "n_cells", "supported"}
        and not column.endswith("genes_detected")
    ]

    counts = metadata.pivot_table(
        index="sample_id",
        columns="cell_type",
        values="n_cells",
        aggfunc="sum",
        fill_value=0,
    )

    shares = counts.div(counts.sum(axis=1), axis=0)
    logs = pd.DataFrame(
        np.log(counts.to_numpy() + pseudocount),
        index=counts.index,
        columns=counts.columns,
    )
    clr = logs.sub(logs.mean(axis=1), axis=0)

    samples = metadata[design].drop_duplicates("sample_id")
    return (
        pd.concat(
            {
                "n_cells": counts.stack(),
                "proportion": shares.stack(),
                "clr_proportion": clr.stack(),
            },
            axis=1,
        )
        .reset_index()
        .merge(samples, on="sample_id", validate="many_to_one")
    )


def _clr_effects(
    proportions: pd.DataFrame,
    spec: ConditionCohortSpec,
    contrast: ContrastSpec,
    *,
    minimum_donors: int,
) -> pd.DataFrame:
    """Fit CLR shares per cell type for one contrast on donor means."""
    frame = proportions.loc[
        proportions["condition"].isin([contrast.target, contrast.reference])
    ].assign(
        is_target=lambda table: table["condition"].eq(contrast.target),
        feature_type="composition_clr",
        feature_id="clr_proportion",
        feature_label="clr_proportion",
        feature_value=lambda table: table["clr_proportion"],
    )

    fitted = donor_regression(
        frame,
        focal="is_target",
        focal_kind="indicator",
        numeric_covariates=[
            c.name for c in spec.covariates if c.kind != "categorical"
        ],
        categorical_covariates=[
            c.name for c in spec.covariates if c.kind == "categorical"
        ],
        minimum_donors=minimum_donors,
    )

    reference_share = _arm_mean_share(
        frame.loc[~frame["is_target"]], "reference_mean_proportion"
    )
    target_share = _arm_mean_share(
        frame.loc[frame["is_target"]], "target_mean_proportion"
    )

    return (
        fitted.merge(reference_share.reset_index(), on="cell_type")
        .merge(target_share.reset_index(), on="cell_type")
        .assign(
            analysis="composition",
            estimand=contrast.estimand,
            role=contrast.role,
            target=contrast.target,
            reference=contrast.reference,
            effect_scale="clr_proportion_difference",
        )
    )


def _decompose(
    units: ConditionUnits,
    proportions: pd.DataFrame,
    contrast: ContrastSpec,
    *,
    scorer: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each feature's tissue-level arm difference into two parts."""
    matrices = _donor_matrices(units, proportions, contrast, scorer=scorer)
    is_target = matrices.donors["condition"].eq(contrast.target).to_numpy()

    point, (composition, within) = _kitagawa(
        matrices.shares,
        matrices.contributions,
        is_target,
        np.arange(is_target.size),
    )

    low, high = _bootstrap_intervals(
        matrices, is_target, n_bootstrap=n_bootstrap, rng=rng
    )

    labels = ("total", "composition", "within")
    split = matrices.features.assign(
        estimand=contrast.estimand,
        role=contrast.role,
        target=contrast.target,
        reference=contrast.reference,
        n_target_donors=int(is_target.sum()),
        n_reference_donors=int((~is_target).sum()),
        **{label: point[:, index] for index, label in enumerate(labels)},
        **{f"{label}_ci_low": low[:, i] for i, label in enumerate(labels)},
        **{f"{label}_ci_high": high[:, i] for i, label in enumerate(labels)},
    )

    parts = _contribution_parts(matrices, composition, within).assign(
        estimand=contrast.estimand
    )
    return split, parts


@dataclass(frozen=True)
class _DonorMatrices:
    """Donor-level shares and share-weighted means for one contrast.

    Attributes:
      donors: Donor identities and arms, in matrix row order.
      cell_types: Cell types, in matrix column order.
      features: Feature identities, in the first axis of `contributions`.
      shares: Donors by cell types: mean sample share.
      contributions: Features by donors by cell types: mean over the donor's
        samples of share times unit mean (zero where the type is absent).
    """

    donors: pd.DataFrame
    cell_types: list[str]
    features: pd.DataFrame
    shares: np.ndarray
    contributions: np.ndarray


def _donor_matrices(
    units: ConditionUnits,
    proportions: pd.DataFrame,
    contrast: ContrastSpec,
    *,
    scorer: str,
) -> _DonorMatrices:
    """Average sample shares and share-weighted unit means per donor."""
    share = proportions.loc[
        proportions["condition"].isin([contrast.target, contrast.reference]),
        ["sample_id", "donor_id", "condition", "cell_type", "proportion"],
    ]

    donors = (
        share[["donor_id", "condition"]]
        .drop_duplicates()
        .sort_values("donor_id")
        .reset_index(drop=True)
    )

    cell_types = sorted(share["cell_type"].unique())
    shares = (
        share.pivot_table(
            index="donor_id",
            columns="cell_type",
            values="proportion",
            aggfunc="mean",
        )
        .reindex(index=donors["donor_id"], columns=cell_types)
        .fillna(0.0)
        .to_numpy()
    )

    identities, contributions = _donor_contributions(
        units, share, donors=donors, cell_types=cell_types, scorer=scorer
    )
    return _DonorMatrices(
        donors=donors,
        cell_types=cell_types,
        features=identities,
        shares=shares,
        contributions=contributions,
    )


def _arm_mean_share(rows: pd.DataFrame, name: str) -> pd.Series:
    """Mean cell-type share over donors, each donor averaged over samples."""
    return (
        rows.groupby(["cell_type", "donor_id"], observed=True)["proportion"]
        .mean()
        .groupby("cell_type", observed=True)
        .mean()
        .rename(name)
    )


def _bootstrap_intervals(
    matrices: _DonorMatrices,
    is_target: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile 95% intervals of the split, resampling donors within arms."""
    target_rows = np.flatnonzero(is_target)
    reference_rows = np.flatnonzero(~is_target)

    draws = np.stack(
        [
            _kitagawa(
                matrices.shares,
                matrices.contributions,
                is_target,
                np.concatenate(
                    [
                        rng.choice(target_rows, target_rows.size),
                        rng.choice(reference_rows, reference_rows.size),
                    ]
                ),
            )[0]
            for _ in range(n_bootstrap)
        ]
    )

    low, high = np.nanquantile(draws, [0.025, 0.975], axis=0)
    return low, high


def _contribution_parts(
    matrices: _DonorMatrices,
    composition: np.ndarray,
    within: np.ndarray,
) -> pd.DataFrame:
    """Long table of each cell type's composition and within contribution."""
    return pd.concat(
        [
            matrices.features.join(
                pd.DataFrame(values, columns=matrices.cell_types)
            )
            .melt(
                id_vars=list(FEATURE_KEYS),
                var_name="cell_type",
                value_name="value",
            )
            .assign(part=name)
            for name, values in (
                ("composition", composition),
                ("within", within),
            )
        ],
        ignore_index=True,
    )


def _donor_contributions(
    units: ConditionUnits,
    share: pd.DataFrame,
    *,
    donors: pd.DataFrame,
    cell_types: list[str],
    scorer: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature identities and their donor x cell-type share-weighted sums.

    Each sample's share times unit value is averaged over the donor's
    samples, so a donor's contributions sum to its tissue-level value.
    """
    features = units.features
    selected = features.loc[
        features["status"].isin(["present", "undetected"])
        & (
            features["feature_type"].isin(["gene_expression", "gene_detection"])
            | features["scorer"].eq(scorer)
        )
    ]
    weighted = selected.merge(
        share, on=["sample_id", "donor_id", "cell_type"], validate="many_to_one"
    )

    n_samples = share.groupby("donor_id")["sample_id"].nunique()
    weighted = weighted.assign(
        product=weighted["proportion"]
        * weighted["feature_value"]
        / weighted["donor_id"].map(n_samples)
    )

    keys = list(FEATURE_KEYS)
    sums = (
        weighted.groupby(
            [*keys, "donor_id", "cell_type"], dropna=False, observed=True
        )["product"]
        .sum()
        .reset_index()
    )
    identities = sums[keys].drop_duplicates().reset_index(drop=True)
    feature_code = pd.MultiIndex.from_frame(identities.fillna("")).get_indexer(
        pd.MultiIndex.from_frame(sums[keys].fillna(""))
    )

    contributions = np.zeros((len(identities), len(donors), len(cell_types)))
    contributions[
        feature_code,
        pd.Index(donors["donor_id"]).get_indexer(pd.Index(sums["donor_id"])),
        pd.Index(cell_types).get_indexer(pd.Index(sums["cell_type"])),
    ] = sums["product"].to_numpy()
    return identities, contributions


def _kitagawa(
    shares: np.ndarray,
    contributions: np.ndarray,
    is_target: np.ndarray,
    rows: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return total, composition and within parts for donors `rows`.

    `shares` is donors by cell types; `contributions` is features by donors
    by cell types. Returns a features-by-3 array and the per-cell-type
    composition and within parts (features by cell types).
    """
    rows = np.asarray(rows)
    target = rows[is_target[rows]]
    reference = rows[~is_target[rows]]

    share_t = shares[target].mean(axis=0)
    share_r = shares[reference].mean(axis=0)
    sum_t = contributions[:, target].mean(axis=1)
    sum_r = contributions[:, reference].mean(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_t = np.where(share_t > 0, sum_t / share_t, np.nan)
        mean_r = np.where(share_r > 0, sum_r / share_r, np.nan)
    # A type absent from one arm takes the other arm's mean, which keeps the
    # split exact: its whole difference is then composition.
    mean_t = np.where(np.isnan(mean_t), mean_r, mean_t)
    mean_r = np.where(np.isnan(mean_r), mean_t, mean_r)
    mean_t, mean_r = np.nan_to_num(mean_t), np.nan_to_num(mean_r)

    composition = (share_t - share_r) * (mean_t + mean_r) / 2.0
    within = (share_t + share_r) / 2.0 * (mean_t - mean_r)

    totals = np.column_stack(
        [
            sum_t.sum(axis=1) - sum_r.sum(axis=1),
            composition.sum(axis=1),
            within.sum(axis=1),
        ]
    )
    return totals, (composition, within)
