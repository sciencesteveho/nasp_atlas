"""Equal-weight paired inference on already aggregated biological units."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats  # type: ignore[import]


__all__ = ["PairedContrastResult", "paired_feature_contrasts"]


@dataclass(frozen=True)
class PairedContrastResult:
    """Feature estimates and all matched/unmatched unit diagnostics."""

    estimates: pd.DataFrame
    donor_differences: pd.DataFrame


def paired_feature_contrasts(
    donor_frame: pd.DataFrame,
    *,
    pair_keys: Sequence[str],
    level_key: str,
    target_level: str,
    reference_level: str,
    minimum_pairs: int = 6,
    alpha: float = 0.05,
) -> PairedContrastResult:
    """Estimate mean target-minus-reference differences across complete pairs.

    Inputs are scoped to one scoring context and scorer. Each feature/person/
    level must occur once; repeated units are errors, never averaged again.
    Cell support is the caller's responsibility. Nonfinite or absent sides
    remain in `donor_differences` with reasons and do not contribute to tests.
    Insufficient support or constant differences retain descriptive estimates
    but have missing inferential quantities. Inputs are not modified.

    Args:
      donor_frame: Aggregated values using feature_type, feature_id,
        feature_label and feature_value, plus the explicit identity keys.
      pair_keys: Columns jointly identifying an independent biological unit.
      level_key: Column identifying the two compared contexts.
      target_level: Context whose value is added.
      reference_level: Context whose value is subtracted.
      minimum_pairs: Minimum complete pairs for t-based inference, at least 2.
      alpha: Two-sided error probability for pointwise confidence intervals.

    Returns:
      Unadjusted paired estimates and unit-level values/support diagnostics.
    """
    _validate_paired_input(
        donor_frame,
        pair_keys=pair_keys,
        level_key=level_key,
        target_level=target_level,
        reference_level=reference_level,
        minimum_pairs=minimum_pairs,
        alpha=alpha,
    )
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    selected = donor_frame.loc[
        donor_frame[level_key].isin([target_level, reference_level])
    ]
    estimates: list[dict[str, object]] = []
    pairs: list[pd.DataFrame] = []
    for identity, group in selected.groupby(feature_keys, observed=True):
        aligned = _align_pairs(
            group,
            pair_keys=pair_keys,
            level_key=level_key,
            target_level=target_level,
            reference_level=reference_level,
        )
        feature = dict(zip(feature_keys, map(str, identity), strict=True))
        estimate = _paired_estimate(
            aligned, minimum_pairs=minimum_pairs, alpha=alpha
        )
        estimates.append({**feature, **estimate})
        pairs.append(aligned.assign(**feature))

    estimate_columns = [
        *feature_keys,
        "n_target_donors",
        "n_reference_donors",
        "n_paired_donors",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "degrees_of_freedom",
        "statistic",
        "pvalue",
        "status",
    ]
    pair_columns = [
        *pair_keys,
        "target_value",
        "reference_value",
        "difference",
        "status",
        *feature_keys,
    ]
    estimate_frame = pd.DataFrame(estimates, columns=estimate_columns)
    pair_frame = (
        pd.concat(pairs, ignore_index=True)
        if pairs
        else pd.DataFrame(columns=pair_columns)
    )
    metadata = {
        "comparison_axis": level_key,
        "target_level": target_level,
        "reference_level": reference_level,
    }
    return PairedContrastResult(
        estimates=estimate_frame.assign(**metadata),
        donor_differences=pair_frame.assign(**metadata),
    )


def _validate_paired_input(
    frame: pd.DataFrame,
    *,
    pair_keys: Sequence[str],
    level_key: str,
    target_level: str,
    reference_level: str,
    minimum_pairs: int,
    alpha: float,
) -> None:
    """Reject ambiguous identities, duplicate units and invalid settings."""
    if isinstance(pair_keys, str) or not pair_keys:
        raise ValueError("pair_keys must be a nonempty sequence of columns")
    if len(set(pair_keys)) != len(pair_keys) or level_key in pair_keys:
        raise ValueError("pair_keys must be distinct and exclude level_key")
    if target_level == reference_level:
        raise ValueError("Target and reference levels must differ")
    if isinstance(minimum_pairs, bool) or not isinstance(minimum_pairs, int):
        raise TypeError("minimum_pairs must be an integer")
    if minimum_pairs < 2 or not 0 < alpha < 1:
        raise ValueError("Require minimum_pairs >= 2 and 0 < alpha < 1")

    feature_keys = ["feature_type", "feature_id", "feature_label"]
    identity = [*feature_keys, *pair_keys, level_key]
    if len(set(identity)) != len(identity):
        raise ValueError("Pair and level keys must not overlap feature keys")
    if missing := {*identity, "feature_value"}.difference(frame.columns):
        raise KeyError(f"Paired frame is missing columns: {sorted(missing)}")
    if frame[identity].isna().any(axis=None):
        raise ValueError(
            "Paired feature and biological identity must not be null"
        )
    unit_keys = ["feature_type", "feature_id", *pair_keys, level_key]
    if frame.duplicated(unit_keys).any():
        raise ValueError("Duplicate feature/person/level units in paired frame")
    labels = frame.groupby(["feature_type", "feature_id"])["feature_label"]
    if labels.nunique().gt(1).any():
        raise ValueError("Each feature must have one consistent feature_label")
    pd.to_numeric(frame["feature_value"], errors="raise")


def _align_pairs(
    group: pd.DataFrame,
    *,
    pair_keys: Sequence[str],
    level_key: str,
    target_level: str,
    reference_level: str,
) -> pd.DataFrame:
    """Outer-align contexts and retain a reason for each incomplete pair."""
    keys = list(pair_keys)
    target = group.loc[
        group[level_key].eq(target_level), [*keys, "feature_value"]
    ].rename(columns={"feature_value": "target_value"})
    reference = group.loc[
        group[level_key].eq(reference_level), [*keys, "feature_value"]
    ].rename(columns={"feature_value": "reference_value"})
    aligned = target.merge(
        reference, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    target_values = aligned["target_value"].to_numpy(dtype=float)
    reference_values = aligned["reference_value"].to_numpy(dtype=float)
    finite = np.isfinite(target_values) & np.isfinite(reference_values)
    differences = np.full(len(aligned), np.nan)
    differences[finite] = target_values[finite] - reference_values[finite]
    aligned["difference"] = differences
    aligned["status"] = np.select(
        [
            aligned["_merge"].eq("left_only"),
            aligned["_merge"].eq("right_only"),
            ~finite,
        ],
        ["missing_reference", "missing_target", "nonfinite_pair"],
        default="complete",
    )
    return aligned.drop(columns="_merge")


def _paired_estimate(
    pairs: pd.DataFrame, *, minimum_pairs: int, alpha: float
) -> dict[str, object]:
    """Compute equal-weight mean, t uncertainty and explicit failure state."""
    differences = pairs.loc[
        pairs["status"].eq("complete"), "difference"
    ].to_numpy(dtype=float)
    n_pairs = differences.size
    result: dict[str, object] = {
        "n_target_donors": int(np.isfinite(pairs["target_value"]).sum()),
        "n_reference_donors": int(np.isfinite(pairs["reference_value"]).sum()),
        "n_paired_donors": n_pairs,
        "estimate": float(differences.mean()) if n_pairs else np.nan,
        "standard_error": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "degrees_of_freedom": n_pairs - 1 if n_pairs else np.nan,
        "statistic": np.nan,
        "pvalue": np.nan,
        "status": "insufficient_pairs",
    }
    if n_pairs < minimum_pairs:
        return result

    sample_sd = float(differences.std(ddof=1))
    if sample_sd == 0:
        result["status"] = "degenerate_variance"
        return result

    effect = float(differences.mean())
    standard_error = sample_sd / np.sqrt(n_pairs)
    critical_value = float(stats.t.ppf(1 - alpha / 2, df=n_pairs - 1))
    statistic = effect / standard_error
    result.update(
        standard_error=standard_error,
        ci_lower=effect - critical_value * standard_error,
        ci_upper=effect + critical_value * standard_error,
        statistic=statistic,
        pvalue=float(2 * stats.t.sf(abs(statistic), df=n_pairs - 1)),
        status="ok",
    )
    return result
