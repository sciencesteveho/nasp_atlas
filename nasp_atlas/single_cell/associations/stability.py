"""Cross-stratum stability summaries for continuous associations."""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "summarize_continuous_association_stability",
]


def summarize_continuous_association_stability(
    regression_results: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    min_tested_strata: int = 2,
    fdr_column: str = "ols_pvalue_fdr",
) -> pd.DataFrame:
    """Summarize the consistency of continuous effects across strata.

    The input is the tidy output of `regress_features_on_continuous`. A
    stratum is considered tested only when its regression was not skipped and
    its slope is finite. Significance is defined by `fdr_column` being at or
    below `fdr_threshold`. Direction consistency is the larger of the positive
    and negative slope counts divided by all tested strata; exact zero slopes
    therefore reduce the fraction.

    Summaries with fewer than `min_tested_strata` retain their descriptive
    values, but are marked `skipped=True` and
    `meets_min_tested_strata=False`. This prevents a single tested stratum from
    being interpreted as evidence of cross-stratum consistency.

    Args:
      regression_results: Results from
        `regress_features_on_continuous`, with one row per feature and stratum.
      fdr_threshold: Inclusive FDR threshold used to count significant strata.
      min_tested_strata: Minimum tested strata required for a stability claim.
      fdr_column: Adjusted p-value column used for significance and for
        selecting the best stratum.

    Returns:
      One row per feature and regression context. The table reports tested and
      significant stratum counts, slope summaries, direction counts and
      consistency, median absolute correlations when available, and the
      stratum with the smallest finite FDR.

    Raises:
      KeyError: If a required regression-result column is absent.
      ValueError: If thresholds are invalid, `skipped` is not Boolean, or a
        feature has duplicate rows for a stratum within one context.
    """
    _validate_parameters(
        fdr_threshold=fdr_threshold,
        min_tested_strata=min_tested_strata,
    )
    group_columns = [
        "feature_type",
        "feature_id",
        "feature_label",
        "predictor",
        "statistical_unit",
        "aggregation",
        "stratify_key",
        "analysis_role",
        "fdr_method",
    ]
    required_columns = [
        *group_columns,
        "stratum",
        "slope",
        "skipped",
        fdr_column,
    ]
    if missing := [
        column
        for column in required_columns
        if column not in regression_results.columns
    ]:
        missing_text = ", ".join(missing)
        raise KeyError(f"regression result columns not found: {missing_text}")
    _validate_skipped_column(regression_results["skipped"])

    output_columns = [
        *group_columns,
        "n_strata",
        "n_tested_strata",
        "n_significant_strata",
        "median_slope",
        "min_slope",
        "max_slope",
        "n_positive_strata",
        "n_negative_strata",
        "dominant_direction",
        "direction_consistency_fraction",
        "median_abs_spearman_r",
        "median_abs_pearson_r",
        "best_stratum",
        "best_fdr",
        "fdr_column",
        "fdr_threshold",
        "min_tested_strata",
        "meets_min_tested_strata",
        "skipped",
        "skip_reason",
    ]
    if regression_results.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, object]] = []
    grouped = regression_results.groupby(
        group_columns,
        observed=True,
        dropna=False,
        sort=False,
    )
    for group_values, group in grouped:
        if group["stratum"].duplicated().any():
            feature_id = group["feature_id"].iloc[0]
            raise ValueError(
                f"duplicate stratum rows for feature and context: {feature_id}"
            )
        context = dict(zip(group_columns, group_values, strict=True))
        records.append(
            {
                **context,
                **_summarize_group(
                    group,
                    fdr_threshold=fdr_threshold,
                    min_tested_strata=min_tested_strata,
                    fdr_column=fdr_column,
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=output_columns)


def _validate_parameters(
    *,
    fdr_threshold: float,
    min_tested_strata: int,
) -> None:
    """Validate public stability-summary parameters."""
    if not np.isfinite(fdr_threshold) or not 0.0 <= fdr_threshold <= 1.0:
        raise ValueError("fdr_threshold must be finite and between 0 and 1")
    if (
        isinstance(min_tested_strata, bool)
        or not isinstance(min_tested_strata, int)
        or min_tested_strata < 1
    ):
        raise ValueError("min_tested_strata must be a positive integer")


def _validate_skipped_column(skipped: pd.Series) -> None:
    """Require explicit Boolean skip indicators."""
    if skipped.isna().any() or not skipped.isin([True, False]).all():
        raise ValueError("skipped must contain only Boolean values")


def _summarize_group(
    group: pd.DataFrame,
    *,
    fdr_threshold: float,
    min_tested_strata: int,
    fdr_column: str,
) -> dict[str, object]:
    """Build one feature-context stability record."""
    slopes = pd.to_numeric(group["slope"], errors="coerce").to_numpy(
        dtype=float
    )
    skipped = group["skipped"].to_numpy(dtype=bool)
    tested = ~skipped & np.isfinite(slopes)
    tested_slopes = slopes[tested]
    n_tested = int(tested.sum())
    fdr = pd.to_numeric(group[fdr_column], errors="coerce").to_numpy(
        dtype=float
    )
    significant = tested & np.isfinite(fdr) & (fdr <= fdr_threshold)
    positive_count = int(np.count_nonzero(tested_slopes > 0.0))
    negative_count = int(np.count_nonzero(tested_slopes < 0.0))
    direction, consistency = _direction_summary(
        n_tested=n_tested,
        positive_count=positive_count,
        negative_count=negative_count,
    )
    best_stratum, best_fdr = _best_stratum(group, tested=tested, fdr=fdr)
    meets_minimum = n_tested >= min_tested_strata
    return {
        "n_strata": int(group.shape[0]),
        "n_tested_strata": n_tested,
        "n_significant_strata": int(significant.sum()),
        "median_slope": _finite_summary(tested_slopes, "median"),
        "min_slope": _finite_summary(tested_slopes, "min"),
        "max_slope": _finite_summary(tested_slopes, "max"),
        "n_positive_strata": positive_count,
        "n_negative_strata": negative_count,
        "dominant_direction": direction,
        "direction_consistency_fraction": consistency,
        "median_abs_spearman_r": _median_absolute_correlation(
            group, "spearman_r", tested
        ),
        "median_abs_pearson_r": _median_absolute_correlation(
            group, "pearson_r", tested
        ),
        "best_stratum": best_stratum,
        "best_fdr": best_fdr,
        "fdr_column": fdr_column,
        "fdr_threshold": fdr_threshold,
        "min_tested_strata": min_tested_strata,
        "meets_min_tested_strata": meets_minimum,
        "skipped": not meets_minimum,
        "skip_reason": "" if meets_minimum else "too_few_tested_strata",
    }


def _direction_summary(
    *,
    n_tested: int,
    positive_count: int,
    negative_count: int,
) -> tuple[str, float]:
    """Return the dominant slope direction and its tested-stratum fraction."""
    if n_tested == 0:
        return "not_tested", np.nan
    if positive_count > negative_count:
        direction = "positive"
    elif negative_count > positive_count:
        direction = "negative"
    elif positive_count == 0:
        direction = "zero"
    else:
        direction = "tie"
    consistency = max(positive_count, negative_count) / n_tested
    return direction, float(consistency)


def _finite_summary(values: np.ndarray, statistic: str) -> float:
    """Return a named finite-value summary or NaN for no values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    if statistic == "median":
        return float(np.median(finite))
    if statistic == "min":
        return float(np.min(finite))
    if statistic == "max":
        return float(np.max(finite))
    raise ValueError(f"unknown summary statistic: {statistic}")


def _median_absolute_correlation(
    group: pd.DataFrame,
    column: str,
    tested: np.ndarray,
) -> float:
    """Return median absolute correlation across tested strata when present."""
    if column not in group.columns:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    return _finite_summary(np.abs(values[tested]), "median")


def _best_stratum(
    group: pd.DataFrame,
    *,
    tested: np.ndarray,
    fdr: np.ndarray,
) -> tuple[str, float]:
    """Return the tested stratum with the smallest finite FDR."""
    candidates = tested & np.isfinite(fdr)
    if not candidates.any():
        return "", np.nan
    candidate_indices = np.flatnonzero(candidates)
    best_position = int(candidate_indices[np.argmin(fdr[candidates])])
    stratum = group["stratum"].iloc[best_position]
    best_stratum = "NA" if pd.isna(stratum) else str(stratum)
    return best_stratum, float(fdr[best_position])
