"""Continuous association tests for aggregated feature frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

from nasp_atlas.single_cell.associations.core import DESCRIPTIVE_UNITS
from nasp_atlas.single_cell.associations.core import MIN_UNITS_FOR_TEST
from nasp_atlas.single_cell.associations.core import ObsSchema
from nasp_atlas.single_cell.associations.core import _linregress_values
from nasp_atlas.single_cell.associations.core import _statistic_pvalue
from nasp_atlas.single_cell.associations.core import benjamini_hochberg


__all__ = [
    "partial_correlation_controlling_tissue",
    "regress_features_on_continuous",
]


def _continuous_pair_stats(
    predictor: pd.Series,
    response: pd.Series,
) -> dict[str, float]:
    """Correlate and OLS-fit a response against a continuous predictor.

    Guards mirror the transcriptomic-clock precedent: fewer than three valid
    paired points, or a constant predictor or response, yield NaN statistics
    rather than raising.

    Args:
    predictor: Continuous predictor values (for example donor age).
    response: Aligned feature values.

    Returns:
    Mapping with paired `n`, Pearson and Spearman coefficients and their
    p-values, and OLS `slope`/`intercept`/`ols_pvalue`.
    """
    mask = predictor.notna() & response.notna()
    x = predictor[mask].to_numpy(dtype=float)
    y = response[mask].to_numpy(dtype=float)
    n_points = int(x.size)
    nan_result = {
        "n": float(n_points),
        "pearson_r": np.nan,
        "pearson_pvalue": np.nan,
        "spearman_r": np.nan,
        "spearman_pvalue": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "ols_pvalue": np.nan,
    }
    if (
        n_points < MIN_UNITS_FOR_TEST
        or np.unique(x).size < 2
        or np.unique(y).size < 2
    ):
        return nan_result
    pearson_r, pearson_pvalue = _statistic_pvalue(stats.pearsonr(x, y))
    spearman_r, spearman_pvalue = _statistic_pvalue(stats.spearmanr(x, y))
    slope, intercept, ols_pvalue = _linregress_values(stats.linregress(x, y))
    return {
        "n": float(n_points),
        "pearson_r": float(pearson_r),
        "pearson_pvalue": float(pearson_pvalue),
        "spearman_r": float(spearman_r),
        "spearman_pvalue": float(spearman_pvalue),
        "slope": float(slope),
        "intercept": float(intercept),
        "ols_pvalue": float(ols_pvalue),
    }


def regress_features_on_continuous(
    unit_frame: pd.DataFrame,
    *,
    predictor_key: str,
    stratify_key: str | None = None,
    fdr_method: str = "benjamini_hochberg",
) -> pd.DataFrame:
    """Regress each feature on a continuous predictor at the unit level.

    Operates on an already-aggregated unit frame so that donor-level predictors
    are tested with one row per donor unit, never per cell. Cell-unit input is
    still accepted but marked descriptive rather than inferential. Features with
    too few units or a constant predictor/response are skipped with a reason.

    Args:
    unit_frame: Aggregated frame from `aggregate_feature_frame` carrying
      `feature_value` and the predictor column.
    predictor_key: Continuous obs column used as the predictor (for example
      the age key or an eQTL-count column).
    stratify_key: Optional column to run the regression within each stratum.
      Missing values form an explicit "NA" stratum.
    fdr_method: FDR method label recorded in output; only Benjamini-Hochberg is
      implemented.

    Returns:
    One row per (feature, stratum) with correlation and OLS statistics, the
    statistical unit, aggregation, an `analysis_role` of `descriptive` or
    `inferential`, FDR-adjusted p-values, and `skipped`/`skip_reason`.
    """
    if predictor_key not in unit_frame.columns:
        raise KeyError(f"predictor column not found: {predictor_key}")
    records: list[dict[str, object]] = []
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    strata_values = (
        unit_frame[stratify_key].astype("string").fillna("NA")
        if stratify_key is not None
        else None
    )
    strata = (
        [None] if strata_values is None else strata_values.unique().tolist()
    )
    for stratum in strata:
        scoped = (
            unit_frame
            if strata_values is None
            else unit_frame[strata_values == stratum]
        )
        records.extend(
            _continuous_regression_record(
                feature_values,
                group,
                predictor_key=predictor_key,
                stratify_key=stratify_key,
                stratum=stratum,
                fdr_method=fdr_method,
            )
            for feature_values, group in scoped.groupby(
                feature_keys, observed=True
            )
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result

    return _adjust_tested_pvalues(
        result,
        ("pearson_pvalue", "spearman_pvalue", "ols_pvalue"),
    )


def _continuous_regression_record(
    feature_values: tuple[object, ...],
    group: pd.DataFrame,
    *,
    predictor_key: str,
    stratify_key: str | None,
    stratum: object | None,
    fdr_method: str,
) -> dict[str, object]:
    """Fit one feature-stratum pair and record why it is non-estimable."""
    if len(feature_values) != 3:
        raise ValueError("feature grouping must contain exactly three keys")
    feature_type, feature_id, feature_label = feature_values
    unit = str(group["statistical_unit"].iloc[0])
    predictor = pd.to_numeric(group[predictor_key], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    response = pd.to_numeric(group["feature_value"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid_mask = predictor.notna() & response.notna()
    n_valid = int(valid_mask.sum())

    if n_valid < MIN_UNITS_FOR_TEST:
        skip_reason = "too_few_units"
    elif predictor[valid_mask].nunique() < 2:
        skip_reason = "constant_predictor"
    elif response[valid_mask].nunique() < 2:
        skip_reason = "constant_response"
    else:
        skip_reason = ""

    return {
        "feature_type": feature_type,
        "feature_id": feature_id,
        "feature_label": feature_label,
        "predictor": predictor_key,
        "statistical_unit": unit,
        "aggregation": str(group["aggregation"].iloc[0]),
        "stratify_key": stratify_key or "",
        "stratum": "" if stratum is None else str(stratum),
        "analysis_role": (
            "descriptive" if unit in DESCRIPTIVE_UNITS else "inferential"
        ),
        "n_units": float(group.shape[0]),
        "fdr_method": fdr_method,
        **_continuous_pair_stats(predictor, response),
        "skipped": bool(skip_reason),
        "skip_reason": skip_reason,
    }


def partial_correlation_controlling_tissue(
    unit_frame: pd.DataFrame,
    *,
    predictor_key: str,
    schema: ObsSchema,
    min_units_per_tissue: int = MIN_UNITS_FOR_TEST,
) -> pd.DataFrame:
    """Correlate each feature with a predictor after removing per-tissue means.

    Centring the feature and predictor within each tissue and correlating the
    pooled residuals is the partial correlation controlling for tissue: it
    removes between-tissue baselines and keeps within-tissue donor variation.
    This mirrors the transcriptomic-clock precedent for donor-aware age
    analysis.

    Args:
    unit_frame: Aggregated unit frame including the tissue key and predictor.
    predictor_key: Continuous predictor column (for example the age key).
    schema: Column-name schema (supplies the tissue key).
    min_units_per_tissue: Minimum distinct predictor values a tissue must
      contribute to be retained.

    Returns:
    One row per feature with pooled within-tissue Pearson/Spearman residual
    correlations with raw and FDR-adjusted p-values, the residual `n`,
    contributing tissue count, and statistical unit.
    """
    tissue_key = schema.tissue_key
    if tissue_key not in unit_frame.columns:
        raise KeyError(f"tissue column not found: {tissue_key}")
    if predictor_key not in unit_frame.columns:
        raise KeyError(f"predictor column not found: {predictor_key}")
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    records: list[dict[str, object]] = []
    records.extend(
        _partial_correlation_record(
            feature_values,
            group,
            predictor_key=predictor_key,
            tissue_key=tissue_key,
            min_units_per_tissue=min_units_per_tissue,
        )
        for feature_values, group in unit_frame.groupby(
            feature_keys, observed=True
        )
    )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result

    return _adjust_tested_pvalues(
        result,
        ("pearson_pvalue", "spearman_pvalue"),
    )


def _partial_correlation_record(
    feature_values: tuple[object, ...],
    group: pd.DataFrame,
    *,
    predictor_key: str,
    tissue_key: str,
    min_units_per_tissue: int,
) -> dict[str, object]:
    """Compute one feature's pooled within-tissue residual correlation."""
    if len(feature_values) != 3:
        raise ValueError("feature grouping must contain exactly three keys")
    feature_type, feature_id, feature_label = feature_values
    unit = str(group["statistical_unit"].iloc[0])
    pooled_x, pooled_y, n_tissues = _within_tissue_residuals(
        group,
        predictor_key=predictor_key,
        tissue_key=tissue_key,
        min_units_per_tissue=min_units_per_tissue,
    )
    if n_tissues == 0:
        return {
            "feature_type": feature_type,
            "feature_id": feature_id,
            "feature_label": feature_label,
            "predictor": predictor_key,
            "statistical_unit": unit,
            "analysis": "partial_correlation_controlling_tissue",
            "n_residual": 0.0,
            "n_tissues": 0.0,
            "pearson_r": np.nan,
            "pearson_pvalue": np.nan,
            "spearman_r": np.nan,
            "spearman_pvalue": np.nan,
            "skipped": True,
            "skip_reason": "no_tissue_meets_min_units",
        }

    valid = (
        pooled_x.size >= MIN_UNITS_FOR_TEST
        and np.unique(pooled_x).size >= 2
        and np.unique(pooled_y).size >= 2
    )
    pearson_r = np.nan
    pearson_pvalue = np.nan
    spearman_r = np.nan
    spearman_pvalue = np.nan
    if valid:
        pearson_r, pearson_pvalue = _statistic_pvalue(
            stats.pearsonr(pooled_x, pooled_y)
        )
        spearman_r, spearman_pvalue = _statistic_pvalue(
            stats.spearmanr(pooled_x, pooled_y)
        )

    return {
        "feature_type": feature_type,
        "feature_id": feature_id,
        "feature_label": feature_label,
        "predictor": predictor_key,
        "statistical_unit": unit,
        "analysis": "partial_correlation_controlling_tissue",
        "n_residual": float(pooled_x.size),
        "n_tissues": float(n_tissues),
        "pearson_r": float(pearson_r),
        "pearson_pvalue": float(pearson_pvalue),
        "spearman_r": float(spearman_r),
        "spearman_pvalue": float(spearman_pvalue),
        "skipped": not valid,
        "skip_reason": "" if valid else "too_few_residual_points",
    }


def _within_tissue_residuals(
    group: pd.DataFrame,
    *,
    predictor_key: str,
    tissue_key: str,
    min_units_per_tissue: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool centered predictor and response values from supported tissues."""
    residual_x: list[np.ndarray] = []
    residual_y: list[np.ndarray] = []
    for _, block in group.groupby(tissue_key, observed=True, dropna=False):
        predictor = pd.to_numeric(
            block[predictor_key], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        response = pd.to_numeric(
            block["feature_value"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        mask = predictor.notna() & response.notna()
        x = predictor[mask].to_numpy(dtype=float)
        y = response[mask].to_numpy(dtype=float)
        if x.size < min_units_per_tissue or np.unique(x).size < 2:
            continue

        residual_x.append(x - x.mean())
        residual_y.append(y - y.mean())

    if not residual_x:
        return np.array([], dtype=float), np.array([], dtype=float), 0

    return (
        np.concatenate(residual_x),
        np.concatenate(residual_y),
        len(residual_x),
    )


def _adjust_tested_pvalues(
    result: pd.DataFrame,
    pvalue_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Add Benjamini-Hochberg columns for estimable result rows."""
    tested = ~result["skipped"].to_numpy(dtype=bool)
    for column in pvalue_columns:
        adjusted = np.full(result.shape[0], np.nan, dtype=float)
        adjusted[tested] = benjamini_hochberg(
            result.loc[tested, column].to_numpy().tolist()
        )
        result[f"{column}_fdr"] = adjusted

    return result
