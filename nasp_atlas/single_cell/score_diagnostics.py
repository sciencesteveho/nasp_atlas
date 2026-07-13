"""Diagnostics comparing independent module-scoring methods."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

from nasp_atlas.single_cell.associations.core import benjamini_hochberg


__all__ = [
    "compare_module_scorers",
]


def compare_module_scorers(
    scores: pd.DataFrame,
    module_ids: Sequence[str],
    *,
    min_cells: int = 3,
) -> pd.DataFrame:
    """Compare Scanpy and AUCell composite scores module by module.

    Rank agreement is the primary diagnostic because Scanpy scores and signed
    AUCell composites do not share a numeric scale. The median absolute
    percentile difference additionally reports cell-level disagreement on a
    zero-to-one scale.

    Args:
      scores: Wide score frame containing `*_score` and `*_auc` columns.
      module_ids: Module identifiers to compare.
      min_cells: Minimum complete cells required for a correlation.

    Returns:
      One row per module with both score columns, including complete-cell
      counts, Pearson and Spearman agreement, p-values and FDR, percentile
      disagreement, and explicit skip reasons.
    """
    if min_cells < 3:
        raise ValueError("min_cells must be at least 3")

    records: list[dict[str, object]] = []
    for module_id in dict.fromkeys(module_ids):
        scanpy_key = f"{module_id}_score"
        aucell_key = f"{module_id}_auc"
        if scanpy_key not in scores.columns or aucell_key not in scores.columns:
            continue
        paired = scores.loc[:, [scanpy_key, aucell_key]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
        scanpy_values = paired[scanpy_key].to_numpy(dtype=float)
        aucell_values = paired[aucell_key].to_numpy(dtype=float)
        if len(paired) < min_cells:
            skip_reason = "too_few_complete_cells"
        elif (
            np.unique(scanpy_values).size < 2
            or np.unique(aucell_values).size < 2
        ):
            skip_reason = "constant_score"
        else:
            skip_reason = ""

        pearson_r = np.nan
        pearson_pvalue = np.nan
        spearman_r = np.nan
        spearman_pvalue = np.nan
        percentile_difference = np.nan
        if not skip_reason:
            pearson_r, pearson_pvalue = _correlation_values(
                stats.pearsonr(scanpy_values, aucell_values)
            )
            spearman_r, spearman_pvalue = _correlation_values(
                stats.spearmanr(scanpy_values, aucell_values)
            )
            scanpy_percentile = paired[scanpy_key].rank(pct=True)
            aucell_percentile = paired[aucell_key].rank(pct=True)
            percentile_difference = float(
                (scanpy_percentile - aucell_percentile).abs().median()
            )
        records.append(
            {
                "module_id": module_id,
                "scanpy_score_key": scanpy_key,
                "aucell_score_key": aucell_key,
                "n_complete_cells": len(paired),
                "pearson_r": pearson_r,
                "pearson_pvalue": pearson_pvalue,
                "spearman_r": spearman_r,
                "spearman_pvalue": spearman_pvalue,
                "median_absolute_percentile_difference": (
                    percentile_difference
                ),
                "skipped": bool(skip_reason),
                "skip_reason": skip_reason,
            }
        )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["pearson_fdr"] = benjamini_hochberg(
        result["pearson_pvalue"].to_numpy(dtype=float).tolist()
    )
    result["spearman_fdr"] = benjamini_hochberg(
        result["spearman_pvalue"].to_numpy(dtype=float).tolist()
    )
    return result


def _correlation_values(result: object) -> tuple[float, float]:
    """Return a coefficient and p-value from a SciPy test result."""
    values = np.asarray(result, dtype=float)
    return float(values[0]), float(values[1])
