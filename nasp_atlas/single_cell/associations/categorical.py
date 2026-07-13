"""Categorical association tests for aggregated feature frames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

from nasp_atlas.single_cell.associations.core import DESCRIPTIVE_UNITS
from nasp_atlas.single_cell.associations.core import MIN_UNITS_FOR_TEST
from nasp_atlas.single_cell.associations.core import ObsSchema
from nasp_atlas.single_cell.associations.core import _statistic_pvalue
from nasp_atlas.single_cell.associations.core import benjamini_hochberg


__all__ = [
    "summarize_feature_groups",
    "test_feature_groups",
]


def summarize_feature_groups(
    unit_frame: pd.DataFrame,
    *,
    group_key: str,
    schema: ObsSchema,
    min_group_size: int = MIN_UNITS_FOR_TEST,
) -> pd.DataFrame:
    """Summarize each feature's values across a categorical grouping.

    Args:
    unit_frame: Aggregated unit frame from `aggregate_feature_frame`.
    group_key: Categorical obs column defining the groups (for example tissue,
      sex, cell type or donor).
    schema: Column-name schema (used to read the statistical unit annotation).
    min_group_size: Minimum units a group needs to be reported.

    Returns:
    One row per (feature, group) with mean/median/std/sem, unit count and
    distinct donor count, the statistical unit and aggregation.
    """
    if group_key not in unit_frame.columns:
        raise KeyError(f"group column not found: {group_key}")
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    records: list[dict[str, object]] = []
    for feature_values, group in unit_frame.groupby(
        feature_keys, observed=True
    ):
        feature_type, feature_id, feature_label = feature_values
        unit = str(group["statistical_unit"].iloc[0])
        aggregation = str(group["aggregation"].iloc[0])
        for group_name, block in group.groupby(
            group_key, observed=True, dropna=False
        ):
            values = (
                pd.to_numeric(block["feature_value"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if values.shape[0] < min_group_size:
                continue
            n_donors = (
                block[schema.donor_key].nunique()
                if schema.donor_key in block.columns
                else np.nan
            )
            std = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
            records.append(
                {
                    "feature_type": feature_type,
                    "feature_id": feature_id,
                    "feature_label": feature_label,
                    "group_key": group_key,
                    "group": str(group_name),
                    "statistical_unit": unit,
                    "aggregation": aggregation,
                    "analysis_role": (
                        "descriptive"
                        if unit in DESCRIPTIVE_UNITS
                        else "inferential"
                    ),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": std,
                    "sem": std / np.sqrt(values.shape[0]),
                    "n_units": float(values.shape[0]),
                    "n_donors": float(n_donors),
                }
            )
    summary_columns = [
        "feature_type",
        "feature_id",
        "feature_label",
        "group_key",
        "group",
        "statistical_unit",
        "aggregation",
        "analysis_role",
        "mean",
        "median",
        "std",
        "sem",
        "n_units",
        "n_donors",
    ]
    return pd.DataFrame.from_records(records, columns=summary_columns)


def test_feature_groups(
    unit_frame: pd.DataFrame,
    *,
    group_key: str,
    parametric: bool = True,
    min_group_size: int = MIN_UNITS_FOR_TEST,
    pairwise: bool = False,
    fdr_method: str = "benjamini_hochberg",
) -> pd.DataFrame:
    """Run omnibus (and optional pairwise) group tests for each feature.

    Two groups use Welch's t-test and Mann-Whitney U; three or more use one-way
    ANOVA and Kruskal-Wallis. Tests run on the supplied statistical unit, so
    donor-collapsed input yields donor-level inference. Cell-unit input is
    marked descriptive.

    Args:
    unit_frame: Aggregated unit frame from `aggregate_feature_frame`.
    group_key: Categorical obs column defining the groups.
    parametric: When True the reported omnibus `pvalue` is the parametric
      test (Welch/ANOVA); the nonparametric test is always reported alongside.
    min_group_size: Minimum units a group needs to enter a test.
    pairwise: When True, emit pairwise two-group comparisons in addition to the
      omnibus row.
    fdr_method: FDR method label recorded in output.

    Returns:
    One row per (feature, comparison) with parametric and nonparametric
    statistics, group counts, statistical unit, `analysis_role`, FDR-adjusted
    p-values and `skipped`/`skip_reason`.
    """
    if group_key not in unit_frame.columns:
        raise KeyError(f"group column not found: {group_key}")
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    records: list[dict[str, object]] = []
    for feature_values, group in unit_frame.groupby(
        feature_keys, observed=True
    ):
        feature_type, feature_id, feature_label = feature_values
        unit = str(group["statistical_unit"].iloc[0])
        aggregation = str(group["aggregation"].iloc[0])
        labels, arrays = _group_arrays(
            group, group_key=group_key, min_group_size=min_group_size
        )
        base = {
            "feature_type": feature_type,
            "feature_id": feature_id,
            "feature_label": feature_label,
            "group_key": group_key,
            "statistical_unit": unit,
            "aggregation": aggregation,
            "analysis_role": (
                "descriptive" if unit in DESCRIPTIVE_UNITS else "inferential"
            ),
            "fdr_method": fdr_method,
        }
        if len(arrays) < 2:
            records.append(
                {
                    **base,
                    "comparison": "omnibus",
                    "group_a": "",
                    "group_b": "",
                    "n_groups": float(len(arrays)),
                    "parametric_test": "",
                    "parametric_stat": np.nan,
                    "parametric_pvalue": np.nan,
                    "nonparametric_test": "",
                    "nonparametric_stat": np.nan,
                    "nonparametric_pvalue": np.nan,
                    "pvalue": np.nan,
                    "skipped": True,
                    "skip_reason": "fewer_than_two_groups",
                }
            )
            continue
        records.append({**base, **_omnibus_row(labels, arrays, parametric)})
        if pairwise:
            records.extend(_pairwise_rows(base, labels, arrays, parametric))
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    for column in ("parametric_pvalue", "nonparametric_pvalue"):
        tested = ~result["skipped"].to_numpy(dtype=bool)
        padj = np.full(result.shape[0], np.nan, dtype=float)
        padj[tested] = benjamini_hochberg(
            result.loc[tested, column].to_numpy().tolist()
        )
        result[f"{column}_fdr"] = padj
    return result


def _group_arrays(
    group: pd.DataFrame,
    *,
    group_key: str,
    min_group_size: int,
) -> tuple[list[str], list[np.ndarray]]:
    """Return group labels and value arrays passing the size threshold.

    Args:
    group: Single-feature slice of a unit frame.
    group_key: Categorical column defining the groups.
    min_group_size: Minimum units for a group to be retained.

    Returns:
    Parallel lists of group labels and their finite value arrays.
    """
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    for group_name, block in group.groupby(
        group_key, observed=True, dropna=False
    ):
        values = (
            pd.to_numeric(block["feature_value"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        if values.size >= min_group_size:
            labels.append(str(group_name))
            arrays.append(values)
    return labels, arrays


def _omnibus_row(
    labels: Sequence[str],
    arrays: Sequence[np.ndarray],
    parametric: bool,
) -> dict[str, object]:
    """Compute the omnibus comparison row for a feature.

    Args:
    labels: Group labels (length matches `arrays`).
    arrays: Per-group finite value arrays.
    parametric: Whether the headline `pvalue` uses the parametric test.

    Returns:
    Mapping of omnibus statistics for the feature.
    """
    if len(arrays) == 2:
        welch_stat, welch_pvalue = _statistic_pvalue(
            stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
        )
        mann_stat, mann_pvalue = _statistic_pvalue(
            stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        )
        parametric_name, parametric_stat, parametric_p = (
            "welch_t",
            float(welch_stat),
            float(welch_pvalue),
        )
        nonparametric_name, nonparametric_stat, nonparametric_p = (
            "mann_whitney_u",
            float(mann_stat),
            float(mann_pvalue),
        )
        group_a, group_b = labels[0], labels[1]
    else:
        anova_stat, anova_pvalue = _statistic_pvalue(stats.f_oneway(*arrays))
        kruskal_stat, kruskal_pvalue = _statistic_pvalue(stats.kruskal(*arrays))
        parametric_name, parametric_stat, parametric_p = (
            "anova",
            float(anova_stat),
            float(anova_pvalue),
        )
        nonparametric_name, nonparametric_stat, nonparametric_p = (
            "kruskal_wallis",
            float(kruskal_stat),
            float(kruskal_pvalue),
        )
        group_a, group_b = "", ""
    return {
        "comparison": "omnibus",
        "group_a": group_a,
        "group_b": group_b,
        "n_groups": float(len(arrays)),
        "parametric_test": parametric_name,
        "parametric_stat": parametric_stat,
        "parametric_pvalue": parametric_p,
        "nonparametric_test": nonparametric_name,
        "nonparametric_stat": nonparametric_stat,
        "nonparametric_pvalue": nonparametric_p,
        "pvalue": parametric_p if parametric else nonparametric_p,
        "skipped": False,
        "skip_reason": "",
    }


def _pairwise_rows(
    base: dict[str, object],
    labels: Sequence[str],
    arrays: Sequence[np.ndarray],
    parametric: bool,
) -> list[dict[str, object]]:
    """Build pairwise two-group comparison rows for a feature.

    Args:
    base: Shared feature-level fields to copy into each row.
    labels: Group labels.
    arrays: Per-group finite value arrays.
    parametric: Whether the headline `pvalue` uses the parametric test.

    Returns:
    One record per unordered group pair.
    """
    rows: list[dict[str, object]] = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            welch_stat, welch_pvalue = _statistic_pvalue(
                stats.ttest_ind(arrays[i], arrays[j], equal_var=False)
            )
            mann_stat, mann_pvalue = _statistic_pvalue(
                stats.mannwhitneyu(
                    arrays[i], arrays[j], alternative="two-sided"
                )
            )
            rows.append(
                {
                    **base,
                    "comparison": "pairwise",
                    "group_a": labels[i],
                    "group_b": labels[j],
                    "n_groups": 2.0,
                    "parametric_test": "welch_t",
                    "parametric_stat": float(welch_stat),
                    "parametric_pvalue": float(welch_pvalue),
                    "nonparametric_test": "mann_whitney_u",
                    "nonparametric_stat": float(mann_stat),
                    "nonparametric_pvalue": float(mann_pvalue),
                    "pvalue": (
                        float(welch_pvalue)
                        if parametric
                        else float(mann_pvalue)
                    ),
                    "skipped": False,
                    "skip_reason": "",
                }
            )
    return rows
