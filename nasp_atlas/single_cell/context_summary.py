"""Summarize donor-aware module scores across biological contexts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


__all__ = [
    "summarize_module_contexts",
]


def summarize_module_contexts(
    unit_frame: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    unit_columns: Sequence[str] = ("unit_id",),
    donor_column: str = "donor_id",
    module_column: str = "feature_label",
    value_column: str = "feature_value",
    cell_count_column: str = "n_cells",
    feature_type_column: str | None = "feature_type",
    module_feature_type: str = "module_score",
    min_units: int = 3,
    min_donors: int = 2,
) -> pd.DataFrame:
    """Rank module-score contexts using donor-aware descriptive summaries.

    The input should contain one row per module and independent statistical
    unit, such as the output of `aggregate_feature_frame`. Raw values are
    summarized on their original scorer-specific scale. Only context medians
    are converted to relative ranks, separately within each module.

    Eligibility is based on units and donors with finite scores. Ineligible
    contexts remain in the result for auditability but do not receive a rank.
    Percentiles run from zero (lowest eligible context) to one (highest);
    singleton and fully tied eligible sets receive a neutral value of 0.5.

    Args:
      unit_frame: Long donor-aware module-score frame.
      context_columns: Columns defining contexts, such as tissue and cell type.
      unit_columns: Columns jointly identifying an independent unit. The
        default uses the identifier emitted by `aggregate_feature_frame`.
      donor_column: Column identifying biological donors.
      module_column: Column containing the module label to preserve.
      value_column: Column containing module-score values.
      cell_count_column: Column containing contributing cells per unit.
      feature_type_column: Optional column used to retain module-score rows.
        Pass None when the input already contains only module scores.
      module_feature_type: Value identifying module scores in the feature-type
        column.
      min_units: Minimum finite-score units required for ranking eligibility.
      min_donors: Minimum donors with finite scores required for eligibility.

    Returns:
      One row per module and context with total and finite unit/donor counts,
      finite coverage, raw-scale descriptive statistics, median contributing
      cells, eligibility, and within-module context rank and percentile.

    Raises:
      KeyError: If a required column is absent.
      TypeError: If score or cell-count values contain non-numeric values.
      ValueError: If identifiers are missing, rows are duplicated, or options
        are invalid.
    """
    contexts = list(context_columns)
    units = list(unit_columns)
    output_columns = [
        module_column,
        *contexts,
        "n_units",
        "n_finite_units",
        "n_donors",
        "n_finite_donors",
        "finite_coverage",
        "mean",
        "median",
        "std",
        "q25",
        "q75",
        "iqr",
        "median_contributing_cells",
        "min_units_required",
        "min_donors_required",
        "eligible",
        "context_rank",
        "context_percentile",
    ]
    scoped, values, cell_counts = _prepare_context_frame(
        unit_frame,
        context_columns=contexts,
        unit_columns=units,
        donor_column=donor_column,
        module_column=module_column,
        value_column=value_column,
        cell_count_column=cell_count_column,
        feature_type_column=feature_type_column,
        module_feature_type=module_feature_type,
        min_units=min_units,
        min_donors=min_donors,
    )
    if scoped.empty:
        return pd.DataFrame(columns=output_columns)

    group_columns = [module_column, *contexts]
    grouped = scoped.groupby(
        group_columns,
        observed=True,
        dropna=False,
        sort=False,
    )
    records: list[dict[str, object]] = [
        _context_summary_record(
            group_key,
            group,
            group_columns=group_columns,
            values=values,
            cell_counts=cell_counts,
            donor_column=donor_column,
            min_units=min_units,
            min_donors=min_donors,
        )
        for group_key, group in grouped
    ]
    result = pd.DataFrame.from_records(records)
    result = _rank_module_contexts(result, module_column=module_column)
    return result.reindex(columns=output_columns)


def _prepare_context_frame(
    unit_frame: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    unit_columns: Sequence[str],
    donor_column: str,
    module_column: str,
    value_column: str,
    cell_count_column: str,
    feature_type_column: str | None,
    module_feature_type: str,
    min_units: int,
    min_donors: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate and normalize the long module-context input frame."""
    _validate_options(
        context_columns,
        unit_columns,
        module_column=module_column,
        min_units=min_units,
        min_donors=min_donors,
    )
    if not unit_frame.columns.is_unique:
        raise ValueError("unit_frame must have unique column names")

    required = [
        *context_columns,
        *unit_columns,
        donor_column,
        module_column,
        value_column,
        cell_count_column,
    ]
    if feature_type_column is not None:
        required.append(feature_type_column)
    _require_columns(unit_frame, required)

    scoped = unit_frame
    if feature_type_column is not None:
        scoped = scoped[
            scoped[feature_type_column].astype(str) == module_feature_type
        ]
    selected_columns = list(
        dict.fromkeys(
            [
                module_column,
                *context_columns,
                *unit_columns,
                donor_column,
                value_column,
                cell_count_column,
            ]
        )
    )
    scoped = scoped.loc[:, selected_columns].copy().reset_index(drop=True)
    if scoped.empty:
        return scoped, pd.Series(dtype=float), pd.Series(dtype=float)

    if scoped[module_column].isna().any():
        raise ValueError(f"{module_column} contains missing module labels")
    if scoped[list(unit_columns)].isna().any(axis=None):
        raise ValueError("unit_columns contain missing unit identifiers")
    if scoped[donor_column].isna().any():
        raise ValueError(f"{donor_column} contains missing donor identifiers")

    duplicate_keys = [module_column, *unit_columns]
    duplicated = scoped.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        example = scoped.loc[duplicated, duplicate_keys].iloc[0].to_dict()
        raise ValueError(
            "unit_frame must have one row per module and statistical unit; "
            f"found a duplicate such as {example}"
        )

    values = _numeric_series(
        scoped[value_column],
        column=value_column,
    ).replace([np.inf, -np.inf], np.nan)
    cell_counts = _numeric_series(
        scoped[cell_count_column],
        column=cell_count_column,
    )
    if cell_counts.dropna().lt(0).any():
        raise ValueError(f"{cell_count_column} contains negative cell counts")
    cell_counts = cell_counts.replace([np.inf, -np.inf], np.nan)

    return scoped, values, cell_counts


def _context_summary_record(
    group_key: object,
    group: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    values: pd.Series,
    cell_counts: pd.Series,
    donor_column: str,
    min_units: int,
    min_donors: int,
) -> dict[str, object]:
    """Summarize finite support and raw scores for one module-context group."""
    key_values = group_key if isinstance(group_key, tuple) else (group_key,)
    group_values = values.loc[group.index]
    finite = group_values.notna()
    finite_values = group_values.loc[finite]
    finite_group = group.loc[finite]
    lower_quartile = float(finite_values.quantile(0.25))
    upper_quartile = float(finite_values.quantile(0.75))
    n_finite_units = int(finite.sum())
    n_finite_donors = int(finite_group[donor_column].nunique())

    return {
        **dict(zip(group_columns, key_values, strict=True)),
        "n_units": len(group),
        "n_finite_units": n_finite_units,
        "n_donors": int(group[donor_column].nunique()),
        "n_finite_donors": n_finite_donors,
        "finite_coverage": n_finite_units / len(group),
        "mean": float(finite_values.mean()),
        "median": float(finite_values.median()),
        "std": float(finite_values.std(ddof=1)),
        "q25": lower_quartile,
        "q75": upper_quartile,
        "iqr": upper_quartile - lower_quartile,
        "median_contributing_cells": float(
            cell_counts.loc[finite_group.index].median()
        ),
        "min_units_required": min_units,
        "min_donors_required": min_donors,
        "eligible": (
            n_finite_units >= min_units and n_finite_donors >= min_donors
        ),
    }


def _rank_module_contexts(
    result: pd.DataFrame,
    *,
    module_column: str,
) -> pd.DataFrame:
    """Assign eligible within-module ranks and restore stable module order."""
    result["context_rank"] = np.nan
    result["context_percentile"] = np.nan
    for _, module_rows in result.groupby(
        module_column,
        observed=True,
        dropna=False,
        sort=False,
    ):
        eligible_index = module_rows.index[
            module_rows["eligible"] & module_rows["median"].notna()
        ]
        medians = result.loc[eligible_index, "median"]
        result.loc[eligible_index, "context_rank"] = medians.rank(
            method="dense",
            ascending=False,
        )
        result.loc[eligible_index, "context_percentile"] = _percentile_rank(
            medians
        )

    module_order = pd.factorize(result[module_column], sort=False)[0]
    result = (
        result.assign(_module_order=module_order)
        .sort_values(
            ["_module_order", "context_rank"],
            kind="stable",
            na_position="last",
        )
        .drop(columns="_module_order")
        .reset_index(drop=True)
    )
    return result


def _validate_options(
    context_columns: Sequence[str],
    unit_columns: Sequence[str],
    *,
    module_column: str,
    min_units: int,
    min_donors: int,
) -> None:
    """Validate context-summary options before inspecting frame values."""
    if not context_columns:
        raise ValueError("context_columns must define at least one context")
    if len(context_columns) != len(set(context_columns)):
        raise ValueError("context_columns must not contain duplicates")
    if not unit_columns:
        raise ValueError("unit_columns must identify statistical units")
    if len(unit_columns) != len(set(unit_columns)):
        raise ValueError("unit_columns must not contain duplicates")
    if module_column in unit_columns:
        raise ValueError("module_column cannot also be a unit column")
    if min_units < 1:
        raise ValueError("min_units must be at least 1")
    if min_donors < 1:
        raise ValueError("min_donors must be at least 1")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise an actionable error when required frame columns are absent."""
    if missing := list(
        dict.fromkeys(column for column in columns if column not in frame)
    ):
        raise KeyError(f"unit_frame missing required columns: {missing}")


def _numeric_series(values: pd.Series, *, column: str) -> pd.Series:
    """Convert numeric-like values while rejecting malformed non-null input."""
    numeric = pd.to_numeric(values, errors="coerce")
    malformed = values.notna() & numeric.isna()
    if malformed.any():
        example = values.loc[malformed].iloc[0]
        raise TypeError(
            f"{column} must contain numeric values; found {example!r}"
        )
    return numeric.astype(float)


def _percentile_rank(values: pd.Series) -> pd.Series:
    """Rank values from zero to one, assigning tied sets a neutral rank."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if values.empty:
        return result
    if len(values) == 1:
        result.loc[:] = 0.5
        return result
    ranks = values.rank(method="average")
    if ranks.nunique() == 1:
        result.loc[:] = 0.5
        return result
    return (ranks - 1.0) / (len(values) - 1.0)
