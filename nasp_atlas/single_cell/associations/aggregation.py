"""Aggregate cell-level association frames to statistical units."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real

import numpy as np
import pandas as pd

from nasp_atlas.single_cell.associations.core import Aggregation
from nasp_atlas.single_cell.associations.core import ObsSchema
from nasp_atlas.single_cell.associations.core import StatisticalUnit
from nasp_atlas.single_cell.associations.core import _unit_group_keys
from nasp_atlas.single_cell.associations.core import metadata_columns


__all__ = [
    "aggregate_feature_frame",
    "aggregate_feature_frame_by_keys",
]


def aggregate_feature_frame(
    cell_frame: pd.DataFrame,
    *,
    statistical_unit: StatisticalUnit,
    aggregation: Aggregation,
    schema: ObsSchema,
    detection_threshold: float = 0.0,
) -> pd.DataFrame:
    """Collapse a cell-level feature frame to a statistical unit.

    For `cell` (and `metacell`, whose rows are already the unit) the frame
    is returned as-is with unit annotations. For donor-based units, cells are
    grouped by the unit keys within each feature and reduced by the chosen
    aggregation, carrying only metadata constant within each unit and both the
    contributing and total cell counts.

    Args:
    cell_frame: Long cell-level frame from `build_cell_feature_frame`.
    statistical_unit: Unit to collapse to.
    aggregation: Reduction applied to `feature_value` within each unit. The
      `fraction_expressing`/`percent_expressing` options use
      `detection_threshold`.
    schema: Column-name schema for grouping and metadata.
    detection_threshold: Per-cell floor above which a value counts as expressed
      for expressing-fraction aggregations.

    Returns:
    Frame with one row per (unit, feature) carrying `feature_value`,
    `statistical_unit`, `aggregation`, `unit_id`, finite-value `n_cells`, total
    `n_cells_total`, and metadata columns. Metadata that varies within a unit
    is set to NaN rather than assigned from an arbitrary cell.
    """
    _validate_detection_threshold(detection_threshold)
    metadata_keys = [
        key for key in metadata_columns(schema) if key in cell_frame.columns
    ]
    if cell_frame.empty:
        return cell_frame.assign(
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            n_cells=pd.Series(dtype=float),
            n_cells_total=pd.Series(dtype=float),
            unit_id=pd.Series(dtype=str),
        )

    group_keys = _unit_group_keys(statistical_unit, schema)
    if not group_keys:
        return _annotate_existing_units(
            cell_frame,
            statistical_unit=statistical_unit,
            aggregation=aggregation,
        )
    if missing := [key for key in group_keys if key not in cell_frame.columns]:
        raise KeyError(
            "cell frame missing grouping columns for unit "
            f"{statistical_unit}: {missing}"
        )

    return _aggregate_by_group_keys(
        cell_frame,
        group_keys=group_keys,
        metadata_keys=metadata_keys,
        statistical_unit=statistical_unit,
        aggregation=aggregation,
        detection_threshold=detection_threshold,
    )


def aggregate_feature_frame_by_keys(
    cell_frame: pd.DataFrame,
    *,
    unit_keys: Sequence[str],
    statistical_unit: str,
    aggregation: Aggregation,
    schema: ObsSchema,
    metadata_keys: Sequence[str] = (),
    detection_threshold: float = 0.0,
) -> pd.DataFrame:
    """Collapse a cell feature frame using caller-defined unit keys.

    Each feature is aggregated independently over the explicit `unit_keys`.
    Available metadata named by `schema` is carried by consensus, together with
    every column in `metadata_keys`. Metadata that varies within a unit is set
    to missing rather than selected from an arbitrary cell. The input frame is
    not modified.

    Args:
      cell_frame: Long cell-level frame containing feature values and metadata.
      unit_keys: Ordered columns whose joint values identify an aggregate unit.
      statistical_unit: Stable label recorded for every aggregate row.
      aggregation: Reduction applied to `feature_value` within each unit. The
        "fraction_expressing" and "percent_expressing" choices use
        `detection_threshold`.
      schema: Column-name schema identifying metadata carried when available.
      metadata_keys: Additional metadata columns that must exist and should be
        carried by consensus.
      detection_threshold: Per-cell floor above which a finite value counts as
        expressed for expressing-fraction aggregations.

    Returns:
      One row per feature and explicit unit, with the aggregated
      `feature_value`, caller-provided `statistical_unit`, `aggregation`, stable
      `unit_id`, non-missing `n_cells`, total `n_cells_total`, grouping columns,
      and consensus metadata.

    Raises:
      TypeError: If a key or the statistical-unit label is not a string.
      ValueError: If keys are empty or repeated, the label is empty, or the
        detection threshold is not finite.
      KeyError: If a grouping, feature, or explicitly requested metadata column
        is absent from `cell_frame`.
    """
    _validate_detection_threshold(detection_threshold)
    resolved_unit_keys = _validate_keys(
        unit_keys,
        parameter_name="unit_keys",
        allow_empty=False,
    )
    resolved_metadata_keys = _validate_keys(
        metadata_keys,
        parameter_name="metadata_keys",
        allow_empty=True,
    )
    if not isinstance(statistical_unit, str):
        raise TypeError("statistical_unit must be a string")
    if not statistical_unit.strip():
        raise ValueError("statistical_unit must not be empty")

    feature_keys = [
        "feature_type",
        "feature_id",
        "feature_label",
        "feature_value",
    ]
    required_keys = list(
        dict.fromkeys(
            [*feature_keys, *resolved_unit_keys, *resolved_metadata_keys]
        )
    )
    if missing := [key for key in required_keys if key not in cell_frame]:
        raise KeyError(f"cell frame missing required columns: {missing}")

    schema_metadata = [
        key for key in metadata_columns(schema) if key in cell_frame.columns
    ]
    carried_metadata = list(
        dict.fromkeys([*schema_metadata, *resolved_metadata_keys])
    )
    if cell_frame.empty:
        return cell_frame.assign(
            statistical_unit=statistical_unit,
            aggregation=aggregation,
            n_cells=pd.Series(dtype=float),
            n_cells_total=pd.Series(dtype=float),
            unit_id=pd.Series(dtype=str),
        )

    return _aggregate_by_group_keys(
        cell_frame,
        group_keys=resolved_unit_keys,
        metadata_keys=carried_metadata,
        statistical_unit=statistical_unit,
        aggregation=aggregation,
        detection_threshold=detection_threshold,
    )


def _aggregate_by_group_keys(
    cell_frame: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    metadata_keys: Sequence[str],
    statistical_unit: str,
    aggregation: Aggregation,
    detection_threshold: float,
) -> pd.DataFrame:
    """Aggregate feature values and consensus metadata over `group_keys`."""
    reducer = _value_reducer(aggregation, detection_threshold)
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    resolved_group_keys = list(group_keys)
    full_group = [*feature_keys, *resolved_group_keys]
    grouped = cell_frame.groupby(full_group, observed=True, dropna=False)
    aggregated = grouped["feature_value"].agg(reducer).reset_index()
    counts = (
        grouped["feature_value"]
        .agg(_finite_value_count)
        .reset_index(name="n_cells")
    )
    totals = grouped.size().reset_index(name="n_cells_total")
    aggregated = aggregated.merge(counts, on=full_group, how="left")
    aggregated = aggregated.merge(totals, on=full_group, how="left")

    if carry := [key for key in metadata_keys if key not in group_keys]:
        consensus = grouped[carry].agg(_constant_value).reset_index()
        aggregated = aggregated.merge(consensus, on=full_group, how="left")

    aggregated["statistical_unit"] = statistical_unit
    aggregated["aggregation"] = aggregation
    aggregated["unit_id"] = _unit_identifiers(
        aggregated,
        resolved_group_keys,
    )

    return aggregated


def _validate_keys(
    keys: Sequence[str],
    *,
    parameter_name: str,
    allow_empty: bool,
) -> list[str]:
    """Return validated, ordered column keys for public aggregation input."""
    if isinstance(keys, str):
        raise TypeError(f"{parameter_name} must be a sequence of column names")
    resolved = list(keys)
    if not allow_empty and not resolved:
        raise ValueError(f"{parameter_name} must contain at least one key")
    if invalid := [key for key in resolved if not isinstance(key, str)]:
        raise TypeError(
            f"{parameter_name} must contain only strings: {invalid}"
        )
    if empty := [key for key in resolved if not key.strip()]:
        raise ValueError(f"{parameter_name} contains empty keys: {empty}")
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{parameter_name} must not contain duplicate keys")
    return resolved


def _validate_detection_threshold(detection_threshold: float) -> None:
    """Reject thresholds that cannot define a finite expression boundary."""
    if (
        isinstance(detection_threshold, bool)
        or not isinstance(detection_threshold, Real)
        or not np.isfinite(float(detection_threshold))
    ):
        raise ValueError("detection_threshold must be a finite real number")


def _annotate_existing_units(
    cell_frame: pd.DataFrame,
    *,
    statistical_unit: StatisticalUnit,
    aggregation: Aggregation,
) -> pd.DataFrame:
    """Annotate rows that already represent the requested statistical unit."""
    annotated = cell_frame.copy()
    annotated["statistical_unit"] = statistical_unit
    annotated["aggregation"] = aggregation
    annotated["n_cells"] = annotated["feature_value"].notna().astype(float)
    annotated["n_cells_total"] = 1.0
    if "obs_name" in annotated.columns:
        annotated["unit_id"] = annotated["obs_name"].astype(str)

    return annotated


def _constant_value(values: pd.Series) -> object:
    """Return a group's sole non-null value, or NaN when it varies."""
    unique = pd.unique(values.dropna())
    return unique[0] if len(unique) == 1 else np.nan


def _finite_value_count(values: pd.Series) -> int:
    """Return the number of finite numeric feature values."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(numeric).sum())


def _unit_identifiers(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    """Return stable, human-readable identifiers for grouped units."""
    identifiers = frame[keys].astype("string").fillna("<NA>")
    return identifiers.apply(
        lambda row: "|".join(f"{key}={row[key]}" for key in keys),
        axis="columns",
    )


def _value_reducer(
    aggregation: Aggregation,
    detection_threshold: float,
) -> Callable[[pd.Series], float]:
    """Return a pandas aggregation callable for the chosen reduction.

    Args:
    aggregation: Reduction name.
    detection_threshold: Floor used by expressing-fraction reductions.

    Returns:
    A callable mapping a Series of values to a scalar.
    """
    if aggregation == "mean":
        return lambda values: _finite_reduction(values, np.mean)
    if aggregation == "median":
        return lambda values: _finite_reduction(values, np.median)
    if aggregation == "sum":
        return lambda values: _finite_reduction(values, np.sum)
    if aggregation in ("fraction_expressing", "percent_expressing"):
        scale = 100.0 if aggregation == "percent_expressing" else 1.0

        def _fraction(values: pd.Series) -> float:
            """Return the finite value fraction above the detection floor."""
            finite = _finite_numeric_values(values)
            if finite.size == 0:
                return np.nan
            return scale * float(np.mean(finite > detection_threshold))

        return _fraction
    raise ValueError(f"unsupported aggregation: {aggregation}")


def _finite_reduction(
    values: pd.Series,
    reducer: Callable[[np.ndarray], np.floating],
) -> float:
    """Reduce finite numeric values or return missing when none exist."""
    finite = _finite_numeric_values(values)
    return float(reducer(finite)) if finite.size else np.nan


def _finite_numeric_values(values: pd.Series) -> np.ndarray:
    """Return only finite numeric values from one aggregation group."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]
