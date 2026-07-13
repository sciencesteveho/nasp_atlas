"""Aggregate cell-level association frames to statistical units."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from nasp_atlas.single_cell.associations.core import Aggregation
from nasp_atlas.single_cell.associations.core import ObsSchema
from nasp_atlas.single_cell.associations.core import StatisticalUnit
from nasp_atlas.single_cell.associations.core import _unit_group_keys
from nasp_atlas.single_cell.associations.core import metadata_columns


__all__ = [
    "aggregate_feature_frame",
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
    metadata_keys = [
        key for key in metadata_columns(schema) if key in cell_frame.columns
    ]
    feature_keys = ["feature_type", "feature_id", "feature_label"]
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

    reducer = _value_reducer(aggregation, detection_threshold)
    full_group = feature_keys + group_keys
    grouped = cell_frame.groupby(full_group, observed=True, dropna=False)
    aggregated = grouped["feature_value"].agg(reducer).reset_index()
    counts = grouped["feature_value"].count().reset_index(name="n_cells")
    totals = grouped.size().reset_index(name="n_cells_total")
    aggregated = aggregated.merge(counts, on=full_group, how="left")
    aggregated = aggregated.merge(totals, on=full_group, how="left")

    if carry := [key for key in metadata_keys if key not in group_keys]:
        consensus = grouped[carry].agg(_constant_value).reset_index()
        aggregated = aggregated.merge(consensus, on=full_group, how="left")

    aggregated["statistical_unit"] = statistical_unit
    aggregated["aggregation"] = aggregation
    aggregated["unit_id"] = _unit_identifiers(aggregated, group_keys)

    return aggregated


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


def _unit_identifiers(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
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
        return lambda values: (
            float(np.nanmean(values)) if len(values) else np.nan
        )
    if aggregation == "median":
        return lambda values: (
            float(np.nanmedian(values)) if len(values) else np.nan
        )
    if aggregation == "sum":
        return lambda values: (
            float(np.nansum(values)) if len(values) else np.nan
        )
    if aggregation in ("fraction_expressing", "percent_expressing"):
        scale = 100.0 if aggregation == "percent_expressing" else 1.0

        def _fraction(values: pd.Series) -> float:
            """Return the finite value fraction above the detection floor."""
            finite = values.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return np.nan
            return scale * float(np.mean(finite > detection_threshold))

        return _fraction
    raise ValueError(f"unsupported aggregation: {aggregation}")
