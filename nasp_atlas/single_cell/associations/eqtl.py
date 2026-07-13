"""eQTL table validation and joins for association frames."""

from __future__ import annotations

import pandas as pd

from nasp_atlas.single_cell.associations.core import EqtlMergeMode
from nasp_atlas.single_cell.associations.core import ObsSchema


EQTL_REQUIRED_COLUMNS: dict[EqtlMergeMode, tuple[str, ...]] = {
    "gene": ("gene_symbol",),
    "tissue": ("tissue",),
    "module": ("module_id",),
    "donor": ("donor_id",),
}

EQTL_VALUE_COLUMNS: tuple[str, ...] = (
    "total_eqtls",
    "tissue_specific_eqtls",
)

__all__ = [
    "merge_eqtl_counts",
    "validate_eqtl_table",
]


def validate_eqtl_table(
    eqtl_table: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
) -> list[str]:
    """Validate an eQTL table against the columns a merge mode requires.

    Args:
    eqtl_table: Candidate eQTL annotation table.
    merge_mode: Level at which the table will be joined.

    Returns:
    The eQTL value columns present in the table.

    Raises:
    KeyError: If a key column required by the merge mode is missing, or no
      recognized eQTL value column is present.
    """
    required = EQTL_REQUIRED_COLUMNS[merge_mode]
    if missing_keys := [
        key for key in required if key not in eqtl_table.columns
    ]:
        raise KeyError(
            f"eqtl table missing key columns for merge_mode={merge_mode}: "
            f"{missing_keys}"
        )
    if value_columns := [
        column for column in EQTL_VALUE_COLUMNS if column in eqtl_table.columns
    ]:
        return value_columns
    else:
        raise KeyError(
            "eqtl table has no recognized value column; expected one of "
            f"{EQTL_VALUE_COLUMNS}"
        )


def merge_eqtl_counts(
    unit_frame: pd.DataFrame,
    eqtl_table: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
    schema: ObsSchema,
) -> pd.DataFrame:
    """Join eQTL burden onto an aggregated unit frame at the correct level.

    eQTL counts are properties of a gene, tissue, module or donor -- never of a
    single cell -- so the join happens on the already-aggregated unit frame. The
    merge mode determines the join key and is recorded in an `eqtl_merge_mode`
    column on the result.

    Args:
    unit_frame: Aggregated unit frame from `aggregate_feature_frame`.
    eqtl_table: Validated eQTL annotation table.
    merge_mode: Level at which to join (gene, tissue, module or donor).
    schema: Column-name schema (supplies the tissue and donor keys).

    Returns:
    The unit frame with eQTL value columns and `eqtl_merge_mode` attached.
    Unmatched rows keep NaN eQTL values.
    """
    value_columns = validate_eqtl_table(eqtl_table, merge_mode=merge_mode)
    left_on, right_on = _eqtl_join_keys(merge_mode, schema)
    if left_on not in unit_frame.columns:
        raise KeyError(
            f"unit frame missing join column for merge_mode={merge_mode}: "
            f"{left_on}"
        )
    right = eqtl_table.loc[:, [right_on, *value_columns]].copy()
    if (
        duplicated := right.loc[
            right[right_on].duplicated(keep=False),
            right_on,
        ]
        .astype(str)
        .unique()
        .tolist()
    ):
        preview = ", ".join(duplicated[:3])
        raise ValueError(
            f"eqtl table has duplicate {right_on!r} keys (for example: "
            f"{preview}); aggregate explicitly or choose a more specific "
            "merge mode"
        )
    merged = unit_frame.merge(
        right,
        how="left",
        left_on=left_on,
        right_on=right_on,
    )
    if right_on != left_on and right_on in merged.columns:
        merged = merged.drop(columns=[right_on])
    merged["eqtl_merge_mode"] = merge_mode
    return merged


def _eqtl_join_keys(
    merge_mode: EqtlMergeMode,
    schema: ObsSchema,
) -> tuple[str, str]:
    """Return the (unit-frame, eqtl-table) join column names for a merge mode.

    Args:
    merge_mode: Level at which to join.
    schema: Column-name schema.

    Returns:
    A tuple of the left (unit frame) and right (eQTL table) column names.
    """
    mapping: dict[EqtlMergeMode, tuple[str, str]] = {
        "gene": ("feature_label", "gene_symbol"),
        "tissue": (schema.tissue_key, "tissue"),
        "module": ("feature_label", "module_id"),
        "donor": (schema.donor_key, "donor_id"),
    }
    return mapping[merge_mode]
