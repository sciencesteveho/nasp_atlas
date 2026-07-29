"""eQTL table preparation, joins, and association frames."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from nasp_atlas.single_cell.associations.continuous import (
    regress_features_on_continuous,
)
from nasp_atlas.single_cell.associations.core import EqtlMergeMode
from nasp_atlas.single_cell.associations.core import ObsSchema


EQTL_REQUIRED_COLUMNS: dict[EqtlMergeMode, tuple[str, ...]] = {
    "gene": ("gene_symbol",),
    "gene_tissue": ("gene_symbol", "tissue"),
    "tissue": ("tissue",),
    "module": ("module_id",),
    "donor": ("donor_id",),
}

EQTL_VALUE_COLUMNS: tuple[str, ...] = (
    "total_eqtls",
    "tissue_specific_eqtls",
)

_COLUMN_ALIASES: dict[str, str] = {
    "gene": "gene_symbol",
    "gene_total_eqtls": "total_eqtls",
    "n_significant_eqtls": "tissue_specific_eqtls",
    "tissue_abbrev": "eqtl_tissue_abbrev",
}

_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "eqtl_tissue_abbrev",
    "eqtl_n_genes",
    "eqtl_source_schema",
    "eqtl_count_unit",
)

_FEATURE_COLUMNS: tuple[str, ...] = (
    "feature_type",
    "feature_id",
    "feature_label",
)

__all__ = [
    "associate_features_with_eqtl_counts",
    "build_eqtl_association_frames",
    "merge_eqtl_counts",
    "prepare_eqtl_table",
    "validate_eqtl_table",
]


def prepare_eqtl_table(
    eqtl_table: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
) -> pd.DataFrame:
    """Return a validated canonical eQTL table for `merge_mode`.

    The canonical aliases for the supplied files are `gene_symbol`,
    `total_eqtls`, and `tissue_specific_eqtls`. Tissue mode explicitly sums
    significant gene-variant pair counts over source genes. Wide input is
    usable for gene-total comparisons; its abbreviated tissue columns cannot
    safely join atlas tissue labels.

    Args:
      eqtl_table: Candidate canonical, long, or wide eQTL count table.
      merge_mode: Level at which the prepared counts will be joined.

    Returns:
      A copy with canonical keys, validated count values, one row per merge
      key, and source-schema/count-unit provenance.

    Raises:
      KeyError: If required keys or a supported count column are missing.
      ValueError: If counts, totals, or key cardinality are inconsistent.
    """
    table = eqtl_table.copy()
    validate_source = "eqtl_source_schema" not in table.columns
    source_schema = _source_schema(table)
    _rename_columns(table)
    for column in _value_columns(table):
        table[column] = _count_series(table[column], column=column)
    if validate_source:
        _validate_supplied_schema(table, source_schema=source_schema)

    if source_schema == "nasp_sensor_eqtl_counts_wide" and merge_mode in {
        "gene_tissue",
        "tissue",
    }:
        raise KeyError(
            "wide sensor eQTL tissue columns use abbreviations without atlas "
            "tissue identifiers; use the long table for "
            f"merge_mode={merge_mode}"
        )
    required = EQTL_REQUIRED_COLUMNS[merge_mode]
    _require_columns(
        table,
        required,
        context=f"eQTL table for merge_mode={merge_mode}",
    )
    for key in required:
        missing = table[key].isna() | (
            table[key].astype("string").str.strip().eq("")
        )
        if missing.any():
            rows = ", ".join(table.index[missing].astype(str)[:3])
            raise ValueError(
                f"eQTL key column {key!r} has missing values at rows: {rows}"
            )
    if not _value_columns(table):
        raise KeyError(
            "eQTL table has no recognized count column; expected one of "
            f"{EQTL_VALUE_COLUMNS}"
        )

    prepared = _prepare_for_merge(table, merge_mode=merge_mode)
    prepared["eqtl_source_schema"] = source_schema
    prepared["eqtl_count_unit"] = (
        _single_string(table["eqtl_count_unit"])
        if "eqtl_count_unit" in table.columns
        else (
            "significant_gene_variant_pairs"
            if source_schema.startswith("nasp_sensor_eqtl_counts_")
            else "reported_eqtl_count"
        )
    )
    return prepared.reset_index(drop=True)


def validate_eqtl_table(
    eqtl_table: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
) -> list[str]:
    """Validate an eQTL table and report its canonical count columns.

    Args:
      eqtl_table: Candidate canonical, long, or wide eQTL count table.
      merge_mode: Level at which the table will be joined.

    Returns:
      Canonical eQTL count columns available after preparation.
    """
    prepared = prepare_eqtl_table(eqtl_table, merge_mode=merge_mode)
    return _value_columns(prepared)


def merge_eqtl_counts(
    unit_frame: pd.DataFrame,
    eqtl_table: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
    schema: ObsSchema,
) -> pd.DataFrame:
    """Join eQTL burden onto an aggregated unit frame.

    Tissue matching normalizes spelling and case only; it does not infer a
    biological crosswalk between different tissue definitions. Original atlas
    and source identifiers remain in the result for auditing.

    Args:
      unit_frame: Aggregated frame from `aggregate_feature_frame`.
      eqtl_table: Candidate canonical, long, or wide eQTL count table.
      merge_mode: Gene, gene-tissue, tissue, module, or donor join level.
      schema: Column-name schema supplying atlas tissue and donor keys.

    Returns:
      A left-preserving copy with canonical counts, source provenance,
      `eqtl_matched`, and `eqtl_merge_mode`.
    """
    prepared = prepare_eqtl_table(eqtl_table, merge_mode=merge_mode)
    key_pairs = _join_keys(merge_mode, schema)
    missing = [
        left_key
        for left_key, _ in key_pairs
        if left_key not in unit_frame.columns
    ]
    if missing:
        raise KeyError(
            "unit frame missing join columns for "
            f"merge_mode={merge_mode}: {missing}"
        )

    left = unit_frame.copy()
    right = prepared.copy()
    value_columns = _value_columns(prepared)
    for column in value_columns:
        right[column] = right[column].astype(float)
    temporary_keys: list[str] = []
    right_keys: list[str] = []
    for index, (left_key, right_key) in enumerate(key_pairs):
        temporary_key = f"__eqtl_join_key_{index}"
        temporary_keys.append(temporary_key)
        right_keys.append(right_key)
        is_tissue = right_key == "tissue"
        left[temporary_key] = _normalized_key(
            left[left_key],
            tissue=is_tissue,
        )
        right[temporary_key] = _normalized_key(
            right[right_key],
            tissue=is_tissue,
        )
        right[f"eqtl_source_{right_key}"] = right[right_key]

    duplicated = right.duplicated(temporary_keys, keep=False)
    if duplicated.any():
        preview = (
            right.loc[duplicated, right_keys]
            .astype(str)
            .drop_duplicates()
            .head(3)
            .to_dict(orient="records")
        )
        raise ValueError(
            "eQTL table has duplicate normalized merge keys "
            f"{tuple(right_keys)}: {preview}"
        )
    payload = [
        column
        for column in right.columns
        if column not in {*right_keys, *temporary_keys}
    ]
    collisions = sorted(set(payload).intersection(unit_frame.columns))
    collisions.extend(
        column
        for column in (
            "eqtl_matched",
            "eqtl_merge_mode",
            "eqtl_predictor_transform",
        )
        if column in unit_frame.columns
    )
    if collisions:
        raise ValueError(
            "unit frame already contains eQTL output columns: "
            f"{sorted(set(collisions))}"
        )

    merged = left.merge(
        right.loc[:, [*temporary_keys, *payload]],
        how="left",
        on=temporary_keys,
        sort=False,
        validate="many_to_one",
    ).drop(columns=temporary_keys)
    merged["eqtl_matched"] = merged[value_columns].notna().any(axis="columns")
    merged["eqtl_merge_mode"] = merge_mode
    merged["eqtl_predictor_transform"] = "none"
    for column in ("eqtl_source_schema", "eqtl_count_unit"):
        merged[column] = _single_string(prepared[column])
    return merged


def build_eqtl_association_frames(
    merged_frame: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
    schema: ObsSchema,
) -> dict[str, pd.DataFrame]:
    """Build the unit-valid frames used for eQTL regressions and plots.

    The returned frames average repeated donor-level observations once at the
    level where the eQTL count varies. Mapping keys are the corresponding
    `analysis_scope` values. Gene-tissue mode returns both the within-gene,
    across-tissue frame and the within-tissue, across-gene frame. Module mode
    returns no frame because module-keyed counts are annotations only.

    Args:
      merged_frame: Unit frame returned by `merge_eqtl_counts`.
      merge_mode: Level used to join the counts.
      schema: Column-name schema supplying tissue and donor keys.

    Returns:
      Analysis-scope mappings to frames with one row per independent
      comparison unit and explicit `statistical_unit` labels.
    """
    predictors = _association_predictors(
        merged_frame,
        merge_mode=merge_mode,
    )
    return _build_eqtl_association_frames(
        merged_frame,
        merge_mode=merge_mode,
        schema=schema,
        predictors=predictors,
    )


def associate_features_with_eqtl_counts(
    merged_frame: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
    schema: ObsSchema,
) -> pd.DataFrame:
    """Estimate descriptive feature/eQTL associations at valid count units.

    Gene mode compares mean expression across curated genes. Gene-tissue mode
    compares each gene across tissues and genes within each tissue. Tissue mode
    compares each module or gene across tissues after averaging donor-level
    values within tissue. This avoids treating a tissue count repeated over
    donors as independent support. Module-keyed counts remain annotations;
    donor-keyed counts retain donors as independent units.

    Results describe selected genes and overlapping tissues. Significant-pair
    counts depend on discovery power and do not establish causal direction.

    Args:
      merged_frame: Unit frame returned by `merge_eqtl_counts`.
      merge_mode: Level used to join the counts.
      schema: Column-name schema supplying tissue and donor keys.

    Returns:
      Regression-style rows with effects, support, FDR values, explicit
      analysis scope, response estimand, and eQTL provenance.
    """
    predictors = _association_predictors(
        merged_frame,
        merge_mode=merge_mode,
    )
    frames = _build_eqtl_association_frames(
        merged_frame,
        merge_mode=merge_mode,
        schema=schema,
        predictors=predictors,
    )

    tables: list[pd.DataFrame] = []
    if merge_mode == "gene":
        tables.extend(
            _regress_predictors(
                frames["eqtl_gene_total_across_genes"],
                predictors=predictors,
                scope="eqtl_gene_total_across_genes",
                estimand=(
                    "mean feature value per sensor gene across analyzed units"
                ),
                provenance=merged_frame,
                descriptive=True,
            )
        )
    elif merge_mode == "gene_tissue":
        tables.extend(
            _regress_predictors(
                frames["eqtl_tissue_count_within_gene"],
                predictors=predictors,
                scope="eqtl_tissue_count_within_gene",
                estimand="mean donor-level gene expression per atlas tissue",
                provenance=merged_frame,
                descriptive=True,
            )
        )
        tables.extend(
            _regress_predictors(
                frames["eqtl_tissue_count_within_tissue_across_genes"],
                predictors=predictors,
                scope="eqtl_tissue_count_within_tissue_across_genes",
                estimand=(
                    "mean donor-level expression per sensor gene and tissue"
                ),
                provenance=merged_frame,
                descriptive=True,
                stratify_key=schema.tissue_key,
            )
        )
    elif merge_mode == "tissue":
        tables.extend(
            _regress_predictors(
                frames["eqtl_sensor_burden_across_tissues"],
                predictors=predictors,
                scope="eqtl_sensor_burden_across_tissues",
                estimand="mean donor-level feature value per atlas tissue",
                provenance=merged_frame,
                descriptive=True,
            )
        )
    elif merge_mode == "donor":
        tables.extend(
            _regress_predictors(
                frames["eqtl_count_across_donors"],
                predictors=predictors,
                scope="eqtl_count_across_donors",
                estimand="mean feature value per donor",
                provenance=merged_frame,
                descriptive=False,
            )
        )

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


def _association_predictors(
    merged_frame: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
) -> list[str]:
    """Return valid predictors after checking the source statistical unit."""
    predictors = _value_columns(merged_frame)
    if not predictors:
        raise KeyError(
            "merged frame has no canonical eQTL count column; expected one of "
            f"{EQTL_VALUE_COLUMNS}"
        )
    if merge_mode in {"gene_tissue", "tissue"}:
        _require_columns(
            merged_frame,
            ("statistical_unit",),
            context="tissue-level eQTL association frame",
        )
        source_units = set(
            merged_frame["statistical_unit"].dropna().astype(str).unique()
        )
        if not source_units <= {"donor_tissue", "tissue"}:
            raise ValueError(
                "tissue-level eQTL associations require donor_tissue or "
                f"tissue input units, found: {sorted(source_units)}"
            )
    if merge_mode == "gene_tissue" and "tissue_specific_eqtls" in predictors:
        return ["tissue_specific_eqtls"]
    return predictors


def _build_eqtl_association_frames(
    merged_frame: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
    schema: ObsSchema,
    predictors: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Construct analysis frames after predictor and unit validation."""
    if merge_mode == "gene":
        genes = merged_frame.loc[
            merged_frame["feature_type"] == "gene_expression"
        ]
        summarized = _summarize_responses(
            genes,
            group_keys=_FEATURE_COLUMNS,
            predictors=predictors,
            statistical_unit="gene",
        )
        return {
            "eqtl_gene_total_across_genes": _as_gene_set(summarized),
        }

    if merge_mode == "gene_tissue":
        summarized = _summarize_responses(
            merged_frame,
            group_keys=(*_FEATURE_COLUMNS, schema.tissue_key),
            predictors=predictors,
            statistical_unit="tissue",
            schema=schema,
        )
        genes = summarized.loc[summarized["feature_type"] == "gene_expression"]
        return {
            "eqtl_tissue_count_within_gene": genes,
            "eqtl_tissue_count_within_tissue_across_genes": _as_gene_set(genes),
        }

    if merge_mode == "tissue":
        summarized = _summarize_responses(
            merged_frame,
            group_keys=(*_FEATURE_COLUMNS, schema.tissue_key),
            predictors=predictors,
            statistical_unit="tissue",
            schema=schema,
        )
        return {
            "eqtl_sensor_burden_across_tissues": summarized,
        }

    if merge_mode == "donor":
        summarized = _summarize_responses(
            merged_frame,
            group_keys=(*_FEATURE_COLUMNS, schema.donor_key),
            predictors=predictors,
            statistical_unit="donor",
            schema=schema,
        )
        return {
            "eqtl_count_across_donors": summarized,
        }

    return {}


def _source_schema(table: pd.DataFrame) -> str:
    """Identify a supported source schema without using its filename."""
    if "eqtl_source_schema" in table.columns:
        return _single_string(table["eqtl_source_schema"])
    if {
        "gene",
        "tissue",
        "n_significant_eqtls",
        "gene_total_eqtls",
    } <= set(table.columns):
        return "nasp_sensor_eqtl_counts_long"
    if {"gene", "gene_total_eqtls"} <= set(table.columns):
        return "nasp_sensor_eqtl_counts_wide"
    return "canonical"


def _rename_columns(table: pd.DataFrame) -> None:
    """Rename source aliases in place after checking collisions."""
    for alias, canonical in _COLUMN_ALIASES.items():
        if alias not in table.columns:
            continue
        if canonical in table.columns:
            left = table[alias].astype("string").fillna("<NA>")
            right = table[canonical].astype("string").fillna("<NA>")
            if not left.equals(right):
                raise ValueError(
                    f"eQTL columns {alias!r} and {canonical!r} disagree"
                )
            table.drop(columns=alias, inplace=True)
        else:
            table.rename(columns={alias: canonical}, inplace=True)


def _validate_supplied_schema(
    table: pd.DataFrame,
    *,
    source_schema: str,
) -> None:
    """Validate the sum and cardinality invariants of supplied NASP tables."""
    if source_schema == "nasp_sensor_eqtl_counts_wide":
        tissue_columns = [
            column
            for column in table.columns
            if column not in {"gene_symbol", "total_eqtls"}
            and not column.startswith("eqtl_")
        ]
        if not tissue_columns:
            raise KeyError(
                "wide eQTL table has no abbreviated tissue count columns"
            )
        for column in tissue_columns:
            table[column] = _count_series(table[column], column=column)
        calculated = table[tissue_columns].sum(axis="columns", min_count=1)
        mismatch = (
            calculated.notna()
            & table["total_eqtls"].notna()
            & calculated.ne(table["total_eqtls"].astype(float))
        )
        if mismatch.any():
            genes = table.loc[mismatch, "gene_symbol"].astype(str).head(3)
            raise ValueError(
                "wide eQTL tissue counts do not sum to gene totals for: "
                f"{', '.join(genes)}"
            )
        return

    if source_schema != "nasp_sensor_eqtl_counts_long":
        return
    _require_columns(
        table,
        ("gene_symbol", "tissue", "tissue_specific_eqtls"),
        context="long eQTL table",
    )
    _assert_unique(table, ("gene_symbol", "tissue"), context="long eQTL table")
    if "total_eqtls" not in table.columns:
        return

    grouped = table.groupby(
        "gene_symbol",
        observed=True,
        dropna=False,
    )
    inconsistent = grouped["total_eqtls"].nunique(dropna=True).gt(1)
    if inconsistent.any():
        genes = inconsistent.index[inconsistent].astype(str)[:3]
        raise ValueError(
            "long eQTL table has inconsistent repeated gene totals for: "
            f"{', '.join(genes)}"
        )
    calculated = grouped["tissue_specific_eqtls"].sum(min_count=1)
    reported = grouped["total_eqtls"].first()
    mismatch = (
        calculated.notna()
        & reported.notna()
        & calculated.ne(reported.astype(float))
    )
    if mismatch.any():
        genes = mismatch.index[mismatch].astype(str)[:3]
        raise ValueError(
            "long eQTL tissue counts do not sum to gene totals for: "
            f"{', '.join(genes)}"
        )


def _count_series(values: pd.Series, *, column: str) -> pd.Series:
    """Return nullable integer counts or raise on invalid source values."""
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float, na_value=np.nan)
    invalid = values.notna() & (
        numeric.isna()
        | ~np.isfinite(array)
        | (numeric < 0)
        | (numeric % 1 != 0)
    )
    if invalid.any():
        rows = ", ".join(values.index[invalid].astype(str)[:3])
        raise ValueError(
            f"eQTL count column {column!r} contains non-finite, negative, or "
            f"non-integer values at rows: {rows}"
        )
    return numeric.astype("Int64")


def _prepare_for_merge(
    table: pd.DataFrame,
    *,
    merge_mode: EqtlMergeMode,
) -> pd.DataFrame:
    """Reduce a validated table to one row per requested merge key."""
    keys = EQTL_REQUIRED_COLUMNS[merge_mode]
    if merge_mode == "gene" and table["gene_symbol"].duplicated().any():
        if "total_eqtls" not in table.columns:
            raise ValueError(
                "gene-mode eQTL table repeats genes without total_eqtls; "
                "use merge_mode='gene_tissue'"
            )
        inconsistent = (
            table.groupby("gene_symbol", observed=True)["total_eqtls"]
            .nunique(dropna=True)
            .gt(1)
        )
        if inconsistent.any():
            genes = inconsistent.index[inconsistent].astype(str)[:3]
            raise ValueError(
                "gene-mode eQTL table has inconsistent totals for: "
                f"{', '.join(genes)}"
            )
        return table.loc[:, ["gene_symbol", "total_eqtls"]].drop_duplicates(
            "gene_symbol"
        )

    if merge_mode == "tissue" and table["tissue"].duplicated().any():
        _require_columns(
            table,
            ("gene_symbol", "tissue_specific_eqtls"),
            context="repeated-tissue eQTL table",
        )
        _assert_unique(
            table,
            ("gene_symbol", "tissue"),
            context="repeated-tissue eQTL table",
        )
        records: list[dict[str, object]] = []
        for tissue, group in table.groupby(
            "tissue",
            observed=True,
            dropna=False,
            sort=False,
        ):
            record: dict[str, object] = {
                "tissue": tissue,
                "tissue_specific_eqtls": group["tissue_specific_eqtls"].sum(
                    min_count=1
                ),
                "eqtl_n_genes": int(group["gene_symbol"].nunique()),
            }
            if "eqtl_tissue_abbrev" in group.columns:
                record["eqtl_tissue_abbrev"] = _constant_value(
                    group["eqtl_tissue_abbrev"]
                )
            records.append(record)
        return pd.DataFrame.from_records(records)

    _assert_unique(table, keys, context="eQTL table")
    columns = [
        *keys,
        *_value_columns(table),
        *[
            column
            for column in _PROVENANCE_COLUMNS
            if column in table.columns and column not in keys
        ],
    ]
    return table.loc[:, list(dict.fromkeys(columns))].copy()


def _require_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    """Raise when `table` lacks required columns."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise KeyError(f"{context} missing columns: {missing}")


def _assert_unique(
    table: pd.DataFrame,
    keys: Sequence[str],
    *,
    context: str,
) -> None:
    """Reject duplicate keys with a deterministic preview."""
    duplicated = table.duplicated(list(keys), keep=False)
    if duplicated.any():
        preview = (
            table.loc[duplicated, list(keys)]
            .astype(str)
            .drop_duplicates()
            .head(3)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{context} has duplicate keys {tuple(keys)}: {preview}"
        )


def _value_columns(table: pd.DataFrame) -> list[str]:
    """Return canonical count columns in public schema order."""
    return [column for column in EQTL_VALUE_COLUMNS if column in table.columns]


def _single_string(values: pd.Series) -> str:
    """Return one non-missing string or reject ambiguous provenance."""
    unique = values.dropna().astype(str).unique()
    if unique.size != 1:
        raise ValueError(
            "eQTL provenance must contain exactly one non-missing value"
        )
    return str(unique[0])


def _join_keys(
    merge_mode: EqtlMergeMode,
    schema: ObsSchema,
) -> tuple[tuple[str, str], ...]:
    """Return ordered (unit-frame, eQTL-table) join-key pairs."""
    mapping: dict[EqtlMergeMode, tuple[tuple[str, str], ...]] = {
        "gene": (("feature_label", "gene_symbol"),),
        "gene_tissue": (
            ("feature_label", "gene_symbol"),
            (schema.tissue_key, "tissue"),
        ),
        "tissue": ((schema.tissue_key, "tissue"),),
        "module": (("feature_label", "module_id"),),
        "donor": ((schema.donor_key, "donor_id"),),
    }
    return mapping[merge_mode]


def _normalized_key(values: pd.Series, *, tissue: bool) -> pd.Series:
    """Normalize safe spelling differences without biological remapping."""
    normalized = values.astype("string").str.strip()
    if tissue:
        normalized = (
            normalized.str.casefold()
            .str.replace(r"[\s-]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
    return normalized


def _summarize_responses(
    frame: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    predictors: Sequence[str],
    statistical_unit: str,
    schema: ObsSchema | None = None,
) -> pd.DataFrame:
    """Average analyzed units once at the eQTL predictor's actual level."""
    if frame.empty:
        return frame.copy()
    _require_columns(
        frame,
        (*group_keys, "feature_value", *predictors),
        context="merged eQTL frame",
    )
    aggregations: dict[str, str | Callable[[pd.Series], object]] = {
        "feature_value": "mean",
        **dict.fromkeys(predictors, _constant_value),
    }
    if "aggregation" in frame.columns:
        aggregations["aggregation"] = _constant_value
    grouped = frame.groupby(
        list(group_keys),
        observed=True,
        dropna=False,
        sort=False,
    )
    summarized = grouped.agg(aggregations).reset_index()
    summarized["n_source_units"] = grouped.size().to_numpy(dtype=float)
    if schema is not None and schema.donor_key in frame.columns:
        summarized["n_donors"] = (
            grouped[schema.donor_key].nunique().to_numpy(dtype=float)
        )
    summarized["statistical_unit"] = statistical_unit
    unit_key = "feature_label"
    if schema is not None and schema.tissue_key in summarized.columns:
        unit_key = schema.tissue_key
    elif schema is not None and schema.donor_key in summarized.columns:
        unit_key = schema.donor_key
    summarized["unit_id"] = summarized[unit_key].astype(str)
    if "aggregation" not in summarized.columns:
        summarized["aggregation"] = "mean"
    return summarized


def _constant_value(values: pd.Series) -> object:
    """Return one non-missing value, NaN if absent, or reject disagreement."""
    unique = pd.unique(values.dropna())
    if len(unique) > 1:
        preview = ", ".join(str(value) for value in unique[:3])
        raise ValueError(
            f"eQTL predictor varies within one comparison unit: {preview}"
        )
    return unique[0] if len(unique) == 1 else np.nan


def _as_gene_set(frame: pd.DataFrame) -> pd.DataFrame:
    """Relabel individual sensor rows as one across-gene response family."""
    relabeled = frame.copy()
    if relabeled.empty:
        return relabeled
    for column in _FEATURE_COLUMNS:
        relabeled[f"source_{column}"] = relabeled[column]
    relabeled["feature_type"] = "gene_expression"
    relabeled["feature_id"] = "sensor_gene_expression_set"
    relabeled["feature_label"] = "Sensor gene expression"
    relabeled["statistical_unit"] = "gene"
    relabeled["unit_id"] = relabeled["source_feature_label"].astype(str)
    return relabeled


def _regress_predictors(
    frame: pd.DataFrame,
    *,
    predictors: Sequence[str],
    scope: str,
    estimand: str,
    provenance: pd.DataFrame,
    descriptive: bool,
    stratify_key: str | None = None,
) -> list[pd.DataFrame]:
    """Run one declared FDR family per eQTL predictor."""
    tables: list[pd.DataFrame] = []
    for predictor in predictors:
        result = regress_features_on_continuous(
            frame,
            predictor_key=predictor,
            stratify_key=stratify_key,
        )
        if result.empty:
            continue
        if descriptive:
            result["analysis_role"] = "descriptive"
        result["analysis_scope"] = scope
        result["response_estimand"] = estimand
        result["eqtl_fdr_family"] = f"{scope}:{predictor}"
        for column in (
            "eqtl_source_schema",
            "eqtl_count_unit",
            "eqtl_predictor_transform",
            "eqtl_table_path",
        ):
            if column in provenance.columns:
                result[column] = _single_string(provenance[column])
        tables.append(result)
    return tables
