"""Donor-paired sensor expression with detection and measurement support."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from nasp_atlas.single_cell.associations.core import benjamini_hochberg
from nasp_atlas.single_cell.associations.paired import PairedContrastResult
from nasp_atlas.single_cell.associations.paired import paired_feature_contrasts
from nasp_atlas.single_cell.sensor_features import resolve_sensor_features


__all__ = [
    "paired_sensor_effects",
    "sensor_donor_expression",
    "unique_sensor_measurements",
]


def sensor_donor_expression(
    expression: ad.AnnData,
    catalog: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    gene_symbol_column: str = "feature_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate prepared expression, reading one sensor at a time.

    X must already have the declared normalization. Detection is X > 0 among
    finite cells. No renormalization or module scoring occurs. Unresolved genes
    retain one unavailable row per donor/context. Inputs are not modified.
    """
    keys = list(group_keys)
    obs, var = expression.obs, expression.var
    if not isinstance(obs, pd.DataFrame) or not isinstance(var, pd.DataFrame):
        raise TypeError("Sensor aggregation requires in-memory obs and var")
    metadata = obs[keys].copy()
    if metadata.isna().any(axis=None) or not expression.obs_names.is_unique:
        raise ValueError(
            "Require unique cells and non-null sensor grouping keys"
        )
    matrix = expression.X
    if matrix is None:
        raise ValueError("Sensor aggregation requires prepared expression in X")
    mapping = resolve_sensor_features(
        var, catalog, gene_symbol_column=gene_symbol_column
    )
    tables = []
    for row in mapping.itertuples(index=False):
        values = np.full(expression.n_obs, np.nan)
        if row.mapping_status == "present":
            position = int(
                expression.var_names.get_indexer(
                    pd.Index([str(row.source_feature_id)])
                )[0]
            )
            column = matrix[:, position : position + 1]
            values = (
                column.toarray().ravel()  # type: ignore[union-attr]
                if sp.issparse(column)
                else np.asarray(column).ravel()
            )
        finite = np.isfinite(values)
        if np.any(values[finite] < 0):
            raise ValueError(
                "Sensor detection requires nonnegative, uncentered expression"
            )
        frame = metadata.assign(
            expression=np.where(finite, values, np.nan),
            detected=np.where(finite, values > 0, np.nan),
        )
        donor = (
            frame.groupby(keys, observed=True)
            .agg(
                n_cells=("expression", "size"),
                n_finite_cells=("expression", "count"),
                mean_expression=("expression", "mean"),
                fraction_detected=("detected", "mean"),
            )
            .reset_index()
        )
        donor["gene"] = row.gene
        donor["status"] = row.mapping_status
        donor["n_nonfinite_cells"] = donor.n_cells - donor.n_finite_cells
        if row.mapping_status == "present":
            donor.loc[donor.n_finite_cells.eq(0), "status"] = "nonfinite"
            donor.loc[donor.fraction_detected.eq(0), "status"] = "undetected"
        tables.append(donor)
    return pd.concat(tables, ignore_index=True), mapping


def unique_sensor_measurements(
    donor_expression: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    context_keys: Sequence[str],
) -> pd.DataFrame:
    """Validate and collapse identical module/scorer copies of each measurement.

    Absent catalog genes are explicitly expanded over observed donor contexts;
    conflicting copies fail instead of selecting an arbitrary module/scorer.
    """
    keys = [*context_keys, "gene"]
    values = [
        "n_cells",
        "n_finite_cells",
        "n_nonfinite_cells",
        "mean_expression",
        "fraction_detected",
        "status",
    ]
    if donor_expression[list(context_keys)].isna().any(axis=None):
        raise ValueError("Sensor donor/context identifiers must not be missing")
    selected = donor_expression.loc[
        donor_expression.gene.isin(catalog.gene), [*keys, *values]
    ]
    if (
        selected.groupby(keys, observed=True)[values]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=None)
    ):
        raise ValueError(
            "Conflicting duplicated sensor measurements across modules/scorers"
        )
    context_support = donor_expression[
        [*context_keys, "n_cells"]
    ].drop_duplicates()
    if context_support.duplicated(list(context_keys)).any():
        raise ValueError("Cell counts disagree within a donor/context")
    complete = context_support.merge(catalog[["gene"]], how="cross")
    result = complete.merge(
        selected.drop_duplicates(keys).drop(columns="n_cells"),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    missing = result.status.isna()
    result.loc[missing, "status"] = "absent_from_saved_table"
    result.loc[missing, "n_finite_cells"] = 0
    result.loc[missing, "n_nonfinite_cells"] = result.loc[missing, "n_cells"]
    if (
        result.n_finite_cells.gt(result.n_cells).any()
        or result.n_finite_cells.lt(0).any()
    ):
        raise ValueError("Invalid finite-cell support")
    if (
        (result.n_finite_cells + result.n_nonfinite_cells)
        .ne(result.n_cells)
        .any()
    ):
        raise ValueError(
            "Finite and nonfinite cell counts must sum to total cells"
        )
    measured = result.n_finite_cells.gt(0)
    if not np.isfinite(
        result.loc[measured, ["mean_expression", "fraction_detected"]]
    ).all(axis=None):
        raise ValueError(
            "Positive finite-cell counts require finite measurements"
        )
    if not result.loc[measured, "fraction_detected"].between(0, 1).all():
        raise ValueError("Detection fractions must be between zero and one")
    return result


def paired_sensor_effects(
    donor_expression: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    pair_keys: Sequence[str],
    level_key: str,
    target: str,
    reference: str,
    minimum_cells: int = 10,
    minimum_pairs: int = 6,
    alpha: float = 0.05,
) -> PairedContrastResult:
    """Estimate equal-person expression differences; detection is descriptive.

    Exactly one assay must contribute per person across the compared levels.
    Expression inference uses paired t intervals and BH across every catalog
    sensor in this comparison. Unavailable tests count as p=1 only in the BH
    denominator; their reported raw and adjusted p-values remain missing.
    Pointwise intervals are not multiplicity adjusted. No primary decisions or
    module-score attribution are made. All support reasons remain in the pairs.
    """
    if minimum_cells < 1:
        raise ValueError("minimum_cells must be positive")
    frame = donor_expression.loc[
        donor_expression[level_key].isin([target, reference])
    ].copy()
    if frame.empty:
        raise ValueError("No donor measurements in the requested comparison")
    if (
        "assay" in frame
        and frame.groupby(list(pair_keys), observed=True)
        .assay.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "Require one assay per person; stratify sensor comparisons"
        )
    frame["support_reason"] = np.select(
        [
            ~frame.status.isin(["present", "undetected"]),
            frame.n_finite_cells.lt(minimum_cells),
        ],
        [frame.status, "below_cell_floor"],
        default="supported",
    )
    frame = frame.assign(
        feature_type="gene_expression",
        feature_id=frame.gene,
        feature_label=frame.gene,
        feature_value=frame.mean_expression.where(
            frame.support_reason.eq("supported")
        ),
    )
    result = paired_feature_contrasts(
        frame,
        pair_keys=pair_keys,
        level_key=level_key,
        target_level=target,
        reference_level=reference,
        minimum_pairs=minimum_pairs,
        alpha=alpha,
    )
    estimates = catalog.merge(
        result.estimates,
        left_on="gene",
        right_on="feature_id",
        how="left",
        validate="one_to_one",
    )
    estimates["status"] = estimates.status.fillna("absent_from_saved_table")
    raw = estimates.pvalue.to_numpy(dtype=float)
    adjusted = benjamini_hochberg(np.where(np.isfinite(raw), raw, 1.0).tolist())
    estimates["pvalue_bh"] = np.where(np.isfinite(raw), adjusted, np.nan)
    estimates["family_size"] = len(catalog)
    estimates["analysis_role"] = "exploratory_sensor_expression"
    estimates["minimum_cells"] = minimum_cells
    estimates["minimum_pairs"] = minimum_pairs

    pairs = result.donor_differences.rename(columns={"feature_id": "gene"})
    keys = [*pair_keys, "gene"]
    support_columns = [
        "n_cells",
        "n_finite_cells",
        "fraction_detected",
        "support_reason",
    ]
    for prefix, level in (("target", target), ("reference", reference)):
        support = frame.loc[
            frame[level_key].eq(level), [*keys, *support_columns]
        ]
        pairs = pairs.merge(
            support.rename(
                columns={key: f"{prefix}_{key}" for key in support_columns}
            ),
            on=keys,
            how="left",
            validate="one_to_one",
        )
        pairs[f"{prefix}_support_reason"] = pairs[
            f"{prefix}_support_reason"
        ].fillna(f"missing_{prefix}")
    complete = pairs.status.eq("complete")
    pairs["detection_difference"] = (
        pairs.target_fraction_detected - pairs.reference_fraction_detected
    ).where(complete)
    summary = (
        pairs.loc[complete]
        .groupby("gene", observed=True)
        .agg(
            mean_target=("target_value", "mean"),
            mean_reference=("reference_value", "mean"),
            detected_target=("target_fraction_detected", "mean"),
            detected_reference=("reference_fraction_detected", "mean"),
            detection_difference=("detection_difference", "mean"),
        )
        .reset_index()
    )
    estimates = estimates.merge(
        summary, on="gene", how="left", validate="one_to_one"
    )
    return PairedContrastResult(estimates=estimates, donor_differences=pairs)
