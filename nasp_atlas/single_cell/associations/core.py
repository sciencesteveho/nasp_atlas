"""Core association types and shared helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


FeatureType = Literal["module_score", "gene_expression"]

StatisticalUnit = Literal[
    "cell",
    "metacell",
    "donor",
    "donor_tissue",
    "donor_tissue_cell_type",
    "donor_tissue_sex",
]

Aggregation = Literal[
    "mean",
    "median",
    "sum",
    "fraction_expressing",
    "percent_expressing",
]

EqtlMergeMode = Literal[
    "gene",
    "gene_tissue",
    "tissue",
    "module",
    "donor",
]

MIN_UNITS_FOR_TEST: int = 3
DESCRIPTIVE_UNITS: tuple[StatisticalUnit, ...] = ("cell", "metacell")

__all__ = [
    "DESCRIPTIVE_UNITS",
    "MIN_UNITS_FOR_TEST",
    "Aggregation",
    "EqtlMergeMode",
    "FeatureSpec",
    "FeatureType",
    "ObsSchema",
    "StatisticalUnit",
    "benjamini_hochberg",
    "metadata_columns",
]


def _statistic_pvalue(result: object) -> tuple[float, float]:
    """Return statistic and p-value from a SciPy test result."""
    values = np.asarray(result, dtype=float)
    return float(values[0]), float(values[1])


def _linregress_values(result: object) -> tuple[float, float, float]:
    """Return slope, intercept and p-value from `stats.linregress`."""
    values = np.asarray(result, dtype=float)
    return float(values[0]), float(values[1]), float(values[3])


@dataclass(frozen=True, kw_only=True)
class ObsSchema:
    """Names of the obs columns that carry association metadata.

    Grouping keys for every statistical unit are derived from these names, so a
    dataset with non-default column names is supported by constructing a schema
    once rather than threading individual key arguments through each function.
    """

    donor_key: str = "donor_id"
    tissue_key: str = "tissue_in_publication"
    cell_type_key: str = "cell_type"
    sex_key: str = "sex"
    assay_key: str = "assay"
    development_stage_key: str = "development_stage"
    age_key: str = "age_years"
    condition_key: str = "disease"
    study_key: str = "dataset_id"


@dataclass(frozen=True, kw_only=True)
class FeatureSpec:
    """A single feature to associate against a predictor.

    Attributes:
    feature_type: Whether the feature is a module score or a gene's expression.
    feature_id: The concrete lookup key -- a score-table column for module
      scores, or an `adata.var_names` entry for gene expression.
    feature_label: Human-facing label (module id or gene symbol).
    source: Which data object supplies the values, `scores` or `expression`.
    """

    feature_type: FeatureType
    feature_id: str
    feature_label: str
    source: Literal["scores", "expression"]


def _unit_group_keys(
    statistical_unit: StatisticalUnit,
    schema: ObsSchema,
) -> list[str]:
    """Return the obs grouping columns that define a statistical unit.

    Args:
    statistical_unit: The unit whose grouping keys are requested.
    schema: Column-name schema for the dataset.

    Returns:
    The obs column names whose unique combinations define one unit. The
    `cell` and `metacell` units group by the per-row identity and so return
    an empty list (each row is already one unit).
    """
    mapping: dict[str, list[str]] = {
        "donor": [schema.donor_key],
        "donor_tissue": [schema.donor_key, schema.tissue_key],
        "donor_tissue_cell_type": [
            schema.donor_key,
            schema.tissue_key,
            schema.cell_type_key,
        ],
        "donor_tissue_sex": [
            schema.donor_key,
            schema.tissue_key,
            schema.sex_key,
        ],
    }
    return mapping.get(statistical_unit, [])


def metadata_columns(schema: ObsSchema) -> list[str]:
    """Return the ordered metadata columns carried through aggregation.

    Args:
    schema: Column-name schema for the dataset.

    Returns:
    Metadata column names in a stable order for tidy output frames.
    """
    return [
        schema.donor_key,
        schema.tissue_key,
        schema.cell_type_key,
        schema.sex_key,
        schema.assay_key,
        schema.development_stage_key,
        schema.age_key,
        schema.condition_key,
        schema.study_key,
    ]


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values.

    A self-contained implementation keeps adjustment behavior stable across
    association methods. NaN inputs are preserved as NaN and excluded from the
    ranking denominator.

    Args:
    pvalues: Raw p-values, possibly containing NaN for skipped tests.

    Returns:
    Array of adjusted p-values aligned to the input order, with NaN where the
    input was NaN.
    """
    raw = np.asarray(pvalues, dtype=float)
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(raw)
    finite = raw[finite_mask]
    n_tests = finite.size
    if n_tests == 0:
        return adjusted
    order = np.argsort(finite, kind="mergesort")
    ranks = np.arange(1, n_tests + 1, dtype=float)
    scaled = finite[order] * n_tests / ranks
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty(n_tests, dtype=float)
    restored[order] = monotone
    adjusted[finite_mask] = restored
    return adjusted
