"""Donor-sample-cell-type units shared by every condition estimand stage."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.sensor_effects import sensor_donor_expression
from nasp_atlas.single_cell.sensor_features import sensor_catalog


logger = logging.getLogger(__name__)


UNIT_KEYS: tuple[str, ...] = ("donor_id", "sample_id", "cell_type")
SENSOR_MEASURES: tuple[str, ...] = ("gene_expression", "gene_detection")


@dataclass(frozen=True)
class ConditionUnits:
    """Aggregates per donor, sample and cell type, with their design.

    Attributes:
      metadata: One row per unit with its arm, covariates, cell count,
        median genes detected per cell and whether it meets minimum support.
      features: One row per unit and feature. `feature_type` is
        "module_score" (mean score, one row per scorer), "gene_expression"
        (mean log1p CP10k, zeros included) or "gene_detection" (fraction of
        cells with expression above zero). `status` keeps absent, ambiguous,
        undetected and non-finite sensors distinct from measured values.
      sensor_mapping: Compendium sensors and their source-feature mapping.
    """

    metadata: pd.DataFrame
    features: pd.DataFrame
    sensor_mapping: pd.DataFrame


def build_condition_units(
    h5ad_path: Path,
    scores: pd.DataFrame,
    spec: ConditionCohortSpec,
    *,
    scorers: Sequence[ScorerName],
    minimum_cells: int = 10,
    unit_keys: Sequence[str] = UNIT_KEYS,
    gene_symbol_column: str = "feature_name",
) -> ConditionUnits:
    """Aggregate module scores and sensor expression into analysis units.

    A unit is one combination of `unit_keys`, by default one donor, sample
    and cell type; the sample carries the arm, covariates and slope
    measurements. Finer keys, such as a cell state within cell type, give
    finer units with the same columns. Every unit is kept, and `supported`
    marks those with at least `minimum_cells` cells, so sparse units are
    counted rather than silently dropped. Sensors come from the compendium
    catalog; expression is the prepared log1p CP10k in X of `h5ad_path`, and
    scores are indexed by obs name. Inputs are not modified.
    """
    if minimum_cells < 1:
        raise ValueError("minimum_cells must be positive")

    keys = list(unit_keys)
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Prepared AnnData must have in-memory obs")

    metadata = _unit_metadata(
        obs, spec, unit_keys=keys, minimum_cells=minimum_cells
    )
    modules = _module_units(obs, scores, scorers=scorers, unit_keys=keys)

    catalog = sensor_catalog()
    sensors = _sensor_columns(adata, catalog, gene_symbol_column, keys)
    del adata
    genes, mapping = sensor_donor_expression(
        sensors,
        catalog,
        group_keys=keys,
        gene_symbol_column=gene_symbol_column,
    )

    features = pd.concat(
        [modules, _gene_units(genes, keys)], ignore_index=True
    ).merge(
        metadata[[*keys, "n_cells"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )

    return ConditionUnits(
        metadata=metadata, features=features, sensor_mapping=mapping
    )


def _unit_metadata(
    obs: pd.DataFrame,
    spec: ConditionCohortSpec,
    *,
    unit_keys: Sequence[str],
    minimum_cells: int,
) -> pd.DataFrame:
    """Describe each unit's design; sample-level values are constant."""
    covariates = [covariate.name for covariate in spec.covariates]
    measures = [
        f"{slope.estimand}_measure" for slope in spec.continuous_contrasts
    ]
    carried = [
        column
        for column in ("condition", "assay", "tissue", *covariates, *measures)
        if column not in unit_keys
    ]

    grouped = obs.groupby(list(unit_keys), observed=True)
    metadata = grouped[list(dict.fromkeys(carried))].first()
    metadata["n_cells"] = grouped.size()
    metadata["median_genes_detected"] = grouped["n_genes_detected"].median()
    metadata["log_median_genes_detected"] = np.log(
        metadata["median_genes_detected"].clip(lower=1.0)
    )

    metadata = metadata.reset_index()
    # Categorical obs columns would carry their alphabetical category order
    # into every downstream sort and map; units hold plain values instead.
    for column in metadata.columns:
        if isinstance(metadata[column].dtype, pd.CategoricalDtype):
            metadata[column] = metadata[column].astype(object)

    metadata["supported"] = metadata["n_cells"].ge(minimum_cells)
    return metadata


def _module_units(
    obs: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    scorers: Sequence[ScorerName],
    unit_keys: Sequence[str],
) -> pd.DataFrame:
    """Average every scored module within each unit, per scorer."""
    catalog = GeneModules()
    aligned = scores.reindex(obs.index.astype(str))
    if aligned.isna().all(axis=1).any():
        missing = int(aligned.isna().all(axis=1).sum())
        raise ValueError(f"{missing} prepared cells have no score row")

    keys = obs[list(unit_keys)].astype(str).set_axis(aligned.index)
    frames = []
    for scorer in scorers:
        for module_id in catalog.module_ids():
            column = module_score_name(
                catalog.get_module(module_id), scorer=scorer
            )
            if column not in aligned:
                logger.warning("Score column %s is absent", column)
                continue

            values = pd.to_numeric(aligned[column], errors="coerce")
            unit = (
                keys.assign(value=values)
                .groupby(list(unit_keys), observed=True)["value"]
                .agg(["mean", "count"])
                .reset_index()
            )

            frames.append(
                unit.rename(
                    columns={"mean": "feature_value", "count": "n_finite"}
                ).assign(
                    feature_type="module_score",
                    feature_id=column,
                    feature_label=module_id,
                    scorer=scorer,
                    status=lambda frame: np.where(
                        frame["n_finite"].gt(0), "present", "nonfinite"
                    ),
                )
            )

    return pd.concat(frames, ignore_index=True)


def _sensor_columns(
    adata: ad.AnnData,
    catalog: pd.DataFrame,
    gene_symbol_column: str,
    unit_keys: Sequence[str],
) -> ad.AnnData:
    """Keep every feature matching a sensor name or alias, plus obs keys.

    All candidate matches are retained, so the downstream resolver still
    sees any ambiguity it would see in the full gene universe.
    """
    labels = {
        token.strip()
        for row in catalog.itertuples(index=False)
        for token in (str(row.gene), *str(row.aliases).split(";"))
        if token.strip()
    }

    var = adata.var
    symbols = (
        var[gene_symbol_column].astype(str)
        if gene_symbol_column in var
        else pd.Series(var.index.astype(str), index=var.index)
    )

    keep = var.index.astype(str).isin(labels) | symbols.isin(labels)
    subset = adata[:, np.flatnonzero(keep)].copy()

    obs = subset.obs
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Prepared AnnData must have in-memory obs")
    subset.obs = obs[list(unit_keys)].astype(str)
    return subset


def _gene_units(genes: pd.DataFrame, unit_keys: Sequence[str]) -> pd.DataFrame:
    """Split sensor aggregates into level and detection features."""
    common = genes[[*unit_keys, "gene", "status", "n_finite_cells"]].rename(
        columns={"n_finite_cells": "n_finite"}
    )
    level, detection = SENSOR_MEASURES

    return (
        pd.concat(
            [
                common.assign(
                    feature_type=level, feature_value=genes["mean_expression"]
                ),
                common.assign(
                    feature_type=detection,
                    feature_value=genes["fraction_detected"],
                ),
            ],
            ignore_index=True,
        )
        .rename(columns={"gene": "feature_label"})
        .assign(
            feature_id=lambda frame: frame["feature_label"],
            scorer=lambda frame: pd.Series(
                pd.NA, index=frame.index, dtype="string"
            ),
        )
    )
