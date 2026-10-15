"""Leave-self-out IFN-I output scores for sensors inside the module (AB3)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.condition_cohorts.units import UNIT_KEYS
from nasp_atlas.analysis.tabula_sapiens.compendium_symbols import (
    COMPENDIUM_SYMBOL_COLUMN,
)
from nasp_atlas.analysis.tabula_sapiens.compendium_symbols import (
    add_compendium_symbol_column,
)
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_modules
from nasp_atlas.single_cell.sensor_features import sensor_catalog


logger = logging.getLogger(__name__)


def sensors_in_module(module_id: str = "IFN_I_OUTPUT") -> list[str]:
    """Return compendium sensors that are also genes of `module_id`.

    Such a sensor is part of its own adjustment score, such as IFIT1, an
    RNA sensor that is also an interferon-stimulated output gene.
    """
    return sorted(
        set(GeneModules.genes(module_id)) & set(sensor_catalog()["gene"])
    )


def leave_self_out_ifn_scores(
    h5ad_path: Path,
    *,
    sensors: Sequence[str],
    scorer: ScorerName,
    random_state: int,
    module_id: str = "IFN_I_OUTPUT",
    unit_keys: Sequence[str] = UNIT_KEYS,
) -> pd.DataFrame:
    """Score `module_id` without each sensor and average it per unit.

    A sensor that belongs to the module would otherwise be adjusted for a
    score that contains it. For each sensor, the compendium module minus
    that gene is scored on the prepared expression (log1p CP10k in X) with
    the scoring stage's scorer, gene matching and seed, and averaged over
    each unit's cells like the published module scores. The curated module
    itself is unchanged; this is a per-sensor adjustment covariate only.

    Returns:
      One row per unit and sensor: `unit_keys`, `feature_label` (the
      sensor), `ifn_output` (unit mean score without it), `n_finite`,
      `score_definition` and `n_module_genes` (genes scored).
    """
    if not sensors:
        raise ValueError("Pass the sensors that belong to the module")

    module = GeneModules.modules(module_id)
    variants = [
        replace(
            module,
            module_id=f"{module_id}_without_{sensor}",
            positive_genes=tuple(
                g for g in module.positive_genes if g != sensor
            ),
            inverse_genes=tuple(g for g in module.inverse_genes if g != sensor),
        )
        for sensor in sensors
    ]

    adata = ad.read_h5ad(h5ad_path)
    add_compendium_symbol_column(adata, source_column="feature_name")
    variant_ids = [variant.module_id for variant in variants]

    if scorer == "scanpy":
        resolved = score_scanpy_modules(
            adata,
            variant_ids,
            gene_symbol_column=COMPENDIUM_SYMBOL_COLUMN,
            random_state=random_state,
            expression_layer=None,
            gene_modules=variants,
        )
        scores = adata.obs
    else:
        _, scores, resolved = score_aucell_modules(
            adata,
            variant_ids,
            gene_symbol_column=COMPENDIUM_SYMBOL_COLUMN,
            random_state=random_state,
            expression_layer=None,
            gene_modules=variants,
        )
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("Leave-self-out scoring needs in-memory scores")

    obs = adata.obs
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Prepared AnnData must have in-memory obs")
    keys = obs[list(unit_keys)].astype(str)

    frames = []
    for sensor, variant in zip(sensors, resolved, strict=True):
        values = pd.to_numeric(
            scores[module_score_name(variant, scorer=scorer)], errors="coerce"
        ).reindex(keys.index)

        unit = (
            keys.assign(value=values)
            .groupby(list(unit_keys), observed=True)["value"]
            .agg(["mean", "count"])
            .reset_index()
        )
        frames.append(
            unit.rename(
                columns={"mean": "ifn_output", "count": "n_finite"}
            ).assign(
                feature_label=sensor,
                score_definition=variant.module_id,
                n_module_genes=len(variant.positive_genes)
                + len(variant.inverse_genes),
            )
        )

        logger.info(
            "%s scored with %d genes for the %s adjustment",
            variant.module_id,
            len(variant.positive_genes) + len(variant.inverse_genes),
            sensor,
        )

    return pd.concat(frames, ignore_index=True)
