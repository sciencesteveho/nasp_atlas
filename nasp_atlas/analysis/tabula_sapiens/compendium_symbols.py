"""Name atlas features as the compendium does, using its curated aliases."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.external_replication.feature_mapping import (
    panel_aliases,
)


logger = logging.getLogger(__name__)

__all__ = ["COMPENDIUM_SYMBOL_COLUMN", "add_compendium_symbol_column"]

# Shared by the scoring and association stages, which must match compendium
# symbols through the same column.
COMPENDIUM_SYMBOL_COLUMN: str = "compendium_symbol"


def add_compendium_symbol_column(
    adata: ad.AnnData,
    *,
    source_column: str,
    target_column: str = COMPENDIUM_SYMBOL_COLUMN,
    panel_path: str | Path | None = None,
) -> dict[str, str]:
    """Add a var column that names features the way the compendium does.

    Atlas annotations can use a newer symbol than the compendium, such as
    RIGI for the curated DDX58, and compendium matching compares symbols
    exactly. For each curated gene absent from `source_column`, the
    compendium's own `aliases` are consulted. The matching feature takes the
    curated symbol only when exactly one alias names exactly one feature, that
    alias is not itself a curated gene, and no other curated gene claims it.
    Every other feature keeps its source symbol, so module membership never
    extends beyond the compendium. `adata.var`, and `adata.raw.var` when
    present, gain `target_column` in place; `source_column` is unchanged.

    Args:
      adata: AnnData whose features are matched to compendium symbols.
      source_column: Var column holding the dataset's gene symbols; feature
        IDs stand in when the column is absent, as in module scoring.
      target_column: New var column used for compendium matching.
      panel_path: Compendium panel; None uses the installed default, the same
        panel `GeneModules` resolves elsewhere in the workflow.

    Returns:
      Applied substitutions as curated symbol -> source symbol.
    """
    catalog = GeneModules(panel_path=panel_path)
    curated = set(catalog.panel["gene_symbol"].astype(str))
    aliases = panel_aliases(catalog.panel)

    applied: dict[str, str] = {}
    frames = [adata.var]
    if adata.raw is not None:
        frames.append(adata.raw.var)
    for var in frames:
        if target_column in var:
            raise ValueError(
                f"var already has a {target_column!r} column; pass another "
                "target_column"
            )
        symbols = (
            var[source_column].astype(object)
            if source_column in var
            else pd.Series(var.index, index=var.index, dtype=object)
        )
        substitutions = _alias_substitutions(
            symbols, curated=curated, aliases=aliases
        )
        var[target_column] = symbols.replace(
            {source: gene for gene, source in substitutions.items()}
        )
        applied.update(substitutions)
    return applied


def _alias_substitutions(
    symbols: pd.Series,
    *,
    curated: set[str],
    aliases: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Return unambiguous curated-symbol to source-symbol alias matches."""
    counts = symbols.dropna().astype(str).value_counts()
    unique_symbols = set(counts.index[counts.eq(1)])
    absent = sorted(curated - set(counts.index))

    claims: dict[str, list[str]] = {}
    for gene in absent:
        hits = [
            alias
            for alias in aliases.get(gene, ())
            if alias in unique_symbols and alias not in curated
        ]
        if len(hits) == 1:
            claims.setdefault(hits[0], []).append(gene)
    substitutions = {
        genes[0]: alias for alias, genes in claims.items() if len(genes) == 1
    }

    if substitutions:
        logger.info(
            "Matched compendium genes through curated aliases: %s",
            substitutions,
        )
    if unresolved := [gene for gene in absent if gene not in substitutions]:
        logger.warning(
            "Compendium genes absent from the dataset after alias matching: %s",
            unresolved,
        )
    return substitutions
