"""Resolve plotting gene identifiers and marker groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class _ResolvedGenes:
    """Resolved plotting genes and display labels."""

    var_names: list[str]
    labels: list[str]


@dataclass(frozen=True)
class _DotplotGenes:
    """Resolved gene list and optional marker group annotations."""

    var_names: list[str]
    labels: list[str]
    group_labels: list[str] | None
    group_positions: list[tuple[int, int]] | None


class _VisualizationGeneMixin:
    """Gene and marker-group resolution for plotting methods."""

    @staticmethod
    def _resolve_genes(
        adata: Any,
        genes: list[str],
        *,
        gene_symbol_column: str | None = None,
    ) -> _ResolvedGenes:
        """Resolve user-facing gene names to adata.var_names."""
        var_names = pd.Index(adata.var_names).astype(str)
        lookup: dict[str, tuple[str, str]] = {
            str(var_name): (str(var_name), str(var_name))
            for var_name in var_names
        }

        if gene_symbol_column is not None:
            if gene_symbol_column not in adata.var.columns:
                raise ValueError(
                    f"gene_symbol_column not found in adata.var: "
                    f"{gene_symbol_column}"
                )
            for var_name, symbol in zip(
                var_names,
                adata.var[gene_symbol_column].astype(str),
                strict=True,
            ):
                if not symbol or symbol == "nan":
                    continue
                lookup.setdefault(symbol, (str(var_name), symbol))

        resolved_var_names = []
        labels = []
        seen = set()
        for gene in genes:
            if gene not in lookup:
                continue
            var_name, label = lookup[gene]
            if var_name in seen:
                continue
            seen.add(var_name)
            resolved_var_names.append(var_name)
            labels.append(label)

        return _ResolvedGenes(var_names=resolved_var_names, labels=labels)

    @staticmethod
    def _resolve_dotplot_genes(
        adata: Any,
        genes: list[str] | None,
        marker_groups: dict[str, list[str]] | None,
        gene_symbol_column: str | None,
    ) -> _DotplotGenes:
        """Resolve the gene list and optional group annotations from inputs."""
        if marker_groups is None:
            resolved = _VisualizationGeneMixin._resolve_genes(
                adata,
                genes or [],
                gene_symbol_column=gene_symbol_column,
            )
            return _DotplotGenes(
                var_names=resolved.var_names,
                labels=resolved.labels,
                group_labels=None,
                group_positions=None,
            )

        group_labels: list[str] = []
        group_positions: list[tuple[int, int]] = []
        var_names: list[str] = []
        labels: list[str] = []
        idx = 0

        for label, group_genes in marker_groups.items():
            resolved = _VisualizationGeneMixin._resolve_genes(
                adata,
                group_genes,
                gene_symbol_column=gene_symbol_column,
            )
            if not resolved.var_names:
                continue
            group_labels.append(label)
            group_positions.append((idx, idx + len(resolved.var_names) - 1))
            var_names.extend(resolved.var_names)
            labels.extend(resolved.labels)
            idx += len(resolved.var_names)

        return _DotplotGenes(var_names, labels, group_labels, group_positions)
