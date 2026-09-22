"""Stable-gene reconciliation without changing curated module membership."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import anndata as ad  # type: ignore[import]
import pandas as pd
from nasp_compendium.types import GeneModule  # type: ignore[import]


__all__ = [
    "FeatureAlignment",
    "align_cohort_features",
    "apply_feature_alignment",
    "panel_aliases",
]


@dataclass(frozen=True)
class FeatureAlignment:
    """Annotation crosswalks, shared background and module gene coverage."""

    reference_features: pd.DataFrame
    external_features: pd.DataFrame
    shared_gene_ids: tuple[str, ...]
    coverage: pd.DataFrame


def panel_aliases(panel: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Read aliases from the compendium public panel without adding members."""
    if "gene_symbol" not in panel:
        raise KeyError("Compendium panel is missing gene_symbol")
    if "aliases" not in panel:
        return {}
    aliases: dict[str, set[str]] = {}
    for gene, value in panel[["gene_symbol", "aliases"]].itertuples(
        index=False, name=None
    ):
        if pd.notna(value):
            aliases.setdefault(str(gene), set()).update(
                token.strip()
                for token in str(value).split(";")
                if token.strip()
            )
    return {gene: tuple(sorted(values)) for gene, values in aliases.items()}


def align_cohort_features(
    reference_var: pd.DataFrame,
    external_var: pd.DataFrame,
    modules: Sequence[GeneModule],
    *,
    reference_id_column: str,
    reference_symbol_column: str,
    external_id_column: str,
    external_symbol_column: str,
    aliases: Mapping[str, Sequence[str]],
) -> FeatureAlignment:
    """Resolve curated symbols through shared stable IDs and observed aliases.

    Each curated symbol must resolve to one stable gene across both files.
    Multiple candidates or two curated symbols naming one gene are ambiguous.
    Missing and ambiguous rows are retained in coverage. Background genes use
    stable IDs as scoring labels, avoiding arbitrary duplicate-symbol loss.
    This changes annotation only, not marker membership or source expression.
    """
    if any(module.gene_id_output != "symbols" for module in modules):
        raise ValueError("Feature alignment requires canonical symbol modules")
    reference = _feature_annotations(
        reference_var, reference_id_column, reference_symbol_column
    )
    external = _feature_annotations(
        external_var, external_id_column, external_symbol_column
    )
    shared = tuple(sorted(set(reference.stable_id) & set(external.stable_id)))
    if not shared:
        raise ValueError("Cohorts have no shared stable gene background")

    observed = pd.concat([reference, external], ignore_index=True)
    symbol_ids = {
        str(symbol): set(group.stable_id)
        for symbol, group in observed.dropna(subset=["source_symbol"]).groupby(
            "source_symbol", observed=True
        )
    }
    candidates: dict[str, set[str]] = {}
    for gene in sorted({gene for module in modules for gene in module.genes}):
        labels = (gene, *aliases.get(gene, ()))
        candidates[gene] = set().union(
            *(symbol_ids.get(label, set()) for label in labels)
        )
    unique = {
        gene: next(iter(ids))
        for gene, ids in candidates.items()
        if len(ids) == 1
    }
    duplicates = pd.Series(unique, dtype=object).duplicated(keep=False)
    ambiguous_symbols = set(duplicates.index[duplicates])
    resolved = {
        gene: stable_id
        for gene, stable_id in unique.items()
        if gene not in ambiguous_symbols
    }
    coverage = _module_coverage(
        modules,
        candidates,
        resolved,
        set(reference.stable_id),
        set(external.stable_id),
    )
    scoring_labels = {stable_id: gene for gene, stable_id in resolved.items()}
    for frame in (reference, external):
        frame["feature_name"] = frame.stable_id.map(scoring_labels).fillna(
            frame.stable_id
        )
        frame["in_shared_background"] = frame.stable_id.isin(shared)
        if frame.feature_name.duplicated().any():
            raise ValueError(
                "Canonical scoring labels collide with background IDs"
            )
    return FeatureAlignment(reference, external, shared, coverage)


def apply_feature_alignment(
    prepared: ad.AnnData,
    features: pd.DataFrame,
    shared_gene_ids: Sequence[str],
) -> ad.AnnData:
    """Subset already-normalized expression to the ordered shared background.

    Original feature IDs and symbols remain in var; var_names become stable
    IDs and feature_name contains canonical scoring labels. Normalization
    denominators are preserved. Returns a copy; no input is modified.
    """
    if "expression_preparation" not in prepared.uns:
        raise ValueError(
            "Normalize on the full source universe before alignment"
        )
    if not prepared.var_names.equals(features.index):
        raise ValueError(
            "Feature crosswalk does not align to prepared source var"
        )
    lookup = features.set_index("stable_id", drop=False)
    selected = lookup.loc[list(shared_gene_ids)]
    output = prepared[:, selected.source_gene_id.tolist()].copy()
    for column in selected.columns:
        output.var[column] = selected[column].to_numpy()
    output.var_names = list(shared_gene_ids)
    output.uns["expression_preparation"] = {
        **prepared.uns["expression_preparation"],
        "shared_background_n_genes": len(shared_gene_ids),
    }
    return output


def _feature_annotations(
    var: pd.DataFrame, id_column: str, symbol_column: str
) -> pd.DataFrame:
    """Preserve original identifiers and validate unique unversioned genes."""
    identifiers = (
        pd.Series(var.index, index=var.index)
        if id_column == "_index"
        else var[id_column]
    )
    stable = identifiers.astype("string").str.replace(r"\.\d+$", "", regex=True)
    if stable.isna().any() or stable.eq("").any() or stable.duplicated().any():
        raise ValueError("Stable gene IDs must be nonempty and one-to-one")
    if not var.index.is_unique:
        raise ValueError("Source feature IDs must be unique")
    return pd.DataFrame(
        {
            "source_gene_id": var.index,
            "source_stable_id": identifiers,
            "stable_id": stable,
            "source_symbol": var[symbol_column].astype("string").str.strip(),
        },
        index=var.index,
    )


def _module_coverage(
    modules: Sequence[GeneModule],
    candidates: Mapping[str, set[str]],
    resolved: Mapping[str, str],
    reference_ids: set[str],
    external_ids: set[str],
) -> pd.DataFrame:
    """Retain every arm member and its mapping or unavailability evidence."""
    rows = []
    for module in modules:
        arms = (
            ("positive", module.positive_genes),
            ("inverse", module.inverse_genes),
            ("context_dependent", module.context_dependent_genes),
        )
        for arm, genes in arms:
            for gene in genes:
                stable_id = resolved.get(gene)
                if not candidates[gene]:
                    status = "absent_both"
                elif stable_id is None:
                    status = "ambiguous"
                elif stable_id not in reference_ids:
                    status = "absent_reference"
                elif stable_id not in external_ids:
                    status = "absent_external"
                else:
                    status = "matched"
                rows.append(
                    {
                        "module_id": module.module_id,
                        "gene": gene,
                        "arm": arm,
                        "stable_id": stable_id,
                        "status": status,
                        "candidate_ids": ";".join(sorted(candidates[gene])),
                        "scored": arm != "context_dependent",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "module_id",
            "gene",
            "arm",
            "stable_id",
            "status",
            "candidate_ids",
            "scored",
        ],
    )
