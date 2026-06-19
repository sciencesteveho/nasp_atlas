"""Score NASP gene modules in single-cell data."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import anndata as ad  # type: ignore
import pandas as pd
import scanpy as sc  # type: ignore
from nasp_compendium import GeneModules  # type: ignore

from nasp_atlas.single_cell.visualization import SCVisualizer


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from pyscenic.aucell import GeneSignature  # type: ignore  # noqa: E402
from pyscenic.aucell import aucell  # type: ignore  # noqa: E402


logger = logging.getLogger(__name__)


def score_scanpy_modules(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    gene_symbol_column: str = "feature_name",
    random_state: int = 0,
) -> list[Any]:
    """Score signed NASP modules with scanpy score_genes."""
    modules = []
    for module_id in module_ids:
        module = GeneModules.modules(
            module_id,
            scorer="scanpy",
            adata=adata,
            gene_symbol_column=gene_symbol_column,
        )
        modules.append(module)

        logger.info(
            "%s missing_pos=%s missing_inv=%s",
            module.module_id,
            module.missing_positive_genes,
            module.missing_inverse_genes,
        )

        for kwargs in module.scanpy_score_kwargs(random_state=random_state):
            sc.tl.score_genes(adata, **kwargs)

        adata.obs[module.score_name] = module.combine_scores(adata.obs)

    return modules


def score_aucell_modules(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    gene_symbol_column: str = "feature_name",
) -> tuple[ad.AnnData, pd.DataFrame, list[Any]]:
    """Score signed NASP modules with pySCENIC AUCell."""
    if adata.isbacked:
        raise ValueError(
            "score_aucell_modules expects an in-memory AnnData. "
            "Call random_cell_subset first for backed data."
        )

    adata_auc = adata.copy()
    symbols = adata_auc.var[gene_symbol_column].astype(str)
    keep = symbols.notna() & (symbols != "") & ~symbols.duplicated()
    expression = adata_auc[:, keep.to_numpy()].X
    if hasattr(expression, "toarray"):
        expression = expression.toarray()  # type: ignore

    expression_df = pd.DataFrame(
        expression,  # type: ignore
        index=adata_auc.obs_names.astype(str),
        columns=symbols[keep].to_numpy(),
    )

    modules = [
        GeneModules.modules(
            module_id,
            scorer="aucell",
            adata=adata_auc,
            gene_symbol_column=gene_symbol_column,
        )
        for module_id in module_ids
    ]
    signatures = [
        GeneSignature(name=name, gene2weight=dict.fromkeys(genes, 1.0))
        for module in modules
        for name, genes in module.gene_sets().items()
    ]

    auc_df = aucell(expression_df, signatures)  # type: ignore
    for module in modules:
        auc_df[module.score_name] = module.combine_scores(auc_df)
        adata_auc.obs[module.score_name] = auc_df[module.score_name]

    return adata_auc, auc_df, modules


def plot_module_gene_umaps(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    viz: SCVisualizer,
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    ncols: int = 6,
    size: float | None = None,
) -> None:
    """Plot one multi-gene UMAP panel per NASP gene module."""
    point_size = size if size is not None else 120000 / adata.n_obs
    for module_id in module_ids:
        module_genes = GeneModules.genes(
            module_id,
            adata=adata,
            gene_symbol_column=gene_symbol_column,
            output="symbols",
        )
        if not module_genes:
            logger.info("%s: no matched genes; skipping UMAPs", module_id)
            continue

        logger.info(
            "%s: plotting %s marker genes",
            module_id,
            len(module_genes),
        )
        viz.plot_multi_gene_umap_panel(
            adata=adata,
            genes=module_genes,
            filename=f"{module_id}_gene_expression_umaps",
            gene_symbol_column=gene_symbol_column,
            expression_layer=expression_layer,
            ncols=ncols,
            size=point_size,
        )
