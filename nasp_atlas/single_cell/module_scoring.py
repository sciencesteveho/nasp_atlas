"""Score NASP gene modules in single-cell data."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Literal, TypeAlias, cast

import anndata as ad  # type: ignore
import pandas as pd
import scanpy as sc  # type: ignore
from nasp_compendium import GeneModules  # type: ignore
from nasp_compendium.types import GeneModule  # type: ignore

from nasp_atlas.single_cell.visualization import SCVisualizer


logger = logging.getLogger(__name__)

ScorerName: TypeAlias = Literal["scanpy", "aucell"]


def positive_module_score_name(
    module: GeneModule,
    *,
    scorer: ScorerName,
) -> str:
    """Return the positive sub-score column name for a module."""
    suffix = "pos" if scorer == "scanpy" else "pos_auc"
    return f"{module.module_id}_{suffix}"


def inverse_module_score_name(
    module: GeneModule,
    *,
    scorer: ScorerName,
) -> str | None:
    """Return the inverse sub-score column name for a module, if needed."""
    if not module.inverse_genes:
        return None
    suffix = "inv" if scorer == "scanpy" else "inv_auc"
    return f"{module.module_id}_{suffix}"


def module_score_name(
    module: GeneModule,
    *,
    scorer: ScorerName,
) -> str:
    """Return the final signed score column name for a module."""
    suffix = "score" if scorer == "scanpy" else "auc"
    return f"{module.module_id}_{suffix}"


def combine_module_scores(
    module: GeneModule,
    scores: pd.DataFrame,
    *,
    scorer: ScorerName,
) -> pd.Series:
    """Combine positive and inverse sub-scores into one signed module score."""
    positive_name = positive_module_score_name(module, scorer=scorer)
    inverse_name = inverse_module_score_name(module, scorer=scorer)
    score_name = module_score_name(module, scorer=scorer)

    if module.positive_genes and inverse_name is not None:
        combined = scores[positive_name] - scores[inverse_name]
        return cast("pd.Series", combined).rename(score_name)
    if module.positive_genes:
        positive = cast("pd.Series", scores[positive_name])
        return positive.rename(score_name)
    if inverse_name is not None:
        inverse = cast("pd.Series", scores[inverse_name])
        return (-inverse).rename(score_name)
    raise ValueError(f"Module {module.module_id!r} has no scorable genes.")


def score_scanpy_module(
    adata: ad.AnnData,
    module: GeneModule,
    *,
    random_state: int = 0,
) -> str:
    """Score one signed module with `scanpy.tl.score_genes`."""
    if module.positive_genes:
        sc.tl.score_genes(
            adata,
            gene_list=list(module.positive_genes),
            score_name=positive_module_score_name(
                module,
                scorer="scanpy",
            ),
            random_state=random_state,
            use_raw=False,
        )

    inverse_name = inverse_module_score_name(module, scorer="scanpy")
    if module.inverse_genes and inverse_name is not None:
        sc.tl.score_genes(
            adata,
            gene_list=list(module.inverse_genes),
            score_name=inverse_name,
            random_state=random_state,
            use_raw=False,
        )

    score_name = module_score_name(module, scorer="scanpy")
    adata.obs[score_name] = combine_module_scores(
        module,
        adata.obs,  # type: ignore
        scorer="scanpy",
    )
    return score_name


def score_scanpy_modules(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    gene_symbol_column: str = "feature_name",
    random_state: int = 0,
) -> list[GeneModule]:
    """Score signed NASP modules with scanpy score_genes."""
    modules: list[GeneModule] = []
    for module_id in module_ids:
        module = GeneModules.modules(
            module_id,
            adata=adata,
            gene_symbol_column=gene_symbol_column,
            output="var_names",
        )
        modules.append(module)

        logger.info(
            "%s missing_pos=%s missing_inv=%s",
            module.module_id,
            module.missing_positive_genes,
            module.missing_inverse_genes,
        )

        score_scanpy_module(
            adata,
            module,
            random_state=random_state,
        )

    return modules


def score_aucell_modules(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    gene_symbol_column: str = "feature_name",
) -> tuple[ad.AnnData, pd.DataFrame, list[GeneModule]]:
    """Score signed NASP modules with pySCENIC AUCell."""
    if adata.isbacked:
        raise ValueError(
            "score_aucell_modules expects an in-memory AnnData. "
            "Load or copy backed data into memory before scoring."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        from pyscenic.aucell import GeneSignature  # type: ignore
        from pyscenic.aucell import aucell  # type: ignore

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
            adata=adata_auc,
            gene_symbol_column=gene_symbol_column,
            output="symbols",
        )
        for module_id in module_ids
    ]
    signatures = []
    for module in modules:
        if module.positive_genes:
            signatures.append(
                GeneSignature(
                    name=positive_module_score_name(
                        module,
                        scorer="aucell",
                    ),
                    gene2weight=dict.fromkeys(module.positive_genes, 1.0),
                )
            )

        inverse_name = inverse_module_score_name(module, scorer="aucell")
        if module.inverse_genes and inverse_name is not None:
            signatures.append(
                GeneSignature(
                    name=inverse_name,
                    gene2weight=dict.fromkeys(module.inverse_genes, 1.0),
                )
            )

    auc_df = aucell(expression_df, signatures)  # type: ignore
    for module in modules:
        score_name = module_score_name(module, scorer="aucell")
        auc_df[score_name] = combine_module_scores(
            module,
            auc_df,
            scorer="aucell",
        )
        adata_auc.obs[score_name] = auc_df[score_name]

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
