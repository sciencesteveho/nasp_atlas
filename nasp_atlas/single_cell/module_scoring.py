"""Score NASP gene modules in single-cell data."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal, TypeAlias, cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore[import]
import scipy.sparse as sp  # type: ignore[import]
from nasp_compendium import GeneModules  # type: ignore[import]
from nasp_compendium.types import GeneModule  # type: ignore[import]
from pyscenic.aucell import GeneSignature  # type: ignore[import]
from pyscenic.aucell import aucell4r  # type: ignore[import]
from pyscenic.aucell import create_rankings  # type: ignore[import]

from nasp_atlas.single_cell.utils import expression_matrix


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


def _z_score_standardize(values: pd.Series) -> pd.Series:
    """Z-score a per-cell sub-score to zero mean and unit variance.

    Per-cell score variance scales with roughly `1 / n_genes`, so a raw
    `positive - inverse` difference is dominated by whichever arm has fewer
    genes. Standardizing each arm before combining equalizes the variance the
    two arms contribute to the signed score. A zero-variance arm is
    mean-centered.
    """
    std = values.std(ddof=0)
    return (
        values - values.mean()
        if np.isclose(std, 0.0)
        else (values - values.mean()) / std
    )


def combine_module_scores(
    module: GeneModule,
    scores: pd.DataFrame,
    *,
    scorer: ScorerName,
    standardize: bool = True,
) -> pd.Series:
    """Combine positive and inverse sub-scores into one signed module score.

    When both arms are present each is z-scored across cells before subtracting
    (`standardize=True`) so the smaller arm cannot dominate the composite
    variance. Single-arm modules are returned in their native scale.
    """
    positive_name = positive_module_score_name(module, scorer=scorer)
    inverse_name = inverse_module_score_name(module, scorer=scorer)
    score_name = module_score_name(module, scorer=scorer)

    if module.positive_genes and inverse_name is not None:
        positive = cast("pd.Series", scores[positive_name])
        inverse = cast("pd.Series", scores[inverse_name])

        if standardize:
            positive = _z_score_standardize(positive)
            inverse = _z_score_standardize(inverse)

        combined = positive - inverse
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
    random_state: int = 42,
    expression_layer: str | None = "log1p",
    use_raw: bool = False,
) -> str:
    """Score one signed module with scanpy.tl.score_genes."""
    if adata.n_obs == 0:
        raise ValueError("Scanpy module scoring requires at least one cell.")

    if expression_layer is not None and use_raw:
        raise ValueError(
            "use_raw=True cannot be combined with expression_layer"
        )

    if module.positive_genes:
        sc.tl.score_genes(
            adata,
            gene_list=list(module.positive_genes),
            score_name=positive_module_score_name(
                module,
                scorer="scanpy",
            ),
            random_state=random_state,
            use_raw=use_raw,
            layer=expression_layer,
        )

    inverse_name = inverse_module_score_name(module, scorer="scanpy")
    if module.inverse_genes and inverse_name is not None:
        sc.tl.score_genes(
            adata,
            gene_list=list(module.inverse_genes),
            score_name=inverse_name,
            random_state=random_state,
            use_raw=use_raw,
            layer=expression_layer,
        )

    score_name = module_score_name(module, scorer="scanpy")
    obs = adata.obs
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("score_scanpy_module requires in-memory AnnData obs.")
    obs[score_name] = combine_module_scores(
        module,
        obs,
        scorer="scanpy",
    )
    return score_name


def score_scanpy_modules(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    gene_symbol_column: str = "feature_name",
    random_state: int = 42,
    expression_layer: str | None = "log1p",
    use_raw: bool = False,
    gene_modules: Sequence[GeneModule] | None = None,
) -> list[GeneModule]:
    """Score compendium or explicitly supplied symbol-space signatures."""
    if adata.n_obs == 0:
        raise ValueError("Scanpy module scoring requires at least one cell.")

    module_source = adata
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True requires adata.raw to be set")
        module_source = ad.AnnData(
            shape=adata.raw.shape,
            var=adata.raw.var,
        )

    if gene_modules is not None and [m.module_id for m in gene_modules] != list(
        module_ids
    ):
        raise ValueError("gene_modules IDs must match module_ids in order")

    supplied = {module.module_id: module for module in gene_modules or ()}
    mapping: dict[str, str] = {}
    ambiguous: set[str] = set()
    if supplied:
        symbols = (
            module_source.var[gene_symbol_column]
            .astype(object)
            .fillna(
                pd.Series(
                    module_source.var_names, index=module_source.var_names
                )
            )
            .astype(str)
            .str.strip()
        )
        mapping = dict(zip(symbols, module_source.var_names, strict=True))
        ambiguous = set(symbols[symbols.duplicated(keep=False)])

    modules: list[GeneModule] = []
    for module_id in module_ids:
        if module_id in supplied:
            module = _resolve_supplied_module(
                supplied[module_id],
                mapping,
                ambiguous,
            )
        else:
            module = GeneModules.modules(
                module_id,
                adata=module_source,  # type: ignore[arg-type]
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
            expression_layer=expression_layer,
            use_raw=use_raw,
        )

    return modules


def score_aucell_modules(
    adata: ad.AnnData,
    module_ids: Sequence[str],
    *,
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = "log1p",
    use_raw: bool = False,
    chunk_size: int = 1_000,
    random_state: int = 42,
    num_workers: int = 1,
    gene_modules: Sequence[GeneModule] | None = None,
) -> tuple[ad.AnnData, pd.DataFrame, list[GeneModule]]:
    """Score signed NASP modules with AUCell.

    The expression matrix is densified one cell-block at a time to manage
    runtime memory. `gene_modules` optionally supplies explicitly derived
    symbol-space signatures (for sensitivity analyses); their IDs must match
    `module_ids` in order. Compendium resolution remains the default.
    """
    if adata.isbacked:
        raise ValueError(
            "score_aucell_modules expects an in-memory AnnData. "
            "Load or copy backed data into memory before scoring."
        )
    if adata.n_obs == 0:
        raise ValueError("AUCell scoring requires at least one cell.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}.")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive; got {num_workers}.")

    source_var_names, source_matrix = expression_matrix(
        adata,
        expression_layer=expression_layer,
        use_raw=use_raw,
    )
    if source_matrix is None:
        raise ValueError("AUCell scoring requires an expression matrix.")
    source_var = (
        adata.raw.var if use_raw and adata.raw is not None else adata.var
    )
    if not isinstance(source_var, pd.DataFrame):
        raise TypeError("AUCell scoring requires in-memory AnnData var.")
    source_var = source_var.loc[source_var_names]

    obs = adata.obs
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("AUCell scoring requires in-memory AnnData obs.")

    source_adata = ad.AnnData(
        shape=(adata.n_obs, len(source_var_names)),
        var=source_var,
    )
    fallback_ids = pd.Series(
        source_var_names,
        index=source_var.index,
        dtype=object,
    )
    symbols = source_var[gene_symbol_column].astype(object)
    gene_ids = symbols.where(symbols.notna(), fallback_ids).astype(str)
    gene_ids = gene_ids.str.strip()
    gene_ids = gene_ids.where(gene_ids != "", fallback_ids)
    keep = (gene_ids != "") & ~gene_ids.duplicated()
    if not bool(keep.any()):
        raise ValueError(
            "AUCell scoring requires at least one non-empty gene identifier."
        )
    keep_mask = keep.to_numpy()
    kept_gene_ids = gene_ids[keep].to_numpy()
    obs_names = adata.obs_names.astype(str)

    modules = (
        list(gene_modules)
        if gene_modules is not None
        else [
            GeneModules.modules(
                module_id,
                adata=source_adata,  # type: ignore[arg-type]
                gene_symbol_column=gene_symbol_column,
                output="symbols",
            )
            for module_id in module_ids
        ]
    )
    if [module.module_id for module in modules] != list(module_ids):
        raise ValueError("gene_modules IDs must match module_ids in order")
    if any(module.gene_id_output != "symbols" for module in modules):
        raise ValueError("AUCell gene_modules must use symbols")

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

    if not signatures:
        raise ValueError(
            "AUCell scoring requires at least one positive or inverse gene "
            "signature."
        )

    available_gene_ids = set(kept_gene_ids)
    if unavailable_signatures := [
        signature.name
        for signature in signatures
        if available_gene_ids.isdisjoint(signature.genes)
    ]:
        unavailable = ", ".join(unavailable_signatures)
        raise ValueError(
            "AUCell signatures have no genes in the ranking matrix: "
            f"{unavailable}."
        )

    n_chunks = (adata.n_obs + chunk_size - 1) // chunk_size
    started_at = time.perf_counter()
    logger.info(
        "AUCell started: cells=%d, genes=%d, signatures=%d, "
        "chunk_size=%d, workers=%d, chunks=%d, seed=%d",
        adata.n_obs,
        len(kept_gene_ids),
        len(signatures),
        chunk_size,
        num_workers,
        n_chunks,
        random_state,
    )

    auc_parts: list[pd.DataFrame] = []
    for chunk_number, start in enumerate(
        range(0, adata.n_obs, chunk_size), start=1
    ):
        stop = min(start + chunk_size, adata.n_obs)
        chunk_started_at = time.perf_counter()
        logger.info(
            "AUCell chunk %d/%d: scoring cells %d-%d",
            chunk_number,
            n_chunks,
            start + 1,
            stop,
        )
        block = source_matrix[start:stop, keep_mask]
        block_values = (
            block.toarray() if sp.issparse(block) else np.asarray(block)  # type: ignore[union-attr]
        )
        block_df = pd.DataFrame(
            block_values,
            index=obs_names[start:stop],
            columns=kept_gene_ids,
        )
        auc_parts.append(
            aucell4r(
                _create_aucell_rankings(block_df, seed=random_state),
                signatures,
                num_workers=num_workers,
            )
        )

        finished_at = time.perf_counter()
        elapsed = finished_at - started_at
        remaining = elapsed * (adata.n_obs - stop) / stop
        logger.info(
            "AUCell chunk %d/%d complete: %d/%d cells (%.1f%%), "
            "chunk %.1fs, elapsed %.1fmin, "
            "estimated remaining %.1fmin (chunks only)",
            chunk_number,
            n_chunks,
            stop,
            adata.n_obs,
            100.0 * stop / adata.n_obs,
            finished_at - chunk_started_at,
            elapsed / 60.0,
            remaining / 60.0,
        )

    logger.info("AUCell chunks finished; assembling signed module scores")
    auc_df = pd.concat(auc_parts)
    for module in modules:
        score_name = module_score_name(module, scorer="aucell")
        auc_df[score_name] = combine_module_scores(
            module,
            auc_df,
            scorer="aucell",
        )

    result_var = adata.var
    if not isinstance(result_var, pd.DataFrame):
        raise TypeError("AUCell scoring requires in-memory AnnData var.")

    adata_auc = ad.AnnData(X=adata.X, obs=obs.copy(), var=result_var)
    adata_auc.uns = adata.uns
    adata_auc.obsm.update(adata.obsm)
    adata_auc.varm.update(adata.varm)
    adata_auc.layers.update(adata.layers)
    adata_auc.obsp.update(adata.obsp)
    adata_auc.varp.update(adata.varp)
    if adata.raw is not None:
        raw_adata = ad.AnnData(X=adata.raw.X, var=adata.raw.var)
        raw_adata.varm.update(
            adata.raw.varm  # type: ignore[reportAttributeAccessIssue]
        )
        adata_auc.raw = raw_adata

    for score_column in auc_df.columns:
        adata_auc.obs[score_column] = auc_df[score_column]

    logger.info(
        "AUCell finished: cells=%d, modules=%d, elapsed %.1fmin",
        adata.n_obs,
        len(modules),
        (time.perf_counter() - started_at) / 60.0,
    )
    return adata_auc, auc_df, modules


def _create_aucell_rankings(
    expression: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Rank each cell with pySCENIC's seeded ties and missing values last."""
    values = expression.to_numpy()
    if values.dtype.kind not in "iufb" or values.dtype.itemsize > 8:
        return create_rankings(expression, seed=seed)

    shuffled = expression.sample(frac=1.0, axis=1, random_state=seed)
    values = shuffled.to_numpy()
    # Complement integers to reverse their order without negation overflow.
    descending_values = ~values if values.dtype.kind in "iub" else -values
    order = np.argsort(descending_values, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.uint32)
    np.put_along_axis(
        ranks,
        order,
        np.arange(order.shape[1], dtype=np.uint32)[None, :],
        axis=1,
    )
    return pd.DataFrame(ranks, index=shuffled.index, columns=shuffled.columns)


def _resolve_supplied_module(
    definition: GeneModule,
    mapping: dict[str, str],
    ambiguous: set[str],
) -> GeneModule:
    """Resolve supplied signature arms, retaining missing-gene diagnostics."""
    if definition.gene_id_output != "symbols":
        raise ValueError("Supplied modules must use symbols")
    if ambiguous.intersection(
        definition.positive_genes + definition.inverse_genes
    ):
        raise ValueError(f"Ambiguous gene symbols in {definition.module_id}")

    return replace(
        definition,
        positive_genes=tuple(
            mapping[g] for g in definition.positive_genes if g in mapping
        ),
        inverse_genes=tuple(
            mapping[g] for g in definition.inverse_genes if g in mapping
        ),
        gene_id_output="var_names",
        missing_positive_genes=tuple(
            g for g in definition.positive_genes if g not in mapping
        ),
        missing_inverse_genes=tuple(
            g for g in definition.inverse_genes if g not in mapping
        ),
    )
