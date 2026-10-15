"""Rank every gene's expression within biological contexts."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]

from nasp_atlas.single_cell.utils import expression_matrix


logger = logging.getLogger(__name__)


def gene_expression_percentiles(
    adata: ad.AnnData,
    *,
    context_keys: Sequence[str],
    donor_key: str,
    n_bins: int,
    min_cells: int,
    min_donors: int,
    expression_layer: str | None = None,
    gene_symbol_column: str | None = None,
    chunk_size: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return each gene's median expression percentile bin across contexts.

    A context is one observed combination of `context_keys`, such as tissue
    and cell type. Its mean expression per gene includes zeros over every
    cell. Within each supported context, a gene's percentile is 100 times the
    fraction of all genes with a strictly lower mean. Tied genes therefore
    share the lower bin, and genes with zero mean occupy bin 0. Percentiles are
    cut into `n_bins` equal-width bins numbered from 0 (lowest) to
    `n_bins - 1`. The median weights every supported context equally. With
    an even number of contexts it can fall halfway between bins; it is then
    rounded up, so every gene reports a whole bin. Context means describe the
    analyzed expression source; they are not donor-level inference.

    The expression matrix is read in row chunks and never densified. Only the
    context-by-gene means are held densely. `adata` is not modified.

    Args:
      adata: AnnData whose cells and genes are summarized.
      context_keys: Obs columns jointly defining one context.
      donor_key: Obs column identifying donors, used for context support.
      n_bins: Number of equal-width percentile bins; at least 2.
      min_cells: Minimum cells in a supported context; at least 1.
      min_donors: Minimum distinct donors in a supported context; at least 1.
      expression_layer: Layer to summarize; None uses `adata.X`.
      gene_symbol_column: Optional var column reported as `gene_symbol`.
      chunk_size: Cells converted to float64 at one time; at least 1.

    Returns:
      A gene table in var order with `var_name`, optional `gene_symbol`,
      `median_percentile_bin`, `n_contexts` and `n_contexts_detected`
      (supported contexts with a positive mean). Also a context table with
      every observed combination of `context_keys`, its `n_cells`, `n_donors`
      and `supported` flag. A context with a missing label is never supported.

    Raises:
      KeyError: If a context, donor or gene-symbol column is absent.
      TypeError: If obs or var is not an in-memory table.
      ValueError: If a setting is invalid, no context is supported, or an
        expression value is non-finite.
    """
    obs = adata.obs
    var = adata.var
    if not isinstance(obs, pd.DataFrame) or not isinstance(var, pd.DataFrame):
        raise TypeError("Percentiles require in-memory obs and var tables")

    for name, value, minimum in (
        ("n_bins", n_bins, 2),
        ("min_cells", min_cells, 1),
        ("min_donors", min_donors, 1),
        ("chunk_size", chunk_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")

    keys = list(context_keys)
    if not keys:
        raise ValueError("context_keys must contain at least one obs column")

    if missing := [key for key in (*keys, donor_key) if key not in obs.columns]:
        raise KeyError(f"obs columns not found for percentiles: {missing}")
    if gene_symbol_column is not None and gene_symbol_column not in var:
        raise KeyError(f"var column not found: {gene_symbol_column}")

    contexts, cell_contexts = _context_support(
        obs,
        context_keys=keys,
        donor_key=donor_key,
        min_cells=min_cells,
        min_donors=min_donors,
    )
    supported = contexts.loc[contexts["supported"]]
    if supported.empty:
        raise ValueError(
            f"no {keys} context has at least {min_cells} cells and "
            f"{min_donors} donors; lower min_cells or min_donors"
        )

    logger.info(
        "Ranking %d genes within %d of %d %s contexts",
        adata.n_vars,
        len(supported),
        len(contexts),
        keys,
    )

    var_names, matrix = expression_matrix(
        adata,
        expression_layer=expression_layer,
    )
    if matrix is None:
        raise ValueError("Percentiles require an expression matrix")

    means = _context_means(
        sp.csr_matrix(matrix),
        cell_contexts,
        n_cells=supported["n_cells"].to_numpy(dtype=np.float64),
        n_genes=len(var_names),
        chunk_size=chunk_size,
    )
    bins = _percentile_bins(means, n_bins=n_bins)

    genes = pd.DataFrame({"var_name": var_names})
    if gene_symbol_column is not None:
        genes["gene_symbol"] = var[gene_symbol_column].to_numpy()

    # Medians of whole bins end in .0 or .5; round half up, unlike np.round.
    genes["median_percentile_bin"] = np.floor(
        np.median(bins, axis=0) + 0.5
    ).astype(np.int64)
    genes["n_contexts"] = len(supported)
    genes["n_contexts_detected"] = (means > 0).sum(axis=0)

    return genes, contexts


def _context_support(
    obs: pd.DataFrame,
    *,
    context_keys: Sequence[str],
    donor_key: str,
    min_cells: int,
    min_donors: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return observed contexts and each cell's supported-context position.

    Cells in unsupported or unlabelled contexts receive position -1.
    """
    grouped = obs.groupby(
        list(context_keys),
        observed=True,
        dropna=False,
        sort=True,
    )
    contexts = grouped.agg(
        n_cells=(donor_key, "size"),
        n_donors=(donor_key, "nunique"),
    ).reset_index()

    labelled = contexts[list(context_keys)].notna().all(axis="columns")
    contexts["supported"] = (
        labelled
        & contexts["n_cells"].ge(min_cells)
        & contexts["n_donors"].ge(min_donors)
    )

    is_supported = contexts["supported"].to_numpy(dtype=bool)
    positions = np.full(len(contexts), -1, dtype=np.int64)
    positions[is_supported] = np.arange(int(is_supported.sum()))
    cell_contexts = positions[grouped.ngroup().to_numpy()]
    return contexts, cell_contexts


def _context_means(
    matrix: sp.csr_matrix,
    cell_contexts: np.ndarray,
    *,
    n_cells: np.ndarray,
    n_genes: int,
    chunk_size: int,
) -> np.ndarray:
    """Return dense context-by-gene means accumulated over cell chunks."""
    n_contexts = len(n_cells)
    n_obs = len(cell_contexts)
    sums = np.zeros((n_contexts, n_genes), dtype=np.float64)

    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        chunk = matrix[start:stop].astype(np.float64)
        if not np.isfinite(chunk.data).all():
            raise ValueError(
                "expression values must be finite to rank genes; remove or "
                "impute non-finite values first"
            )

        codes = cell_contexts[start:stop]
        kept = np.flatnonzero(codes >= 0)
        indicator = sp.csr_matrix(
            (np.ones(kept.size), (codes[kept], kept)),
            shape=(n_contexts, stop - start),
        )
        sums += (indicator @ chunk).toarray()

    return sums / n_cells[:, np.newaxis]


def _percentile_bins(means: np.ndarray, *, n_bins: int) -> np.ndarray:
    """Return each gene's equal-width percentile bin within every context."""
    n_genes = means.shape[1]
    bins = np.empty(means.shape, dtype=np.min_scalar_type(n_bins - 1))

    for row, values in enumerate(means):
        # The left insertion point counts genes with a strictly lower mean.
        n_lower = np.searchsorted(np.sort(values), values, side="left")
        bins[row] = n_lower * n_bins // n_genes

    return bins
