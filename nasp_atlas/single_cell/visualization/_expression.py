"""Memory-aware expression-matrix helpers for visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp  # type: ignore[import]


def _expression_column_means(matrix: Any) -> np.ndarray:
    """Return per-gene means without densifying a sparse matrix."""
    n_obs, n_vars = matrix.shape
    if sp.issparse(matrix):
        sparse_matrix = matrix
        coo = sparse_matrix.tocoo(copy=False)
        nan_mask = np.isnan(coo.data)
        if bool(nan_mask.any()):
            sparse_matrix = sparse_matrix.copy()
            sparse_matrix.data[np.isnan(sparse_matrix.data)] = 0
            nan_counts = np.bincount(
                coo.col[nan_mask],
                minlength=n_vars,
            )
        else:
            nan_counts = np.zeros(n_vars, dtype=int)
        sums = np.asarray(sparse_matrix.sum(axis=0)).reshape(-1)
        counts = n_obs - nan_counts
    else:
        values = np.asarray(matrix)
        nan_mask = np.isnan(values)
        sums = np.nansum(values, axis=0)
        counts = n_obs - nan_mask.sum(axis=0)

    return np.divide(
        sums,
        counts,
        out=np.full(n_vars, np.nan, dtype=float),
        where=counts > 0,
    )
