"""Validated library-size log normalization of sparse count matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp  # type: ignore[import]


@dataclass(frozen=True)
class NormalizedCounts:
    """Validated counts, their log expression and per-cell library sizes.

    Attributes:
      counts: Copy of the validated integer-valued count matrix (CSR).
      expression: Natural-log `log1p` of counts scaled to `target_sum` (CSR).
      totals: Per-cell library size used for scaling.
    """

    counts: sp.csr_matrix
    expression: sp.csr_matrix
    totals: np.ndarray


def log_normalize_counts(
    matrix: object,
    *,
    target_sum: float = 10_000.0,
    source: str = "counts",
    dtype: type[np.floating] = np.float64,
) -> NormalizedCounts:
    """Scale each cell to `target_sum` counts and apply natural `log1p`.

    Counts must be sparse, finite, nonnegative and integer-valued, although
    floating-point storage is accepted. Every cell needs a positive library;
    a zero or non-finite library fails rather than producing NaN expression.
    Normalization uses the complete supplied gene universe, so subset genes
    only afterwards. The input matrix is not modified.

    Args:
      matrix: Sparse cells-by-genes count matrix.
      target_sum: Positive, finite library size after scaling.
      source: Name of the matrix used in error messages.
      dtype: Floating type of the expression values. float32 halves memory
        for atlas-scale inputs; library sizes stay float64.

    Returns:
      The validated counts, log expression and library sizes.

    Raises:
      TypeError: If `matrix` is not sparse.
      ValueError: If `target_sum`, any count or any library is invalid.
    """
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("Normalization target_sum must be positive and finite")
    if not sp.issparse(matrix):
        raise TypeError("This cohort workflow requires sparse count storage")

    counts = sp.csr_matrix(matrix, copy=True)
    values = counts.data
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(f"Invalid counts in {source}")

    totals = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError(
            "Selected cells have nonfinite or zero count libraries"
        )

    expression = (
        counts.astype(dtype)
        .multiply((target_sum / totals).astype(dtype)[:, None])
        .tocsr()
    )
    expression.data = np.log1p(expression.data)

    return NormalizedCounts(counts=counts, expression=expression, totals=totals)
