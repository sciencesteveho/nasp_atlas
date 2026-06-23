"""Coverage-based metacell (pseudobulk) aggregation for single-cell data."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore

from nasp_atlas.single_cell.utils import snake_case


logger = logging.getLogger(__name__)

METACELL_ID_COLUMN = "metacell_id"
N_CELLS_COLUMN = "n_cells"
GROUP_KEY_DELIMITER = "___"


CountMatrix = sp.spmatrix | np.ndarray


def aggregate_metacells(
    adata: ad.AnnData,
    group_by: Sequence[str],
    *,
    counts_layer: str | None = "decontXcounts",
    coverage_threshold: float = 1e6,
    carry_obs: Sequence[str] = (),
    shuffle: bool = False,
    random_seed: int | None = None,
    metacell_prefix: str = "metacell",
    cell_assignment_key: str | None = None,
    coverage_column: str = "cumulative_coverage",
) -> ad.AnnData:
    """Aggregate single cells into metacells within obs groups.

    Cells are first split into the unique combinations of `group_by`, then
    coverage-aggregated within each combination. The result is a metacell-level
    AnnData whose `.X` holds summed counts in the original gene space, suitable
    as the entry point for the transcriptomic-clock preprocessing pipeline.

    Args:
      adata: Cell-level AnnData (cellxgene-standardized gene space).
      group_by: obs columns whose unique combinations define the strata within
        which metacells are built (e.g. ["donor_id", "tissue"]).
      counts_layer: Layer holding counts to sum. None uses `adata.X`.
      coverage_threshold: Minimum cumulative counts per metacell.
      carry_obs: Additional obs columns to propagate when constant within a
        group (e.g. a numeric donor age column for age acceleration).
      shuffle: Permute cells before accumulation within each group.
      random_seed: Seed used when `shuffle` is True.
      metacell_prefix: Prefix for generated metacell identifiers.
      cell_assignment_key: When set, the source `adata.obs` gains this column
        mapping each cell to its metacell identifier, enabling per-cell
        broadcasts of metacell-level results.
      coverage_column: Name of the metacell obs column storing cumulative
        coverage.

    Returns:
      A metacell-level AnnData with summed counts in `.X`, the grouping columns
      plus carried obs, `n_cells`, and `cumulative_coverage` in `.obs`, and the
      original `.var` preserved.
    """
    if missing := [
        column for column in group_by if column not in adata.obs.columns
    ]:
        raise KeyError(f"group_by columns not found in obs: {missing}")
    if missing_carry := [
        column for column in carry_obs if column not in adata.obs.columns
    ]:
        raise KeyError(f"carry_obs columns not found in obs: {missing_carry}")

    counts_matrix = adata.layers[counts_layer] if counts_layer else adata.X
    if counts_matrix is None:
        raise ValueError("AnnData has no count matrix in .X.")
    counts_matrix = cast(CountMatrix, counts_matrix)
    metadata = cast(pd.DataFrame, adata.obs)
    composite_key = _group_keys(metadata, group_by)
    unique_groups = composite_key.unique()

    logger.info(
        "[metacells] group_by=%s | groups=%d | coverage_threshold=%.0f",
        list(group_by),
        len(unique_groups),
        coverage_threshold,
    )

    block_counts: list[np.ndarray] = []
    block_obs: list[dict[str, object]] = []
    cell_metacell_ids = np.empty(adata.n_obs, dtype=object)
    metacell_counter = 0

    for group_value in unique_groups:
        cell_indices = np.flatnonzero((composite_key == group_value).to_numpy())
        group_counts = cast(CountMatrix, cast(Any, counts_matrix)[cell_indices])

        metacell_counts, cells_per_metacell, coverage, cell_assignment = (
            aggregate_counts_by_coverage(
                group_counts,
                coverage_threshold=coverage_threshold,
                shuffle=shuffle,
                random_seed=random_seed,
            )
        )

        key_values = str(group_value).split(GROUP_KEY_DELIMITER)
        carried = _carry_constant_obs(metadata, cell_indices, carry_obs)
        group_metacell_ids = [
            f"{metacell_prefix}_{metacell_counter + local_index}"
            for local_index in range(metacell_counts.shape[0])
        ]

        for local_index, metacell_id in enumerate(group_metacell_ids):
            record: dict[str, object] = dict(
                zip(group_by, key_values, strict=True)
            )
            record |= carried
            record[METACELL_ID_COLUMN] = metacell_id
            record[N_CELLS_COLUMN] = int(cells_per_metacell[local_index])
            record[coverage_column] = float(coverage[local_index])
            block_obs.append(record)

        cell_metacell_ids[cell_indices] = [
            group_metacell_ids[local_group] for local_group in cell_assignment
        ]
        metacell_counter += metacell_counts.shape[0]
        block_counts.append(metacell_counts)

    metacell_matrix = np.vstack(block_counts)
    obs_frame = pd.DataFrame(block_obs).set_index(METACELL_ID_COLUMN)

    metacell_adata = ad.AnnData(
        X=metacell_matrix,
        obs=obs_frame,
        var=cast(pd.DataFrame, adata.var).copy(),
    )

    if cell_assignment_key is not None:
        adata.obs[cell_assignment_key] = pd.Categorical(cell_metacell_ids)

    logger.info(
        "[metacells] built %d metacells (median cells/metacell=%.0f)",
        metacell_adata.n_obs,
        float(np.median(obs_frame[N_CELLS_COLUMN])),
    )
    return metacell_adata


def _as_dense_counts(matrix: CountMatrix) -> np.ndarray:
    """Return a dense float64 copy of a count matrix."""
    if sp.issparse(matrix):
        sparse_matrix = cast(Any, matrix)
        return np.asarray(sparse_matrix.todense(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _assign_coverage_groups(
    coverage_per_cell: np.ndarray,
    coverage_threshold: float,
) -> np.ndarray:
    """Assign cells to metacells by walking cumulative coverage.

    Cells are consumed in their given order and coverage accumulates until it
    reaches `coverage_threshold`, which closes the current metacell and opens
    the next. The trailing metacell is retained.

    Args:
      coverage_per_cell: Per-cell total counts.
      coverage_threshold: Cumulative counts to close a metacell.

    Returns:
      A 0-based group index per cell, aligned to the input order. Indices are
      contiguous and only cover metacells that actually received cells.
    """
    assignments = np.empty(coverage_per_cell.shape[0], dtype=np.int64)
    current_group = 0
    current_coverage = 0.0
    for position in range(coverage_per_cell.shape[0]):
        current_coverage += float(coverage_per_cell[position])
        assignments[position] = current_group
        if current_coverage >= coverage_threshold:
            current_group += 1
            current_coverage = 0.0
    return assignments


def aggregate_counts_by_coverage(
    counts: CountMatrix,
    *,
    coverage_threshold: float = 1e6,
    shuffle: bool = False,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a cells x genes count matrix into metacells by coverage.

    Args:
      counts: A (cells, genes) count matrix (sparse or dense).
      coverage_threshold: Minimum cumulative counts per metacell.
      shuffle: Permute cells before accumulation to break input ordering.
      random_seed: Seed used when `shuffle` is True.

    Returns:
      (metacell_counts, n_cells, coverage, cell_assignment):
        metacell_counts: Dense (metacells, genes) summed counts.
        n_cells: Number of cells per metacell.
        coverage: Total counts per metacell.
        cell_assignment: 0-based metacell index per input cell, in input order.
    """
    n_cells = counts.shape[0]
    if n_cells == 0:
        raise ValueError("Cannot aggregate an empty count matrix.")

    summed_counts = cast(Any, counts).sum(axis=1)
    coverage_per_cell = np.asarray(summed_counts).reshape(-1).astype(np.float64)

    order = np.arange(n_cells)
    if shuffle:
        order = np.random.default_rng(random_seed).permutation(n_cells)

    ordered_coverage = coverage_per_cell[order]
    group_per_position = _assign_coverage_groups(
        ordered_coverage, coverage_threshold
    )
    n_groups = int(group_per_position.max()) + 1

    indicator = sp.csr_matrix(
        (np.ones(n_cells), (group_per_position, order)),
        shape=(n_groups, n_cells),
    )

    metacell_counts = _as_dense_counts(indicator @ counts)
    cells_per_metacell = (
        np.asarray(indicator.sum(axis=1)).reshape(-1).astype(np.int64)
    )
    coverage_per_metacell = np.asarray(indicator @ coverage_per_cell).reshape(
        -1
    )

    cell_assignment = np.empty(n_cells, dtype=np.int64)
    cell_assignment[order] = group_per_position

    return (
        metacell_counts,
        cells_per_metacell,
        coverage_per_metacell,
        cell_assignment,
    )


def _group_keys(metadata: pd.DataFrame, group_by: Sequence[str]) -> pd.Series:
    """Build a single composite key per cell from the grouping columns."""
    string_frame = metadata[list(group_by)].astype(str)
    return string_frame.agg(GROUP_KEY_DELIMITER.join, axis=1)


def _carry_constant_obs(
    metadata: pd.DataFrame,
    cell_indices: np.ndarray,
    carry_obs: Sequence[str],
) -> dict[str, object]:
    """Carry constant obs columns within a cell group.

    Columns that vary within the group resolve to NaN so the ambiguity is
    explicit rather than silently taking the first value.

    Args:
      metadata: Full cell-level obs frame.
      cell_indices: Positional indices of the cells in this group.
      carry_obs: obs columns to propagate to the metacell.

    Returns:
      Mapping of column name to the constant value (or NaN when not constant).
    """
    carried: dict[str, object] = {}
    group_frame = metadata.iloc[cell_indices]
    for column in carry_obs:
        unique_values = group_frame[column].dropna().unique()
        carried[column] = (
            unique_values[0] if len(unique_values) == 1 else np.nan
        )
    return carried


def _metacell_stratum_label(values: Sequence[str]) -> str:
    """Return a filesystem-safe stratum label from grouping values."""
    return snake_case(GROUP_KEY_DELIMITER.join(str(value) for value in values))
