"""Single-cell IO helpers."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem


def random_obs_indices(
    n_obs: int,
    *,
    fraction: float,
    random_state: int = 0,
) -> npt.NDArray[np.integer[Any]]:
    """Return sorted random observation indices.

    Used to sample a subset of a given AnnData.
    """
    if not 0 < fraction <= 1:
        raise ValueError(
            "fraction must be greater than 0 and less than or equal to 1."
        )

    rng = np.random.default_rng(random_state)
    n_subset = max(1, round(n_obs * fraction))
    return np.sort(rng.choice(n_obs, size=n_subset, replace=False))


def read_csr_rows(
    x_group: h5py.Group,
    obs_indices: npt.NDArray[np.integer[Any]],
) -> sp.csr_matrix:
    """Read selected rows from an H5AD CSR matrix group."""
    if x_group.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError("Expected /X to be stored as a CSR matrix.")

    shape = _matrix_shape(x_group)
    data_dataset = _require_dataset(x_group, "data")
    indices_dataset = _require_dataset(x_group, "indices")
    indptr = np.asarray(_require_dataset(x_group, "indptr")[()])
    row_lengths = indptr[obs_indices + 1] - indptr[obs_indices]
    out_indptr = np.concatenate([[0], np.cumsum(row_lengths)])

    nnz = int(out_indptr[-1])
    data = np.empty(nnz, dtype=data_dataset.dtype)
    indices = np.empty(nnz, dtype=indices_dataset.dtype)

    offset = 0
    for row_index in obs_indices:
        start = int(indptr[row_index])
        stop = int(indptr[row_index + 1])
        length = stop - start
        if length == 0:
            continue
        data[offset : offset + length] = data_dataset[start:stop]
        indices[offset : offset + length] = indices_dataset[start:stop]
        offset += length

    return sp.csr_matrix(
        (data, indices, out_indptr),
        shape=(len(obs_indices), shape[1]),
    )


def read_random_h5ad_subset(
    path: str | Path,
    *,
    fraction: float,
    random_state: int = 0,
    obsm_keys: Sequence[str] = ("X_umap",),
) -> tuple[ad.AnnData, int]:
    """Read a random cell subset from an H5AD CSR matrix without slicing."""
    with h5py.File(path, "r") as h5:
        x_group = _require_group(h5, "X")
        n_obs = _matrix_shape(x_group)[0]
        obs_indices = random_obs_indices(
            n_obs,
            fraction=fraction,
            random_state=random_state,
        )

        x = read_csr_rows(x_group, obs_indices)
        obs = _read_dataframe(h5, "obs").iloc[obs_indices].copy()
        var = _read_dataframe(h5, "var").copy()
        uns = _read_color_uns(h5)
        obsm_values = {
            key: np.asarray(_read_elem_from_group(h5, "obsm", key))[
                obs_indices
            ].copy()
            for key in obsm_keys
            if _has_group_key(h5, "obsm", key)
        }

    adata = ad.AnnData(X=x, obs=obs, var=var, uns=uns)
    for key, value in obsm_values.items():
        adata.obsm[key] = value
    return adata, n_obs


def random_cell_subset(
    adata: ad.AnnData,
    *,
    fraction: float,
    random_state: int = 0,
) -> ad.AnnData:
    """Return a reproducible random cell subset."""
    obs_indices = random_obs_indices(
        adata.n_obs,
        fraction=fraction,
        random_state=random_state,
    )
    subset = adata[obs_indices, :]
    if subset.isbacked and subset.filename is not None:
        sampled, _ = read_random_h5ad_subset(
            subset.filename,
            fraction=fraction,
            random_state=random_state,
        )
        return sampled
    return subset.copy()


def _matrix_shape(x_group: h5py.Group) -> tuple[int, int]:
    """Return the stored sparse matrix shape."""
    shape = np.asarray(x_group.attrs["shape"]).astype(int).tolist()
    if len(shape) != 2:
        raise ValueError("Expected sparse matrix shape to have two dimensions.")
    return int(shape[0]), int(shape[1])


def _require_group(parent: h5py.Group, key: str) -> h5py.Group:
    """Return a child HDF5 group by key."""
    child = parent[key]
    if not isinstance(child, h5py.Group):
        raise TypeError(f"Expected {key!r} to be an HDF5 group.")
    return child


def _require_dataset(parent: h5py.Group, key: str) -> h5py.Dataset:
    """Return a child HDF5 dataset by key."""
    child = parent[key]
    if not isinstance(child, h5py.Dataset):
        raise TypeError(f"Expected {key!r} to be an HDF5 dataset.")
    return child


def _read_dataframe(parent: h5py.Group, key: str) -> pd.DataFrame:
    """Read a dataframe element from an H5AD group."""
    value = read_elem(_require_storage(parent, key))
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Expected {key!r} to decode as a dataframe.")
    return value


def _read_elem_from_group(
    parent: h5py.Group,
    group_key: str,
    key: str,
) -> Any:
    """Read an element from a nested H5AD group."""
    group = _require_group(parent, group_key)
    return read_elem(_require_storage(group, key))


def _read_color_uns(h5: h5py.File) -> dict[str, Any]:
    """Read Scanpy color entries from uns."""
    if "uns" not in h5:
        return {}

    uns_group = _require_group(h5, "uns")
    return {
        key: deepcopy(read_elem(_require_storage(uns_group, key)))
        for key in uns_group.keys()
        if str(key).endswith("_colors")
    }


def _has_group_key(parent: h5py.Group, group_key: str, key: str) -> bool:
    """Return whether a nested HDF5 group contains a key."""
    if group_key not in parent:
        return False
    group = parent[group_key]
    return isinstance(group, h5py.Group) and key in group


def _require_storage(
    parent: h5py.Group,
    key: str,
) -> h5py.Group | h5py.Dataset:
    """Return a child HDF5 storage object readable by anndata."""
    child = parent[key]
    if isinstance(child, h5py.Group | h5py.Dataset):
        return child
    raise TypeError(f"Expected {key!r} to be an HDF5 group or dataset.")
