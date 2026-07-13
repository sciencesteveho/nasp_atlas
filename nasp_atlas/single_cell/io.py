"""Single-cell I/O helpers."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import anndata as ad  # type: ignore[import]
import h5py  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import pandas as pd
import scanpy as sc  # type: ignore[import]
import scipy.sparse as sp  # type: ignore[import]
from anndata.io import read_elem  # type: ignore[import]


def _random_obs_indices(
    n_obs: int,
    *,
    fraction: float,
    random_state: int = 42,
) -> npt.NDArray[np.integer[Any]]:
    """Return sorted random observation indices to sample a subset of a given
    AnnData.

    Args:
      n_obs: Total number of observations in the AnnData.
      fraction: Fraction of observations to sample (0 < fraction <= 1).
      random_state: Random seed for reproducibility.
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


def _read_random_h5ad_subset(
    path: str | Path,
    *,
    fraction: float,
    random_state: int = 42,
    read_x: bool = True,
    layer_keys: Sequence[str] | None = None,
    read_raw: bool = True,
    obsm_keys: Sequence[str] = ("X_umap",),
) -> tuple[ad.AnnData, int]:
    """Read a random cell subset from an H5AD file without AnnData slicing."""
    with h5py.File(path, "r") as h5:
        x_storage = _require_storage(h5, "X")
        n_obs = _get_stored_matrix_shape(x_storage)[0]
        obs_indices = _random_obs_indices(
            n_obs,
            fraction=fraction,
            random_state=random_state,
        )

        if read_x:
            x = _read_matrix_rows(h5, "X", obs_indices)
        else:
            n_vars = _get_stored_matrix_shape(x_storage)[1]
            x = sp.csr_matrix((len(obs_indices), n_vars), dtype=np.float32)

        obs = _read_dataframe(h5, "obs").iloc[obs_indices].copy()
        var = _read_dataframe(h5, "var").copy()
        uns = _read_color_uns(h5)

        layers = _read_layer_rows(h5, obs_indices, layer_keys=layer_keys)
        obsm_values = {
            key: np.asarray(
                _read_elem_from_group(
                    parent=h5,
                    group_key="obsm",
                    key=key,
                )
            )[obs_indices].copy()
            for key in obsm_keys
            if _has_group_key(parent=h5, group_key="obsm", key=key)
        }
        raw = _read_raw_rows(h5, obs, obs_indices) if read_raw else None

    adata = ad.AnnData(X=x, obs=obs, var=var, uns=uns, layers=layers)
    for key, value in obsm_values.items():
        adata.obsm[key] = value

    if raw is not None:
        adata.raw = raw

    return adata, n_obs


def read_h5ad(
    path: str | Path,
    *,
    subset_fraction: float | None = None,
    random_state: int = 42,
    read_x: bool = True,
    layer_keys: Sequence[str] | None = None,
    read_raw: bool = True,
    obsm_keys: Sequence[str] = ("X_umap",),
) -> tuple[ad.AnnData, int]:
    """Read a full h5ad or a reproducible selective random subset."""
    if subset_fraction is not None:
        return _read_random_h5ad_subset(
            path,
            fraction=subset_fraction,
            random_state=random_state,
            read_x=read_x,
            layer_keys=layer_keys,
            read_raw=read_raw,
            obsm_keys=obsm_keys,
        )

    adata = sc.read_h5ad(path)
    if not read_x:
        adata.X = sp.csr_matrix(adata.shape, dtype=np.float32)

    if layer_keys is not None:
        if missing := [key for key in layer_keys if key not in adata.layers]:
            raise KeyError(f"requested layers are absent from h5ad: {missing}")
        for key in set(adata.layers).difference(layer_keys):
            del adata.layers[key]

    if not read_raw:
        del adata.raw

    for key in set(adata.obsm).difference(obsm_keys):
        del adata.obsm[key]

    return adata, adata.n_obs


def read_h5ad_rows(
    path: str | Path,
    obs_names: Sequence[str],
    *,
    layer_keys: Sequence[str] = (),
    read_x: bool = True,
    read_raw: bool = False,
    obsm_keys: Sequence[str] = (),
) -> ad.AnnData:
    """Read exact H5AD observation rows without loading full matrices.

    Intended for workflows that already have a score table or other manifest
    defining the exact cell population. Only requested layers, raw, and
    embeddings are loaded, to avoid materializing unused matrices.

    Args:
      path: Source H5AD path.
      obs_names: Observation names to load, in the desired output order.
      layer_keys: Layer matrices to include. No layers are loaded by default.
      read_x: Load source `X`. When False, a shape-compatible zero matrix is
        used and callers must read expression from a requested layer or raw.
      read_raw: Include the source raw expression matrix.
      obsm_keys: Embeddings to include. No embeddings are loaded by default.

    Returns:
      In-memory AnnData containing exactly the requested observations.

    Raises:
      ValueError: If names are empty, duplicated, or absent from the source.
      KeyError: If a requested layer or embedding is absent.
    """
    requested = pd.Index([str(name) for name in obs_names], dtype=object)
    if requested.empty:
        raise ValueError("obs_names must contain at least one observation")
    if not requested.is_unique:
        raise ValueError("obs_names index is not unique")

    with h5py.File(path, "r") as h5:
        source_obs = _read_dataframe(h5, "obs")
        source_index = pd.Index(source_obs.index).astype(str)
        if not source_index.is_unique:
            raise ValueError("source h5ad obs names must be unique")

        positions = source_index.get_indexer(requested)
        if bool((positions < 0).any()):
            missing = requested[positions < 0].tolist()
            if len(missing) == len(requested):
                raise ValueError(
                    "no obs_name values requested from the score table are "
                    "present in the source h5ad"
                )
            preview = ", ".join(missing[:3])
            raise ValueError(
                f"{len(missing)} requested obs names are absent from the "
                f"source h5ad (for example: {preview})"
            )

        sort_order = np.argsort(positions)
        sorted_positions = positions[sort_order]
        restore_order = np.argsort(sort_order)
        source_x = _require_storage(h5, "X")
        n_vars = _get_stored_matrix_shape(source_x)[1]

        if read_x:
            x = _read_matrix_rows(h5, "X", sorted_positions)[restore_order, :]
        else:
            x = sp.csr_matrix((len(requested), n_vars), dtype=np.float32)

        obs = source_obs.iloc[positions].copy()
        obs.index = requested
        var = _read_dataframe(h5, "var").copy()
        uns = _read_color_uns(h5)
        layers = _read_selected_layer_rows(
            h5,
            layer_keys,
            sorted_positions,
            restore_order,
        )
        obsm_values = _read_selected_obsm_rows(h5, obsm_keys, positions)
        raw = _read_selected_raw_rows(
            h5,
            obs,
            sorted_positions,
            restore_order,
            read_raw=read_raw,
        )

    adata = ad.AnnData(X=x, obs=obs, var=var, uns=uns, layers=layers)
    for key, value in obsm_values.items():
        adata.obsm[key] = value

    if raw is not None:
        adata.raw = raw

    return adata


def random_cell_subset(
    adata: ad.AnnData,
    *,
    fraction: float,
    random_state: int = 42,
) -> ad.AnnData:
    """Return a reproducible random cell subset."""
    obs_indices = _random_obs_indices(
        adata.n_obs,
        fraction=fraction,
        random_state=random_state,
    )
    subset = adata[obs_indices, :]
    if subset.isbacked and subset.filename is not None:
        sampled, _ = _read_random_h5ad_subset(
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


def _get_stored_matrix_shape(
    storage: h5py.Group | h5py.Dataset,
) -> tuple[int, int]:
    """Return the row/column shape of a stored H5AD matrix."""
    if isinstance(storage, h5py.Dataset):
        shape = storage.shape
        return int(shape[0]), int(shape[1])
    return _matrix_shape(storage)


def _read_matrix_rows(
    parent: h5py.Group,
    key: str,
    obs_indices: npt.NDArray[np.integer[Any]],
) -> Any:
    """Read selected rows from a dense or sparse H5AD matrix element."""
    storage = _require_storage(parent, key)

    if isinstance(storage, h5py.Dataset):
        return storage[obs_indices, :].copy()
    if storage.attrs.get("encoding-type") == "csr_matrix":
        return read_csr_rows(storage, obs_indices)

    matrix = read_elem(storage)

    return matrix[obs_indices, :].copy()  # type: ignore[index,union-attr]


def _read_layer_rows(
    h5: h5py.File,
    obs_indices: npt.NDArray[np.integer[Any]],
    *,
    layer_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read all layer matrices for selected rows."""
    if "layers" not in h5:
        return {}

    layers_group = _require_group(h5, "layers")
    keys = list(layers_group.keys()) if layer_keys is None else list(layer_keys)
    if missing := [key for key in keys if key not in layers_group]:
        raise KeyError(f"requested layers are absent from h5ad: {missing}")

    return {
        key: _read_matrix_rows(layers_group, key, obs_indices)
        for key in dict.fromkeys(keys)
    }


def _read_selected_layer_rows(
    h5: h5py.File,
    layer_keys: Sequence[str],
    sorted_positions: npt.NDArray[np.integer[Any]],
    restore_order: npt.NDArray[np.integer[Any]],
) -> dict[str, Any]:
    """Read selected rows from explicitly requested layer matrices."""
    if not layer_keys:
        return {}
    if "layers" not in h5:
        raise KeyError(f"requested layers are absent from h5ad: {layer_keys}")
    layers_group = _require_group(h5, "layers")
    if missing := [key for key in layer_keys if key not in layers_group]:
        raise KeyError(f"requested layers are absent from h5ad: {missing}")
    return {
        key: _read_matrix_rows(layers_group, key, sorted_positions)[
            restore_order, :
        ]
        for key in dict.fromkeys(layer_keys)
    }


def _read_selected_obsm_rows(
    h5: h5py.File,
    obsm_keys: Sequence[str],
    positions: npt.NDArray[np.integer[Any]],
) -> dict[str, np.ndarray]:
    """Read selected rows from explicitly requested embeddings."""
    if missing := [
        key
        for key in obsm_keys
        if not _has_group_key(parent=h5, group_key="obsm", key=key)
    ]:
        raise KeyError(
            f"requested obsm entries are absent from h5ad: {missing}"
        )
    return {
        key: np.asarray(
            _read_elem_from_group(parent=h5, group_key="obsm", key=key)
        )[positions].copy()
        for key in dict.fromkeys(obsm_keys)
    }


def _read_selected_raw_rows(
    h5: h5py.File,
    obs: pd.DataFrame,
    sorted_positions: npt.NDArray[np.integer[Any]],
    restore_order: npt.NDArray[np.integer[Any]],
    *,
    read_raw: bool,
) -> ad.AnnData | None:
    """Read selected raw rows when explicitly requested."""
    if not read_raw:
        return None

    if "raw" not in h5:
        raise ValueError("read_raw=True requires raw data in the source h5ad")

    raw_group = _require_group(h5, "raw")
    raw_x = _read_matrix_rows(raw_group, "X", sorted_positions)[
        restore_order, :
    ]

    return ad.AnnData(
        X=raw_x,
        obs=obs.copy(),
        var=_read_dataframe(raw_group, "var").copy(),
    )


def _read_raw_rows(
    h5: h5py.File,
    obs: pd.DataFrame,
    obs_indices: npt.NDArray[np.integer[Any]],
) -> ad.AnnData | None:
    """Read selected rows from `.raw`, when present."""
    if "raw" not in h5:
        return None

    raw_group = _require_group(h5, "raw")

    return ad.AnnData(
        X=_read_matrix_rows(raw_group, "X", obs_indices),
        obs=obs.copy(),
        var=_read_dataframe(raw_group, "var").copy(),
    )


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
    *,
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


def _has_group_key(*, parent: h5py.Group, group_key: str, key: str) -> bool:
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
