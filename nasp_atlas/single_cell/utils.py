"""Reusable single-cell utility functions."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Literal, cast

import anndata as ad  # type: ignore
import h5py  # type: ignore
import numpy as np
import pandas as pd

from nasp_atlas.single_cell.io import random_cell_subset
from nasp_atlas.single_cell.io import read_csr_rows
from nasp_atlas.single_cell.io import read_h5ad


H5adCompression = Literal["gzip", "lzf"]
SplitCompression = H5adCompression | Literal["source"] | None


def split_anndata_by_obs(
    adata_or_path: ad.AnnData | str | Path,
    *,
    output_dir: str | Path,
    obs_key: str,
    output_name: str,
    subset_fraction: float | None = None,
    random_state: int = 0,
    compression: SplitCompression = "source",
) -> dict[str, Path]:
    """Write one AnnData file per value in `adata.obs[obs_key]`.

    By default, path-backed inputs preserve the source H5AD matrix
    compression for the output files. Pass ``compression=None`` to disable
    output compression.
    """
    close_backed = False
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path
        if subset_fraction is not None:
            adata = random_cell_subset(
                adata,
                fraction=subset_fraction,
                random_state=random_state,
            )
    elif subset_fraction is None:
        adata = ad.read_h5ad(adata_or_path, backed="r")
        close_backed = True
    else:
        adata, _ = read_h5ad(
            adata_or_path,
            subset_fraction=subset_fraction,
            random_state=random_state,
        )

    try:
        return _write_obs_value_h5ads(
            obs_key,
            adata,
            output_dir,
            output_name,
            compression=compression,
        )
    finally:
        if close_backed:
            adata.file.close()


def _write_obs_value_h5ads(
    obs_key: str,
    adata: ad.AnnData,
    output_dir: str | Path,
    output_name: str,
    *,
    compression: SplitCompression = "source",
) -> dict[str, Path]:
    """Write one h5ad file for each non-null value in an obs column."""
    if obs_key not in adata.obs:
        raise ValueError(f"obs_key not found in adata.obs: {obs_key}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = snake_case(output_name)
    if not suffix:
        raise ValueError(
            "output_name must contain at least one alphanumeric character."
        )

    written: dict[str, Path] = {}
    used_stems: set[str] = set()
    write_compression, compression_opts = _resolve_write_compression(
        adata,
        compression,
    )
    for value, obs_indices in _iter_obs_value_indices(adata.obs[obs_key]):
        value_slug = snake_case(value)
        if not value_slug:
            continue

        stem = dedupe_stem(f"{value_slug}_{suffix}", used_stems)
        out = output_path / f"{stem}.h5ad"
        subset = _materialize_obs_subset(adata, obs_indices)
        normalize_h5ad_string_storage(subset)
        subset.write_h5ad(
            out,
            compression=write_compression,
            compression_opts=compression_opts,
        )
        written[value] = out
        del subset
        gc.collect()

    return written


def snake_case(value: object) -> str:
    """Return a filesystem-safe snake_case label."""
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_")


def dedupe_stem(stem: str, used_stems: set[str]) -> str:
    """Return a unique file stem, preserving the first name."""
    if stem not in used_stems:
        used_stems.add(stem)
        return stem

    index = 2
    while f"{stem}_{index}" in used_stems:
        index += 1
    deduped = f"{stem}_{index}"
    used_stems.add(deduped)
    return deduped


def _iter_obs_value_indices(
    obs_values: pd.Series,
) -> list[tuple[str, np.ndarray]]:
    """Return sorted non-null obs value labels and integer row indices."""
    valid = obs_values.notna()
    if not bool(valid.any()):
        return []

    positions = pd.Series(np.arange(len(obs_values)), index=obs_values.index)
    labels = obs_values.loc[valid].astype(str)
    grouped = positions.loc[valid].groupby(labels, sort=True)
    return [
        (str(value), group.to_numpy(dtype=np.intp, copy=False))
        for value, group in grouped
    ]


def _resolve_write_compression(
    adata: ad.AnnData,
    compression: SplitCompression,
) -> tuple[H5adCompression | None, Any | None]:
    """Return output compression settings for split h5ads."""
    if compression == "source":
        return _infer_h5ad_compression(adata)
    return compression, None


def _infer_h5ad_compression(
    adata: ad.AnnData,
) -> tuple[H5adCompression | None, Any | None]:
    """Infer H5AD matrix compression from a backed AnnData source."""
    if adata.filename is None:
        return None, None

    with h5py.File(adata.filename, "r") as h5:
        for dataset in _iter_h5ad_matrix_datasets(h5):
            compression = dataset.compression
            if compression in {"gzip", "lzf"}:
                return (
                    cast(H5adCompression, compression),
                    dataset.compression_opts,
                )

    return None, None


def _iter_h5ad_matrix_datasets(h5: h5py.File) -> list[h5py.Dataset]:
    """Return datasets that represent X or raw.X storage."""
    datasets: list[h5py.Dataset] = []

    if "X" in h5:
        datasets.extend(_matrix_datasets(h5["X"]))

    raw = h5.get("raw")
    if isinstance(raw, h5py.Group) and "X" in raw:
        datasets.extend(_matrix_datasets(raw["X"]))

    return datasets


def _matrix_datasets(storage: h5py.Group | h5py.Dataset) -> list[h5py.Dataset]:
    """Return datasets backing a dense or sparse H5AD matrix storage object."""
    if isinstance(storage, h5py.Dataset):
        return [storage]

    datasets: list[h5py.Dataset] = []
    for key in ("data", "indices", "indptr"):
        child = storage.get(key)
        if isinstance(child, h5py.Dataset):
            datasets.append(child)
    return datasets


def normalize_h5ad_string_storage(adata: ad.AnnData) -> None:
    """Convert pandas Arrow-backed string columns to object dtype for h5ad
    writes.
    """
    adata.obs.index = pd.Index(adata.obs_names.astype(str), dtype=object)
    adata.var.index = pd.Index(adata.var_names.astype(str), dtype=object)
    _normalize_dataframe_string_storage(cast(pd.DataFrame, adata.obs))
    _normalize_dataframe_string_storage(cast(pd.DataFrame, adata.var))
    if adata.raw is not None:
        adata.raw.var.index = pd.Index(
            adata.raw.var_names.astype(str),
            dtype=object,
        )
        _normalize_dataframe_string_storage(cast(pd.DataFrame, adata.raw.var))


def _normalize_dataframe_string_storage(frame: pd.DataFrame) -> None:
    """Convert Arrow-backed string storage in a dataframe for h5ad writes."""
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.StringDtype):
            frame[column] = series.astype(object)
        elif isinstance(series.dtype, pd.CategoricalDtype):
            categories = series.cat.categories
            if isinstance(categories.dtype, pd.StringDtype):
                frame[column] = series.cat.set_categories(
                    pd.Index(categories.astype(object), dtype=object)
                )


def _materialize_obs_subset(
    adata: ad.AnnData,
    obs_indices: np.ndarray,
) -> ad.AnnData:
    """Return an in-memory AnnData subset for the given obs indices."""
    subset = adata[obs_indices, :]
    if not subset.isbacked:
        return subset.copy()

    if subset.raw is None:
        return subset.to_memory()

    raw = subset.raw
    raw_x = _read_backed_raw_x_rows(adata, obs_indices)
    raw_var = cast(pd.DataFrame, raw.var.copy())
    raw_any = cast(Any, raw)
    raw_varm = {key: value.copy() for key, value in raw_any.varm.items()}

    subset._raw = None
    in_memory = subset.to_memory()
    in_memory.raw = ad.AnnData(
        X=raw_x,
        obs=cast(pd.DataFrame, in_memory.obs.copy()),
        var=raw_var,
        varm=cast(dict[str, Any], raw_varm),
    )
    return in_memory


def _read_backed_raw_x_rows(
    adata: ad.AnnData,
    obs_indices: np.ndarray,
) -> object:
    """Read selected rows from a backed AnnData raw matrix."""
    if adata.filename is None:
        raise ValueError("Backed AnnData raw matrix has no source filename.")

    with h5py.File(adata.filename, "r") as h5:
        raw_group = h5["raw"]
        if not isinstance(raw_group, h5py.Group):
            raise TypeError("Expected backed AnnData raw to be an HDF5 group.")

        raw_x = raw_group["X"]
        if isinstance(raw_x, h5py.Dataset):
            return raw_x[obs_indices, :].copy()
        if isinstance(raw_x, h5py.Group):
            if raw_x.attrs.get("encoding-type") == "csr_matrix":
                return read_csr_rows(raw_x, obs_indices)
            raise ValueError(
                "Expected backed AnnData raw.X to be dense or CSR sparse."
            )

    raise TypeError(
        "Expected backed AnnData raw.X to be an HDF5 dataset/group."
    )
