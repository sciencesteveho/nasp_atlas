"""Reusable single-cell utility functions."""

from __future__ import annotations

import gc
import re
from pathlib import Path

import anndata as ad  # type: ignore
import pandas as pd

from nasp_atlas.single_cell.io import random_cell_subset
from nasp_atlas.single_cell.io import read_h5ad


def split_anndata_by_obs(
    adata_or_path: ad.AnnData | str | Path,
    *,
    output_dir: str | Path,
    obs_key: str,
    output_name: str,
    subset_fraction: float | None = None,
    random_state: int = 0,
) -> dict[str, Path]:
    """Write one AnnData file per value in `adata.obs[obs_key]`."""
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
        return _write_obs_value_h5ads(obs_key, adata, output_dir, output_name)
    finally:
        if close_backed:
            adata.file.close()


def _write_obs_value_h5ads(
    obs_key: str,
    adata: ad.AnnData,
    output_dir: str | Path,
    output_name: str,
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
    obs_values = adata.obs[obs_key]
    obs_values_text = obs_values.astype(str)
    for value in sorted(obs_values.dropna().astype(str).unique()):
        value_slug = snake_case(value)
        if not value_slug:
            continue

        stem = dedupe_stem(f"{value_slug}_{suffix}", used_stems)
        out = output_path / f"{stem}.h5ad"
        subset = _materialize_obs_subset(adata, obs_values_text == value)
        normalize_h5ad_string_storage(subset)
        subset.write_h5ad(out)
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


def normalize_h5ad_string_storage(adata: ad.AnnData) -> None:
    """Convert pandas Arrow-backed string columns to object dtype for h5ad
    writes.
    """
    adata.obs.index = pd.Index(adata.obs_names.astype(str), dtype=object)
    adata.var.index = pd.Index(adata.var_names.astype(str), dtype=object)
    for frame in (adata.obs, adata.var):
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
    obs_mask: pd.Series,
) -> ad.AnnData:
    """Return an in-memory AnnData subset for the given obs mask."""
    subset = adata[obs_mask.to_numpy(), :]
    return subset.to_memory() if subset.isbacked else subset.copy()
