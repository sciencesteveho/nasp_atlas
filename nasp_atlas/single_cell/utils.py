"""Reusable single-cell utility functions."""

from __future__ import annotations

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
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path
        if subset_fraction is not None:
            adata = random_cell_subset(
                adata,
                fraction=subset_fraction,
                random_state=random_state,
            )
    else:
        adata, _ = read_h5ad(
            adata_or_path,
            subset_fraction=subset_fraction,
            random_state=random_state,
        )

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
        subset = adata[obs_values_text == value].copy()
        normalize_h5ad_string_storage(subset)
        subset.write_h5ad(out)
        written[value] = out

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
            if isinstance(frame[column].dtype, pd.StringDtype):
                frame[column] = frame[column].astype(object)
