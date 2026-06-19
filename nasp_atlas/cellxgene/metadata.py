"""Helpers for CELLxGENE observation metadata."""

from __future__ import annotations

from typing import Any

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd

from nasp_atlas.cellxgene.categorize import stage_age_value


def add_development_stage_age_obs(
    adata: ad.AnnData,
    *,
    stage_column: str = "development_stage",
    age_column: str = "age_years",
) -> ad.AnnData:
    """Add numeric age in years from CELLxGENE development-stage labels.

    Args:
      adata: AnnData object with CELLxGENE observation metadata.
      stage_column: Observation column with development-stage labels.
      age_column: Observation column to create with numeric age values.

    Returns:
      The input AnnData object, modified in place.
    """
    age_values = pd.to_numeric(
        adata.obs[stage_column].astype(str).map(stage_age_value),
        errors="coerce",
    )
    adata.obs[age_column] = age_values.replace(
        [float("inf"), float("inf") - 1.0],
        np.nan,
    )
    return adata


def category_color_map_from_uns(
    adata: ad.AnnData,
    obs_key: str,
) -> dict[Any, str]:
    """Return category-color mapping from Scanpy-style uns colors."""
    adata.obs[obs_key] = adata.obs[obs_key].astype("category")
    categories = list(adata.obs[obs_key].cat.categories)
    colors = list(adata.uns[f"{obs_key}_colors"])

    if len(categories) != len(colors):
        raise ValueError(
            f"{obs_key} has {len(categories)} categories but "
            f"{obs_key}_colors has {len(colors)} colors."
        )

    return dict(zip(categories, colors, strict=True))
