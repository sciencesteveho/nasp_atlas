"""UMAP panel input and resolution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, Required, TypedDict, cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap


__all__ = [
    "UmapPanel",
    "UmapPanelSpec",
    "embedding_xy",
    "is_categorical_obs",
    "resolve_umap_panel_specs",
]


class UmapPanelSpec(TypedDict, total=False):
    """User-facing specification for one observation-colored UMAP panel.

    Example Usage:
      >>> panel: UmapPanelSpec = {
      ...     "obs_key": "cell_type",
      ...     "title": "Cell type",
      ... }
      >>> panel["obs_key"]
      'cell_type'
    """

    obs_key: Required[str]
    title: NotRequired[str]
    kind: NotRequired[Literal["categorical", "numeric"]]
    color_map: NotRequired[dict[str, str]]
    cmap: NotRequired[Colormap | str]
    legend_loc: NotRequired[Literal["right", "bottom"]]
    legend_ncol: NotRequired[int]
    vmin: NotRequired[float | None]
    vmax: NotRequired[float | None]
    cbar_ticks: NotRequired[Sequence[float] | None]


@dataclass(frozen=True, kw_only=True, slots=True)
class UmapPanel:
    """Resolved UMAP panel configuration."""

    obs_key: str
    title: str
    kind: Literal["categorical", "numeric"] = "categorical"
    color_map: dict[str, str] | None = None
    cmap: Colormap | str = "viridis"
    legend_loc: Literal["right", "bottom"] = "right"
    legend_ncol: int = 1
    vmin: float | None = None
    vmax: float | None = None
    cbar_ticks: Sequence[float] | None = None


def resolve_umap_panel_specs(
    adata: ad.AnnData,
    panels: Sequence[str | UmapPanelSpec],
) -> list[UmapPanel]:
    """Normalize user panel inputs while preserving requested order."""
    resolved: list[UmapPanel] = []
    for panel in panels:
        if isinstance(panel, str):
            spec: UmapPanelSpec = {"obs_key": panel}
        else:
            spec = cast(UmapPanelSpec, dict(panel))

        obs_key = spec["obs_key"]
        if obs_key not in adata.obs.columns:
            raise KeyError(f"obs column not found for UMAP panel: {obs_key}")

        title = spec.get("title", obs_key)
        kind = spec.get("kind")
        if kind is None:
            if pd.api.types.is_numeric_dtype(adata.obs[obs_key]):
                kind = "numeric"
            else:
                kind = "categorical"

        resolved.append(
            UmapPanel(
                obs_key=obs_key,
                title=title,
                kind=kind,
                color_map=spec.get("color_map"),
                cmap=spec.get("cmap", "viridis"),
                legend_loc=spec.get("legend_loc", "right"),
                legend_ncol=spec.get("legend_ncol", 1),
                vmin=spec.get("vmin"),
                vmax=spec.get("vmax"),
                cbar_ticks=spec.get("cbar_ticks"),
            )
        )

    return resolved


def embedding_xy(adata: ad.AnnData, *, basis: str) -> np.ndarray:
    """Return first two embedding coordinates as a dense numpy array."""
    if basis not in adata.obsm:
        raise KeyError(f"embedding basis not found: {basis}")

    xy = np.asarray(adata.obsm[basis])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            f"embedding basis {basis!r} must have at least two columns"
        )

    return xy[:, :2]


def is_categorical_obs(adata: ad.AnnData, key: str) -> bool:
    """Return True if key is a categorical obs column."""
    return hasattr(adata.obs[key], "cat") if key in adata.obs.columns else False
