"""Configuration objects for single-cell analysis pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Literal


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding pipeline.

    Captures every parameter that defines a preprocessing + embedding run.
    Serializable to dict/JSON so the exact recipe can be stored alongside the
    resulting AnnData in `adata.uns`.

    Attributes:
      name: Human-readable identifier used as the output subdirectory
      pipeline: Normalization strategy
      harmony_key: obs column for Harmony batch correction (None to skip)
      force_directed: Force-directed layout algorithm ("fa" or "fr"), or None
        to skip
      n_top_genes: Number of HVGs to select
      n_pcs: Number of PCA components
      n_neighbors: k for the kNN graph
      umap_min_dist: Minimum distance for UMAP
      target_sum: Normalization target sum per cell (standard pipeline)
      scale: Whether to z-score scale prior to PCA (standard pipeline)
      exclude_highly_expressed: Exclude highly expressed genes during
        normalization (standard pipeline)
      max_fraction: Max fraction for highly expressed gene exclusion
        (standard pipeline)
      regress_out: obs column name(s) to regress from the expression matrix
        prior to PCA. None to skip
      hvg_kwargs: Extra kwargs for sc.pp.highly_variable_genes
      pearson_residuals_kwargs: Extra kwargs for
        sc.experimental.pp.recipe_pearson_residuals
    """

    # Identity
    name: str

    # Strategy
    pipeline: Literal["standard", "pearson_residuals"] = "standard"
    harmony_key: str | None = None
    force_directed: Literal["fa", "fr"] | None = None

    # Params
    n_top_genes: int | None = None
    n_pcs: int = 30
    n_neighbors: int = 15
    umap_min_dist: float = 0.5
    target_sum: float = 1e4
    scale: bool = False
    exclude_highly_expressed: bool = True
    max_fraction: float = 0.03
    regress_out: list[str] | None = None

    # Passthrough kwargs
    hvg_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    pearson_residuals_kwargs: dict[str, Any] | None = field(
        default=None, repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON or adata.uns storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingConfig:
        """Reconstruct from a dict (e.g. from adata.uns).

        Args:
          data: Dictionary of config parameters
        """
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string.

        Args:
          indent: JSON indentation level
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> EmbeddingConfig:
        """Reconstruct from a JSON string.

        Args:
          raw: JSON string of config parameters
        """
        return cls.from_dict(json.loads(raw))
