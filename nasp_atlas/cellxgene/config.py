"""Configuration for CELLxGENE metadata queries."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Self

from nasp_atlas.cellxgene.categorize import CategorySchema
from nasp_atlas.cellxgene.categorize import load_category_schema


@dataclass(frozen=True)
class CXGMetadataConfig:
    """Configuration for CELLxGENE metadata queries."""

    census_version: str = "2025-11-08"
    obs_cols: tuple[str, ...] = (
        "dataset_id",
        "donor_id",
        "assay",
        "cell_type",
        "development_stage",
        "disease",
        "sex",
        "tissue",
        "suspension_type",
    )
    dataset_cols: tuple[str, ...] = (
        "dataset_id",
        "collection_name",
        "collection_id",
        "dataset_total_cell_count",
        "dataset_h5ad_path",
    )
    category_schema: CategorySchema = field(
        default_factory=load_category_schema
    )

    @classmethod
    def from_category_schema(
        cls,
        source: str | Path,
        **kwargs: object,
    ) -> Self:
        """Build config using a custom category-schema YAML file or URL."""
        return cls(category_schema=load_category_schema(source), **kwargs)  # type: ignore

    def categorize_disease(self, raw: object) -> str:
        """Map a raw disease label to a configured broad category."""
        return self.category_schema.categorize_disease(raw)

    def categorize_tissue(self, raw: object) -> str:
        """Map a raw tissue label to a configured broad category."""
        return self.category_schema.categorize_tissue(raw)

    def display_name(self, label: object) -> str:
        """Return configured display name or a simple human-readable
        fallback.
        """
        label_string = str(label)
        return self.category_schema.display_names.get(
            label_string,
            label_string.replace("_", " ").capitalize(),
        )
