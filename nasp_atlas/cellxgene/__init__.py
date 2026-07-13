"""CELLxGENE querying and visualization utilities."""

from nasp_atlas.cellxgene.categorize import CategorySchema
from nasp_atlas.cellxgene.categorize import categorize_development_stage
from nasp_atlas.cellxgene.categorize import categorize_disease
from nasp_atlas.cellxgene.categorize import categorize_tissue
from nasp_atlas.cellxgene.categorize import collapse_sex_series
from nasp_atlas.cellxgene.categorize import load_category_schema
from nasp_atlas.cellxgene.categorize import stage_age_value
from nasp_atlas.cellxgene.categorize import summarize_development_stage
from nasp_atlas.cellxgene.config import CXGMetadataConfig
from nasp_atlas.cellxgene.metadata import add_development_stage_age_obs
from nasp_atlas.cellxgene.metadata import category_color_map_from_uns
from nasp_atlas.cellxgene.query import CXGMetadata


__all__ = [
    "CXGMetadata",
    "CXGMetadataConfig",
    "CategorySchema",
    "add_development_stage_age_obs",
    "categorize_development_stage",
    "categorize_disease",
    "categorize_tissue",
    "category_color_map_from_uns",
    "collapse_sex_series",
    "load_category_schema",
    "stage_age_value",
    "summarize_development_stage",
]
