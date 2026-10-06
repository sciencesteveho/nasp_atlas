"""Tests for CELLxGENE metadata workflows and plots."""

from __future__ import annotations

import pandas as pd

from nasp_atlas.cellxgene import CategorySchema
from nasp_atlas.cellxgene import CXGMetadata
from nasp_atlas.cellxgene import CXGMetadataConfig


def _metadata_workflow() -> CXGMetadata:
    """Return a small metadata workflow with deterministic categories."""
    datasets = pd.DataFrame(
        {
            "dataset_id": ["dataset_a", "dataset_b"],
            "collection_name": ["Collection A", "Collection B"],
        }
    )
    obs = pd.DataFrame(
        {
            "dataset_id": ["dataset_a", "dataset_a", "dataset_b"],
            "disease": ["healthy", "influenza", "healthy"],
            "tissue": ["upper lung", "lower lung", "liver"],
            "development_stage": [
                "20-year-old human stage",
                "40-year-old human stage",
                "60-year-old human stage",
            ],
        }
    )
    schema = CategorySchema(
        disease_patterns={
            "normal": ("healthy",),
            "infectious": ("influenza",),
        },
        tissue_patterns={"lung": ("lung",), "liver": ("liver",)},
    )
    config = CXGMetadataConfig(category_schema=schema)
    return CXGMetadata(datasets=datasets, obs=obs, config=config)


def test_metadata_workflow_categorizes_before_filtering() -> None:
    """Default categories are available when filtering raw metadata."""
    metadata = _metadata_workflow()

    result = metadata.filter_tissues(["lung"])

    assert result is metadata
    assert metadata.obs["tissue"].tolist() == ["upper lung", "lower lung"]
    assert metadata.obs["tissue_category"].tolist() == ["lung", "lung"]
