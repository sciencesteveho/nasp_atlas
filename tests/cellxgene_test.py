"""Tests for CELLxGENE metadata workflows and plots."""

from __future__ import annotations

import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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


def test_metadata_barplot_writes_composition_figure(tmp_path) -> None:
    """Metadata composition plotting writes a non-empty image."""
    metadata = _metadata_workflow()
    output_path = tmp_path / "tissue_composition.png"

    metadata.metadata_barplot(
        label_column="tissue",
        outpath=output_path,
    )

    assert output_path.stat().st_size > 0


def test_metadata_sankey_writes_category_flow_figure(tmp_path) -> None:
    """Metadata Sankey plotting writes a non-empty category-flow image."""
    metadata = _metadata_workflow()
    output_path = tmp_path / "tissue_sankey.png"

    written_path = metadata.metadata_sankey(
        label_column="tissue",
        outpath=output_path,
    )

    assert written_path == output_path
    assert output_path.stat().st_size > 0


def test_plot_age_ranges_writes_development_stage_figure(tmp_path) -> None:
    """Age-range plotting writes a non-empty development-stage image."""
    metadata = _metadata_workflow()
    output_path = tmp_path / "age_ranges.png"

    written_path = metadata.plot_age_ranges(output_path)

    assert written_path == output_path
    assert output_path.stat().st_size > 0
