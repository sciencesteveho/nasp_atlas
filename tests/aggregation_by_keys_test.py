"""Tests for explicit-key feature aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.single_cell.associations.aggregation import (
    aggregate_feature_frame_by_keys,
)
from nasp_atlas.single_cell.associations.core import Aggregation
from nasp_atlas.single_cell.associations.core import ObsSchema


def test_explicit_keys_keep_same_donor_separate_between_studies() -> None:
    """A repeated donor label remains distinct in different studies."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 4,
            "feature_id": ["NASP_score"] * 4,
            "feature_label": ["NASP"] * 4,
            "feature_value": [1.0, 3.0, 10.0, 14.0],
            "dataset_id": ["study_a", "study_a", "study_b", "study_b"],
            "donor_id": ["D1"] * 4,
        }
    )

    result = aggregate_feature_frame_by_keys(
        cell_frame,
        unit_keys=["dataset_id", "donor_id"],
        statistical_unit="study_donor",
        aggregation="mean",
        schema=ObsSchema(),
    )
    by_study = result.set_index("dataset_id")

    assert by_study["feature_value"].to_dict() == {
        "study_a": 2.0,
        "study_b": 12.0,
    }
    assert set(result["unit_id"]) == {
        "dataset_id=study_a|donor_id=D1",
        "dataset_id=study_b|donor_id=D1",
    }
    assert set(result["statistical_unit"]) == {"study_donor"}


def test_explicit_keys_retain_assay_values_and_cell_counts() -> None:
    """Assay units retain means and observed versus total cell counts."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 5,
            "feature_id": ["NASP_score"] * 5,
            "feature_label": ["NASP"] * 5,
            "feature_value": [1.0, np.nan, 5.0, 3.0, 7.0],
            "dataset_id": ["study_a"] * 5,
            "donor_id": ["D1"] * 5,
            "tissue": ["lung"] * 5,
            "cell_type": ["macrophage"] * 5,
            "assay": ["10x", "10x", "10x", "smartseq", "smartseq"],
        }
    )

    result = aggregate_feature_frame_by_keys(
        cell_frame,
        unit_keys=[
            "dataset_id",
            "donor_id",
            "tissue",
            "cell_type",
            "assay",
        ],
        statistical_unit="study_donor_tissue_cell_type_assay",
        aggregation="mean",
        schema=ObsSchema(tissue_key="tissue"),
    ).set_index("assay")

    assert result["feature_value"].to_dict() == {"10x": 3.0, "smartseq": 5.0}
    assert result["n_cells"].to_dict() == {"10x": 2, "smartseq": 2}
    assert result["n_cells_total"].to_dict() == {"10x": 3, "smartseq": 2}


def test_explicit_keys_carry_only_consensus_metadata() -> None:
    """Constant metadata is retained while heterogeneous metadata is missing."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["module_score", "module_score"],
            "feature_id": ["NASP_score", "NASP_score"],
            "feature_label": ["NASP", "NASP"],
            "feature_value": [1.0, 3.0],
            "donor_id": ["D1", "D1"],
            "dataset_id": ["study_a", "study_a"],
            "disease": ["normal", "normal"],
            "assay": ["10x", "smartseq"],
            "cohort": ["discovery", "discovery"],
            "library_batch": ["batch_1", "batch_2"],
        }
    )

    result = aggregate_feature_frame_by_keys(
        cell_frame,
        unit_keys=["donor_id"],
        statistical_unit="donor",
        aggregation="mean",
        schema=ObsSchema(),
        metadata_keys=["cohort", "library_batch"],
    ).iloc[0]

    assert result["dataset_id"] == "study_a"
    assert result["disease"] == "normal"
    assert result["cohort"] == "discovery"
    assert pd.isna(result["assay"])
    assert pd.isna(result["library_batch"])


def test_explicit_key_aggregation_does_not_mutate_input() -> None:
    """Explicit-key aggregation leaves the supplied cell frame unchanged."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["module_score", "module_score"],
            "feature_id": ["NASP_score", "NASP_score"],
            "feature_label": ["NASP", "NASP"],
            "feature_value": [1.0, 3.0],
            "donor_id": ["D1", "D1"],
        }
    )
    original = cell_frame.copy(deep=True)

    aggregate_feature_frame_by_keys(
        cell_frame,
        unit_keys=["donor_id"],
        statistical_unit="donor",
        aggregation="mean",
        schema=ObsSchema(),
    )

    pd.testing.assert_frame_equal(cell_frame, original)


def test_explicit_key_aggregation_uses_detection_threshold() -> None:
    """Expressing fractions use finite values above the requested threshold."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["gene_expression"] * 5,
            "feature_id": ["CGAS"] * 5,
            "feature_label": ["CGAS"] * 5,
            "feature_value": [-1.0, 0.2, 0.8, np.nan, np.inf],
            "donor_id": ["D1"] * 5,
        }
    )

    result = aggregate_feature_frame_by_keys(
        cell_frame,
        unit_keys=["donor_id"],
        statistical_unit="donor",
        aggregation="fraction_expressing",
        schema=ObsSchema(),
        detection_threshold=0.5,
    ).iloc[0]

    assert result["feature_value"] == pytest.approx(1 / 3)
    assert result["n_cells"] == 3
    assert result["n_cells_total"] == 5


@pytest.mark.parametrize(
    "detection_threshold",
    [np.nan, np.inf, -np.inf, "0.0", None, True],
)
def test_explicit_key_aggregation_rejects_invalid_detection_threshold(
    detection_threshold: object,
) -> None:
    """Expression thresholds must define one finite numeric boundary."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["gene_expression"],
            "feature_id": ["CGAS"],
            "feature_label": ["CGAS"],
            "feature_value": [1.0],
            "donor_id": ["D1"],
        }
    )

    with pytest.raises(ValueError, match="finite real number"):
        aggregate_feature_frame_by_keys(
            cell_frame,
            unit_keys=["donor_id"],
            statistical_unit="donor",
            aggregation="fraction_expressing",
            schema=ObsSchema(),
            detection_threshold=detection_threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("aggregation", ["mean", "median", "sum"])
def test_numeric_reducers_use_only_finite_values(
    aggregation: Aggregation,
) -> None:
    """Reducers align their values with finite contributing-cell counts."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["module_score"] * 5,
            "feature_id": ["NASP_score"] * 5,
            "feature_label": ["NASP"] * 5,
            "feature_value": [1.0, np.inf, np.nan, np.inf, np.nan],
            "donor_id": ["D1", "D1", "D1", "D2", "D2"],
        }
    )

    result = aggregate_feature_frame_by_keys(
        cell_frame,
        unit_keys=["donor_id"],
        statistical_unit="donor",
        aggregation=aggregation,
        schema=ObsSchema(),
    ).set_index("donor_id")

    assert result.loc["D1", "feature_value"] == pytest.approx(1.0)
    assert result.loc["D1", "n_cells"] == 1
    assert pd.isna(result.loc["D2", "feature_value"])
    assert result.loc["D2", "n_cells"] == 0


@pytest.mark.parametrize(
    ("unit_keys", "metadata_keys", "missing_key"),
    [
        (["missing_unit"], (), "missing_unit"),
        (["donor_id"], ("missing_batch",), "missing_batch"),
    ],
)
def test_explicit_key_aggregation_rejects_missing_columns(
    unit_keys: list[str],
    metadata_keys: tuple[str, ...],
    missing_key: str,
) -> None:
    """Missing grouping or requested metadata keys fail with their names."""
    cell_frame = pd.DataFrame(
        {
            "feature_type": ["module_score"],
            "feature_id": ["NASP_score"],
            "feature_label": ["NASP"],
            "feature_value": [1.0],
            "donor_id": ["D1"],
        }
    )

    with pytest.raises(KeyError, match=missing_key):
        aggregate_feature_frame_by_keys(
            cell_frame,
            unit_keys=unit_keys,
            statistical_unit="custom_unit",
            aggregation="mean",
            schema=ObsSchema(),
            metadata_keys=metadata_keys,
        )
