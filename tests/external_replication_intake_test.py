"""Real sparse-file experiments for explicit external cohort preparation."""

from __future__ import annotations

from dataclasses import replace

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp  # type: ignore[import]
import yaml  # type: ignore[import]

from nasp_atlas.analysis.external_replication import prepare_cohort_counts
from nasp_atlas.analysis.external_replication import read_cohort_metadata
from nasp_atlas.analysis.external_replication import read_cohort_spec
from nasp_atlas.analysis.external_replication import select_cohort
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import aggregate_feature_frame_by_keys
from nasp_atlas.single_cell.associations import paired_feature_contrasts


@pytest.mark.parametrize("source_name", ["raw/X", "layers/counts"])
def test_count_source_and_row_identity_survive_donor_aggregation(
    tmp_path, source_name
) -> None:
    """Read the declared matrix, normalize before subsetting and pair people."""
    obs = pd.DataFrame(
        {
            "person": ["d1", "d2", "d3", "d4"] * 2 + ["unmatched"],
            "site": ["spleen"] * 4 + ["blood"] * 4 + ["spleen"],
            "label": "mono",
            "technology": "10x",
            "library": ["a", "b", "c"] * 3,
        },
        index=pd.Index([f"c{i}" for i in range(9)], dtype=object),
        dtype=object,
    )
    counts = np.column_stack([np.arange(1, 10), np.full(9, 10)])
    var = pd.DataFrame(
        {"symbol": ["A", "B"]},
        index=pd.Index(["ens1", "ens2"], dtype=object),
        dtype=object,
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(np.full((9, 2), 99)),
        obs=obs,
        var=var,
        layers={"counts": sp.csr_matrix(counts)},
    )
    adata.raw = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    data_path = tmp_path / "fixture.hdf"
    adata.write_h5ad(data_path)
    config = {
        "cohort_id": "fixture",
        "input_path": data_path.name,
        "counts_source": source_name,
        "gene_id_column": "_index",
        "gene_symbol_column": "symbol",
        "obs_columns": {
            "donor_id": "person",
            "tissue": "site",
            "assay": "technology",
        },
        "obs_constants": {"modality": "cells"},
        "filters": {"technology": ["10x"]},
        "populations": {"monocyte": {"column": "label", "labels": ["mono"]}},
        "library_keys": ["library"],
        "comparison": {
            "level_key": "tissue",
            "target": "spleen",
            "reference": "blood",
            "minimum_cells": 1,
            "minimum_pairs": 4,
        },
        "source_notes": "Synthetic counts and deliberately incompatible X.",
    }
    config_path = tmp_path / "cohort.yaml"
    config_path.write_text(yaml.safe_dump(config))
    spec = read_cohort_spec(config_path)
    metadata, _ = read_cohort_metadata(spec)
    intake = select_cohort(metadata, spec)
    assert len(intake.selected_obs) == 8
    assert not intake.donor_support.set_index("donor_id").loc[
        "unmatched", "eligible"
    ]
    requested = intake.selected_obs.iloc[::-1]
    prepared = prepare_cohort_counts(spec, requested)
    assert prepared.obs_names.tolist() == requested.index.tolist()
    positions = obs.index.get_indexer(requested.index)
    selected_counts = counts[positions]
    np.testing.assert_array_equal(
        prepared.layers["counts"].toarray(), selected_counts
    )
    expected = np.log1p(
        selected_counts / selected_counts.sum(axis=1)[:, None] * 10000
    )
    np.testing.assert_allclose(prepared.X.toarray(), expected)
    assert sp.issparse(prepared.X)

    frame = requested.assign(
        feature_type="gene_expression",
        feature_id="ens1",
        feature_label="A",
        feature_value=prepared.X[:, 0].toarray().ravel(),
    )
    donor = aggregate_feature_frame_by_keys(
        frame,
        unit_keys=["cohort_id", "donor_id", "tissue"],
        statistical_unit="donor_context",
        aggregation="mean",
        schema=ObsSchema(study_key="cohort_id", tissue_key="tissue"),
    )
    estimate = paired_feature_contrasts(
        donor,
        pair_keys=("cohort_id", "donor_id"),
        level_key="tissue",
        target_level="spleen",
        reference_level="blood",
        minimum_pairs=4,
    ).estimates.iloc[0]
    all_expression = np.log1p(counts / counts.sum(axis=1)[:, None] * 10000)
    assert estimate["estimate"] == pytest.approx(
        (all_expression[:4, 0] - all_expression[4:8, 0]).mean()
    )
    assert estimate["n_paired_donors"] == 4

    overlapping = replace(
        spec,
        populations=(
            spec.populations[0],
            replace(spec.populations[0], name="second"),
        ),
    )
    with pytest.raises(ValueError, match="Overlapping"):
        select_cohort(metadata, overlapping)
    with pytest.raises(ValueError, match="donor identity"):
        select_cohort(metadata.assign(person=pd.NA), spec)
