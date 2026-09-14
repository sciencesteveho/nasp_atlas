"""Operational acceptance checks for model scopes and score checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.analysis.tabula_sapiens import mixed_model_scope_analysis
from nasp_atlas.analysis.tabula_sapiens.score_checkpoint import (
    score_checkpoint_matches,
)
from nasp_atlas.analysis.tabula_sapiens.score_checkpoint import (
    write_score_checkpoint,
)
from nasp_atlas.single_cell import ObsSchema


@pytest.mark.parametrize(
    "combined,per_tissue",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_selected_scopes_use_all_their_cells(
    tmp_path: Path, combined: bool, per_tissue: bool
) -> None:
    """Each enabled scope fits its complete population and publishes support."""
    rng = np.random.default_rng(2)
    records = []
    for donor in range(6):
        for tissue in ["Muscle", "Lung"]:
            for cell_type in ["E", "M"]:
                for cell in range(2):
                    records.append(
                        {
                            "obs_name": f"{donor}_{tissue}_{cell_type}_{cell}",
                            "donor_id": str(donor),
                            "tissue_in_publication": tissue,
                            "cell_type": cell_type,
                            "feature_type": "module_score",
                            "feature_id": "M_score",
                            "feature_label": "M",
                            "feature_value": donor * 0.2
                            + (tissue == "Muscle")
                            + (cell_type == "E")
                            + rng.normal(0, 0.2),
                        }
                    )

    result = mixed_model_scope_analysis(
        pd.DataFrame(records),
        output_dir=tmp_path,
        provenance=pd.DataFrame([{"scorer": "scanpy"}]),
        schema=ObsSchema(),
        combined=combined,
        per_tissue=per_tissue,
        minimum_cells=2,
        plot_visualizations=False,
    )

    manifest = pd.read_csv(
        tmp_path / "association_tables" / "association_mixed_model_scopes.csv"
    )
    current = manifest.loc[manifest.status.eq("completed")]
    assert len(current) == int(combined) + 2 * int(per_tissue)
    assert (result is not None) == combined
    for scope in current.itertuples():
        assert scope.n_cells == (48 if scope.scope == "combined_input" else 24)
        diagnostics = pd.read_csv(
            Path(scope.output_dir)
            / "association_tables"
            / "association_mixed_model_diagnostics.csv"
        )
        assert diagnostics.n_independent_units.eq(6).all()
        assert diagnostics.n_observations.eq(
            24 if scope.scope == "combined_input" else 12
        ).all()


def test_failed_scope_does_not_publish_completion(tmp_path: Path) -> None:
    """Failed inference remains explicit and raises to the worker."""
    frame = pd.DataFrame(
        {"obs_name": ["a"], "tissue_in_publication": ["Muscle"]}
    )

    with pytest.raises(KeyError, match="mixed-model columns"):
        mixed_model_scope_analysis(
            frame,
            output_dir=tmp_path,
            provenance=pd.DataFrame(),
            schema=ObsSchema(),
            per_tissue=True,
            plot_visualizations=False,
        )

    manifest = pd.read_csv(
        tmp_path / "association_tables" / "association_mixed_model_scopes.csv"
    )
    assert manifest.status.eq("failed").any()
    assert not manifest.status.eq("completed").any()


def test_resume_rejects_changed_selection_and_corrupt_scores(
    tmp_path: Path,
) -> None:
    """Subset checkpoints and truncated scores cannot represent full runs."""
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        {"obs_name": ["a"], "scoring_scorers": ["scanpy"], "M_score": [1.0]}
    ).to_csv(path, index=False)
    request = {"tissue_label": "Muscle", "subset_fraction": 0.5}
    write_score_checkpoint(path, request)

    assert score_checkpoint_matches(path, request)
    assert not score_checkpoint_matches(
        path, {"tissue_label": None, "subset_fraction": None}
    )
    path.write_text("obs_name,scoring_scorers\na,scanpy\n")
    assert not score_checkpoint_matches(path, request)
