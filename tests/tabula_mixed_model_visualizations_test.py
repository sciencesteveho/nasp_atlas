"""Tests for the fixed Tabula Sapiens mixed-model figure set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.tabula_sapiens import visualizations


def test_fixed_plot_helper_removes_stale_module_file_for_gene_only_results(
    tmp_path: Path,
) -> None:
    """A prior module figure is not reported when only genes are estimable."""
    stale = tmp_path / "nasp_mixed_adjusted_cell_type_effects.png"
    stale.write_bytes(b"stale")
    contrasts = pd.DataFrame(
        {
            "analysis": ["adjusted_context"],
            "estimand": ["adjusted_cell_type"],
            "feature_type": ["gene_expression"],
            "estimable": [True],
            "status": ["ok"],
        }
    )

    paths = visualizations.plot_tabula_sapiens_mixed_model_inference(
        output_dir=tmp_path,
        contrasts=contrasts,
        variance_components=pd.DataFrame(),
    )

    assert paths == []
    assert not stale.exists()
