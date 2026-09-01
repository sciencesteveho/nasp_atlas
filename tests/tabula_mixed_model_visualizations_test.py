"""Tests for the fixed Tabula Sapiens mixed-model figure set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.tabula_sapiens import visualizations


def test_fixed_mixed_model_plot_set_calls_every_estimand(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """All five contrasts and variance receive their stable filename."""
    effect_specs = [
        (
            "adjusted_context",
            "adjusted_cell_type",
            "nasp_mixed_adjusted_cell_type_effects",
        ),
        (
            "condition_by_cell_type",
            "condition_by_cell_type",
            "nasp_mixed_condition_effects_by_cell_type",
        ),
        (
            "age_by_cell_type",
            "age_by_cell_type",
            "nasp_mixed_age_slopes_by_cell_type",
        ),
        (
            "paired_tissue",
            "paired_tissue",
            "nasp_mixed_paired_tissue_effects",
        ),
        (
            "adjusted_context",
            "assay_batch_effects",
            "nasp_mixed_assay_batch_effects",
        ),
    ]
    contrasts = pd.DataFrame(
        {
            "analysis": [spec[0] for spec in effect_specs],
            "estimand": [spec[1] for spec in effect_specs],
            "feature_type": ["module_score"] * len(effect_specs),
            "estimable": [True] * len(effect_specs),
            "status": ["ok"] * len(effect_specs),
        }
    )
    variance = pd.DataFrame(
        {
            "analysis": ["adjusted_context"],
            "feature_type": ["module_score"],
            "estimable": [True],
            "status": ["ok"],
        }
    )
    calls: list[tuple[str, str, str]] = []

    class RecordingPlotter:
        """Record plot requests while materializing their path contract."""

        def __init__(self, output_dir: str | Path) -> None:
            """Create the output directory used by the plotting helper."""
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def plot_mixed_model_effects(
            self,
            effects: pd.DataFrame,
            *,
            analysis: str,
            estimand: str,
            filename: str,
            **kwargs: object,
        ) -> None:
            """Record one effect request and stand in for figure saving."""
            assert effects is contrasts
            calls.append((analysis, estimand, filename))
            (self.output_dir / f"{filename}.png").write_bytes(b"plot")

        def plot_mixed_model_variance(
            self,
            values: pd.DataFrame,
            *,
            analysis: str,
            filename: str,
            **kwargs: object,
        ) -> None:
            """Record the variance request and stand in for figure saving."""
            assert values is variance
            calls.append((analysis, "variance_decomposition", filename))
            (self.output_dir / f"{filename}.png").write_bytes(b"plot")

    monkeypatch.setattr(visualizations, "MixedModelPlotter", RecordingPlotter)

    paths = visualizations.plot_tabula_sapiens_mixed_model_inference(
        output_dir=tmp_path,
        contrasts=contrasts,
        variance_components=variance,
    )

    assert calls == [
        *effect_specs,
        (
            "adjusted_context",
            "variance_decomposition",
            "nasp_mixed_variance_decomposition",
        ),
    ]
    assert [path.stem for path in paths] == [call[2] for call in calls]


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
