"""Tests for transcriptomic-clock model application."""

from __future__ import annotations

import os


os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import importlib.util
from pathlib import Path
from typing import cast

import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import nasp_atlas.analysis.clock as clock_analysis
from nasp_atlas.single_cell.clocks.model import ClockModel
from nasp_atlas.single_cell.clocks.model import load_clock
from nasp_atlas.single_cell.clocks.model import model_feature_coverage
from nasp_atlas.single_cell.clocks.model import predict_metacells


class _Estimator:
    """Minimal estimator used to exercise the prediction contract."""

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return X.sum(axis=1).to_numpy(dtype=float)


def test_predict_metacells_has_stable_dataframe_return_type() -> None:
    """Prediction metadata does not change the function's return shape."""
    clock = ClockModel(
        name="test",
        estimator=_Estimator(),
        feature_names=("a", "b"),
        supports_std=False,
    )
    features = pd.DataFrame({"a": [1.0], "extra": [5.0]}, index=["cell"])

    prediction = predict_metacells(clock, features, species="unknown")

    assert isinstance(prediction, pd.DataFrame)
    assert prediction.loc["cell", "tage"] == 1.0
    assert pd.isna(prediction.loc["cell", "tage_std"])
    assert model_feature_coverage(features, clock) == 0.5


def test_load_clock_patches_legacy_simple_imputer_fill_dtype(tmp_path) -> None:
    """Loaded legacy sklearn imputers predict under current sklearn."""
    training = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [2.0, 4.0, 6.0],
        }
    )
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("regressor", LinearRegression()),
        ]
    )
    estimator.fit(training, np.array([1.0, 2.0, 3.0]))
    delattr(estimator.named_steps["imputer"], "_fill_dtype")
    model_path = tmp_path / "legacy_clock.pkl"
    joblib.dump(estimator, model_path)

    clock = load_clock(model_path)
    prediction = predict_metacells(
        clock,
        pd.DataFrame({"a": [np.nan], "b": [5.0]}, index=["cell"]),
        species="unknown",
    )

    loaded = cast(Pipeline, clock.estimator)
    tage = cast(float, prediction.loc["cell", "tage"])
    assert hasattr(loaded.named_steps["imputer"], "_fill_dtype")
    assert np.isfinite(tage)


def test_predict_stratum_keeps_coverage_separate_per_clock(monkeypatch) -> None:
    """Coverage from one clock cannot overwrite another clock's coverage."""
    features = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0]},
        index=["one", "two"],
    )
    clocks = [
        ClockModel(
            name="BR_Chronoage_test_scaleddiff",
            estimator=_Estimator(),
            feature_names=("a", "missing"),
            supports_std=False,
        ),
        ClockModel(
            name="BR_Mortality_test_scaleddiff",
            estimator=_Estimator(),
            feature_names=("a", "b"),
            supports_std=False,
        ),
    ]
    monkeypatch.setattr(
        clock_analysis,
        "preprocess_metacells",
        lambda *args, **kwargs: {"scaled_diff": features},
    )

    result = clock_analysis._predict_stratum(
        features,
        pd.DataFrame(index=features.index),
        clocks=clocks,
        human_map=pd.Series(dtype=str),
        mouse_map=pd.Series(dtype=str),
        config=clock_analysis.ClockConfig(species="unknown"),
    )

    chronoage_coverage = result[
        "chronoage_scaleddiff_feature_coverage"
    ].unique()
    mortality_coverage = result[
        "mortality_scaleddiff_feature_coverage"
    ].unique()
    assert chronoage_coverage.tolist() == [0.5]
    assert mortality_coverage.tolist() == [1.0]


def test_clock_regression_plots_written_for_prediction_columns(
    tmp_path,
) -> None:
    """Clock regression helper writes one plot per prediction column."""
    tidy = pd.DataFrame(
        {
            "age_years": [20.0, 30.0, 40.0, 50.0],
            "chronoage_scaleddiff_tage": [18.0, 29.0, 42.0, 51.0],
            "chronoage_scaleddiff_tage_std": [1.0, 1.0, 1.0, 1.0],
        }
    )

    clock_analysis._plot_clock_regressions(
        tidy,
        output_dir=tmp_path,
        level="tissue",
        age_key="age_years",
    )

    assert (
        tmp_path / "clock_tissue_chronoage_scaleddiff_tage_regression.png"
    ).exists()


def test_combined_clock_regression_dev_module_combines_tissue_tables(
    tmp_path,
) -> None:
    """Development helper pools per-tissue clock tables before plotting."""
    module_path = (
        Path(__file__).parents[1]
        / "development"
        / "plot_combined_clock_regressions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_combined_clock_regressions",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    root = tmp_path / "clock_root"
    for tissue, ages, predictions in (
        ("liver", [20.0, 30.0], [18.0, 31.0]),
        ("lung", [40.0, 50.0], [39.0, 52.0]),
    ):
        tissue_dir = root / tissue
        tissue_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "age_years": ages,
                "chronoage_scaleddiff_tage": predictions,
            }
        ).to_csv(tissue_dir / "clock_tissue_metacells.csv", index=False)

    output_dir = tmp_path / "combined"
    results = module.plot_combined_clock_regressions(
        clock_root=root,
        output_dir=output_dir,
        levels=("tissue",),
    )

    combined = results["tissue"]
    assert combined.shape[0] == 4
    assert combined["clock_source_dir"].tolist() == [
        "liver",
        "liver",
        "lung",
        "lung",
    ]
    assert (output_dir / "clock_tissue_combined_metacells.csv").exists()
    assert (
        output_dir
        / "clock_combined_tissue_chronoage_scaleddiff_tage_regression.png"
    ).exists()
