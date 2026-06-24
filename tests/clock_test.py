"""Tests for transcriptomic-clock model application."""

from __future__ import annotations

import os


os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import pandas as pd

import nasp_atlas.analysis.clock as clock_analysis
from nasp_atlas.single_cell.clocks.model import ClockModel
from nasp_atlas.single_cell.clocks.model import model_feature_coverage
from nasp_atlas.single_cell.clocks.model import predict_metacells


class _Estimator:
    """Minimal estimator used to exercise the prediction contract."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features.sum(axis=1).to_numpy()


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
    assert np.isnan(prediction.loc["cell", "tage_std"])
    assert model_feature_coverage(features, clock) == 0.5


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
