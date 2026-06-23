"""Transcriptomic-clock model loading and prediction."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast, overload

import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.pipeline  # type: ignore
from sklearn.exceptions import InconsistentVersionWarning  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

SPECIES_MAX_LIFESPAN = {
    "human": 122.5,
    "mouse": 48.0,
    "rat": 50.4,
    "monkey": 39.0,
}


class _ClockEstimator(Protocol):
    """Estimator interface used by clock prediction."""

    @overload
    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]: ...

    @overload
    def predict(
        self,
        X: pd.DataFrame,
        *,
        return_std: Literal[True],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        """Predict from an aligned feature matrix."""


class _FeaturedEstimator(_ClockEstimator, Protocol):
    """Non-pipeline estimator carrying training feature names."""

    feature_names: tuple[object, ...]


@dataclass(frozen=True)
class ClockModel:
    """A loaded transcriptomic-clock model and its feature space.

    Attributes:
      name: Identifier derived from the model filename.
      estimator: Fitted scikit-learn estimator or Pipeline.
      feature_names: Ordered model feature names (mouse Entrez strings).
      supports_std: Whether `predict` can return a predictive standard
        deviation (BayesianRidge does; ElasticNet does not).
    """

    name: str
    estimator: _ClockEstimator
    feature_names: tuple[str, ...]
    supports_std: bool


def _patch_simple_imputer(imputer: SimpleImputer) -> None:
    """Restore the `_fill_dtype` attribute on imputers from older sklearn."""
    if not hasattr(imputer, "_fill_dtype"):
        statistics = getattr(imputer, "statistics_", None)
        imputer._fill_dtype = (  # type: ignore
            np.float64 if statistics is None else statistics.dtype
        )


def _final_estimator(estimator: _ClockEstimator) -> object:
    """Return the terminal estimator of a Pipeline, or the estimator itself."""
    if isinstance(estimator, sklearn.pipeline.Pipeline):
        return estimator.steps[-1][1]
    return estimator


def load_clock(
    model_path: str | Path,
    *,
    std_capable_estimators: tuple[str, ...] = (
        "BayesianRidge",
        "ARDRegression",
    ),
) -> ClockModel:
    """Load a serialized clock model and extract its feature space.

    Args:
      model_path: Path to a joblib-serialized estimator or Pipeline.
      std_capable_estimators: Final estimator class names whose `predict`
        method can return predictive standard deviations.

    Returns:
      A `ClockModel` wrapping the estimator, its ordered features, and whether
      it supports predictive standard deviations.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    estimator = cast(_ClockEstimator, joblib.load(model_path))

    if isinstance(estimator, sklearn.pipeline.Pipeline):
        for _, step in estimator.steps:
            if isinstance(step, SimpleImputer):
                _patch_simple_imputer(step)
        feature_names = tuple(str(name) for name in estimator.feature_names_in_)
    else:
        estimator_with_features = cast(_FeaturedEstimator, estimator)
        feature_names = tuple(
            str(name) for name in estimator_with_features.feature_names
        )

    estimator_name = type(_final_estimator(estimator)).__name__
    supports_std = estimator_name in std_capable_estimators

    return ClockModel(
        name=model_path.stem,
        estimator=estimator,
        feature_names=feature_names,
        supports_std=supports_std,
    )


def align_to_model_features(
    features: pd.DataFrame,
    clock: ClockModel,
) -> tuple[pd.DataFrame, float]:
    """Align a relative-feature matrix to a model's ordered features.

    Features absent from the matrix are filled with zero, representing zero
    relative deviation from the stratum reference.

    Args:
      features: Metacells x mouse-Entrez relative features.
      clock: Target clock model.

    Returns:
      A tuple of (aligned_features, coverage) where coverage is the fraction of
      model features present in the input matrix.
    """
    present = features.columns.intersection(pd.Index(clock.feature_names))
    coverage = len(present) / len(clock.feature_names)
    aligned = features.reindex(columns=clock.feature_names, fill_value=0.0)
    return aligned, coverage


def predict_metacells(
    clock: ClockModel,
    features: pd.DataFrame,
    *,
    species: str = "human",
    return_std: bool = True,
) -> pd.DataFrame:
    """Predict transcriptomic age for metacells with one clock model.

    Args:
      clock: Loaded clock model.
      features: Metacells x mouse-Entrez relative features for one stratum.
      species: Query species; selects the max-lifespan adjustment applied to
        the models' relative output.
      return_std: Request the predictive standard deviation when supported.

    Returns:
      DataFrame indexed by metacell with columns "tage" and "tage_std"; the std
      column is NaN when the model does not support it.
    """
    aligned, coverage = align_to_model_features(features, clock)
    logger.info(
        "[clock.model] %s | feature coverage=%.1f%%",
        clock.name,
        coverage * 100.0,
    )

    emit_std = return_std and clock.supports_std
    if emit_std:
        point, std = clock.estimator.predict(aligned, return_std=True)
    else:
        point = clock.estimator.predict(aligned)
        std = np.full(point.shape, np.nan)

    adjustment = SPECIES_MAX_LIFESPAN.get(species)
    if adjustment is not None:
        point = point * adjustment
        std = std * adjustment

    return pd.DataFrame(
        {"tage": point, "tage_std": std},
        index=features.index,
    )
