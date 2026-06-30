"""Transcriptomic-clock model loading and prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import joblib  # type: ignore
import numpy as np
import pandas as pd
import sklearn.base  # type: ignore
import sklearn.impute  # type: ignore
import sklearn.pipeline  # type: ignore


logger = logging.getLogger(__name__)

SPECIES_MAX_LIFESPAN = {
    "human": 122.5,
    "mouse": 48.0,
    "rat": 50.4,
    "monkey": 39.0,
}

NORMALIZED_AGE_SCALE = 100.0
MODEL_TYPE_PREFIXES = ("br", "en")
CHRONOLOGICAL_CLOCK_KEYS = ("chronoage", "chronologicalage")
NORMALIZED_AGE_CLOCK_KEYS = ("normalizedage", "normalized_age")


class _ClockEstimator(Protocol):
    """Estimator interface used by clock prediction."""

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


def _final_estimator(estimator: _ClockEstimator) -> object:
    """Return the terminal estimator of a Pipeline, or the estimator itself."""
    if isinstance(estimator, sklearn.pipeline.Pipeline):
        return estimator.steps[-1][1]
    return estimator


def _patch_sklearn_estimator_compatibility(estimator: object) -> None:
    """Patch loaded sklearn estimators for known serialization drift."""
    for _, step in _iter_sklearn_estimators(estimator):
        if (
            isinstance(step, sklearn.impute.SimpleImputer)
            and not hasattr(step, "_fill_dtype")
            and hasattr(step, "_fit_dtype")
        ):
            step._fill_dtype = step._fit_dtype  # type: ignore


def _iter_sklearn_estimators(estimator: object) -> list[tuple[str, object]]:
    """Return an estimator or its direct Pipeline steps."""
    if isinstance(estimator, sklearn.pipeline.Pipeline):
        return [
            (name, step)
            for name, step in estimator.steps
            if step != "passthrough"
        ]
    if isinstance(estimator, sklearn.base.BaseEstimator):
        return [(type(estimator).__name__, estimator)]
    return []


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
    _patch_sklearn_estimator_compatibility(estimator)

    if isinstance(estimator, sklearn.pipeline.Pipeline):
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


def _clock_key(clock_name: str | Path) -> str:
    """Infer the biological clock target from a serialized model name."""
    parts = Path(clock_name).stem.lower().split("_")
    if len(parts) > 1 and parts[0] in MODEL_TYPE_PREFIXES:
        return parts[1]
    return parts[0] if parts else ""


def clock_prediction_scale(clock_name: str | Path, species: str) -> float:
    """Return the unit conversion factor for one clock output.

    Chronological-age clocks are trained in relative lifespan units and are
    converted to species-specific years. Normalized-age clocks are converted to
    percent of expected lifespan. Mortality and lifespan-effect clocks are
    already in their model units and are not lifespan-scaled.

    Args:
      clock_name: Serialized model name or path.
      species: Query species used for chronological-age clock scaling.

    Returns:
      Multiplicative factor applied to point predictions and predictive std.
    """
    key = _clock_key(clock_name)
    if key in CHRONOLOGICAL_CLOCK_KEYS:
        return SPECIES_MAX_LIFESPAN.get(species, 1.0)
    return NORMALIZED_AGE_SCALE if key in NORMALIZED_AGE_CLOCK_KEYS else 1.0


def align_to_model_features(
    features: pd.DataFrame,
    clock: ClockModel,
) -> tuple[pd.DataFrame, float]:
    """Align a relative-feature matrix to a model's ordered features.

    Features absent from the matrix are left as NaN so the fitted model
    pipeline's training-median imputer handles them.

    Args:
      features: Metacells x mouse-Entrez relative features.
      clock: Target clock model.

    Returns:
      A tuple of (aligned_features, coverage) where coverage is the fraction of
      model features present in the input matrix.
    """
    coverage = model_feature_coverage(features, clock)
    aligned = features.reindex(columns=clock.feature_names)
    return aligned, coverage


def model_feature_coverage(
    features: pd.DataFrame,
    clock: ClockModel,
) -> float:
    """Return the fraction of model features present in an input matrix."""
    present = features.columns.intersection(pd.Index(clock.feature_names))
    return len(present) / len(clock.feature_names)


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
      species: Query species; selects chronological-age lifespan scaling.
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

    adjustment = clock_prediction_scale(clock.name, species)
    point = point * adjustment
    std = std * adjustment

    return pd.DataFrame(
        {"tage": point, "tage_std": std},
        index=features.index,
    )
