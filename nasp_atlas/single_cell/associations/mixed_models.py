"""Repeated-measures mixed-model inference for aggregated feature frames."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from itertools import combinations
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
import patsy  # type: ignore[import]
import statsmodels.formula.api as smf  # type: ignore[import]
from scipy import stats  # type: ignore[import]
from statsmodels.regression.mixed_linear_model import (  # type: ignore[import]
    MixedLM,
)
from statsmodels.regression.mixed_linear_model import (  # type: ignore[import]
    MixedLMResultsWrapper,
)

from nasp_atlas.single_cell.associations.core import benjamini_hochberg


ContrastMode = Literal[
    "categorical_vs_mean",
    "categorical_pairwise",
    "categorical_vs_reference",
    "continuous_slope",
]

__all__ = [
    "MixedModelContrast",
    "MixedModelInferenceResult",
    "MixedModelSpec",
    "mixed_model_inference",
]


@dataclass(frozen=True, kw_only=True)
class MixedModelSpec:
    """Specify one reusable family of per-feature mixed models.

    Predictors are named in the caller's schema. The implementation aliases
    them before constructing a Patsy formula, so spaces and punctuation in
    source column names do not alter the model. Every interaction includes its
    two main effects. Variance-component keys describe random intercepts nested
    within the top-level `group_key`.

    Attributes:
      analysis: Stable analysis identifier written to every result table.
      response_key: Numeric response column fitted within each feature.
      group_key: Top-level random-intercept grouping column.
      independent_unit_key: Column counted as independent biological support.
        Defaults to `group_key`.
      continuous_keys: Numeric fixed-effect predictors and covariates.
      categorical_keys: Categorical fixed-effect predictors and covariates.
      interactions: Pairs of declared predictors whose interaction is fitted.
      variance_component_keys: Nested random-intercept classifications.
      minimum_groups: Minimum top-level random-effect groups needed to fit.
      minimum_independent_units: Minimum independent units needed to fit.
      minimum_focal_units: Minimum independent support for each contrast arm.
      reml: Whether to fit by restricted maximum likelihood.
      alpha: Two-sided confidence-interval and test error rate.
      optimizer_methods: Optimizers attempted in order by `statsmodels`.
      max_iterations: Maximum iterations passed to each optimizer.
    """

    analysis: str
    response_key: str = "feature_value"
    group_key: str = "donor_id"
    independent_unit_key: str | None = None
    continuous_keys: tuple[str, ...] = ()
    categorical_keys: tuple[str, ...] = ()
    interactions: tuple[tuple[str, str], ...] = ()
    variance_component_keys: tuple[str, ...] = ()
    minimum_groups: int = 3
    minimum_independent_units: int = 3
    minimum_focal_units: int = 3
    reml: bool = True
    alpha: float = 0.05
    optimizer_methods: tuple[str, ...] = ("lbfgs", "powell")
    max_iterations: int = 1_000


@dataclass(frozen=True, kw_only=True)
class MixedModelContrast:
    """Specify biologically interpretable marginal model contrasts.

    Attributes:
      mode: Contrast construction mode. "categorical_vs_mean" compares every
        selected level with the equally weighted marginal mean;
        "categorical_pairwise" compares paired levels; and
        "categorical_vs_reference" compares levels with `reference`.
        "continuous_slope" estimates a per-unit linear slope.
      predictor_key: Categorical or continuous predictor being contrasted.
      estimand: Stable scientific estimand name and FDR-family identifier.
      reference: Reference level for "categorical_vs_reference".
      by_key: Optional categorical modifier for simple effects or slopes.
      levels: Optional predictor levels to test, in requested output order.
      by_levels: Optional modifier levels to test, in requested output order.
      slope_step: Predictor increment defining a continuous slope.
      effect_scale: Human-readable unit carried to tables and figures.
    """

    mode: ContrastMode
    predictor_key: str
    estimand: str
    reference: str | None = None
    by_key: str | None = None
    levels: tuple[str, ...] | None = None
    by_levels: tuple[str, ...] | None = None
    slope_step: float = 1.0
    effect_scale: str = "score_difference"


@dataclass(frozen=True)
class MixedModelInferenceResult:
    """Tidy estimates and diagnostics from a mixed-model analysis family.

    Attributes:
      contrasts: Planned marginal effects used for scientific interpretation.
      fixed_effects: Raw fixed-effect coefficients for model audit.
      term_tests: Coefficient-block Wald tests for fixed-effect formula terms.
      variance_components: Conditional random and residual variance estimates.
      diagnostics: One fit-status and provenance row per modeled feature.
    """

    contrasts: pd.DataFrame
    fixed_effects: pd.DataFrame
    term_tests: pd.DataFrame
    variance_components: pd.DataFrame
    diagnostics: pd.DataFrame

    @classmethod
    def empty(cls) -> MixedModelInferenceResult:
        """Return empty result tables with every stable public column."""
        return cls(
            contrasts=pd.DataFrame(
                columns=(
                    "analysis",
                    "feature_type",
                    "feature_id",
                    "feature_label",
                    "estimand",
                    "contrast",
                    "predictor",
                    "level",
                    "reference",
                    "by_key",
                    "by_level",
                    "effect_scale",
                    "estimate",
                    "standard_error",
                    "ci_low",
                    "ci_high",
                    "statistic",
                    "pvalue",
                    "pvalue_fdr",
                    "n_independent_units",
                    "n_level_units",
                    "n_reference_units",
                    "n_paired_units",
                    "estimable",
                    "status",
                    "reason",
                )
            ),
            fixed_effects=pd.DataFrame(
                columns=(
                    "analysis",
                    "feature_type",
                    "feature_id",
                    "feature_label",
                    "term",
                    "estimate",
                    "standard_error",
                    "ci_low",
                    "ci_high",
                    "statistic",
                    "pvalue",
                    "pvalue_fdr",
                    "n_observations",
                    "n_groups",
                    "n_independent_units",
                    "status",
                    "reason",
                )
            ),
            term_tests=pd.DataFrame(
                columns=(
                    "analysis",
                    "feature_type",
                    "feature_id",
                    "feature_label",
                    "term",
                    "statistic",
                    "degrees_of_freedom",
                    "pvalue",
                    "pvalue_fdr",
                    "n_observations",
                    "n_groups",
                    "n_independent_units",
                    "status",
                    "reason",
                )
            ),
            variance_components=pd.DataFrame(
                columns=(
                    "analysis",
                    "feature_type",
                    "feature_id",
                    "feature_label",
                    "component",
                    "component_type",
                    "variance",
                    "variance_fraction",
                    "n_observations",
                    "n_groups",
                    "n_independent_units",
                    "estimable",
                    "status",
                    "reason",
                )
            ),
            diagnostics=pd.DataFrame(
                columns=(
                    "analysis",
                    "feature_type",
                    "feature_id",
                    "feature_label",
                    "formula",
                    "model_formula",
                    "alias_mapping",
                    "group_key",
                    "independent_unit_key",
                    "variance_component_keys",
                    "n_input_observations",
                    "n_observations",
                    "n_dropped_missing",
                    "n_dropped_nonfinite",
                    "n_groups",
                    "n_independent_units",
                    "max_observations_per_independent_unit",
                    "independent_unit_covariance",
                    "min_observations_per_group",
                    "max_observations_per_group",
                    "n_fixed_effects",
                    "design_rank",
                    "minimum_groups",
                    "minimum_independent_units",
                    "minimum_focal_units",
                    "reml",
                    "alpha",
                    "optimizer_methods",
                    "max_iterations",
                    "converged",
                    "warning_count",
                    "warning_messages",
                    "log_likelihood",
                    "aic",
                    "bic",
                    "statsmodels_version",
                    "status",
                    "reason",
                )
            ),
        )


@dataclass(frozen=True)
class _PreparedFeature:
    """Prepared safe-name model data and explicit filtering diagnostics."""

    data: pd.DataFrame
    aliases: dict[str, str]
    formula: str
    display_formula: str
    variance_formulas: dict[str, str]
    variance_names: dict[str, str]
    n_input: int
    n_dropped_missing: int
    n_dropped_nonfinite: int


@dataclass(frozen=True)
class _FitOutcome:
    """One feature's model objects, status, and captured warnings."""

    prepared: _PreparedFeature
    model: MixedLM | None
    result: MixedLMResultsWrapper | None
    status: str
    reason: str
    warning_messages: tuple[str, ...]
    n_groups: int
    n_independent_units: int
    min_group_size: int
    max_group_size: int
    n_fixed_effects: int
    design_rank: int


def mixed_model_inference(
    frame: pd.DataFrame,
    *,
    spec: MixedModelSpec,
    contrasts: Sequence[MixedModelContrast] = (),
    feature_keys: Sequence[str] = (
        "feature_type",
        "feature_id",
        "feature_label",
    ),
) -> MixedModelInferenceResult:
    """Fit repeated-measures mixed models to long aggregated features.

    One model is fitted per unique `feature_keys` combination. Rows are
    observational units such as donor-tissue-cell-type aggregates, while
    `spec.independent_unit_key` defines biological support and `spec.group_key`
    receives an explicit random intercept. Missing, invalid numeric, and
    non-finite inputs are excluded with counts in `diagnostics`; no alternate
    model is fitted when the requested model is unsupported or fails.

    Planned contrasts use marginal fixed-effect design rows standardized over
    the model population. Pairwise categorical contrasts standardize over and
    require donors observed in both levels. Raw interaction coefficients remain
    available for audit, but simple slopes and effects in `contrasts` are the
    intended inferential output. Confidence intervals and P values are
    asymptotic two-sided Wald z inferences; they describe observational
    associations, not causal or mechanistic effects. Inputs are not modified.

    Args:
      frame: Long aggregated frame with one response row per observational unit
        and feature.
      spec: Model structure, support thresholds, and fitting policy.
      contrasts: Planned marginal contrasts to estimate after every fit.
      feature_keys: Canonical feature metadata columns whose unique
        combinations define separate models. Entries must be selected from
        `feature_type`, `feature_id`, and `feature_label` so result identities
        remain representable in the stable output schema.

    Returns:
      Tidy planned contrasts, fixed coefficients, coefficient-block tests,
      conditional variance components, and fit diagnostics.

    Raises:
      KeyError: If a configured model, contrast, or feature column is absent.
      ValueError: If the model or contrast specification is internally invalid.
    """
    _validate_spec(spec, contrasts)
    feature_columns = tuple(feature_keys)
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_keys contains duplicate columns")
    supported_feature_columns = {
        "feature_type",
        "feature_id",
        "feature_label",
    }
    if unsupported := set(feature_columns).difference(
        supported_feature_columns
    ):
        raise ValueError(
            "feature_keys contains unsupported metadata columns: "
            f"{sorted(unsupported)}"
        )
    _validate_input_columns(frame, spec, contrasts, feature_columns)

    contrast_records: list[dict[str, object]] = []
    fixed_records: list[dict[str, object]] = []
    term_records: list[dict[str, object]] = []
    variance_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []

    for feature, feature_frame in _feature_groups(frame, feature_columns):
        prepared = _prepare_feature(feature_frame, spec)
        outcome = _fit_feature_model(prepared, spec)

        contrast_records.extend(
            _contrast_records(feature, outcome, spec, contrasts)
        )
        fixed_records.extend(_fixed_effect_records(feature, outcome, spec))
        term_records.extend(_term_test_records(feature, outcome, spec))
        variance_records.extend(_variance_records(feature, outcome, spec))
        diagnostic_records.append(_diagnostic_record(feature, outcome, spec))

    empty_result = MixedModelInferenceResult.empty()
    contrast_table = pd.DataFrame.from_records(
        contrast_records,
        columns=empty_result.contrasts.columns,
    )
    fixed_table = pd.DataFrame.from_records(
        fixed_records,
        columns=empty_result.fixed_effects.columns,
    )
    term_table = pd.DataFrame.from_records(
        term_records,
        columns=empty_result.term_tests.columns,
    )

    return MixedModelInferenceResult(
        contrasts=_adjust_fdr(contrast_table, family_columns=("estimand",)),
        fixed_effects=_adjust_fdr(fixed_table, family_columns=("term",)),
        term_tests=_adjust_fdr(term_table, family_columns=("term",)),
        variance_components=pd.DataFrame.from_records(
            variance_records,
            columns=empty_result.variance_components.columns,
        ),
        diagnostics=pd.DataFrame.from_records(
            diagnostic_records,
            columns=empty_result.diagnostics.columns,
        ),
    )


def _validate_spec(
    spec: MixedModelSpec,
    contrasts: Sequence[MixedModelContrast],
) -> None:
    """Validate model and contrast configuration before touching data."""
    if not spec.analysis:
        raise ValueError("analysis must be a non-empty identifier")
    if not (0.0 < spec.alpha < 1.0):
        raise ValueError(f"alpha must be between 0 and 1: {spec.alpha}")
    for name, value in (
        ("minimum_groups", spec.minimum_groups),
        ("minimum_independent_units", spec.minimum_independent_units),
        ("minimum_focal_units", spec.minimum_focal_units),
        ("max_iterations", spec.max_iterations),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive: {value}")
    if not spec.optimizer_methods:
        raise ValueError("optimizer_methods must contain at least one method")

    continuous = set(spec.continuous_keys)
    categorical = set(spec.categorical_keys)
    if overlap := continuous.intersection(categorical):
        raise ValueError(
            "predictors cannot be both continuous and categorical: "
            f"{sorted(overlap)}"
        )
    if len(continuous) != len(spec.continuous_keys):
        raise ValueError("continuous_keys contains duplicate columns")
    if len(categorical) != len(spec.categorical_keys):
        raise ValueError("categorical_keys contains duplicate columns")
    if len(set(spec.variance_component_keys)) != len(
        spec.variance_component_keys
    ):
        raise ValueError("variance_component_keys contains duplicate columns")
    if spec.group_key in set(spec.variance_component_keys):
        raise ValueError("group_key cannot also be a variance component")

    fixed = continuous | categorical
    seen_interactions: set[frozenset[str]] = set()
    for left, right in spec.interactions:
        if left not in fixed or right not in fixed:
            raise ValueError(
                "interaction columns must be declared predictors: "
                f"{left}, {right}"
            )
        if left == right:
            raise ValueError(f"self-interactions are not supported: {left}")
        interaction = frozenset((left, right))
        if interaction in seen_interactions:
            raise ValueError(f"duplicate interaction: {left}, {right}")
        seen_interactions.add(interaction)

    for contrast in contrasts:
        _validate_contrast(contrast, spec)


def _validate_contrast(
    contrast: MixedModelContrast,
    spec: MixedModelSpec,
) -> None:
    """Validate one planned contrast against its model specification."""
    categorical_modes = {
        "categorical_vs_mean",
        "categorical_pairwise",
        "categorical_vs_reference",
    }
    if not contrast.estimand:
        raise ValueError("contrast estimand must be non-empty")
    if not contrast.effect_scale:
        raise ValueError("contrast effect_scale must be non-empty")
    for name, requested in (
        ("levels", contrast.levels),
        ("by_levels", contrast.by_levels),
    ):
        if requested is not None and len(set(requested)) != len(requested):
            raise ValueError(f"contrast {name} contains duplicate levels")
    if contrast.mode in categorical_modes:
        if contrast.predictor_key not in spec.categorical_keys:
            raise ValueError(
                "categorical contrast predictor was not declared categorical: "
                f"{contrast.predictor_key}"
            )
    elif contrast.mode == "continuous_slope":
        if contrast.predictor_key not in spec.continuous_keys:
            raise ValueError(
                "slope predictor was not declared continuous: "
                f"{contrast.predictor_key}"
            )
        if not np.isfinite(contrast.slope_step) or contrast.slope_step <= 0:
            raise ValueError(
                f"slope_step must be finite and positive: {contrast.slope_step}"
            )
    else:
        raise ValueError(f"unsupported contrast mode: {contrast.mode}")

    if contrast.mode == "categorical_vs_reference":
        if contrast.reference is None:
            raise ValueError(
                "categorical_vs_reference requires an explicit reference"
            )
        if (
            contrast.levels is not None
            and contrast.reference in contrast.levels
        ):
            raise ValueError("contrast levels cannot contain its reference")
    elif contrast.reference is not None:
        raise ValueError(
            f"reference is unsupported for contrast mode {contrast.mode}"
        )

    if contrast.by_key is not None:
        if contrast.mode not in {
            "categorical_vs_reference",
            "continuous_slope",
        }:
            raise ValueError(
                f"by_key is unsupported for contrast mode {contrast.mode}"
            )
        if contrast.by_key not in spec.categorical_keys:
            raise ValueError(
                "contrast by_key was not declared categorical: "
                f"{contrast.by_key}"
            )
        if contrast.by_key == contrast.predictor_key:
            raise ValueError("contrast by_key must differ from predictor_key")
    elif contrast.by_levels is not None:
        raise ValueError("by_levels requires by_key")


def _validate_input_columns(
    frame: pd.DataFrame,
    spec: MixedModelSpec,
    contrasts: Sequence[MixedModelContrast],
    feature_keys: Sequence[str],
) -> None:
    """Raise an actionable error for absent configured input columns."""
    independent_key = spec.independent_unit_key or spec.group_key
    required = {
        *feature_keys,
        spec.response_key,
        spec.group_key,
        independent_key,
        *spec.continuous_keys,
        *spec.categorical_keys,
        *spec.variance_component_keys,
    }
    required.update(contrast.predictor_key for contrast in contrasts)
    required.update(
        contrast.by_key for contrast in contrasts if contrast.by_key is not None
    )
    if missing := sorted(required.difference(frame.columns)):
        raise KeyError(
            f"mixed-model frame missing configured columns: {missing}"
        )


def _feature_groups(
    frame: pd.DataFrame,
    feature_keys: Sequence[str],
) -> Iterator[tuple[dict[str, object], pd.DataFrame]]:
    """Yield feature metadata and per-feature frames in input order."""
    if not feature_keys:
        yield {}, frame
        return

    grouped = frame.groupby(
        list(feature_keys),
        observed=True,
        dropna=False,
        sort=False,
    )
    for values, group in grouped:
        tuple_values = values if isinstance(values, tuple) else (values,)
        yield dict(zip(feature_keys, tuple_values, strict=True)), group


def _prepare_feature(
    feature_frame: pd.DataFrame,
    spec: MixedModelSpec,
) -> _PreparedFeature:
    """Prepare one feature without silently changing its requested model."""
    independent_key = spec.independent_unit_key or spec.group_key
    required = list(
        dict.fromkeys(
            (
                spec.response_key,
                spec.group_key,
                independent_key,
                *spec.continuous_keys,
                *spec.categorical_keys,
                *spec.variance_component_keys,
            )
        )
    )
    working = feature_frame.loc[:, required].copy()
    missing_mask = working.isna().any(axis="columns")
    nonfinite_mask = pd.Series(False, index=working.index)

    numeric_keys = {spec.response_key, *spec.continuous_keys}
    for key in required:
        if key not in numeric_keys:
            flags = working[key].map(_is_nonfinite_scalar).to_numpy(dtype=bool)
            nonfinite_mask |= pd.Series(
                flags,
                index=working.index,
                dtype=bool,
            )
            continue
        converted = pd.to_numeric(working[key], errors="coerce")
        invalid = working[key].notna() & converted.isna()
        finite = converted.notna() & np.isfinite(
            converted.to_numpy(dtype=float)
        )
        nonfinite_mask |= invalid | (converted.notna() & ~finite)
        working[key] = converted.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~(missing_mask | nonfinite_mask)
    complete = working.loc[valid_mask].copy()
    n_dropped_missing = int(missing_mask.sum())
    n_dropped_nonfinite = int((nonfinite_mask & ~missing_mask).sum())

    aliases = {key: f"_mm_value_{index}" for index, key in enumerate(required)}
    prepared = pd.DataFrame(index=complete.index)
    for key in required:
        alias = aliases[key]
        if key in numeric_keys:
            prepared[alias] = complete[key].to_numpy(dtype=float)
        else:
            prepared[alias] = _collision_safe_labels(complete[key]).to_numpy()
    prepared = prepared.reset_index(drop=True)

    formula, display_formula = _model_formulas(spec, aliases)
    variance_formulas: dict[str, str] = {}
    variance_names: dict[str, str] = {}
    for index, key in enumerate(spec.variance_component_keys):
        component = f"_mm_component_{index}"
        variance_formulas[component] = f"0 + C({aliases[key]})"
        variance_names[component] = key

    return _PreparedFeature(
        data=prepared,
        aliases=aliases,
        formula=formula,
        display_formula=display_formula,
        variance_formulas=variance_formulas,
        variance_names=variance_names,
        n_input=int(feature_frame.shape[0]),
        n_dropped_missing=n_dropped_missing,
        n_dropped_nonfinite=n_dropped_nonfinite,
    )


def _is_nonfinite_scalar(value: object) -> bool:
    """Return whether a non-numeric-role value is an actual infinite scalar."""
    return isinstance(value, Real) and not bool(np.isfinite(float(value)))


def _collision_safe_labels(values: pd.Series) -> pd.Series:
    """Stringify categorical values without collapsing unlike identities."""
    identities = values.map(_typed_identity)
    displays = values.map(str)
    identity_counts = (
        pd.DataFrame({"display": displays, "identity": identities})
        .groupby("display", observed=True, dropna=False)["identity"]
        .nunique()
    )
    collisions = set(identity_counts.loc[identity_counts.gt(1)].index)
    labels = [
        (
            f"{type(value).__name__}:{value}"
            if display in collisions
            else display
        )
        for value, display in zip(values, displays, strict=True)
    ]
    return pd.Series(labels, index=values.index, dtype=str)


def _typed_identity(value: object) -> str:
    """Return a stable type-qualified identity for metadata comparison."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _model_formulas(
    spec: MixedModelSpec,
    aliases: dict[str, str],
) -> tuple[str, str]:
    """Return safe executable and human-readable fixed-effect formulas."""
    safe_terms = [aliases[key] for key in spec.continuous_keys]
    safe_terms.extend(f"C({aliases[key]})" for key in spec.categorical_keys)
    display_terms = list(spec.continuous_keys)
    display_terms.extend(f"C({key})" for key in spec.categorical_keys)

    for left, right in spec.interactions:
        left_safe = _fixed_term(left, spec, aliases)
        right_safe = _fixed_term(right, spec, aliases)
        safe_terms.append(f"{left_safe}:{right_safe}")
        left_display = _display_fixed_term(left, spec)
        right_display = _display_fixed_term(right, spec)
        display_terms.append(f"{left_display}:{right_display}")

    safe_rhs = " + ".join(safe_terms) if safe_terms else "1"
    display_rhs = " + ".join(display_terms) if display_terms else "1"
    return (
        f"{aliases[spec.response_key]} ~ {safe_rhs}",
        f"{spec.response_key} ~ {display_rhs}",
    )


def _fixed_term(
    key: str,
    spec: MixedModelSpec,
    aliases: dict[str, str],
) -> str:
    """Return one safe Patsy fixed-effect term."""
    return (
        f"C({aliases[key]})" if key in spec.categorical_keys else aliases[key]
    )


def _display_fixed_term(key: str, spec: MixedModelSpec) -> str:
    """Return one human-readable fixed-effect term."""
    return f"C({key})" if key in spec.categorical_keys else key


def _fit_feature_model(
    prepared: _PreparedFeature,
    spec: MixedModelSpec,
) -> _FitOutcome:
    """Fit one prepared feature or preserve why it is not estimable."""
    data = prepared.data
    group_alias = prepared.aliases[spec.group_key]
    independent_key = spec.independent_unit_key or spec.group_key
    independent_alias = prepared.aliases[independent_key]
    group_sizes = (
        data[group_alias].value_counts() if not data.empty else pd.Series()
    )
    independent_sizes = (
        data[independent_alias].value_counts()
        if not data.empty
        else pd.Series()
    )
    n_groups = int(group_sizes.shape[0])
    n_independent = (
        int(data[independent_alias].nunique()) if not data.empty else 0
    )
    min_group_size = int(group_sizes.min()) if not group_sizes.empty else 0
    max_group_size = int(group_sizes.max()) if not group_sizes.empty else 0

    status = "ok"
    reason = ""
    if data.empty:
        status, reason = "no_complete_observations", "no_complete_observations"
    elif n_groups < spec.minimum_groups:
        status, reason = "insufficient_groups", "insufficient_groups"
    elif n_independent < spec.minimum_independent_units:
        status, reason = (
            "insufficient_independent_units",
            "insufficient_independent_units",
        )
    elif (
        independent_key != spec.group_key
        and independent_key not in spec.variance_component_keys
        and not independent_sizes.empty
        and int(independent_sizes.max()) > 1
    ):
        status, reason = (
            "unmodeled_repeated_independent_unit",
            f"repeated_independent_unit_requires_random_effect:{independent_key}",
        )
    elif unidentifiable_component := _unidentifiable_variance_component(
        data,
        prepared=prepared,
        spec=spec,
    ):
        status, reason = (
            "unidentifiable_variance_component",
            unidentifiable_component,
        )
    elif data[prepared.aliases[spec.response_key]].nunique() < 2:
        status, reason = "constant_response", "constant_response"
    elif max_group_size < 2:
        status, reason = (
            "no_within_group_replication",
            "no_within_group_replication",
        )

    if status != "ok":
        return _FitOutcome(
            prepared=prepared,
            model=None,
            result=None,
            status=status,
            reason=reason,
            warning_messages=(),
            n_groups=n_groups,
            n_independent_units=n_independent,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            n_fixed_effects=0,
            design_rank=0,
        )

    try:
        model = smf.mixedlm(
            prepared.formula,
            data,
            groups=group_alias,
            re_formula="1",
            vc_formula=(prepared.variance_formulas or None),
            missing="raise",
            eval_env=-1,
        )
        fixed_design = model.exog
        if fixed_design is None:
            raise ValueError("mixed model has no fixed-effect design matrix")
        n_fixed = int(fixed_design.shape[1])
        design_rank = int(np.linalg.matrix_rank(fixed_design))
    except (
        np.linalg.LinAlgError,
        TypeError,
        ValueError,
        patsy.PatsyError,
    ) as error:
        return _FitOutcome(
            prepared=prepared,
            model=None,
            result=None,
            status="model_construction_failed",
            reason=f"model_construction_failed:{type(error).__name__}",
            warning_messages=(),
            n_groups=n_groups,
            n_independent_units=n_independent,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            n_fixed_effects=0,
            design_rank=0,
        )
    if design_rank < n_fixed:
        return _FitOutcome(
            prepared=prepared,
            model=model,
            result=None,
            status="rank_deficient",
            reason="rank_deficient_fixed_effects",
            warning_messages=(),
            n_groups=n_groups,
            n_independent_units=n_independent,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            n_fixed_effects=n_fixed,
            design_rank=design_rank,
        )
    if model.nobs <= n_fixed:
        return _FitOutcome(
            prepared=prepared,
            model=model,
            result=None,
            status="insufficient_residual_degrees",
            reason="insufficient_residual_degrees",
            warning_messages=(),
            n_groups=n_groups,
            n_independent_units=n_independent,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            n_fixed_effects=n_fixed,
            design_rank=design_rank,
        )

    captured: list[warnings.WarningMessage]
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = model.fit(
                reml=spec.reml,
                method=list(spec.optimizer_methods),
                full_output=True,
                maxiter=spec.max_iterations,
                disp=False,
            )
    except (
        np.linalg.LinAlgError,
        TypeError,
        ValueError,
        patsy.PatsyError,
    ) as error:
        return _FitOutcome(
            prepared=prepared,
            model=model,
            result=None,
            status="fit_failed",
            reason=f"fit_failed:{type(error).__name__}",
            warning_messages=(),
            n_groups=n_groups,
            n_independent_units=n_independent,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            n_fixed_effects=n_fixed,
            design_rank=design_rank,
        )

    warning_messages = tuple(
        f"{warning.category.__name__}: {warning.message}"
        for warning in captured
    )
    fit_status = "ok" if bool(result.converged) else "non_converged"
    fit_reason = "" if bool(result.converged) else "optimizer_did_not_converge"
    if fit_status == "ok":
        hessian_invalid = any(
            "hessian" in message.casefold()
            and "not positive definite" in message.casefold()
            for message in warning_messages
        )
        try:
            covariance_valid = (
                not hessian_invalid and _fixed_covariance_is_valid(result)
            )
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            covariance_valid = False
        if not covariance_valid:
            fit_status = "invalid_covariance"
            fit_reason = (
                "hessian_not_positive_definite"
                if hessian_invalid
                else "fixed_effect_covariance_invalid"
            )

    return _FitOutcome(
        prepared=prepared,
        model=model,
        result=result,
        status=fit_status,
        reason=fit_reason,
        warning_messages=warning_messages,
        n_groups=n_groups,
        n_independent_units=n_independent,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        n_fixed_effects=n_fixed,
        design_rank=design_rank,
    )


def _unidentifiable_variance_component(
    data: pd.DataFrame,
    *,
    prepared: _PreparedFeature,
    spec: MixedModelSpec,
) -> str:
    """Return why a nested variance component is confounded, if any."""
    if data.empty:
        return ""
    group_alias = prepared.aliases[spec.group_key]
    for key in spec.variance_component_keys:
        component_alias = prepared.aliases[key]
        component_sizes = data.groupby(
            [group_alias, component_alias],
            observed=True,
            dropna=False,
        ).size()
        if component_sizes.empty or int(component_sizes.max()) < 2:
            return f"variance_component_all_singletons:{key}"
        levels_per_group = data.groupby(
            group_alias,
            observed=True,
            dropna=False,
        )[component_alias].nunique(dropna=False)
        if levels_per_group.empty or int(levels_per_group.max()) < 2:
            return f"variance_component_confounded_with_group:{key}"
    for left, right in combinations(spec.variance_component_keys, 2):
        left_alias = prepared.aliases[left]
        right_alias = prepared.aliases[right]
        unique_pairs = data[
            [group_alias, left_alias, right_alias]
        ].drop_duplicates()
        left_mapping = unique_pairs.groupby(
            [group_alias, left_alias],
            observed=True,
            dropna=False,
        )[right_alias].nunique(dropna=False)
        right_mapping = unique_pairs.groupby(
            [group_alias, right_alias],
            observed=True,
            dropna=False,
        )[left_alias].nunique(dropna=False)
        if (
            not left_mapping.empty
            and not right_mapping.empty
            and int(left_mapping.max()) == 1
            and int(right_mapping.max()) == 1
        ):
            return f"variance_components_confounded:{left}|{right}"
    return ""


def _fixed_covariance_is_valid(result: MixedLMResultsWrapper) -> bool:
    """Return whether fixed estimates have valid covariance uncertainty."""
    names = list(result.model.exog_names)
    covariance = result.cov_params().loc[names, names].to_numpy(dtype=float)
    estimates = result.fe_params.reindex(names).to_numpy(dtype=float)
    if not np.isfinite(covariance).all() or not np.isfinite(estimates).all():
        return False
    if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-12):
        return False
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0):
        return False
    symmetric = (covariance + covariance.T) / 2.0
    tolerance = max(float(np.max(diagonal)), 1.0) * 1e-10
    return bool(np.linalg.eigvalsh(symmetric).min() >= -tolerance)


def _contrast_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
    contrasts: Sequence[MixedModelContrast],
) -> list[dict[str, object]]:
    """Build every requested contrast row for one fitted feature."""
    records: list[dict[str, object]] = []
    for contrast in contrasts:
        try:
            if contrast.mode == "categorical_vs_mean":
                records.extend(
                    _categorical_vs_mean_records(
                        feature,
                        outcome,
                        spec,
                        contrast,
                    )
                )
            elif contrast.mode == "categorical_pairwise":
                records.extend(
                    _categorical_pairwise_records(
                        feature,
                        outcome,
                        spec,
                        contrast,
                    )
                )
            elif contrast.mode == "categorical_vs_reference":
                records.extend(
                    _categorical_reference_records(
                        feature,
                        outcome,
                        spec,
                        contrast,
                    )
                )
            else:
                records.extend(
                    _continuous_slope_records(
                        feature,
                        outcome,
                        spec,
                        contrast,
                    )
                )
        except (
            np.linalg.LinAlgError,
            TypeError,
            ValueError,
            patsy.PatsyError,
        ) as error:
            records.append(
                _nonestimable_contrast(
                    feature,
                    outcome,
                    spec,
                    contrast,
                    level="",
                    reference=contrast.reference or "",
                    reason=f"design_construction_failed:{type(error).__name__}",
                )
            )
    return records


def _categorical_vs_mean_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
    contrast: MixedModelContrast,
) -> list[dict[str, object]]:
    """Compare categorical levels with their equally weighted marginal mean."""
    data = outcome.prepared.data
    predictor = outcome.prepared.aliases[contrast.predictor_key]
    levels = _requested_levels(data, predictor, contrast.levels)
    if len(levels) < 2:
        return [
            _nonestimable_contrast(
                feature,
                outcome,
                spec,
                contrast,
                level=levels[0] if levels else "",
                reference="marginal_mean",
                reason="fewer_than_two_levels",
            )
        ]

    independent = outcome.prepared.aliases[
        spec.independent_unit_key or spec.group_key
    ]
    observed = set(data[predictor])
    support_by_level = {
        level: set(data.loc[data[predictor] == level, independent])
        for level in levels
        if level in observed
    }
    supported_levels = [
        level
        for level in levels
        if len(support_by_level.get(level, set())) >= spec.minimum_focal_units
    ]
    design_by_level = (
        {
            level: _marginal_design(
                outcome,
                data,
                set_values={predictor: level},
            )
            for level in supported_levels
        }
        if outcome.status == "ok"
        else {}
    )
    mean_design = (
        np.mean(list(design_by_level.values()), axis=0)
        if len(design_by_level) >= 2
        else None
    )
    reference_units = set().union(
        *(support_by_level[level] for level in supported_levels)
    )
    records: list[dict[str, object]] = []
    for level in levels:
        n_level = len(support_by_level.get(level, set()))
        reason = ""
        if level not in observed:
            reason = "level_absent"
        elif n_level < spec.minimum_focal_units:
            reason = "insufficient_level_support"
        elif len(supported_levels) < 2:
            reason = "fewer_than_two_supported_levels"
        vector = None
        if not reason and mean_design is not None:
            vector = design_by_level[level] - mean_design
        records.append(
            _contrast_record(
                feature,
                outcome,
                spec,
                contrast,
                vector=vector,
                level=level,
                reference="marginal_mean",
                by_level="",
                n_independent=n_level,
                n_level=n_level,
                n_reference=len(reference_units),
                n_paired=np.nan,
                support_reason=reason,
            )
        )
    return records


def _categorical_pairwise_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
    contrast: MixedModelContrast,
) -> list[dict[str, object]]:
    """Estimate within-independent-unit pairwise categorical contrasts."""
    data = outcome.prepared.data
    predictor = outcome.prepared.aliases[contrast.predictor_key]
    levels = _requested_levels(data, predictor, contrast.levels)
    if len(levels) < 2:
        return [
            _nonestimable_contrast(
                feature,
                outcome,
                spec,
                contrast,
                level=levels[0] if levels else "",
                reference="",
                reason="fewer_than_two_levels",
            )
        ]

    independent = outcome.prepared.aliases[
        spec.independent_unit_key or spec.group_key
    ]
    records: list[dict[str, object]] = []
    for level, reference in combinations(levels, 2):
        level_units = set(data.loc[data[predictor] == level, independent])
        reference_units = set(
            data.loc[data[predictor] == reference, independent]
        )
        paired_units = level_units.intersection(reference_units)
        n_level = len(level_units)
        n_reference = len(reference_units)
        n_paired = len(paired_units)
        reason = ""
        if level not in set(data[predictor]):
            reason = "level_absent"
        elif reference not in set(data[predictor]):
            reason = "reference_level_absent"
        elif n_level < spec.minimum_focal_units:
            reason = "insufficient_level_support"
        elif n_reference < spec.minimum_focal_units:
            reason = "insufficient_reference_support"
        elif n_paired < spec.minimum_focal_units:
            reason = "insufficient_paired_support"

        vector: np.ndarray | None = None
        if not reason and outcome.status == "ok":
            paired_data = data[data[independent].isin(paired_units)]
            level_design = _marginal_design(
                outcome,
                paired_data,
                set_values={predictor: level},
            )
            reference_design = _marginal_design(
                outcome,
                paired_data,
                set_values={predictor: reference},
            )
            vector = level_design - reference_design
        records.append(
            _contrast_record(
                feature,
                outcome,
                spec,
                contrast,
                vector=vector,
                level=level,
                reference=reference,
                by_level="",
                n_independent=n_paired,
                n_level=n_level,
                n_reference=n_reference,
                n_paired=n_paired,
                support_reason=reason,
            )
        )
    return records


def _categorical_reference_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
    contrast: MixedModelContrast,
) -> list[dict[str, object]]:
    """Estimate categorical effects versus a reference, optionally by level."""
    assert contrast.reference is not None
    data = outcome.prepared.data
    predictor = outcome.prepared.aliases[contrast.predictor_key]
    observed = _observed_levels(data, predictor)
    levels = (
        list(contrast.levels)
        if contrast.levels is not None
        else [level for level in observed if level != contrast.reference]
    )
    by_alias = (
        outcome.prepared.aliases[contrast.by_key]
        if contrast.by_key is not None
        else None
    )
    by_levels = (
        _requested_levels(data, by_alias, contrast.by_levels)
        if by_alias is not None
        else [""]
    )
    if not levels:
        return [
            _nonestimable_contrast(
                feature,
                outcome,
                spec,
                contrast,
                level="",
                reference=contrast.reference,
                reason="fewer_than_two_levels",
            )
        ]

    independent = outcome.prepared.aliases[
        spec.independent_unit_key or spec.group_key
    ]
    records: list[dict[str, object]] = []
    for by_level in by_levels:
        base = data if by_alias is None else data[data[by_alias] == by_level]
        for level in levels:
            level_mask = data[predictor].eq(level)
            reference_mask = data[predictor].eq(contrast.reference)
            if by_alias is not None:
                level_mask &= data[by_alias].eq(by_level)
                reference_mask &= data[by_alias].eq(by_level)
            level_units = set(data.loc[level_mask, independent])
            reference_units = set(data.loc[reference_mask, independent])
            n_level = len(level_units)
            n_reference = len(reference_units)
            reason = ""
            if by_alias is not None and by_level not in set(data[by_alias]):
                reason = "by_level_absent"
            elif level not in set(data[predictor]):
                reason = "level_absent"
            elif contrast.reference not in set(data[predictor]):
                reason = "reference_level_absent"
            elif n_level < spec.minimum_focal_units:
                reason = "insufficient_level_support"
            elif n_reference < spec.minimum_focal_units:
                reason = "insufficient_reference_support"

            vector: np.ndarray | None = None
            if not reason and outcome.status == "ok":
                level_values = {predictor: level}
                reference_values = {predictor: contrast.reference}
                if by_alias is not None:
                    level_values[by_alias] = by_level
                    reference_values[by_alias] = by_level
                vector = _marginal_design(
                    outcome,
                    base,
                    set_values=level_values,
                ) - _marginal_design(
                    outcome,
                    base,
                    set_values=reference_values,
                )
            records.append(
                _contrast_record(
                    feature,
                    outcome,
                    spec,
                    contrast,
                    vector=vector,
                    level=level,
                    reference=contrast.reference,
                    by_level=by_level,
                    n_independent=len(level_units.union(reference_units)),
                    n_level=n_level,
                    n_reference=n_reference,
                    n_paired=np.nan,
                    support_reason=reason,
                )
            )
    return records


def _continuous_slope_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
    contrast: MixedModelContrast,
) -> list[dict[str, object]]:
    """Estimate a continuous simple slope, optionally by categorical level."""
    data = outcome.prepared.data
    predictor = outcome.prepared.aliases[contrast.predictor_key]
    by_alias = (
        outcome.prepared.aliases[contrast.by_key]
        if contrast.by_key is not None
        else None
    )
    by_levels = (
        _requested_levels(data, by_alias, contrast.by_levels)
        if by_alias is not None
        else [""]
    )
    independent = outcome.prepared.aliases[
        spec.independent_unit_key or spec.group_key
    ]
    records: list[dict[str, object]] = []
    half_step = contrast.slope_step / 2.0
    for by_level in by_levels:
        base = data if by_alias is None else data[data[by_alias] == by_level]
        n_level = int(base[independent].nunique())
        reason = ""
        if by_alias is not None and by_level not in set(data[by_alias]):
            reason = "by_level_absent"
        elif n_level < spec.minimum_focal_units:
            reason = "insufficient_level_support"
        elif base[predictor].nunique() < 2:
            reason = "constant_continuous_predictor"

        vector: np.ndarray | None = None
        if not reason and outcome.status == "ok":
            set_values = {by_alias: by_level} if by_alias is not None else {}
            upper = _marginal_design(
                outcome,
                base,
                set_values=set_values,
                shifts={predictor: half_step},
            )
            lower = _marginal_design(
                outcome,
                base,
                set_values=set_values,
                shifts={predictor: -half_step},
            )
            vector = (upper - lower) / contrast.slope_step
        records.append(
            _contrast_record(
                feature,
                outcome,
                spec,
                contrast,
                vector=vector,
                level=by_level or contrast.predictor_key,
                reference="",
                by_level=by_level,
                n_independent=n_level,
                n_level=n_level,
                n_reference=np.nan,
                n_paired=np.nan,
                support_reason=reason,
            )
        )
    return records


def _requested_levels(
    data: pd.DataFrame,
    alias: str | None,
    requested: tuple[str, ...] | None,
) -> list[str]:
    """Return requested levels or deterministic observed levels."""
    if requested is not None:
        return list(dict.fromkeys(requested))
    if alias is None:
        return []
    return _observed_levels(data, alias)


def _observed_levels(data: pd.DataFrame, alias: str) -> list[str]:
    """Return sorted observed string levels for an aliased column."""
    return sorted(data[alias].dropna().astype(str).unique().tolist())


def _marginal_design(
    outcome: _FitOutcome,
    base: pd.DataFrame,
    *,
    set_values: Mapping[str, object],
    shifts: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return a population-mean fixed-effect design row for one scenario."""
    if outcome.model is None or base.empty:
        raise ValueError("marginal design requires a fitted model population")
    scenario = base.copy()
    for key, value in set_values.items():
        scenario[key] = pd.Series(value, index=scenario.index)
    for key, shift in (shifts or {}).items():
        scenario[key] = scenario[key].to_numpy(dtype=float) + shift

    matrix = patsy.build_design_matrices(  # type: ignore[attr-defined]
        [outcome.model.data.design_info],
        scenario,
        NA_action="raise",
        return_type="dataframe",
    )[0]
    return np.asarray(matrix, dtype=float).mean(axis=0)


def _contrast_record(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
    contrast: MixedModelContrast,
    *,
    vector: np.ndarray | None,
    level: str,
    reference: str,
    by_level: str,
    n_independent: int | float,
    n_level: int | float,
    n_reference: int | float,
    n_paired: int | float,
    support_reason: str,
) -> dict[str, object]:
    """Estimate one supported contrast or preserve why it is unavailable."""
    status = outcome.status
    reason = outcome.reason
    if status == "ok" and support_reason:
        status, reason = "non_estimable", support_reason
    if status == "ok" and vector is None:
        status, reason = "non_estimable", "contrast_vector_unavailable"

    estimate = np.nan
    standard_error = np.nan
    ci_low = np.nan
    ci_high = np.nan
    statistic = np.nan
    pvalue = np.nan
    estimable = False
    if status == "ok" and vector is not None:
        result = outcome.result
        assert result is not None
        names = list(result.model.exog_names)
        beta = result.fe_params.reindex(names).to_numpy(dtype=float)
        covariance = result.cov_params().loc[names, names].to_numpy(dtype=float)
        variance = float(vector @ covariance @ vector)
        if not np.isfinite(variance) or variance <= 0:
            status, reason = "non_estimable", "non_positive_contrast_variance"
        else:
            estimate = float(vector @ beta)
            standard_error = float(np.sqrt(variance))
            statistic = estimate / standard_error
            pvalue = float(2.0 * stats.norm.sf(abs(statistic)))
            quantile = float(stats.norm.ppf(1.0 - spec.alpha / 2.0))
            ci_low = estimate - quantile * standard_error
            ci_high = estimate + quantile * standard_error
            estimable = True

    return {
        "analysis": spec.analysis,
        **_standard_feature(feature),
        "estimand": contrast.estimand,
        "contrast": _contrast_label(
            contrast,
            level=level,
            reference=reference,
            by_level=by_level,
        ),
        "predictor": contrast.predictor_key,
        "level": str(level),
        "reference": str(reference),
        "by_key": contrast.by_key or "",
        "by_level": str(by_level),
        "effect_scale": contrast.effect_scale,
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "statistic": statistic,
        "pvalue": pvalue,
        "pvalue_fdr": np.nan,
        "n_independent_units": float(n_independent),
        "n_level_units": float(n_level),
        "n_reference_units": float(n_reference),
        "n_paired_units": float(n_paired),
        "estimable": estimable,
        "status": status,
        "reason": reason,
    }


def _nonestimable_contrast(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
    contrast: MixedModelContrast,
    *,
    level: str,
    reference: str,
    reason: str,
) -> dict[str, object]:
    """Return a visible non-estimable row for an unsupported contrast."""
    return _contrast_record(
        feature,
        outcome,
        spec,
        contrast,
        vector=None,
        level=level,
        reference=reference,
        by_level="",
        n_independent=0,
        n_level=0,
        n_reference=0,
        n_paired=np.nan,
        support_reason=reason,
    )


def _contrast_label(
    contrast: MixedModelContrast,
    *,
    level: str,
    reference: str,
    by_level: str,
) -> str:
    """Return a stable human-readable label for one planned contrast."""
    if contrast.mode == "continuous_slope":
        suffix = f" within {by_level}" if by_level else ""
        return f"{contrast.predictor_key} slope{suffix}"
    if reference == "marginal_mean":
        return f"{level} vs marginal mean"
    if reference:
        return f"{level} vs {reference}"
    return str(level)


def _fixed_effect_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
) -> list[dict[str, object]]:
    """Extract raw fixed effects for audit without covariance parameters."""
    result = outcome.result
    if result is None:
        return []
    names = list(result.model.exog_names)
    beta = result.fe_params.reindex(names).to_numpy(dtype=float)
    covariance = (
        result.cov_params().loc[names, names].to_numpy(dtype=float)
        if outcome.status == "ok"
        else np.full((len(names), len(names)), np.nan)
    )
    quantile = float(stats.norm.ppf(1.0 - spec.alpha / 2.0))
    records: list[dict[str, object]] = []
    for index, name in enumerate(names):
        variance = float(covariance[index, index])
        standard_error = np.sqrt(variance) if variance > 0 else np.nan
        statistic = beta[index] / standard_error
        pvalue = float(2.0 * stats.norm.sf(abs(statistic)))
        records.append(
            {
                "analysis": spec.analysis,
                **_standard_feature(feature),
                "term": _restore_aliases(name, outcome.prepared.aliases),
                "estimate": float(beta[index]),
                "standard_error": float(standard_error),
                "ci_low": float(beta[index] - quantile * standard_error),
                "ci_high": float(beta[index] + quantile * standard_error),
                "statistic": float(statistic),
                "pvalue": pvalue,
                "pvalue_fdr": np.nan,
                "n_observations": float(outcome.prepared.data.shape[0]),
                "n_groups": float(outcome.n_groups),
                "n_independent_units": float(outcome.n_independent_units),
                "status": outcome.status,
                "reason": outcome.reason,
            }
        )
    return records


def _term_test_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
) -> list[dict[str, object]]:
    """Extract coefficient-block Wald tests for each fixed-effect term."""
    result = outcome.result
    if result is None:
        return []
    names = list(result.model.exog_names)
    beta = result.fe_params.reindex(names).to_numpy(dtype=float)
    covariance = (
        result.cov_params().loc[names, names].to_numpy(dtype=float)
        if outcome.status == "ok"
        else np.full((len(names), len(names)), np.nan)
    )
    records: list[dict[str, object]] = []
    for (
        term,
        term_slice,
    ) in result.model.data.design_info.term_name_slices.items():
        width = int(term_slice.stop - term_slice.start)
        status = outcome.status
        reason = outcome.reason
        statistic = np.nan
        pvalue = np.nan
        if status == "ok" and width == 0:
            status = "non_estimable"
            reason = "zero_degree_term"
        elif status == "ok":
            term_beta = beta[term_slice]
            term_covariance = covariance[term_slice, term_slice]
            try:
                statistic = float(
                    term_beta @ np.linalg.solve(term_covariance, term_beta)
                )
            except np.linalg.LinAlgError:
                status = "non_estimable"
                reason = "singular_term_covariance"
            if status == "ok":
                if not np.isfinite(statistic) or statistic < -1e-10:
                    status = "non_estimable"
                    reason = "invalid_term_statistic"
                    statistic = np.nan
                else:
                    statistic = max(statistic, 0.0)
                    pvalue = float(stats.chi2.sf(statistic, width))
        records.append(
            {
                "analysis": spec.analysis,
                **_standard_feature(feature),
                "term": _restore_aliases(
                    str(term),
                    outcome.prepared.aliases,
                ),
                "statistic": statistic,
                "degrees_of_freedom": float(width),
                "pvalue": pvalue,
                "pvalue_fdr": np.nan,
                "n_observations": float(outcome.prepared.data.shape[0]),
                "n_groups": float(outcome.n_groups),
                "n_independent_units": float(outcome.n_independent_units),
                "status": status,
                "reason": reason,
            }
        )
    return records


def _variance_records(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
) -> list[dict[str, object]]:
    """Extract conditional random and residual variance components."""
    component_specs = [
        (spec.group_key, "group_random_intercept"),
        *((key, "variance_component") for key in spec.variance_component_keys),
        ("residual", "residual"),
    ]
    result = outcome.result
    values: dict[str, float] = {}
    if result is not None and outcome.status == "ok":
        values[spec.group_key] = float(result.cov_re.iloc[0, 0])
        values.update(
            {
                outcome.prepared.variance_names[name]: float(value)
                for name, value in zip(
                    result.model.exog_vc.names,
                    np.asarray(result.vcomp, dtype=float),
                    strict=True,
                )
            }
        )
        values["residual"] = float(result.scale)
    total: float = float(sum(values.values())) if values else float("nan")

    records: list[dict[str, object]] = []
    for component, component_type in component_specs:
        variance: float = (
            float(values[component]) if component in values else float("nan")
        )
        fraction: float = (
            variance / total
            if np.isfinite(variance) and np.isfinite(total) and total > 0
            else np.nan
        )
        records.append(
            {
                "analysis": spec.analysis,
                **_standard_feature(feature),
                "component": component,
                "component_type": component_type,
                "variance": float(variance),
                "variance_fraction": float(fraction),
                "n_observations": float(outcome.prepared.data.shape[0]),
                "n_groups": float(outcome.n_groups),
                "n_independent_units": float(outcome.n_independent_units),
                "estimable": bool(
                    outcome.status == "ok"
                    and np.isfinite(variance)
                    and np.isfinite(fraction)
                ),
                "status": outcome.status,
                "reason": outcome.reason,
            }
        )
    return records


def _diagnostic_record(
    feature: dict[str, object],
    outcome: _FitOutcome,
    spec: MixedModelSpec,
) -> dict[str, object]:
    """Return one complete fit provenance and diagnostic record."""
    result = outcome.result
    converged = bool(result.converged) if result is not None else False
    log_likelihood = float(result.llf) if result is not None else np.nan
    aic = float(result.aic) if result is not None else np.nan
    bic = float(result.bic) if result is not None else np.nan
    aliases = ";".join(
        f"{key}={value}" for key, value in outcome.prepared.aliases.items()
    )
    independent_key = spec.independent_unit_key or spec.group_key
    independent_alias = outcome.prepared.aliases[independent_key]
    independent_sizes = outcome.prepared.data[independent_alias].value_counts()
    max_independent_size = (
        int(independent_sizes.max()) if not independent_sizes.empty else 0
    )
    if independent_key == spec.group_key:
        independent_covariance = "group_random_intercept"
    elif independent_key in spec.variance_component_keys:
        independent_covariance = "variance_component"
    elif max_independent_size <= 1:
        independent_covariance = "singleton_residual"
    else:
        independent_covariance = "unmodeled"
    return {
        "analysis": spec.analysis,
        **_standard_feature(feature),
        "formula": outcome.prepared.display_formula,
        "model_formula": outcome.prepared.formula,
        "alias_mapping": aliases,
        "group_key": spec.group_key,
        "independent_unit_key": spec.independent_unit_key or spec.group_key,
        "variance_component_keys": "|".join(spec.variance_component_keys),
        "n_input_observations": float(outcome.prepared.n_input),
        "n_observations": float(outcome.prepared.data.shape[0]),
        "n_dropped_missing": float(outcome.prepared.n_dropped_missing),
        "n_dropped_nonfinite": float(outcome.prepared.n_dropped_nonfinite),
        "n_groups": float(outcome.n_groups),
        "n_independent_units": float(outcome.n_independent_units),
        "max_observations_per_independent_unit": float(max_independent_size),
        "independent_unit_covariance": independent_covariance,
        "min_observations_per_group": float(outcome.min_group_size),
        "max_observations_per_group": float(outcome.max_group_size),
        "n_fixed_effects": float(outcome.n_fixed_effects),
        "design_rank": float(outcome.design_rank),
        "minimum_groups": float(spec.minimum_groups),
        "minimum_independent_units": float(spec.minimum_independent_units),
        "minimum_focal_units": float(spec.minimum_focal_units),
        "reml": spec.reml,
        "alpha": spec.alpha,
        "optimizer_methods": "|".join(spec.optimizer_methods),
        "max_iterations": float(spec.max_iterations),
        "converged": converged,
        "warning_count": float(len(outcome.warning_messages)),
        "warning_messages": " | ".join(outcome.warning_messages),
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "statsmodels_version": version("statsmodels"),
        "status": outcome.status,
        "reason": outcome.reason,
    }


def _standard_feature(feature: dict[str, object]) -> dict[str, object]:
    """Return stable feature metadata, leaving unavailable labels empty."""
    return {
        "feature_type": feature.get("feature_type", ""),
        "feature_id": feature.get("feature_id", ""),
        "feature_label": feature.get("feature_label", ""),
    }


def _restore_aliases(text: str, aliases: dict[str, str]) -> str:
    """Replace safe formula aliases with caller-facing source column names."""
    restored = text
    replacements = sorted(
        ((alias, key) for key, alias in aliases.items()),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for alias, key in replacements:
        restored = restored.replace(alias, key)
    return restored


def _adjust_fdr(
    table: pd.DataFrame,
    *,
    family_columns: Sequence[str],
) -> pd.DataFrame:
    """Adjust estimable finite p-values within declared analysis families."""
    if table.empty:
        return table
    adjusted_table = table.copy()
    adjusted_table["pvalue_fdr"] = np.nan
    group_columns = ["analysis", *family_columns]
    for _, index in adjusted_table.groupby(
        group_columns,
        observed=True,
        dropna=False,
    ).groups.items():
        positions = np.asarray(list(index), dtype=int)
        finite = np.isfinite(
            adjusted_table.loc[positions, "pvalue"].to_numpy(dtype=float)
        )
        if "estimable" in adjusted_table.columns:
            finite &= adjusted_table.loc[positions, "estimable"].to_numpy(
                dtype=bool
            )
        if "status" in adjusted_table.columns:
            finite &= (
                adjusted_table.loc[positions, "status"].eq("ok").to_numpy()
            )
        tested_positions = positions[finite]
        if tested_positions.size == 0:
            continue
        adjusted_table.loc[tested_positions, "pvalue_fdr"] = benjamini_hochberg(
            adjusted_table.loc[tested_positions, "pvalue"].tolist()
        )
    return adjusted_table
