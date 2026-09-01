"""Mixed-model estimands for donor-repeated atlas contexts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from nasp_atlas.single_cell.associations import Aggregation
from nasp_atlas.single_cell.associations import MixedModelContrast
from nasp_atlas.single_cell.associations import MixedModelInferenceResult
from nasp_atlas.single_cell.associations import MixedModelSpec
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import aggregate_feature_frame_by_keys
from nasp_atlas.single_cell.associations import benjamini_hochberg
from nasp_atlas.single_cell.associations import mixed_model_inference


__all__ = [
    "TabulaMixedModelResults",
    "tabula_sapiens_mixed_model_inference",
]

_INDEPENDENT_UNIT_KEY: str = "__mixed_independent_unit"
_CONTEXT_UNIT_KEY: str = "__mixed_context_unit"
_OBSERVATIONAL_UNIT_KEY: str = "__mixed_observational_unit"
_AGE_DECADES_KEY: str = "__mixed_age_decades_centered"
_FEATURE_KEYS: tuple[str, ...] = (
    "feature_type",
    "feature_id",
    "feature_label",
)


@dataclass(frozen=True, kw_only=True)
class TabulaMixedModelResults:
    """Tables produced by the prespecified Tabula Sapiens mixed models.

    Attributes:
      contrasts: Primary marginal contrasts and simple slopes.
      fixed_effects: Raw fixed-effect coefficients for model auditing.
      term_tests: Joint tests for fixed-effect model terms.
      variance_components: Model-based random and residual variances.
      diagnostics: Per-feature fit, support, and convergence diagnostics.
      availability: Dataset-level status of every requested analysis.
    """

    contrasts: pd.DataFrame
    fixed_effects: pd.DataFrame
    term_tests: pd.DataFrame
    variance_components: pd.DataFrame
    diagnostics: pd.DataFrame
    availability: pd.DataFrame

    @classmethod
    def empty(cls) -> TabulaMixedModelResults:
        """Return stable empty Tabula Sapiens mixed-model output schemas."""
        core = MixedModelInferenceResult.empty()
        provenance_columns = (
            "age_center_years",
            "aggregation",
            "detection_threshold",
            "configured_condition_reference",
            "minimum_cells_per_observation",
            "minimum_donors",
            "minimum_studies",
            "minimum_repeated_contexts",
            "observational_unit",
            "independent_unit",
        )

        def with_provenance(table: pd.DataFrame) -> pd.DataFrame:
            """Add empty workflow provenance columns to a core table."""
            enriched = table.copy()
            enriched["condition_comparison"] = pd.Series(dtype=object)
            for column in provenance_columns:
                enriched[column] = pd.Series(dtype=object)
            return enriched

        def diagnostic_schema(table: pd.DataFrame) -> pd.DataFrame:
            """Add stable support-diagnostic fields to an empty table."""
            enriched = with_provenance(table)
            for column in (
                "support_filter",
                "n_observations_before_support_filter",
                "n_observations_excluded_support_filter",
                "excluded_support_levels",
                "n_aggregates_before_minimum_cells",
                "n_aggregates_excluded_minimum_cells",
                "n_aggregates_after_minimum_cells",
            ):
                enriched[column] = pd.Series(dtype=object)
            return enriched

        return cls(
            contrasts=with_provenance(core.contrasts),
            fixed_effects=with_provenance(core.fixed_effects),
            term_tests=with_provenance(core.term_tests),
            variance_components=with_provenance(core.variance_components),
            diagnostics=diagnostic_schema(core.diagnostics),
            availability=pd.DataFrame(
                columns=(
                    "analysis",
                    "estimable",
                    "status",
                    "reason",
                    "n_estimable_results",
                    "minimum_cells",
                    "minimum_donors",
                    "minimum_studies",
                    "minimum_repeated_contexts",
                    "detection_threshold",
                    "configured_condition_reference",
                    "observational_unit",
                    "independent_unit",
                )
            ),
        )


def tabula_sapiens_mixed_model_inference(
    cell_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    aggregation: Aggregation = "mean",
    detection_threshold: float = 0.0,
    condition_reference: str | None = "normal",
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    minimum_studies: int = 3,
    minimum_repeated_contexts: int = 3,
) -> TabulaMixedModelResults:
    """Estimate prespecified effects from donor-repeated context scores.

    Cells are first reduced to equally weighted donor, optional study, tissue,
    cell-type, and assay observations. The adjusted-context model estimates
    cell-type and assay contrasts and supplies the variance decomposition.
    Tissue contrasts use a separate adjusted model fitted to the connected
    network of donors observed in multiple tissues; reported pairs additionally
    require direct paired-donor support. Age interactions and separate
    condition-versus-reference interactions are fitted after excluding sparse
    cell-type strata, so their primary outputs are supported cell-type-specific
    decade slopes and condition effects rather than raw interaction
    coefficients. A donor random intercept is always used. When enough studies
    are available, study becomes the top-level random intercept and donors are
    modeled as nested variance components.

    The independent statistical unit is the donor (qualified by study when a
    study column is available). Each aggregate row receives equal weight;
    contributing cell counts describe measurement support, not replication.
    The function returns new tables and does not mutate `cell_frame`.

    Args:
      cell_frame: Long cell-level feature frame with score or expression values
        and observation metadata.
      schema: Names of donor, context, assay, condition, and study columns.
      aggregation: Cell-to-observation reduction applied within each feature.
      detection_threshold: Threshold used by expressing-fraction reductions.
      condition_reference: Reference condition for within-cell-type contrasts,
        or None to use the first donor-supported level. Each non-reference
        condition is fitted separately with the same reference.
      minimum_cells: Minimum finite cell values supporting an aggregate
        observation for each feature.
      minimum_donors: Minimum independent donors required for inference and for
        each reported focal level or pair.
      minimum_studies: Minimum top-level studies required before study is
        modeled as a random intercept rather than a fixed adjustment.
      minimum_repeated_contexts: Minimum donor-tissue-cell-type contexts with
        repeated assay observations required for a context variance component.

    Returns:
      Mixed-model contrasts, secondary coefficients and term tests, variance
      components, fit diagnostics, and explicit analysis availability.

    Raises:
      KeyError: If donor, tissue, or cell-type metadata is unavailable.
      ValueError: If thresholds are invalid or donor metadata is missing.
    """
    if minimum_cells < 1:
        raise ValueError("minimum_cells must be at least 1")
    if (
        isinstance(detection_threshold, bool)
        or not isinstance(detection_threshold, Real)
        or not np.isfinite(detection_threshold)
    ):
        raise ValueError("detection_threshold must be a finite real number")
    if minimum_donors < 3:
        raise ValueError("minimum_donors must be at least 3")
    if minimum_studies < 3:
        raise ValueError("minimum_studies must be at least 3")
    if minimum_repeated_contexts < 3:
        raise ValueError("minimum_repeated_contexts must be at least 3")

    unit_frame, minimum_cell_support = _mixed_model_unit_frame(
        cell_frame,
        schema=schema,
        aggregation=aggregation,
        detection_threshold=detection_threshold,
        minimum_cells=minimum_cells,
    )
    study_mode = _study_random_effect_is_estimable(
        unit_frame,
        schema=schema,
        minimum_studies=minimum_studies,
    )
    minimum_groups = minimum_studies if study_mode else minimum_donors
    age_center = _add_centered_age(unit_frame, schema=schema)
    group_key, variance_component_keys = _random_effect_structure(
        unit_frame,
        schema=schema,
        study_mode=study_mode,
        minimum_repeated_contexts=minimum_repeated_contexts,
    )
    observational_unit = _observational_unit_label(unit_frame)

    results: list[MixedModelInferenceResult] = []
    main_result = _adjusted_context_inference(
        unit_frame,
        schema=schema,
        group_key=group_key,
        variance_component_keys=variance_component_keys,
        study_mode=study_mode,
        minimum_groups=minimum_groups,
        minimum_donors=minimum_donors,
    )
    if main_result is not None:
        results.append(main_result)

    paired_tissue_result = _paired_tissue_inference(
        unit_frame,
        schema=schema,
        group_key=group_key,
        variance_component_keys=variance_component_keys,
        study_mode=study_mode,
        minimum_groups=minimum_groups,
        minimum_donors=minimum_donors,
    )
    if paired_tissue_result is not None:
        results.append(paired_tissue_result)

    age_result = _age_by_cell_type_inference(
        unit_frame,
        schema=schema,
        group_key=group_key,
        variance_component_keys=variance_component_keys,
        study_mode=study_mode,
        minimum_groups=minimum_groups,
        minimum_donors=minimum_donors,
    )
    if age_result is not None:
        results.append(age_result)

    condition_result = _condition_by_cell_type_inference(
        unit_frame,
        schema=schema,
        group_key=group_key,
        variance_component_keys=variance_component_keys,
        study_mode=study_mode,
        condition_reference=condition_reference,
        minimum_groups=minimum_groups,
        minimum_donors=minimum_donors,
    )
    if condition_result is not None:
        results.append(condition_result)

    combined = _combine_inference_results(
        results,
        age_center=age_center,
        schema=schema,
        aggregation=aggregation,
        detection_threshold=detection_threshold,
        condition_reference=condition_reference,
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
        minimum_studies=minimum_studies,
        minimum_repeated_contexts=minimum_repeated_contexts,
        observational_unit=observational_unit,
        minimum_cell_support=minimum_cell_support,
    )
    availability = _analysis_availability(
        unit_frame,
        contrasts=combined.contrasts,
        variance_components=combined.variance_components,
        diagnostics=combined.diagnostics,
        schema=schema,
        study_mode=study_mode,
        detection_threshold=detection_threshold,
        condition_reference=condition_reference,
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
        minimum_studies=minimum_studies,
        minimum_repeated_contexts=minimum_repeated_contexts,
        observational_unit=observational_unit,
        minimum_cell_support=minimum_cell_support,
    )
    return TabulaMixedModelResults(
        contrasts=combined.contrasts,
        fixed_effects=combined.fixed_effects,
        term_tests=combined.term_tests,
        variance_components=combined.variance_components,
        diagnostics=combined.diagnostics,
        availability=availability,
    )


def _mixed_model_unit_frame(
    cell_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    aggregation: Aggregation,
    detection_threshold: float,
    minimum_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return donor-context-assay observations eligible by cell support."""
    required = [schema.donor_key, schema.tissue_key, schema.cell_type_key]
    if missing := [key for key in required if key not in cell_frame]:
        raise KeyError(f"cell frame missing mixed-model columns: {missing}")
    if cell_frame[schema.donor_key].isna().any():
        raise ValueError(
            f"{schema.donor_key} contains missing donor identifiers"
        )

    unit_keys = [
        key
        for key in [
            schema.study_key,
            schema.donor_key,
            schema.tissue_key,
            schema.cell_type_key,
            schema.assay_key,
        ]
        if key in cell_frame
    ]
    unit_label = " x ".join(unit_keys)
    aggregated = aggregate_feature_frame_by_keys(
        cell_frame,
        unit_keys=unit_keys,
        statistical_unit=unit_label,
        aggregation=aggregation,
        schema=schema,
        detection_threshold=detection_threshold,
    )
    support = (
        aggregated.assign(
            _excluded=aggregated["n_cells"].lt(minimum_cells),
        )
        .groupby(list(_FEATURE_KEYS), observed=True, dropna=False)
        .agg(
            n_aggregates_before_minimum_cells=("n_cells", "size"),
            n_aggregates_excluded_minimum_cells=("_excluded", "sum"),
        )
        .reset_index()
    )
    support["n_aggregates_after_minimum_cells"] = (
        support["n_aggregates_before_minimum_cells"]
        - support["n_aggregates_excluded_minimum_cells"]
    )
    aggregated = aggregated.loc[aggregated["n_cells"].ge(minimum_cells)].copy()
    aggregated.attrs["observational_unit"] = unit_label

    independent_keys = [
        key for key in (schema.study_key, schema.donor_key) if key in aggregated
    ]
    aggregated[_INDEPENDENT_UNIT_KEY] = _factorized_identifier(
        aggregated,
        keys=independent_keys,
        prefix="donor_",
    )
    context_keys = [
        *independent_keys,
        schema.tissue_key,
        schema.cell_type_key,
    ]
    aggregated[_CONTEXT_UNIT_KEY] = _factorized_identifier(
        aggregated,
        keys=context_keys,
        prefix="context_",
    )
    aggregated[_OBSERVATIONAL_UNIT_KEY] = _factorized_identifier(
        aggregated,
        keys=unit_keys,
        prefix="observation_",
    )
    return aggregated, support


def _factorized_identifier(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    prefix: str,
) -> pd.Series:
    """Return collision-free codes for typed tuples of metadata values."""
    identities = [
        tuple(_typed_metadata_identity(value) for value in row)
        for row in frame.loc[:, keys].itertuples(index=False, name=None)
    ]
    codes, _ = pd.factorize(
        pd.Series(identities, index=frame.index, dtype=object),
        sort=False,
    )
    return pd.Series(
        [f"{prefix}{int(code)}" for code in codes],
        index=frame.index,
        dtype=str,
    )


def _typed_metadata_identity(value: object) -> str:
    """Return a type-qualified scalar identity for grouping-code creation."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _observational_unit_label(unit_frame: pd.DataFrame) -> str:
    """Return the exact aggregation keys recorded on modeled rows."""
    if label := unit_frame.attrs.get("observational_unit"):
        return str(label)
    if "statistical_unit" not in unit_frame or unit_frame.empty:
        return "unavailable"
    labels = (
        unit_frame["statistical_unit"].dropna().astype(str).unique().tolist()
    )
    return labels[0] if labels else "unavailable"


def _study_random_effect_is_estimable(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    minimum_studies: int,
) -> bool:
    """Return whether study-level random variation has enough support."""
    if schema.study_key not in unit_frame:
        return False
    studies = unit_frame[schema.study_key].dropna()
    return studies.nunique() >= minimum_studies


def _add_centered_age(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
) -> float:
    """Add donor-weighted centered age in decades and return its center."""
    if schema.age_key not in unit_frame:
        unit_frame[_AGE_DECADES_KEY] = np.nan
        return np.nan

    donor_age = unit_frame.loc[
        :, [_INDEPENDENT_UNIT_KEY, schema.age_key]
    ].copy()
    donor_age[schema.age_key] = pd.to_numeric(
        donor_age[schema.age_key], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    age_counts = donor_age.groupby(_INDEPENDENT_UNIT_KEY, dropna=False)[
        schema.age_key
    ].nunique(dropna=True)
    if inconsistent := age_counts[age_counts > 1].index.astype(str).tolist():
        preview = ", ".join(inconsistent[:3])
        raise ValueError(
            f"{schema.age_key} varies within donor for {len(inconsistent)} "
            f"donors (for example: {preview})"
        )

    unique_ages = donor_age.groupby(
        _INDEPENDENT_UNIT_KEY,
        observed=True,
        dropna=False,
    )[schema.age_key].first()
    center = float(unique_ages.mean())
    if not np.isfinite(center):
        unit_frame[_AGE_DECADES_KEY] = np.nan
        return np.nan
    numeric_age = pd.to_numeric(
        unit_frame[schema.age_key], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    unit_frame[_AGE_DECADES_KEY] = (numeric_age - center) / 10.0
    return center


def _random_effect_structure(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    study_mode: bool,
    minimum_repeated_contexts: int,
) -> tuple[str, tuple[str, ...]]:
    """Return the supported study, donor, and context random intercepts."""
    variance_components: list[str] = []
    if study_mode:
        group_key = schema.study_key
        variance_components.append(_INDEPENDENT_UNIT_KEY)
    else:
        group_key = _INDEPENDENT_UNIT_KEY

    independent_rows = unit_frame.drop_duplicates(_OBSERVATIONAL_UNIT_KEY)
    repeated_contexts = independent_rows.groupby(
        _CONTEXT_UNIT_KEY,
        observed=True,
        dropna=False,
    ).size()
    n_repeated = int((repeated_contexts >= 2).sum())
    if n_repeated >= minimum_repeated_contexts:
        variance_components.append(_CONTEXT_UNIT_KEY)
    return group_key, tuple(variance_components)


def _available_keys(
    frame: pd.DataFrame,
    keys: list[str],
) -> tuple[str, ...]:
    """Return present predictors with at least one non-missing value."""
    return tuple(
        key for key in keys if key in frame and frame[key].notna().any()
    )


def _adjusted_context_inference(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    group_key: str,
    variance_component_keys: tuple[str, ...],
    study_mode: bool,
    minimum_groups: int,
    minimum_donors: int,
) -> MixedModelInferenceResult | None:
    """Fit the shared adjusted model for context and technical contrasts."""
    categorical = _available_keys(
        unit_frame,
        [
            schema.cell_type_key,
            schema.tissue_key,
            schema.assay_key,
            schema.sex_key,
            schema.condition_key,
            *(() if study_mode else (schema.study_key,)),
        ],
    )
    continuous = (
        (_AGE_DECADES_KEY,)
        if unit_frame[_AGE_DECADES_KEY].notna().any()
        else ()
    )
    contrasts: list[MixedModelContrast] = []
    contrasts.extend(
        MixedModelContrast(
            mode="categorical_vs_mean",
            predictor_key=key,
            estimand=estimand,
            effect_scale=scale,
        )
        for key, estimand, scale in (
            (
                schema.cell_type_key,
                "adjusted_cell_type",
                "adjusted_score_deviation_from_supported_cell_type_mean",
            ),
            (
                schema.assay_key,
                "assay_batch_effects",
                "adjusted_score_deviation_from_supported_assay_mean",
            ),
        )
        if key in categorical
    )
    return mixed_model_inference(
        unit_frame,
        spec=MixedModelSpec(
            analysis="adjusted_context",
            group_key=group_key,
            independent_unit_key=_INDEPENDENT_UNIT_KEY,
            continuous_keys=continuous,
            categorical_keys=categorical,
            variance_component_keys=variance_component_keys,
            minimum_groups=minimum_groups,
            minimum_independent_units=minimum_donors,
            minimum_focal_units=minimum_donors,
        ),
        contrasts=contrasts,
    )


def _paired_tissue_inference(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    group_key: str,
    variance_component_keys: tuple[str, ...],
    study_mode: bool,
    minimum_groups: int,
    minimum_donors: int,
) -> MixedModelInferenceResult | None:
    """Fit tissue contrasts using only donors observed in multiple tissues."""
    if (
        schema.tissue_key not in unit_frame
        or unit_frame[schema.tissue_key].dropna().astype(str).nunique() < 2
    ):
        return None
    categorical = _available_keys(
        unit_frame,
        [
            schema.cell_type_key,
            schema.tissue_key,
            schema.assay_key,
            schema.sex_key,
            schema.condition_key,
            *(() if study_mode else (schema.study_key,)),
        ],
    )
    continuous = (
        (_AGE_DECADES_KEY,)
        if unit_frame[_AGE_DECADES_KEY].notna().any()
        else ()
    )
    filtered_frame, support_summary = _filter_multitissue_donor_support(
        unit_frame,
        tissue_key=schema.tissue_key,
        categorical_keys=categorical,
        continuous_keys=continuous,
        group_key=group_key,
        variance_component_keys=variance_component_keys,
    )
    tissue_levels = _observed_string_levels(unit_frame, schema.tissue_key)
    result = mixed_model_inference(
        filtered_frame,
        spec=MixedModelSpec(
            analysis="paired_tissue",
            group_key=group_key,
            independent_unit_key=_INDEPENDENT_UNIT_KEY,
            continuous_keys=continuous,
            categorical_keys=categorical,
            variance_component_keys=variance_component_keys,
            minimum_groups=minimum_groups,
            minimum_independent_units=minimum_donors,
            minimum_focal_units=minimum_donors,
        ),
        contrasts=(
            MixedModelContrast(
                mode="categorical_pairwise",
                predictor_key=schema.tissue_key,
                estimand="paired_tissue",
                levels=tissue_levels,
                effect_scale="adjusted_within_donor_score_difference",
            ),
        ),
    )
    return _attach_support_diagnostics(
        result,
        support_summary=support_summary,
        filter_name="donors_observed_in_multiple_tissues",
    )


def _age_by_cell_type_inference(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    group_key: str,
    variance_component_keys: tuple[str, ...],
    study_mode: bool,
    minimum_groups: int,
    minimum_donors: int,
) -> MixedModelInferenceResult | None:
    """Fit adjusted per-decade slopes within each supported cell type."""
    if (
        schema.cell_type_key not in unit_frame
        or not unit_frame[schema.cell_type_key].notna().any()
        or not unit_frame[_AGE_DECADES_KEY].notna().any()
    ):
        return None
    categorical = _available_keys(
        unit_frame,
        [
            schema.cell_type_key,
            schema.tissue_key,
            schema.assay_key,
            schema.sex_key,
            schema.condition_key,
            *(() if study_mode else (schema.study_key,)),
        ],
    )
    filtered_frame, by_levels, support = _filter_age_interaction_support(
        unit_frame,
        schema=schema,
        categorical_keys=categorical,
        group_key=group_key,
        variance_component_keys=variance_component_keys,
        minimum_donors=minimum_donors,
    )
    result = mixed_model_inference(
        filtered_frame,
        spec=MixedModelSpec(
            analysis="age_by_cell_type",
            group_key=group_key,
            independent_unit_key=_INDEPENDENT_UNIT_KEY,
            continuous_keys=(_AGE_DECADES_KEY,),
            categorical_keys=categorical,
            interactions=((_AGE_DECADES_KEY, schema.cell_type_key),),
            variance_component_keys=variance_component_keys,
            minimum_groups=minimum_groups,
            minimum_independent_units=minimum_donors,
            minimum_focal_units=minimum_donors,
        ),
        contrasts=(
            MixedModelContrast(
                mode="continuous_slope",
                predictor_key=_AGE_DECADES_KEY,
                by_key=schema.cell_type_key,
                by_levels=by_levels,
                estimand="age_by_cell_type",
                effect_scale="score_change_per_10_years",
            ),
        ),
    )
    return _annotate_interaction_support(
        result,
        support=support,
        estimand="age_by_cell_type",
        filter_name="age_by_cell_type_donor_support",
    )


def _condition_by_cell_type_inference(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    group_key: str,
    variance_component_keys: tuple[str, ...],
    study_mode: bool,
    condition_reference: str | None,
    minimum_groups: int,
    minimum_donors: int,
) -> MixedModelInferenceResult | None:
    """Fit separate adjusted condition-reference effects by cell type."""
    if (
        schema.condition_key not in unit_frame
        or schema.cell_type_key not in unit_frame
        or not unit_frame[schema.condition_key].notna().any()
        or unit_frame[schema.condition_key].dropna().astype(str).nunique() < 2
        or not unit_frame[schema.cell_type_key].notna().any()
    ):
        return None
    categorical = _available_keys(
        unit_frame,
        [
            schema.condition_key,
            schema.cell_type_key,
            schema.tissue_key,
            schema.assay_key,
            schema.sex_key,
            *(() if study_mode else (schema.study_key,)),
        ],
    )
    continuous = (
        (_AGE_DECADES_KEY,)
        if unit_frame[_AGE_DECADES_KEY].notna().any()
        else ()
    )
    observed_conditions = sorted(
        unit_frame[schema.condition_key].dropna().astype(str).unique().tolist()
    )
    resolved_reference = condition_reference or _first_supported_condition(
        unit_frame,
        condition_key=schema.condition_key,
        minimum_donors=minimum_donors,
    )
    comparison_levels = tuple(
        level for level in observed_conditions if level != resolved_reference
    )
    comparison_results: list[MixedModelInferenceResult] = []
    for comparison_level in comparison_levels:
        filtered_frame, by_levels, support = (
            _filter_condition_interaction_support(
                unit_frame,
                schema=schema,
                categorical_keys=categorical,
                continuous_keys=continuous,
                group_key=group_key,
                variance_component_keys=variance_component_keys,
                reference=resolved_reference,
                comparison_levels=(comparison_level,),
                minimum_donors=minimum_donors,
            )
        )
        pair_frame = filtered_frame.loc[
            filtered_frame[schema.condition_key]
            .astype(str)
            .isin((resolved_reference, comparison_level))
        ].copy()
        result = mixed_model_inference(
            pair_frame,
            spec=MixedModelSpec(
                analysis="condition_by_cell_type",
                group_key=group_key,
                independent_unit_key=_INDEPENDENT_UNIT_KEY,
                continuous_keys=continuous,
                categorical_keys=categorical,
                interactions=((schema.condition_key, schema.cell_type_key),),
                variance_component_keys=variance_component_keys,
                minimum_groups=minimum_groups,
                minimum_independent_units=minimum_donors,
                minimum_focal_units=minimum_donors,
            ),
            contrasts=(
                MixedModelContrast(
                    mode="categorical_vs_reference",
                    predictor_key=schema.condition_key,
                    reference=resolved_reference,
                    by_key=schema.cell_type_key,
                    levels=(comparison_level,),
                    by_levels=by_levels,
                    estimand="condition_by_cell_type",
                    effect_scale="adjusted_condition_score_difference",
                ),
            ),
        )
        annotated = _annotate_interaction_support(
            result,
            support=support,
            estimand="condition_by_cell_type",
            filter_name="condition_by_cell_type_donor_support",
        )
        comparison_results.append(
            _tag_model_comparison(
                annotated,
                comparison=f"{comparison_level} vs {resolved_reference}",
            )
        )
    return _combine_condition_results(comparison_results)


def _first_supported_condition(
    unit_frame: pd.DataFrame,
    *,
    condition_key: str,
    minimum_donors: int,
) -> str:
    """Return the first deterministic condition with independent support."""
    support = unit_frame.groupby(
        condition_key,
        observed=True,
        dropna=True,
    )[_INDEPENDENT_UNIT_KEY].nunique()
    if supported := sorted(
        support.loc[support.ge(minimum_donors)].index.astype(str).tolist()
    ):
        return supported[0]
    observed = _observed_string_levels(unit_frame, condition_key)
    return observed[0] if observed else ""


def _tag_model_comparison(
    result: MixedModelInferenceResult,
    *,
    comparison: str,
) -> MixedModelInferenceResult:
    """Tag every condition-model table with its fitted two-arm comparison."""

    def tagged(table: pd.DataFrame) -> pd.DataFrame:
        """Return a tagged table without mutating the inference result."""
        copied = table.copy()
        copied["condition_comparison"] = comparison
        return copied

    return MixedModelInferenceResult(
        contrasts=tagged(result.contrasts),
        fixed_effects=tagged(result.fixed_effects),
        term_tests=tagged(result.term_tests),
        variance_components=tagged(result.variance_components),
        diagnostics=tagged(result.diagnostics),
    )


def _combine_condition_results(
    results: list[MixedModelInferenceResult],
) -> MixedModelInferenceResult:
    """Combine pairwise condition fits and restore shared FDR families."""
    if not results:
        return MixedModelInferenceResult.empty()

    def combine(attribute: str) -> pd.DataFrame:
        """Concatenate one result table across condition comparisons."""
        return pd.concat(
            [getattr(result, attribute) for result in results],
            ignore_index=True,
            sort=False,
        )

    return MixedModelInferenceResult(
        contrasts=_recompute_fdr(
            combine("contrasts"),
            family_columns=("analysis", "estimand"),
        ),
        fixed_effects=_recompute_fdr(
            combine("fixed_effects"),
            family_columns=("analysis", "term"),
        ),
        term_tests=_recompute_fdr(
            combine("term_tests"),
            family_columns=("analysis", "term"),
        ),
        variance_components=combine("variance_components"),
        diagnostics=combine("diagnostics"),
    )


def _recompute_fdr(
    table: pd.DataFrame,
    *,
    family_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Recompute BH FDR after separately fitted results are combined."""
    if table.empty:
        return table
    adjusted = table.copy()
    adjusted["pvalue_fdr"] = np.nan
    for _, indices in adjusted.groupby(
        list(family_columns),
        observed=True,
        dropna=False,
    ).groups.items():
        positions = list(indices)
        eligible = adjusted.loc[positions, "status"].astype(str).eq("ok")
        if "estimable" in adjusted:
            eligible &= adjusted.loc[positions, "estimable"].fillna(False)
        pvalues = pd.to_numeric(
            adjusted.loc[positions, "pvalue"],
            errors="coerce",
        )
        eligible &= np.isfinite(pvalues.to_numpy(dtype=float))
        if tested := list(pd.Index(positions)[eligible.to_numpy(dtype=bool)]):
            adjusted.loc[tested, "pvalue_fdr"] = benjamini_hochberg(
                adjusted.loc[tested, "pvalue"].tolist()
            )
    return adjusted


def _filter_age_interaction_support(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    categorical_keys: tuple[str, ...],
    group_key: str,
    variance_component_keys: tuple[str, ...],
    minimum_donors: int,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """Exclude sparse interaction levels without hiding their requested rows."""
    levels = _observed_string_levels(unit_frame, schema.cell_type_key)
    required_keys = tuple(
        dict.fromkeys(
            (
                "feature_value",
                _AGE_DECADES_KEY,
                schema.cell_type_key,
                group_key,
                _INDEPENDENT_UNIT_KEY,
                *categorical_keys,
                *variance_component_keys,
            )
        )
    )
    return _filter_interaction_support(
        unit_frame,
        by_key=schema.cell_type_key,
        levels=levels,
        required_keys=required_keys,
        numeric_keys=("feature_value", _AGE_DECADES_KEY),
        minimum_donors=minimum_donors,
        condition_key=None,
        reference=None,
        comparison_levels=(),
    )


def _filter_multitissue_donor_support(
    unit_frame: pd.DataFrame,
    *,
    tissue_key: str,
    categorical_keys: tuple[str, ...],
    continuous_keys: tuple[str, ...],
    group_key: str,
    variance_component_keys: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict each feature to donors contributing within-donor tissue data."""
    required_keys = tuple(
        dict.fromkeys(
            (
                "feature_value",
                tissue_key,
                group_key,
                _INDEPENDENT_UNIT_KEY,
                *continuous_keys,
                *categorical_keys,
                *variance_component_keys,
            )
        )
    )
    retained: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    grouped = unit_frame.groupby(
        list(_FEATURE_KEYS),
        observed=True,
        dropna=False,
        sort=False,
    )
    for feature_values, feature_frame in grouped:
        values = (
            feature_values
            if isinstance(feature_values, tuple)
            else (feature_values,)
        )
        feature: dict[str, object] = dict(
            zip(_FEATURE_KEYS, values, strict=True)
        )
        complete = feature_frame.loc[
            _complete_support_mask(
                feature_frame,
                required_keys=required_keys,
                numeric_keys=("feature_value", *continuous_keys),
            )
        ]
        tissue_counts = complete.groupby(
            _INDEPENDENT_UNIT_KEY,
            observed=True,
            dropna=False,
        )[tissue_key].nunique(dropna=True)
        if multi_tissue_units := set(tissue_counts[tissue_counts >= 2].index):
            retained_frame = feature_frame.loc[
                feature_frame[_INDEPENDENT_UNIT_KEY].isin(multi_tissue_units)
            ].copy()
            n_excluded = int(feature_frame.shape[0] - retained_frame.shape[0])
        else:
            retained_frame = feature_frame.copy()
            retained_frame["feature_value"] = np.nan
            n_excluded = int(feature_frame.shape[0])
        retained.append(retained_frame)
        summaries.append(
            {
                **feature,
                "n_observations_before_support_filter": int(
                    feature_frame.shape[0]
                ),
                "n_observations_excluded_support_filter": n_excluded,
                "excluded_support_levels": "",
            }
        )
    filtered = (
        pd.concat(retained, axis="index").sort_index(kind="stable")
        if retained
        else unit_frame.copy()
    )
    return filtered, pd.DataFrame.from_records(summaries)


def _filter_condition_interaction_support(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    categorical_keys: tuple[str, ...],
    continuous_keys: tuple[str, ...],
    group_key: str,
    variance_component_keys: tuple[str, ...],
    reference: str,
    comparison_levels: tuple[str, ...],
    minimum_donors: int,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """Retain cell types supported across every modeled condition arm."""
    levels = _observed_string_levels(unit_frame, schema.cell_type_key)
    required_keys = tuple(
        dict.fromkeys(
            (
                "feature_value",
                schema.condition_key,
                schema.cell_type_key,
                group_key,
                _INDEPENDENT_UNIT_KEY,
                *continuous_keys,
                *categorical_keys,
                *variance_component_keys,
            )
        )
    )
    return _filter_interaction_support(
        unit_frame,
        by_key=schema.cell_type_key,
        levels=levels,
        required_keys=required_keys,
        numeric_keys=("feature_value", *continuous_keys),
        minimum_donors=minimum_donors,
        condition_key=schema.condition_key,
        reference=reference,
        comparison_levels=comparison_levels,
    )


def _filter_interaction_support(
    unit_frame: pd.DataFrame,
    *,
    by_key: str,
    levels: tuple[str, ...],
    required_keys: tuple[str, ...],
    numeric_keys: tuple[str, ...],
    minimum_donors: int,
    condition_key: str | None,
    reference: str | None,
    comparison_levels: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """Filter unsupported per-feature interaction levels before fitting."""
    retained: list[pd.DataFrame] = []
    support_records: list[dict[str, object]] = []
    grouped = unit_frame.groupby(
        list(_FEATURE_KEYS),
        observed=True,
        dropna=False,
        sort=False,
    )
    for feature_values, feature_frame in grouped:
        values = (
            feature_values
            if isinstance(feature_values, tuple)
            else (feature_values,)
        )
        feature: dict[str, object] = dict(
            zip(_FEATURE_KEYS, values, strict=True)
        )
        complete = feature_frame.loc[
            _complete_support_mask(
                feature_frame,
                required_keys=required_keys,
                numeric_keys=numeric_keys,
            )
        ]
        supported_levels: list[str] = []
        level_records: list[dict[str, object]] = []
        for level in levels:
            level_frame = complete.loc[complete[by_key].astype(str).eq(level)]
            if condition_key is None:
                level_records.append(
                    _age_support_record(
                        feature,
                        level=level,
                        level_frame=level_frame,
                        minimum_donors=minimum_donors,
                    )
                )
                if bool(level_records[-1]["supported"]):
                    supported_levels.append(level)
                continue

            assert reference is not None
            condition_records = _condition_support_records(
                feature,
                level=level,
                level_frame=level_frame,
                condition_key=condition_key,
                reference=reference,
                comparison_levels=comparison_levels,
                minimum_donors=minimum_donors,
            )
            level_records.extend(condition_records)
            if condition_records and all(
                bool(record["supported"]) for record in condition_records
            ):
                supported_levels.append(level)

        if supported_levels:
            retained_frame = feature_frame.loc[
                feature_frame[by_key].astype(str).isin(supported_levels)
            ].copy()
            excluded_levels = set(levels).difference(supported_levels)
            for record in level_records:
                record["excluded_from_model"] = (
                    record["by_level"] in excluded_levels
                )
        else:
            retained_frame = feature_frame.copy()
            retained_frame["feature_value"] = np.nan
            for record in level_records:
                record["excluded_from_model"] = True
        n_excluded = (
            int(feature_frame.shape[0] - retained_frame.shape[0])
            if supported_levels
            else int(feature_frame.shape[0])
        )
        for record in level_records:
            record["n_rows_before_focal_filter"] = int(feature_frame.shape[0])
            record["n_rows_excluded_focal_filter"] = n_excluded
        retained.append(retained_frame)
        support_records.extend(level_records)

    filtered = (
        pd.concat(retained, axis="index").sort_index(kind="stable")
        if retained
        else unit_frame.copy()
    )
    return filtered, levels, pd.DataFrame.from_records(support_records)


def _complete_support_mask(
    frame: pd.DataFrame,
    *,
    required_keys: tuple[str, ...],
    numeric_keys: tuple[str, ...],
) -> pd.Series:
    """Return rows that the configured model can use for support counting."""
    mask = frame.loc[:, list(required_keys)].notna().all(axis="columns")
    numeric = set(numeric_keys)
    for key in required_keys:
        if key in numeric:
            values = pd.to_numeric(frame[key], errors="coerce").to_numpy(
                dtype=float
            )
            mask &= np.isfinite(values)
        else:
            mask &= ~frame[key].map(_is_infinite_scalar).to_numpy(dtype=bool)
    return mask


def _is_infinite_scalar(value: object) -> bool:
    """Return whether a metadata value is an infinite numeric scalar."""
    return isinstance(value, Real) and not bool(np.isfinite(float(value)))


def _age_support_record(
    feature: dict[str, object],
    *,
    level: str,
    level_frame: pd.DataFrame,
    minimum_donors: int,
) -> dict[str, object]:
    """Return donor and age support for one feature-cell-type slope."""
    n_donors = int(level_frame[_INDEPENDENT_UNIT_KEY].nunique())
    n_ages = int(level_frame[_AGE_DECADES_KEY].nunique())
    supported = n_donors >= minimum_donors and n_ages >= 2
    reason = ""
    if n_donors < minimum_donors:
        reason = "insufficient_level_support"
    elif n_ages < 2:
        reason = "constant_continuous_predictor"
    return {
        **feature,
        "level": level,
        "by_level": level,
        "n_independent_units_support": n_donors,
        "n_level_units_support": n_donors,
        "n_reference_units_support": np.nan,
        "support_reason": reason,
        "supported": supported,
    }


def _condition_support_records(
    feature: dict[str, object],
    *,
    level: str,
    level_frame: pd.DataFrame,
    condition_key: str,
    reference: str,
    comparison_levels: tuple[str, ...],
    minimum_donors: int,
) -> list[dict[str, object]]:
    """Return arm support for every condition contrast in one cell type."""
    condition_values = level_frame[condition_key].astype(str)
    units_by_condition = {
        condition: set(
            level_frame.loc[
                condition_values.eq(condition),
                _INDEPENDENT_UNIT_KEY,
            ]
        )
        for condition in (reference, *comparison_levels)
    }
    reference_units = units_by_condition[reference]
    full_cross_supported = all(
        len(units) >= minimum_donors for units in units_by_condition.values()
    )
    records: list[dict[str, object]] = []
    for condition in comparison_levels:
        level_units = units_by_condition[condition]
        reason = ""
        if len(reference_units) < minimum_donors:
            reason = "insufficient_reference_support"
        elif len(level_units) < minimum_donors:
            reason = "insufficient_level_support"
        elif not full_cross_supported:
            reason = "incomplete_condition_cross_support"
        records.append(
            {
                **feature,
                "level": condition,
                "by_level": level,
                "n_independent_units_support": len(
                    level_units.union(reference_units)
                ),
                "n_level_units_support": len(level_units),
                "n_reference_units_support": len(reference_units),
                "support_reason": reason,
                "supported": full_cross_supported,
            }
        )
    return records


def _observed_string_levels(
    frame: pd.DataFrame,
    key: str,
) -> tuple[str, ...]:
    """Return deterministic non-missing levels for a categorical column."""
    return tuple(sorted(frame[key].dropna().astype(str).unique().tolist()))


def _annotate_interaction_support(
    result: MixedModelInferenceResult,
    *,
    support: pd.DataFrame,
    estimand: str,
    filter_name: str,
) -> MixedModelInferenceResult:
    """Restore support and diagnostics for focal levels excluded before fit."""
    if support.empty:
        return result
    merge_keys = [*_FEATURE_KEYS, "level", "by_level"]
    support_columns = [
        *merge_keys,
        "n_independent_units_support",
        "n_level_units_support",
        "n_reference_units_support",
        "support_reason",
        "excluded_from_model",
    ]
    contrasts = result.contrasts.copy()
    if not contrasts.empty and set(merge_keys).issubset(contrasts.columns):
        contrasts = contrasts.merge(
            support.loc[:, support_columns],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
        excluded = contrasts["estimand"].astype(str).eq(estimand) & contrasts[
            "excluded_from_model"
        ].fillna(False).astype(bool)
        for target, source in (
            ("n_independent_units", "n_independent_units_support"),
            ("n_level_units", "n_level_units_support"),
            ("n_reference_units", "n_reference_units_support"),
        ):
            contrasts.loc[excluded, target] = contrasts.loc[excluded, source]
        contrasts.loc[excluded, "reason"] = contrasts.loc[
            excluded, "support_reason"
        ]
        contrasts = contrasts.drop(
            columns=[
                "n_independent_units_support",
                "n_level_units_support",
                "n_reference_units_support",
                "support_reason",
                "excluded_from_model",
            ]
        )

    diagnostic_support = support.groupby(
        list(_FEATURE_KEYS),
        observed=True,
        dropna=False,
    ).agg(
        n_observations_before_support_filter=(
            "n_rows_before_focal_filter",
            "first",
        ),
        n_observations_excluded_support_filter=(
            "n_rows_excluded_focal_filter",
            "first",
        ),
    )
    excluded_levels = (
        support.loc[support["excluded_from_model"].astype(bool)]
        .groupby(list(_FEATURE_KEYS), observed=True, dropna=False)["by_level"]
        .agg(lambda values: "|".join(sorted(set(values.astype(str)))))
        .rename("excluded_support_levels")
    )
    diagnostic_support = (
        diagnostic_support.join(excluded_levels, how="left")
        .fillna({"excluded_support_levels": ""})
        .reset_index()
    )
    annotated = MixedModelInferenceResult(
        contrasts=contrasts,
        fixed_effects=result.fixed_effects,
        term_tests=result.term_tests,
        variance_components=result.variance_components,
        diagnostics=result.diagnostics,
    )
    return _attach_support_diagnostics(
        annotated,
        support_summary=diagnostic_support,
        filter_name=filter_name,
    )


def _attach_support_diagnostics(
    result: MixedModelInferenceResult,
    *,
    support_summary: pd.DataFrame,
    filter_name: str,
) -> MixedModelInferenceResult:
    """Attach explicit pre-fit support exclusions to model diagnostics."""
    diagnostics = result.diagnostics.copy()
    if not diagnostics.empty and set(_FEATURE_KEYS).issubset(
        diagnostics.columns
    ):
        diagnostics = diagnostics.merge(
            support_summary,
            on=list(_FEATURE_KEYS),
            how="left",
            validate="one_to_one",
        )
        diagnostics["support_filter"] = filter_name
    return MixedModelInferenceResult(
        contrasts=result.contrasts,
        fixed_effects=result.fixed_effects,
        term_tests=result.term_tests,
        variance_components=result.variance_components,
        diagnostics=diagnostics,
    )


def _combine_inference_results(
    results: list[MixedModelInferenceResult],
    *,
    age_center: float,
    schema: ObsSchema,
    aggregation: Aggregation,
    detection_threshold: float,
    condition_reference: str | None,
    minimum_cells: int,
    minimum_donors: int,
    minimum_studies: int,
    minimum_repeated_contexts: int,
    observational_unit: str,
    minimum_cell_support: pd.DataFrame,
) -> MixedModelInferenceResult:
    """Concatenate model families and attach the age transformation center."""

    def combine(attribute: str) -> pd.DataFrame:
        """Return one provenance-annotated table for `attribute`."""
        tables = [getattr(result, attribute) for result in results]
        if not tables:
            return pd.DataFrame()
        combined = pd.concat(tables, ignore_index=True, sort=False)
        if "condition_comparison" not in combined:
            combined["condition_comparison"] = ""
        combined["age_center_years"] = age_center
        combined["aggregation"] = aggregation
        combined["detection_threshold"] = detection_threshold
        combined["configured_condition_reference"] = condition_reference
        combined["minimum_cells_per_observation"] = minimum_cells
        combined["minimum_donors"] = minimum_donors
        combined["minimum_studies"] = minimum_studies
        combined["minimum_repeated_contexts"] = minimum_repeated_contexts
        combined["observational_unit"] = observational_unit
        combined["independent_unit"] = "donor"
        return combined

    contrasts = combine("contrasts")
    display_age = f"{schema.age_key}_centered_per_10_years"
    if not contrasts.empty and "predictor" in contrasts:
        contrasts.loc[
            contrasts["predictor"] == _AGE_DECADES_KEY,
            "predictor",
        ] = schema.age_key
        contrasts = _replace_table_text(
            contrasts,
            columns=("contrast",),
            old=_AGE_DECADES_KEY,
            new=display_age,
        )
    fixed_effects = _replace_table_text(
        combine("fixed_effects"),
        columns=("term",),
        old=_AGE_DECADES_KEY,
        new=display_age,
    )
    term_tests = _replace_table_text(
        combine("term_tests"),
        columns=("term",),
        old=_AGE_DECADES_KEY,
        new=display_age,
    )
    variance_components = combine("variance_components")
    if "component" in variance_components:
        variance_components["component"] = variance_components[
            "component"
        ].replace(
            {
                _INDEPENDENT_UNIT_KEY: "donor",
                _CONTEXT_UNIT_KEY: "context",
                schema.study_key: "study",
            }
        )
    diagnostics = _replace_table_text(
        combine("diagnostics"),
        columns=("formula", "alias_mapping"),
        old=_AGE_DECADES_KEY,
        new=display_age,
    )
    diagnostics = _replace_table_text(
        diagnostics,
        columns=("variance_component_keys",),
        old=_INDEPENDENT_UNIT_KEY,
        new="donor",
    )
    diagnostics = _replace_table_text(
        diagnostics,
        columns=("variance_component_keys",),
        old=_CONTEXT_UNIT_KEY,
        new="context",
    )
    if "group_key" in diagnostics:
        diagnostics["group_key"] = diagnostics["group_key"].replace(
            {
                _INDEPENDENT_UNIT_KEY: "donor",
                schema.study_key: "study",
            }
        )
    if "independent_unit_key" in diagnostics:
        diagnostics["independent_unit_key"] = "donor"
    diagnostics = _attach_minimum_cell_support(
        diagnostics,
        minimum_cell_support=minimum_cell_support,
        age_center=age_center,
        aggregation=aggregation,
        detection_threshold=detection_threshold,
        condition_reference=condition_reference,
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
        minimum_studies=minimum_studies,
        minimum_repeated_contexts=minimum_repeated_contexts,
        observational_unit=observational_unit,
    )
    declared_diagnostic_columns = list(
        TabulaMixedModelResults.empty().diagnostics.columns
    )
    for column in declared_diagnostic_columns:
        if column not in diagnostics:
            diagnostics[column] = np.nan
    extra_diagnostic_columns = [
        column
        for column in diagnostics.columns
        if column not in declared_diagnostic_columns
    ]
    diagnostics = diagnostics.reindex(
        columns=[*declared_diagnostic_columns, *extra_diagnostic_columns]
    )
    return MixedModelInferenceResult(
        contrasts=contrasts,
        fixed_effects=fixed_effects,
        term_tests=term_tests,
        variance_components=variance_components,
        diagnostics=diagnostics,
    )


def _attach_minimum_cell_support(
    diagnostics: pd.DataFrame,
    *,
    minimum_cell_support: pd.DataFrame,
    age_center: float,
    aggregation: Aggregation,
    detection_threshold: float,
    condition_reference: str | None,
    minimum_cells: int,
    minimum_donors: int,
    minimum_studies: int,
    minimum_repeated_contexts: int,
    observational_unit: str,
) -> pd.DataFrame:
    """Attach aggregate support and add rows for fully filtered features."""
    support_columns = [
        *_FEATURE_KEYS,
        "n_aggregates_before_minimum_cells",
        "n_aggregates_excluded_minimum_cells",
        "n_aggregates_after_minimum_cells",
    ]
    if minimum_cell_support.empty:
        return diagnostics
    merged = diagnostics.merge(
        minimum_cell_support.loc[:, support_columns],
        on=list(_FEATURE_KEYS),
        how="left",
        validate="many_to_one",
    )
    represented = set(
        zip(
            *(merged[key].astype(str) for key in _FEATURE_KEYS),
            strict=True,
        )
    )
    missing_support = minimum_cell_support.loc[
        minimum_cell_support.apply(
            lambda row: (
                tuple(str(row[key]) for key in _FEATURE_KEYS) not in represented
            ),
            axis="columns",
        )
    ]
    records: list[dict[str, object]] = []
    for row in missing_support.to_dict(orient="records"):
        record: dict[str, object] = {
            str(column): np.nan for column in merged.columns
        }
        for key in _FEATURE_KEYS:
            record[key] = row[key]
        record |= {
            "analysis": "input_support",
            "n_input_observations": row["n_aggregates_before_minimum_cells"],
            "n_observations": row["n_aggregates_after_minimum_cells"],
            "age_center_years": age_center,
            "aggregation": aggregation,
            "detection_threshold": detection_threshold,
            "configured_condition_reference": condition_reference,
            "minimum_cells_per_observation": minimum_cells,
            "minimum_donors": minimum_donors,
            "minimum_studies": minimum_studies,
            "minimum_repeated_contexts": minimum_repeated_contexts,
            "observational_unit": observational_unit,
            "independent_unit": "donor",
            "status": "no_observations_after_minimum_cells",
            "reason": (f"all_aggregates_below_minimum_cells:{minimum_cells}"),
            "n_aggregates_before_minimum_cells": row[
                "n_aggregates_before_minimum_cells"
            ],
            "n_aggregates_excluded_minimum_cells": row[
                "n_aggregates_excluded_minimum_cells"
            ],
            "n_aggregates_after_minimum_cells": row[
                "n_aggregates_after_minimum_cells"
            ],
        }
        records.append(record)
    if records:
        merged = pd.concat(
            (merged, pd.DataFrame.from_records(records)),
            ignore_index=True,
            sort=False,
        )
    return merged


def _replace_table_text(
    table: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    old: str,
    new: str,
) -> pd.DataFrame:
    """Replace an internal predictor name in selected provenance columns."""
    for column in columns:
        if column in table:
            table[column] = (
                table[column]
                .astype("string")
                .str.replace(
                    old,
                    new,
                    regex=False,
                )
            )
    return table


def _analysis_availability(
    unit_frame: pd.DataFrame,
    *,
    contrasts: pd.DataFrame,
    variance_components: pd.DataFrame,
    diagnostics: pd.DataFrame,
    schema: ObsSchema,
    study_mode: bool,
    detection_threshold: float,
    condition_reference: str | None,
    minimum_cells: int,
    minimum_donors: int,
    minimum_studies: int,
    minimum_repeated_contexts: int,
    observational_unit: str,
    minimum_cell_support: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize estimability without hiding unavailable requested analyses."""
    analyses = [
        ("adjusted_cell_type", schema.cell_type_key),
        ("condition_by_cell_type", schema.condition_key),
        ("age_by_cell_type", schema.age_key),
        ("paired_tissue", schema.tissue_key),
        ("assay_batch_effects", schema.assay_key),
        ("variance_decomposition", None),
    ]
    records = [
        _availability_record(
            unit_frame,
            contrasts=contrasts,
            variance_components=variance_components,
            diagnostics=diagnostics,
            analysis=analysis,
            predictor_key=predictor_key,
            minimum_cells=minimum_cells,
            minimum_donors=minimum_donors,
            minimum_studies=minimum_studies,
            minimum_repeated_contexts=minimum_repeated_contexts,
            detection_threshold=detection_threshold,
            condition_reference=condition_reference,
            minimum_cell_support=minimum_cell_support,
            observational_unit=observational_unit,
        )
        for analysis, predictor_key in analyses
    ]
    successful_study_features = 0
    if study_mode and {"status", "group_key", "feature_id"}.issubset(
        diagnostics.columns
    ):
        successful_study_features = int(
            diagnostics.loc[
                diagnostics["status"].astype(str).eq("ok")
                & diagnostics["group_key"].astype(str).eq("study"),
                "feature_id",
            ].nunique()
        )
    study_estimated = study_mode and successful_study_features > 0
    if study_estimated:
        study_reason = ""
    elif study_mode:
        study_reason = "no_successful_study_structured_fit"
    else:
        study_reason = _study_unavailable_reason(
            unit_frame,
            schema=schema,
            minimum_studies=minimum_studies,
        )
    records.append(
        {
            "analysis": "multi_study_structure",
            "estimable": study_estimated,
            "status": "estimated" if study_estimated else "not_estimable",
            "reason": study_reason,
            "n_estimable_results": float(successful_study_features),
            "minimum_cells": minimum_cells,
            "minimum_donors": minimum_donors,
            "minimum_studies": minimum_studies,
            "minimum_repeated_contexts": minimum_repeated_contexts,
            "detection_threshold": detection_threshold,
            "configured_condition_reference": condition_reference,
            "observational_unit": observational_unit,
            "independent_unit": "donor",
        }
    )
    return pd.DataFrame.from_records(records)


def _availability_record(
    unit_frame: pd.DataFrame,
    *,
    contrasts: pd.DataFrame,
    variance_components: pd.DataFrame,
    diagnostics: pd.DataFrame,
    analysis: str,
    predictor_key: str | None,
    minimum_cells: int,
    minimum_donors: int,
    minimum_studies: int,
    minimum_repeated_contexts: int,
    detection_threshold: float,
    condition_reference: str | None,
    minimum_cell_support: pd.DataFrame,
    observational_unit: str,
) -> dict[str, object]:
    """Return one requested analysis's dataset-level support status."""
    if analysis == "variance_decomposition":
        scoped = (
            variance_components.loc[
                variance_components["analysis"] == "adjusted_context"
            ]
            if "analysis" in variance_components
            else variance_components
        )
    elif "estimand" in contrasts:
        scoped = contrasts.loc[contrasts["estimand"] == analysis]
    else:
        scoped = pd.DataFrame()
    if "estimable" in scoped:
        estimable = scoped["estimable"].fillna(False).astype(bool)
    elif "status" in scoped:
        estimable = scoped["status"].astype(str).eq("ok")
    else:
        estimable = pd.Series(dtype=bool)
    n_estimable = int(estimable.sum())
    if n_estimable:
        status = "estimated" if bool(estimable.all()) else "partially_estimable"
        reason = ""
    else:
        status = "not_estimable"
        reason = _unavailable_reason(
            unit_frame,
            predictor_key=predictor_key,
            scoped=scoped,
            diagnostics=diagnostics,
            minimum_cell_support=minimum_cell_support,
        )
    return {
        "analysis": analysis,
        "estimable": bool(n_estimable),
        "status": status,
        "reason": reason,
        "n_estimable_results": float(n_estimable),
        "minimum_cells": minimum_cells,
        "minimum_donors": minimum_donors,
        "minimum_studies": minimum_studies,
        "minimum_repeated_contexts": minimum_repeated_contexts,
        "detection_threshold": detection_threshold,
        "configured_condition_reference": condition_reference,
        "observational_unit": observational_unit,
        "independent_unit": "donor",
    }


def _unavailable_reason(
    unit_frame: pd.DataFrame,
    *,
    predictor_key: str | None,
    scoped: pd.DataFrame,
    diagnostics: pd.DataFrame,
    minimum_cell_support: pd.DataFrame,
) -> str:
    """Return the most actionable reason a requested analysis is unavailable."""
    if (
        not minimum_cell_support.empty
        and minimum_cell_support["n_aggregates_after_minimum_cells"].eq(0).all()
    ):
        return "all_aggregates_below_minimum_cells"
    if predictor_key is not None:
        if predictor_key not in unit_frame:
            return f"missing_predictor:{predictor_key}"
        values = unit_frame[predictor_key].dropna()
        if values.empty:
            return f"missing_predictor_values:{predictor_key}"
        if values.nunique() < 2:
            return f"single_predictor_level:{predictor_key}"
    if not scoped.empty and "reason" in scoped:
        reasons = scoped["reason"].dropna().astype(str)
        reasons = reasons[reasons.ne("")]
        if not reasons.empty:
            return reasons.iloc[0]
    if not diagnostics.empty and "reason" in diagnostics:
        reasons = diagnostics["reason"].dropna().astype(str)
        reasons = reasons[reasons.ne("")]
        if not reasons.empty:
            return reasons.iloc[0]
    return "no_estimable_model_results"


def _study_unavailable_reason(
    unit_frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    minimum_studies: int,
) -> str:
    """Return why a study random intercept cannot be fitted."""
    if schema.study_key not in unit_frame:
        return f"missing_study_column:{schema.study_key}"
    n_studies = unit_frame[schema.study_key].dropna().nunique()
    return f"too_few_studies:{n_studies}<{minimum_studies}"
