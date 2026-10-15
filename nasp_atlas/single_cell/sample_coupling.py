"""Separate sample-shared from within-sample coupling of unit features."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.single_cell.associations import MixedModelContrast
from nasp_atlas.single_cell.associations import MixedModelInferenceResult
from nasp_atlas.single_cell.associations import MixedModelSpec
from nasp_atlas.single_cell.associations import mixed_model_inference


@dataclass(frozen=True)
class SampleCouplingResults:
    """Shared-sample variance and within/between-sample coupling.

    Attributes:
      variance_partition: Independent-unit, sample and residual variance of
        every context-centered feature.
      coupling: Within-sample, between-sample and contextual slopes for each
        requested predictor-response pair, including unfitted pairs.
      diagnostics: Context-centering support and per-model fit diagnostics.
    """

    variance_partition: pd.DataFrame
    coupling: pd.DataFrame
    diagnostics: pd.DataFrame


def sample_coupling(
    unit_frame: pd.DataFrame,
    *,
    predictors: Sequence[str],
    responses: Sequence[str],
    independent_unit_key: str,
    sample_key: str,
    context_keys: Sequence[str],
    excluded_pairs: Collection[tuple[str, str]] = (),
    minimum_independent_units: int = 3,
) -> SampleCouplingResults:
    """Separate sample-shared from within-sample coupling of unit features.

    Rows are observational units, such as donor-tissue-cell-type aggregates,
    nested in samples (donor-tissue) nested in independent units (donors),
    with one `feature_value` per unit and feature. Each feature is first
    centered on its context mean (for example tissue, cell type and assay)
    over contexts observed in at least `minimum_independent_units`
    independent units, so fixed context baselines drive no estimate.

    The variance partition fits an intercept-only mixed model per feature
    with an independent-unit random intercept and, when any independent unit
    has several samples, a nested sample variance component. Large sample and
    independent-unit fractions mean that different contexts of one sample
    move together, as expected for a shared paracrine, systemic, processing
    or ambient signal. The residual also contains finite-cell sampling noise.

    Each predictor-response pair uses a within-between hybrid model with the
    same random effects: `response ~ within + sample_mean`, where
    `sample_mean` is the predictor's mean over the pair's complete units in
    the sample and `within` is the deviation from it. The within-sample slope
    compares contexts of one sample, removing additive sample-level
    confounding; it does not remove reverse causation, shared within-unit
    measurement or context-specific responses to a shared signal. The
    between-sample slope compares sample means and absorbs shared exposures
    and confounders. The contextual slope, from `response ~ centered +
    sample_mean`, estimates their difference. Each estimand is BH-adjusted
    across all fitted pairs. Every estimate is an observational association.
    Inputs are not modified.

    Args:
      unit_frame: Long frame with `feature_type`, `feature_id`,
        `feature_label`, `feature_value` and every key column.
      predictors: Feature labels used as predictors, such as sensor genes.
      responses: Feature labels used as responses, such as output modules.
      independent_unit_key: Column identifying independent units (donors).
      sample_key: Column identifying samples nested in independent units.
      context_keys: Columns whose joint levels define centering contexts.
      excluded_pairs: (predictor, response) pairs reported without fitting,
        for example a gene that belongs to the response module. A feature
        paired with itself is always excluded.
      minimum_independent_units: Minimum independent units supporting a
        context, a model, and within-sample variation of a pair.

    Returns:
      Variance partition, coupling estimates and diagnostics.

    Raises:
      KeyError: If a required column is absent.
      ValueError: If identifiers are missing, labels are ambiguous, units are
        duplicated, samples span independent units, or the threshold is
        below 3.
    """
    if minimum_independent_units < 3:
        raise ValueError("minimum_independent_units must be at least 3")

    unit_keys = list(
        dict.fromkeys([independent_unit_key, sample_key, *context_keys])
    )
    _validate_unit_frame(
        unit_frame,
        unit_keys=unit_keys,
        independent_unit_key=independent_unit_key,
        sample_key=sample_key,
    )

    centered, support = _context_centered_features(
        unit_frame,
        unit_keys=unit_keys,
        independent_unit_key=independent_unit_key,
        context_keys=context_keys,
        minimum_independent_units=minimum_independent_units,
    )
    samples_per_unit = centered.groupby(independent_unit_key, observed=True)[
        sample_key
    ].nunique()
    variance_components = (sample_key,) if samples_per_unit.gt(1).any() else ()

    partition = mixed_model_inference(
        centered,
        spec=MixedModelSpec(
            analysis="sample_variance_partition",
            response_key="centered_value",
            group_key=independent_unit_key,
            variance_component_keys=variance_components,
            minimum_groups=minimum_independent_units,
            minimum_independent_units=minimum_independent_units,
        ),
    )

    pairs, pair_support = _pair_frame(
        centered,
        predictors=predictors,
        responses=responses,
        excluded_pairs=excluded_pairs,
        unit_keys=unit_keys,
        independent_unit_key=independent_unit_key,
        sample_key=sample_key,
        minimum_independent_units=minimum_independent_units,
    )

    within_between, contextual = _fit_pair_models(
        pairs,
        independent_unit_key=independent_unit_key,
        variance_components=variance_components,
        minimum_independent_units=minimum_independent_units,
    )

    # Features without a supported context keep an explicit diagnostic row.
    partition_diagnostics = partition.diagnostics.merge(
        support,
        on=["feature_type", "feature_id", "feature_label"],
        how="right",
        validate="one_to_one",
    )
    unfitted = partition_diagnostics["analysis"].isna()
    partition_diagnostics.loc[unfitted, "analysis"] = (
        "sample_variance_partition"
    )
    partition_diagnostics.loc[unfitted, ["status", "reason"]] = (
        "no_supported_context"
    )

    diagnostics = pd.concat(
        [
            partition_diagnostics,
            within_between.diagnostics,
            contextual.diagnostics,
        ],
        ignore_index=True,
    )

    return SampleCouplingResults(
        variance_partition=partition.variance_components,
        coupling=_coupling_table(
            [within_between.contrasts, contextual.contrasts],
            pair_support,
        ),
        diagnostics=diagnostics,
    )


def sensor_output_pairs(
    units: pd.DataFrame,
    output_modules: Sequence[str],
) -> tuple[list[str], list[str], set[tuple[str, str]]]:
    """Return gene predictors, scored output modules and in-module pairs.

    Predictors are the gene-expression features of `units`; responses are
    the `output_modules` present as module scores. A gene that belongs to an
    output module is excluded from that pair, because it would be regressed
    on a score that contains it.
    """
    sensors = sorted(
        units.loc[
            units["feature_type"].eq("gene_expression"), "feature_label"
        ].unique()
    )
    modules = set(
        units.loc[units["feature_type"].eq("module_score"), "feature_label"]
    )
    responses = [module for module in output_modules if module in modules]
    excluded = {
        (sensor, module)
        for module in responses
        for sensor in set(GeneModules.genes(module)).intersection(sensors)
    }
    return sensors, responses, excluded


def _validate_unit_frame(
    unit_frame: pd.DataFrame,
    *,
    unit_keys: Sequence[str],
    independent_unit_key: str,
    sample_key: str,
) -> None:
    """Reject frames whose unit, sample or feature identity is ambiguous."""
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    required = [*feature_keys, "feature_value", *unit_keys]
    if missing := [key for key in required if key not in unit_frame]:
        raise KeyError(f"sample coupling frame missing columns: {missing}")

    if unit_frame[list(unit_keys)].isna().any(axis=None):
        raise ValueError(
            f"Unit identifiers {list(unit_keys)} must not be missing"
        )

    identities = unit_frame[feature_keys].drop_duplicates()
    if identities.feature_label.duplicated().any():
        ambiguous = identities.loc[
            identities.feature_label.duplicated(), "feature_label"
        ].tolist()
        raise ValueError(
            f"Feature labels map to several features: {ambiguous[:3]}"
        )

    duplicated = unit_frame.duplicated([*unit_keys, "feature_label"])
    if duplicated.any():
        example = unit_frame.loc[duplicated, list(unit_keys)].iloc[0]
        raise ValueError(
            "Require one row per unit and feature; duplicated unit "
            f"{example.to_dict()}"
        )

    owners = unit_frame.groupby(sample_key, observed=True)[
        independent_unit_key
    ].nunique()
    if owners.gt(1).any():
        raise ValueError(
            f"Samples must be nested in one {independent_unit_key}; "
            f"{owners.gt(1).sum()} samples span several"
        )


def _context_centered_features(
    unit_frame: pd.DataFrame,
    *,
    unit_keys: Sequence[str],
    independent_unit_key: str,
    context_keys: Sequence[str],
    minimum_independent_units: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Center finite values on supported context means and count exclusions."""
    feature_keys = ["feature_type", "feature_id", "feature_label"]
    frame = unit_frame[[*feature_keys, "feature_value", *unit_keys]].copy()
    frame["feature_value"] = pd.to_numeric(
        frame["feature_value"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    finite = frame["feature_value"].notna()

    context_groups = [*feature_keys, *context_keys]
    context_support = (
        frame.loc[finite]
        .groupby(context_groups, observed=True)[independent_unit_key]
        .transform("nunique")
    )
    supported = context_support.reindex(frame.index).ge(
        minimum_independent_units
    )

    centered = frame.loc[supported].copy()
    centered["centered_value"] = centered["feature_value"] - centered.groupby(
        context_groups, observed=True
    )["feature_value"].transform("mean")

    support = (
        frame.assign(
            _nonfinite=~finite,
            _unsupported=finite & ~supported,
            _centered=supported,
        )
        .groupby(feature_keys, observed=True)
        .agg(
            n_units_input=("feature_value", "size"),
            n_units_nonfinite=("_nonfinite", "sum"),
            n_units_unsupported_context=("_unsupported", "sum"),
            n_units_centered=("_centered", "sum"),
        )
    )
    support["n_contexts_centered"] = (
        centered.drop_duplicates(context_groups)
        .groupby(feature_keys, observed=True)
        .size()
        .reindex(support.index, fill_value=0)
    )

    return centered, support.reset_index()


def _pair_frame(
    centered: pd.DataFrame,
    *,
    predictors: Sequence[str],
    responses: Sequence[str],
    excluded_pairs: Collection[tuple[str, str]],
    unit_keys: Sequence[str],
    independent_unit_key: str,
    sample_key: str,
    minimum_independent_units: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each predictor into sample means and within-sample deviations.

    The sample mean uses only the pair's complete units, so the within and
    between terms partition exactly the predictor values being modeled.
    """
    wide = centered.pivot(
        index=list(unit_keys),
        columns="feature_label",
        values="centered_value",
    )
    excluded = set(excluded_pairs)

    records: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for response in dict.fromkeys(responses):
        for predictor in dict.fromkeys(predictors):
            record, frame = _pair_units(
                wide,
                predictor=predictor,
                response=response,
                excluded=(predictor, response) in excluded,
                independent_unit_key=independent_unit_key,
                sample_key=sample_key,
                minimum_independent_units=minimum_independent_units,
            )
            records.append(record)
            if frame is not None:
                frames.append(frame)

    pair_support = pd.DataFrame.from_records(
        records,
        columns=[
            "pair_id",
            "predictor",
            "response",
            "status",
            "reason",
            "n_units",
            "n_samples",
            "n_independent_units",
            "n_samples_with_within_variation",
            "n_independent_units_with_within_variation",
        ],
    )
    if pair_support.pair_id.duplicated().any():
        raise ValueError(
            "Predictor and response labels produce duplicate pair identifiers"
        )

    pairs = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "feature_type",
                "feature_id",
                "feature_label",
                independent_unit_key,
                sample_key,
                "response_centered",
                "predictor_centered",
                "predictor_sample_mean",
                "predictor_within_sample",
            ]
        )
    )
    return pairs, pair_support


def _pair_units(
    wide: pd.DataFrame,
    *,
    predictor: str,
    response: str,
    excluded: bool,
    independent_unit_key: str,
    sample_key: str,
    minimum_independent_units: int,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    """Return one pair's support record and, if it can be fitted, its rows.

    A pair is excluded (a self pair or a declared exclusion), unavailable (a
    feature without a supported context), short of within-sample support,
    or fitted; only a fitted pair has model rows.
    """
    record: dict[str, object] = {
        "pair_id": f"{predictor}::{response}",
        "predictor": predictor,
        "response": response,
        "status": "fitted",
        "reason": "",
    }
    if predictor == response or excluded:
        record["status"] = "excluded"
        record["reason"] = (
            "self_pair" if predictor == response else "excluded_pair"
        )
        return record, None
    if predictor not in wide or response not in wide:
        record["status"] = "unavailable"
        record["reason"] = "feature_without_supported_context"
        return record, None

    paired = wide[[predictor, response]].dropna().reset_index()
    sample_sizes = paired.groupby(
        [independent_unit_key, sample_key], observed=True
    ).size()
    repeated = sample_sizes.loc[sample_sizes.ge(2)]
    n_within_units = repeated.index.get_level_values(
        independent_unit_key
    ).nunique()
    record.update(
        n_units=len(paired),
        n_samples=len(sample_sizes),
        n_independent_units=paired[independent_unit_key].nunique(),
        n_samples_with_within_variation=len(repeated),
        n_independent_units_with_within_variation=n_within_units,
    )
    if n_within_units < minimum_independent_units:
        record["status"] = "insufficient_within_sample_support"
        record["reason"] = (
            "independent_units_with_repeated_sample_contexts_below_"
            f"{minimum_independent_units}"
        )
        return record, None

    sample_mean = paired.groupby(sample_key, observed=True)[
        predictor
    ].transform("mean")
    frame = pd.DataFrame(
        {
            "feature_type": "predictor_response_pair",
            "feature_id": record["pair_id"],
            "feature_label": record["pair_id"],
            independent_unit_key: paired[independent_unit_key],
            sample_key: paired[sample_key],
            "response_centered": paired[response],
            "predictor_centered": paired[predictor],
            "predictor_sample_mean": sample_mean,
            "predictor_within_sample": paired[predictor] - sample_mean,
        }
    )
    return record, frame


def _fit_pair_models(
    pairs: pd.DataFrame,
    *,
    independent_unit_key: str,
    variance_components: tuple[str, ...],
    minimum_independent_units: int,
) -> tuple[MixedModelInferenceResult, MixedModelInferenceResult]:
    """Fit the within-between and contextual parameterizations of each pair.

    Both parameterizations describe the same model: the contextual slope on
    the sample mean equals the between-sample minus the within-sample slope.
    """
    effect_scale = "centered_response_change_per_centered_predictor_unit"
    spec = MixedModelSpec(
        analysis="within_between_sample",
        response_key="response_centered",
        group_key=independent_unit_key,
        continuous_keys=("predictor_within_sample", "predictor_sample_mean"),
        variance_component_keys=variance_components,
        minimum_groups=minimum_independent_units,
        minimum_independent_units=minimum_independent_units,
        minimum_focal_units=minimum_independent_units,
    )
    within_between = mixed_model_inference(
        pairs,
        spec=spec,
        contrasts=(
            MixedModelContrast(
                mode="continuous_slope",
                predictor_key="predictor_within_sample",
                estimand="within_sample_slope",
                effect_scale=effect_scale,
            ),
            MixedModelContrast(
                mode="continuous_slope",
                predictor_key="predictor_sample_mean",
                estimand="between_sample_slope",
                effect_scale=effect_scale,
            ),
        ),
    )

    contextual = mixed_model_inference(
        pairs,
        spec=replace(
            spec,
            analysis="sample_contextual",
            continuous_keys=("predictor_centered", "predictor_sample_mean"),
        ),
        contrasts=(
            MixedModelContrast(
                mode="continuous_slope",
                predictor_key="predictor_sample_mean",
                estimand="sample_contextual_slope",
                effect_scale=effect_scale,
            ),
        ),
    )
    return within_between, contextual


def _coupling_table(
    contrast_tables: Sequence[pd.DataFrame],
    pair_support: pd.DataFrame,
) -> pd.DataFrame:
    """Join fitted slopes to pair support and keep unfitted pairs explicit."""
    estimands = (
        "within_sample_slope",
        "between_sample_slope",
        "sample_contextual_slope",
    )
    estimates = pd.concat(contrast_tables, ignore_index=True).rename(
        columns={"feature_id": "pair_id"}
    )
    complete = pair_support.merge(
        pd.DataFrame({"estimand": estimands}), how="cross"
    ).merge(
        estimates[
            [
                "pair_id",
                "analysis",
                "estimand",
                "effect_scale",
                "estimate",
                "standard_error",
                "ci_low",
                "ci_high",
                "statistic",
                "pvalue",
                "pvalue_fdr",
                "estimable",
                "status",
                "reason",
            ]
        ],
        on=["pair_id", "estimand"],
        how="left",
        validate="one_to_one",
        suffixes=("_pair", ""),
    )
    unfitted = complete["status"].isna()
    complete.loc[unfitted, "status"] = complete.loc[unfitted, "status_pair"]
    complete.loc[unfitted, "reason"] = complete.loc[unfitted, "reason_pair"]
    complete["estimable"] = (
        complete["estimable"].astype("boolean").fillna(False)
    )

    columns = [
        "pair_id",
        "predictor",
        "response",
        "analysis",
        "estimand",
        "effect_scale",
        "estimate",
        "standard_error",
        "ci_low",
        "ci_high",
        "statistic",
        "pvalue",
        "pvalue_fdr",
        "n_units",
        "n_samples",
        "n_independent_units",
        "n_samples_with_within_variation",
        "n_independent_units_with_within_variation",
        "estimable",
        "status",
        "reason",
    ]
    return complete.reindex(columns=columns)
