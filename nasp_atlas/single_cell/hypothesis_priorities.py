"""Prioritize interpretable hypotheses from relative NASP evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd


MechanisticEdgeSpec: TypeAlias = tuple[str, str, str]
_HypothesisSpec: TypeAlias = tuple[
    str,
    str,
    pd.Series,
    pd.Series,
    str,
    str,
    str,
]

__all__ = [
    "MechanisticEdgeSpec",
    "expected_module_coupling_report",
    "rank_nasp_hypotheses",
]


def expected_module_coupling_report(
    coupling: pd.DataFrame,
    *,
    edge_specs: Sequence[MechanisticEdgeSpec],
    module_a_column: str = "module_a",
    module_b_column: str = "module_b",
    source_output_column: str = "source_module",
    target_output_column: str = "target_module",
    edge_label_output_column: str = "mechanistic_edge",
) -> pd.DataFrame:
    """Annotate coupling results for caller-declared mechanistic edges.

    Module correlations are symmetric, whereas a biological hypothesis often
    has a source and target. Each edge specification supplies that expected
    direction without treating the correlation as causal. A pair is matched
    in either orientation, so sorted all-pairs correlation output can be used
    directly.

    All input columns are retained unchanged. In particular, separate raw and
    within-context rows, raw p-values, FDR values, and marker-overlap fields
    remain auditable in the result. The directed source and target columns are
    annotations and do not replace the original pair columns.

    Args:
      coupling: All-pairs module-coupling table.
      edge_specs: Ordered `(source, target, edge label)` specifications.
      module_a_column: First module-pair column in `coupling`.
      module_b_column: Second module-pair column in `coupling`.
      source_output_column: Output column for the expected source module.
      target_output_column: Output column for the expected target module.
      edge_label_output_column: Output column for the mechanistic edge label.

    Returns:
      Matching coupling rows in edge-specification order, annotated with the
      expected direction and label. Edge specifications absent from
      `coupling` do not create placeholder rows.
    """
    _require_columns(
        coupling,
        [module_a_column, module_b_column],
        frame_name="coupling",
    )
    annotation_columns = [
        source_output_column,
        target_output_column,
        edge_label_output_column,
    ]
    if len(annotation_columns) != len(set(annotation_columns)):
        raise ValueError("coupling annotation column names must be distinct")
    if collisions := [
        column for column in annotation_columns if column in coupling.columns
    ]:
        raise ValueError(
            "coupling already contains requested annotation columns: "
            f"{collisions}"
        )
    if coupling[[module_a_column, module_b_column]].isna().any(axis=None):
        raise ValueError(
            "coupling module-pair columns cannot contain missing values"
        )

    validated_specs = _validate_edge_specs(edge_specs)
    output_columns = [*annotation_columns, *coupling.columns.tolist()]
    if coupling.empty or not validated_specs:
        return pd.DataFrame(columns=output_columns)

    annotated: list[pd.DataFrame] = []
    module_a = coupling[module_a_column]
    module_b = coupling[module_b_column]
    for edge_order, (source, target, edge_label) in enumerate(validated_specs):
        matches = ((module_a == source) & (module_b == target)) | (
            (module_a == target) & (module_b == source)
        )
        if not matches.any():
            continue
        edge_rows = coupling.loc[matches].copy()
        edge_rows.insert(0, edge_label_output_column, edge_label)
        edge_rows.insert(0, target_output_column, target)
        edge_rows.insert(0, source_output_column, source)
        edge_rows["__edge_spec_order"] = edge_order
        edge_rows["__coupling_row_order"] = np.flatnonzero(matches.to_numpy())
        annotated.append(edge_rows)

    if not annotated:
        return pd.DataFrame(columns=output_columns)
    result = pd.concat(annotated, axis="index", ignore_index=True)
    result = result.sort_values(
        ["__edge_spec_order", "__coupling_row_order"],
        kind="stable",
    )
    return result.drop(
        columns=["__edge_spec_order", "__coupling_row_order"]
    ).reset_index(drop=True)


def rank_nasp_hypotheses(
    profiles: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    competence_column: str = "relative_competence",
    output_column: str = "relative_output",
    restriction_column: str = "relative_restriction",
    feedback_column: str = "relative_feedback",
    post_column: str = "relative_post",
    unit_count_column: str | None = "n_units",
    donor_count_column: str | None = "n_donors",
    min_units: int | None = 3,
    min_donors: int | None = 2,
    min_priority_score: float = 0.0,
    max_per_hypothesis: int | None = None,
) -> pd.DataFrame:
    """Rank testable NASP hypotheses from relative context profiles.

    The input must contain one row per caller-declared context. Relative axes
    are expected on the zero-to-one scale emitted by
    `relative_nasp_evidence_profiles`. The function expands each eligible
    context into a long table of non-causal evidence patterns:

    * active-like uses joint competence and output support;
    * responsive-like uses positive output-minus-competence contrast, weighted
      by output;
    * restricted/buffered uses positive restriction-minus-output contrast,
      weighted by the mean of restriction and competence when competence is
      observed;
    * feedback-dominant uses positive feedback-minus-output contrast, weighted
      by feedback; and
    * post-without-NASP uses positive post-minus-maximum-NASP contrast,
      weighted by the post-NASP axis.

    Positive priority scores are ranked independently within each hypothesis,
    within each context, and across the full report. Count thresholds filter
    under-supported contexts but do not otherwise reward large atlases. A
    missing competence value is allowed only for restricted/buffered ranking,
    where restriction alone supplies the weighting term.

    Args:
      profiles: Relative NASP profiles with one row per context.
      context_columns: Columns that jointly identify a ranked context.
      competence_column: Relative sensing-competence axis column.
      output_column: Relative active-output axis column.
      restriction_column: Relative restriction/checkpoint axis column.
      feedback_column: Relative induced-feedback axis column.
      post_column: Relative post-NASP phenotype axis column.
      unit_count_column: Column holding independent-unit counts, or None to
        disable the unit-count filter and omit that field.
      donor_count_column: Column holding donor counts, or None to disable the
        donor-count filter and omit that field.
      min_units: Minimum independent units, or None to disable this filter.
      min_donors: Minimum donors, or None to disable this filter.
      min_priority_score: Inclusive minimum positive priority score.
      max_per_hypothesis: Optional maximum contexts retained per hypothesis.

    Returns:
      A long ranked hypothesis table with standardized axis values, contrasts,
      cautious interpretations, and experimental follow-up labels.
    """
    axis_columns = {
        "competence": competence_column,
        "output": output_column,
        "restriction": restriction_column,
        "feedback": feedback_column,
        "post": post_column,
    }
    count_columns = _validate_hypothesis_inputs(
        profiles,
        context_columns=context_columns,
        axis_columns=axis_columns,
        unit_count_column=unit_count_column,
        donor_count_column=donor_count_column,
        min_units=min_units,
        min_donors=min_donors,
        min_priority_score=min_priority_score,
        max_per_hypothesis=max_per_hypothesis,
    )
    base_columns = list(dict.fromkeys([*context_columns, *count_columns]))
    output_columns = [
        *base_columns,
        "hypothesis",
        "priority_score",
        "rank_within_hypothesis",
        "rank_within_context",
        "overall_priority_rank",
        "contrast_name",
        "contrast_value",
        "priority_basis",
        "competence",
        "output",
        "restriction",
        "feedback",
        "post",
        "interpretation",
        "experimental_follow_up",
    ]
    if profiles.empty:
        return pd.DataFrame(columns=output_columns)

    base = _eligible_hypothesis_profiles(
        profiles,
        context_columns=context_columns,
        axis_columns=axis_columns,
        count_columns=count_columns,
        unit_count_column=unit_count_column,
        donor_count_column=donor_count_column,
        min_units=min_units,
        min_donors=min_donors,
    )
    if base.empty:
        return pd.DataFrame(columns=output_columns)

    state_specs = _hypothesis_specs(base)
    result = _score_hypothesis_states(
        base,
        state_specs=state_specs,
        min_priority_score=min_priority_score,
        max_per_hypothesis=max_per_hypothesis,
    )
    if result.empty:
        return pd.DataFrame(columns=output_columns)

    return _rank_hypothesis_states(
        result,
        context_columns=context_columns,
        state_specs=state_specs,
        output_columns=output_columns,
    )


def _validate_hypothesis_inputs(
    profiles: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    axis_columns: dict[str, str],
    unit_count_column: str | None,
    donor_count_column: str | None,
    min_units: int | None,
    min_donors: int | None,
    min_priority_score: float,
    max_per_hypothesis: int | None,
) -> list[str]:
    """Validate context identity, support thresholds, and required columns."""
    if not context_columns:
        raise ValueError("context_columns must identify each ranked context")
    if len(context_columns) != len(set(context_columns)):
        raise ValueError("context_columns must be unique")

    _validate_optional_threshold(min_units, name="min_units")
    _validate_optional_threshold(min_donors, name="min_donors")
    if not 0.0 <= min_priority_score <= 1.0:
        raise ValueError("min_priority_score must be between zero and one")
    if max_per_hypothesis is not None and max_per_hypothesis < 1:
        raise ValueError("max_per_hypothesis must be at least 1")
    if min_units is not None and unit_count_column is None:
        raise ValueError(
            "unit_count_column is required when min_units is enabled"
        )
    if min_donors is not None and donor_count_column is None:
        raise ValueError(
            "donor_count_column is required when min_donors is enabled"
        )

    count_columns = [
        column
        for column in (unit_count_column, donor_count_column)
        if column is not None
    ]
    required_columns = list(
        dict.fromkeys(
            [
                *context_columns,
                *axis_columns.values(),
                *count_columns,
            ]
        )
    )
    _require_columns(profiles, required_columns, frame_name="profiles")
    if not profiles.columns.is_unique:
        raise ValueError("profiles must have unique column names")
    if profiles.duplicated(list(context_columns), keep=False).any():
        raise ValueError(
            "profiles must have one row per context_columns combination"
        )

    return count_columns


def _eligible_hypothesis_profiles(
    profiles: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    axis_columns: dict[str, str],
    count_columns: Sequence[str],
    unit_count_column: str | None,
    donor_count_column: str | None,
    min_units: int | None,
    min_donors: int | None,
) -> pd.DataFrame:
    """Return supported contexts with normalized evidence-axis columns."""
    working = profiles.reset_index(drop=True)
    axes = {
        axis_name: _relative_axis(working, column)
        for axis_name, column in axis_columns.items()
    }
    counts = {
        column: _count_values(working, column) for column in count_columns
    }

    eligible = pd.Series(True, index=working.index, dtype=bool)
    if min_units is not None and unit_count_column is not None:
        eligible &= counts[unit_count_column].ge(min_units)
    if min_donors is not None and donor_count_column is not None:
        eligible &= counts[donor_count_column].ge(min_donors)

    base_columns = list(dict.fromkeys([*context_columns, *count_columns]))
    base = working.loc[eligible, base_columns].copy()
    for axis_name, values in axes.items():
        base[axis_name] = values.loc[eligible]

    return base


def _hypothesis_specs(base: pd.DataFrame) -> list[_HypothesisSpec]:
    """Define hypothesis contrasts, scores, and cautious interpretations."""
    competence = base["competence"]
    output = base["output"]
    restriction = base["restriction"]
    feedback = base["feedback"]
    post = base["post"]

    joint_active = pd.concat([competence, output], axis="columns").min(
        axis="columns",
        skipna=False,
    )
    responsive_gap = output - competence
    restriction_gap = restriction - output
    feedback_gap = feedback - output
    nasp_evidence = pd.concat([competence, output], axis="columns").max(
        axis="columns",
        skipna=False,
    )
    post_gap = post - nasp_evidence
    restriction_weight = pd.concat(
        [restriction, competence],
        axis="columns",
    ).mean(axis="columns", skipna=True)

    return [
        (
            "active_like",
            "joint_competence_and_output",
            joint_active,
            joint_active,
            "min(competence, output)",
            (
                "Joint relative competence and output are compatible with an "
                "active-like transcriptomic state, but do not establish "
                "sensor engagement or pathway causality."
            ),
            "matched_sensor_or_adaptor_perturbation",
        ),
        (
            "responsive_like",
            "output_minus_competence",
            responsive_gap,
            responsive_gap.clip(lower=0.0) * output,
            "positive(output - competence) * output",
            (
                "Output exceeding measured competence is compatible with a "
                "paracrine response, sensor dropout, or unmeasured sensing; "
                "it is not evidence of a sender-receiver link."
            ),
            "co_culture_and_receptor_blockade",
        ),
        (
            "restricted_buffered",
            "restriction_minus_output",
            restriction_gap,
            restriction_gap.clip(lower=0.0) * restriction_weight,
            "positive(restriction - output) * mean(restriction, competence)",
            (
                "Restriction exceeding output, especially with competence, "
                "is compatible with buffering; expression alone does not "
                "demonstrate functional suppression."
            ),
            "restriction_factor_perturbation_and_ligand_challenge",
        ),
        (
            "feedback_dominant",
            "feedback_minus_output",
            feedback_gap,
            feedback_gap.clip(lower=0.0) * feedback,
            "positive(feedback - output) * feedback",
            (
                "Feedback exceeding output is compatible with dampened, "
                "recent, or resolving activation; a snapshot cannot resolve "
                "the temporal explanation."
            ),
            "ligand_time_course_with_feedback_perturbation",
        ),
        (
            "post_without_nasp",
            "post_minus_maximum_nasp_evidence",
            post_gap,
            post_gap.clip(lower=0.0) * post,
            "positive(post - max(competence, output)) * post",
            (
                "A post-NASP phenotype exceeding competence and output "
                "provides weak evidence for NASP causality; parallel or "
                "downstream inflammation remains plausible."
            ),
            "compare_nasp_and_parallel_pathway_perturbations",
        ),
    ]


def _score_hypothesis_states(
    base: pd.DataFrame,
    *,
    state_specs: Sequence[_HypothesisSpec],
    min_priority_score: float,
    max_per_hypothesis: int | None,
) -> pd.DataFrame:
    """Expand eligible contexts into supported hypothesis-score rows."""
    state_tables: list[pd.DataFrame] = []
    for (
        hypothesis,
        contrast_name,
        contrast,
        priority,
        priority_basis,
        interpretation,
        follow_up,
    ) in state_specs:
        state = base.copy()
        state["hypothesis"] = hypothesis
        state["contrast_name"] = contrast_name
        state["contrast_value"] = contrast
        state["priority_score"] = priority.clip(lower=0.0, upper=1.0)
        state["priority_basis"] = priority_basis
        state["interpretation"] = interpretation
        state["experimental_follow_up"] = follow_up

        supported = state["priority_score"].notna()
        supported &= state["priority_score"].gt(0.0)
        supported &= state["priority_score"].ge(min_priority_score)
        state = state.loc[supported]
        if max_per_hypothesis is not None:
            state = state.sort_values(
                "priority_score",
                ascending=False,
                kind="stable",
            ).head(max_per_hypothesis)

        state_tables.append(state)

    return pd.concat(state_tables, axis="index", ignore_index=True)


def _rank_hypothesis_states(
    result: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    state_specs: Sequence[_HypothesisSpec],
    output_columns: Sequence[str],
) -> pd.DataFrame:
    """Rank supported hypotheses within class, context, and full report."""
    result["rank_within_hypothesis"] = (
        result.groupby("hypothesis", observed=True)["priority_score"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    result["rank_within_context"] = (
        result.groupby(list(context_columns), observed=True, dropna=False)[
            "priority_score"
        ]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    result["overall_priority_rank"] = (
        result["priority_score"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    hypothesis_order = {
        state_spec[0]: order for order, state_spec in enumerate(state_specs)
    }
    result["__hypothesis_order"] = result["hypothesis"].map(hypothesis_order)
    result = result.sort_values(
        ["__hypothesis_order", "rank_within_hypothesis"],
        kind="stable",
    ).drop(columns="__hypothesis_order")

    return result.reindex(columns=list(output_columns)).reset_index(drop=True)


def _validate_edge_specs(
    edge_specs: Sequence[MechanisticEdgeSpec],
) -> list[MechanisticEdgeSpec]:
    """Validate edge specifications while preserving caller order."""
    validated: list[MechanisticEdgeSpec] = []
    seen: set[MechanisticEdgeSpec] = set()
    for index, edge_spec in enumerate(edge_specs):
        if len(edge_spec) != 3:
            raise ValueError(
                "each edge specification must contain source, target, and "
                f"label; item {index} has length {len(edge_spec)}"
            )
        source, target, edge_label = edge_spec
        if not all(
            isinstance(value, str) and value.strip()
            for value in (source, target, edge_label)
        ):
            raise ValueError(
                "edge specification values must be non-empty strings; "
                f"invalid item at index {index}"
            )
        if source == target:
            raise ValueError(
                f"edge specification cannot be a self-edge: {source!r}"
            )
        normalized = (source, target, edge_label)
        if normalized in seen:
            raise ValueError(f"duplicate edge specification: {normalized}")
        seen.add(normalized)
        validated.append(normalized)
    return validated


def _relative_axis(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a validated zero-to-one relative evidence axis."""
    if not pd.api.types.is_numeric_dtype(frame[column]):
        raise TypeError(f"relative evidence column must be numeric: {column}")
    values = frame[column].astype(float).replace([np.inf, -np.inf], np.nan)
    observed = values.dropna()
    outside = observed.lt(0.0) | observed.gt(1.0)
    if outside.any():
        example = float(observed.loc[outside].iloc[0])
        raise ValueError(
            f"relative evidence column {column!r} must be within [0, 1]; "
            f"found {example}"
        )
    return values


def _count_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return validated non-negative support counts."""
    if not pd.api.types.is_numeric_dtype(frame[column]):
        raise TypeError(f"support-count column must be numeric: {column}")
    values = frame[column].astype(float).replace([np.inf, -np.inf], np.nan)
    observed = values.dropna()
    if observed.lt(0.0).any():
        raise ValueError(f"support-count column cannot be negative: {column}")
    return values


def _validate_optional_threshold(value: int | None, *, name: str) -> None:
    """Validate an optional positive count threshold."""
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1 or None")


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    """Raise an actionable error when required frame columns are absent."""
    if missing := [column for column in columns if column not in frame.columns]:
        raise KeyError(f"{frame_name} missing required columns: {missing}")
