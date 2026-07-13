"""Analyze relationships among existing NASP module scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from nasp_compendium.types import GeneModule  # type: ignore[import]
from scipy import stats  # type: ignore[import]

from nasp_atlas.single_cell.associations.core import benjamini_hochberg


EvidenceRole: TypeAlias = Literal[
    "competence",
    "output",
    "restriction",
    "feedback",
    "post",
]
RoleAssignment: TypeAlias = EvidenceRole | Sequence[EvidenceRole]

_EVIDENCE_ROLES: tuple[EvidenceRole, ...] = (
    "competence",
    "output",
    "restriction",
    "feedback",
    "post",
)
_CORRELATION_COLUMNS = (
    "module_a",
    "module_b",
    "statistical_unit",
    "aggregation",
    "unit_columns",
    "strata_columns",
    "centered_within_strata",
    "n_units",
    "n_strata",
    "pearson_r",
    "pearson_pvalue",
    "pearson_fdr",
    "spearman_r",
    "spearman_pvalue",
    "spearman_fdr",
    "skipped",
    "skip_reason",
)
_OVERLAP_COLUMNS = (
    "module_a",
    "module_b",
    "gene_id_output",
    "n_signed_genes_a",
    "n_signed_genes_b",
    "n_positive_positive",
    "n_inverse_inverse",
    "n_positive_inverse",
    "n_inverse_positive",
    "n_concordant_shared",
    "n_cross_arm_shared",
    "n_shared_signed_genes",
    "n_union_signed_genes",
    "signed_gene_jaccard",
    "shared_sign_concordance",
    "signed_cosine",
    "concordant_shared_genes",
    "cross_arm_shared_genes",
    "n_context_shared",
    "context_shared_genes",
)

__all__ = [
    "EvidenceRole",
    "RoleAssignment",
    "module_gene_overlap",
    "pairwise_module_correlations",
    "relative_nasp_evidence_profiles",
]


def pairwise_module_correlations(
    unit_frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str],
    strata_columns: Sequence[str] = (),
    center_within_strata: bool = False,
    module_column: str = "feature_label",
    value_column: str = "feature_value",
    feature_type_column: str | None = "feature_type",
    module_feature_type: str = "module_score",
    module_pairs: Sequence[tuple[str, str]] | None = None,
    min_units: int = 3,
) -> pd.DataFrame:
    """Correlate module scores after aligning independent statistical units.

    The input is the long output of `aggregate_feature_frame`, or an
    equivalently shaped frame. Correlations are computed after pivoting on the
    caller-declared unit identity, so repeated cells cannot inflate the sample
    size. Optional within-stratum centering removes stratum mean differences
    before both Pearson and Spearman tests.

    Args:
      unit_frame: Long frame with one value per module and statistical unit.
      unit_columns: Columns that jointly identify an independent unit, such as
        donor and tissue for donor-tissue aggregation.
      strata_columns: Columns defining baselines to remove when centering, such
        as tissue or cell type. They are also included in row alignment.
      center_within_strata: Subtract each module's stratum-specific mean before
        computing correlations.
      module_column: Column identifying modules. The default uses the module
        labels emitted by the association feature builder.
      value_column: Numeric module-score value column.
      feature_type_column: Optional column used to retain module-score rows.
        Pass None when the input already contains modules only.
      module_feature_type: Value in `feature_type_column` denoting modules.
      module_pairs: Optional explicit module pairs to test. By default all
        unordered pairs are tested.
      min_units: Minimum complete independent units required for a test.

    Returns:
      One row per unordered module pair with complete-unit counts, Pearson and
      Spearman coefficients, raw p-values, and Benjamini-Hochberg FDR values.
      Constant or underpowered pairs are retained with NaN test statistics and
      an explicit skip reason.
    """
    if not unit_columns:
        raise ValueError(
            "unit_columns must identify independent statistical units"
        )
    if min_units < 3:
        raise ValueError("min_units must be at least 3")
    if center_within_strata and not strata_columns:
        raise ValueError(
            "strata_columns are required when center_within_strata is True"
        )

    identity_columns = list(dict.fromkeys([*unit_columns, *strata_columns]))
    scoped = _prepare_correlation_frame(
        unit_frame,
        identity_columns=identity_columns,
        unit_columns=unit_columns,
        strata_columns=strata_columns,
        center_within_strata=center_within_strata,
        module_column=module_column,
        value_column=value_column,
        feature_type_column=feature_type_column,
        module_feature_type=module_feature_type,
    )
    modules = sorted(scoped[module_column].unique().tolist())
    tested_pairs = _resolve_module_pairs(modules, module_pairs)
    if not tested_pairs:
        return pd.DataFrame(columns=_CORRELATION_COLUMNS)

    wide = scoped.pivot(
        index=identity_columns,
        columns=module_column,
        values=value_column,
    )
    context: dict[str, object] = {
        "statistical_unit": _single_label(unit_frame, "statistical_unit"),
        "aggregation": _single_label(unit_frame, "aggregation"),
        "unit_columns": ";".join(unit_columns),
        "strata_columns": ";".join(strata_columns),
        "centered_within_strata": center_within_strata,
    }
    records = [
        {
            **context,
            **_module_pair_correlation(
                wide,
                pair,
                strata_columns=strata_columns,
                min_units=min_units,
            ),
        }
        for pair in tested_pairs
    ]

    result = pd.DataFrame.from_records(records)
    result["pearson_fdr"] = benjamini_hochberg(
        result["pearson_pvalue"].to_numpy(dtype=float).tolist()
    )
    result["spearman_fdr"] = benjamini_hochberg(
        result["spearman_pvalue"].to_numpy(dtype=float).tolist()
    )
    return result.reindex(columns=_CORRELATION_COLUMNS)


def module_gene_overlap(modules: Sequence[GeneModule]) -> pd.DataFrame:
    """Annotate signed marker-gene overlap for every module pair.

    Positive-positive and inverse-inverse intersections are concordant;
    positive-inverse intersections are cross-arm. The Jaccard coefficient
    measures unsigned membership overlap among the scored arms, while signed
    cosine ranges from -1 (opposite directions) to 1 (matching directions).
    Context-dependent genes are reported separately and excluded from both
    signed metrics. Callers comparing catalog definitions should therefore
    pass complete symbol-space modules rather than dataset-filtered modules.

    Args:
      modules: Gene modules expressed in one common gene identifier space.

    Returns:
      One row per unordered module pair with directional intersections,
      concordant and cross-arm totals, Jaccard, shared-sign concordance, signed
      cosine, and stable semicolon-delimited overlap annotations.
    """
    _validate_overlap_modules(modules)
    records = [
        _module_overlap_record(module_a, module_b)
        for module_a, module_b in combinations(modules, 2)
    ]
    return pd.DataFrame.from_records(records, columns=_OVERLAP_COLUMNS)


def relative_nasp_evidence_profiles(
    scores: pd.DataFrame,
    *,
    module_roles: Mapping[str, RoleAssignment],
    reference_columns: Sequence[str] = (),
    high_percentile: float = 0.8,
    low_percentile: float = 0.2,
    min_modules_per_axis: int = 1,
) -> pd.DataFrame:
    """Build relative, non-causal NASP evidence axes from existing scores.

    Each mapped module is percentile-ranked independently, optionally within
    caller-declared reference strata. Ranks are then averaged by biological
    role. This scale harmonization supports mismatch screens without implying
    that raw scores from different gene sets are directly comparable.

    Composite flags deliberately require extreme evidence on both relevant
    axes: active means high competence and output, responsive means high output
    with low competence, and competent means high competence with low output.
    Restriction, feedback, and post-NASP flags describe relative expression
    states, not causal pathway activity. Singleton and fully tied reference
    groups receive a neutral percentile of 0.5.

    Args:
      scores: Wide frame with one row per cell or aggregated unit and one
        numeric column per already-computed module score.
      module_roles: Mapping from score column to one role or a sequence of
        roles. Assigning a module to multiple roles is allowed and explicit.
      reference_columns: Optional score-frame columns defining percentile
        reference groups, such as tissue and cell type.
      high_percentile: Inclusive percentile threshold for high evidence.
      low_percentile: Inclusive percentile threshold for low evidence.
      min_modules_per_axis: Minimum non-missing mapped module ranks required to
        report an axis for a row.

    Returns:
      A frame aligned to `scores.index` containing per-module percentile ranks,
      the five relative evidence axes, observed-module counts, signed mismatch
      gaps, and nullable relative-state flags.
    """
    _validate_profile_options(
        scores,
        module_roles=module_roles,
        high_percentile=high_percentile,
        low_percentile=low_percentile,
        min_modules_per_axis=min_modules_per_axis,
    )

    role_modules = _group_modules_by_role(module_roles)
    mapped_columns = list(module_roles)
    _require_columns(
        scores,
        [*reference_columns, *mapped_columns],
        frame_name="scores",
    )
    for column in mapped_columns:
        if not pd.api.types.is_numeric_dtype(scores[column]):
            raise TypeError(f"module score column must be numeric: {column}")

    percentiles, percentile_columns = _rank_module_scores(
        scores,
        mapped_columns=mapped_columns,
        reference_columns=reference_columns,
    )
    axes = _aggregate_evidence_axes(
        percentiles,
        role_modules=role_modules,
        percentile_columns=percentile_columns,
        min_modules_per_axis=min_modules_per_axis,
    )
    states = _relative_state_columns(
        axes,
        high_percentile=high_percentile,
        low_percentile=low_percentile,
    )

    return pd.concat([percentiles, axes, states], axis=1)


def _validate_overlap_modules(modules: Sequence[GeneModule]) -> None:
    """Require unique modules in one valid signed identifier space."""
    module_ids = [module.module_id for module in modules]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("module IDs must be unique for pairwise overlap")

    outputs = {module.gene_id_output for module in modules}
    if len(outputs) > 1:
        raise ValueError("modules must use the same gene identifier output")

    for module in modules:
        if overlap := set(module.positive_genes) & set(module.inverse_genes):
            raise ValueError(
                f"module {module.module_id!r} has genes in both signed arms: "
                f"{sorted(overlap)}"
            )


def _module_overlap_record(
    module_a: GeneModule,
    module_b: GeneModule,
) -> dict[str, object]:
    """Compute directional marker overlap for one module pair."""
    positive_a = set(module_a.positive_genes)
    inverse_a = set(module_a.inverse_genes)
    positive_b = set(module_b.positive_genes)
    inverse_b = set(module_b.inverse_genes)

    positive_positive = positive_a & positive_b
    inverse_inverse = inverse_a & inverse_b
    positive_inverse = positive_a & inverse_b
    inverse_positive = inverse_a & positive_b
    concordant = positive_positive | inverse_inverse
    cross_arm = positive_inverse | inverse_positive
    signed_a = positive_a | inverse_a
    signed_b = positive_b | inverse_b
    shared = signed_a & signed_b
    union = signed_a | signed_b
    context_shared = set(module_a.context_dependent_genes) & set(
        module_b.context_dependent_genes
    )

    dot_product = len(concordant) - len(cross_arm)
    denominator = np.sqrt(len(signed_a) * len(signed_b))

    return {
        "module_a": module_a.module_id,
        "module_b": module_b.module_id,
        "gene_id_output": module_a.gene_id_output,
        "n_signed_genes_a": len(signed_a),
        "n_signed_genes_b": len(signed_b),
        "n_positive_positive": len(positive_positive),
        "n_inverse_inverse": len(inverse_inverse),
        "n_positive_inverse": len(positive_inverse),
        "n_inverse_positive": len(inverse_positive),
        "n_concordant_shared": len(concordant),
        "n_cross_arm_shared": len(cross_arm),
        "n_shared_signed_genes": len(shared),
        "n_union_signed_genes": len(union),
        "signed_gene_jaccard": (
            len(shared) / len(union) if union else float("nan")
        ),
        "shared_sign_concordance": (
            dot_product / len(shared) if shared else float("nan")
        ),
        "signed_cosine": (
            dot_product / denominator if denominator else float("nan")
        ),
        "concordant_shared_genes": ";".join(sorted(concordant)),
        "cross_arm_shared_genes": ";".join(sorted(cross_arm)),
        "n_context_shared": len(context_shared),
        "context_shared_genes": ";".join(sorted(context_shared)),
    }


def _prepare_correlation_frame(
    unit_frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    unit_columns: Sequence[str],
    strata_columns: Sequence[str],
    center_within_strata: bool,
    module_column: str,
    value_column: str,
    feature_type_column: str | None,
    module_feature_type: str,
) -> pd.DataFrame:
    """Validate and normalize long module scores for pairwise tests."""
    required = [*identity_columns, module_column, value_column]
    if feature_type_column is not None:
        required.append(feature_type_column)
    _require_columns(unit_frame, required, frame_name="unit_frame")

    scoped = unit_frame
    if feature_type_column is not None:
        scoped = scoped[
            scoped[feature_type_column].astype(str) == module_feature_type
        ]
    scoped = scoped[[*identity_columns, module_column, value_column]].copy()

    if scoped[module_column].isna().any():
        raise ValueError(f"{module_column} contains missing module identifiers")
    if scoped[list(unit_columns)].isna().any(axis=None):
        raise ValueError("unit_columns contain missing unit identifiers")

    scoped[module_column] = scoped[module_column].astype(str)
    scoped[value_column] = pd.to_numeric(
        scoped[value_column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    duplicate_keys = [*identity_columns, module_column]
    duplicated = scoped.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        example = scoped.loc[duplicated, duplicate_keys].iloc[0].to_dict()
        raise ValueError(
            "unit_frame must have one row per module and statistical unit; "
            f"found a duplicate such as {example}"
        )

    if center_within_strata:
        means = scoped.groupby(
            [module_column, *strata_columns],
            observed=True,
            dropna=False,
        )[value_column].transform("mean")
        scoped[value_column] = scoped[value_column] - means

    return scoped


def _resolve_module_pairs(
    modules: Sequence[str],
    requested_pairs: Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Resolve unique unordered pairs and reject invalid requests."""
    if requested_pairs is None:
        return list(combinations(modules, 2))

    available = set(modules)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for module_a, module_b in requested_pairs:
        if module_a == module_b:
            raise ValueError("module_pairs cannot contain self-pairs")
        if module_a not in available or module_b not in available:
            raise KeyError(
                "module_pairs contain identifiers absent from unit_frame: "
                f"{(module_a, module_b)}"
            )

        pair = (
            (module_a, module_b)
            if module_a < module_b
            else (module_b, module_a)
        )
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return pairs


def _module_pair_correlation(
    wide: pd.DataFrame,
    pair: tuple[str, str],
    *,
    strata_columns: Sequence[str],
    min_units: int,
) -> dict[str, object]:
    """Compute one complete-unit Pearson and Spearman comparison."""
    module_a, module_b = pair
    paired = wide[[module_a, module_b]].dropna()
    values_a = paired[module_a].to_numpy(dtype=float)
    values_b = paired[module_b].to_numpy(dtype=float)
    n_units = int(paired.shape[0])

    skip_reason = ""
    if n_units < min_units:
        skip_reason = "too_few_complete_units"
    elif np.unique(values_a).size < 2 or np.unique(values_b).size < 2:
        skip_reason = "constant_module_score"

    pearson_r = np.nan
    pearson_pvalue = np.nan
    spearman_r = np.nan
    spearman_pvalue = np.nan
    if not skip_reason:
        pearson_r, pearson_pvalue = _correlation_values(
            stats.pearsonr(values_a, values_b)
        )
        spearman_r, spearman_pvalue = _correlation_values(
            stats.spearmanr(values_a, values_b)
        )

    return {
        "module_a": module_a,
        "module_b": module_b,
        "n_units": n_units,
        "n_strata": _count_strata(paired, strata_columns),
        "pearson_r": pearson_r,
        "pearson_pvalue": pearson_pvalue,
        "spearman_r": spearman_r,
        "spearman_pvalue": spearman_pvalue,
        "skipped": bool(skip_reason),
        "skip_reason": skip_reason,
    }


def _validate_profile_options(
    scores: pd.DataFrame,
    *,
    module_roles: Mapping[str, RoleAssignment],
    high_percentile: float,
    low_percentile: float,
    min_modules_per_axis: int,
) -> None:
    """Validate profile-wide mappings and percentile thresholds."""
    if not module_roles:
        raise ValueError("module_roles must map at least one score column")
    if not 0.0 <= low_percentile < high_percentile <= 1.0:
        raise ValueError(
            "percentile thresholds must satisfy "
            "0 <= low_percentile < high_percentile <= 1"
        )
    if min_modules_per_axis < 1:
        raise ValueError("min_modules_per_axis must be at least 1")
    if not scores.columns.is_unique:
        raise ValueError("scores must have unique column names")


def _group_modules_by_role(
    module_roles: Mapping[str, RoleAssignment],
) -> dict[EvidenceRole, list[str]]:
    """Validate role assignments and group their score columns."""
    grouped: dict[EvidenceRole, list[str]] = {
        role: [] for role in _EVIDENCE_ROLES
    }
    for module_column, assignment in module_roles.items():
        roles = (
            (assignment,) if isinstance(assignment, str) else tuple(assignment)
        )
        if not roles:
            raise ValueError(
                f"module {module_column!r} must have at least one role"
            )

        for role in dict.fromkeys(roles):
            if role not in _EVIDENCE_ROLES:
                raise ValueError(
                    f"unsupported evidence role {role!r} for {module_column!r}"
                )
            grouped[role].append(module_column)

    return grouped


def _rank_module_scores(
    scores: pd.DataFrame,
    *,
    mapped_columns: Sequence[str],
    reference_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Percentile-rank each module within the requested reference strata."""
    percentiles = pd.DataFrame(index=scores.index.copy())
    percentile_columns: dict[str, str] = {}
    groupers = [scores[column] for column in reference_columns]

    for module_column in mapped_columns:
        percentile_column = f"percentile__{module_column}"
        percentile_columns[module_column] = percentile_column
        numeric = (
            scores[module_column]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )

        if groupers:
            percentiles[percentile_column] = numeric.groupby(
                groupers,
                observed=True,
                dropna=False,
            ).transform(_percentile_rank)
        else:
            percentiles[percentile_column] = _percentile_rank(numeric)

    return percentiles, percentile_columns


def _aggregate_evidence_axes(
    percentiles: pd.DataFrame,
    *,
    role_modules: Mapping[EvidenceRole, Sequence[str]],
    percentile_columns: Mapping[str, str],
    min_modules_per_axis: int,
) -> pd.DataFrame:
    """Average mapped module ranks into observed relative evidence axes."""
    axes = pd.DataFrame(index=percentiles.index.copy())
    for role in _EVIDENCE_ROLES:
        rank_columns = [
            percentile_columns[module] for module in role_modules[role]
        ]
        observed_column = f"n_{role}_modules_observed"
        axis_column = f"relative_{role}"

        if not rank_columns:
            axes[observed_column] = 0
            axes[axis_column] = np.nan
            continue

        observed = percentiles[rank_columns].notna().sum(axis=1)
        mean_rank = percentiles[rank_columns].mean(axis=1, skipna=True)
        axes[observed_column] = observed
        axes[axis_column] = mean_rank.where(observed >= min_modules_per_axis)

    return axes


def _relative_state_columns(
    axes: pd.DataFrame,
    *,
    high_percentile: float,
    low_percentile: float,
) -> pd.DataFrame:
    """Derive relative state flags and signed mismatch gaps from axes."""
    high = {
        role: _threshold_flag(axes[f"relative_{role}"], high_percentile, True)
        for role in _EVIDENCE_ROLES
    }
    low = {
        role: _threshold_flag(axes[f"relative_{role}"], low_percentile, False)
        for role in _EVIDENCE_ROLES
    }

    states = pd.DataFrame(index=axes.index.copy())
    for role in _EVIDENCE_ROLES:
        states[f"relative_{role}_high"] = high[role]

    states["relative_nasp_active"] = high["competence"] & high["output"]
    states["relative_nasp_responsive"] = low["competence"] & high["output"]
    states["relative_nasp_competent"] = high["competence"] & low["output"]
    states["relative_nasp_restricted"] = high["restriction"]
    states["relative_feedback_output_mismatch"] = (
        high["feedback"] & low["output"]
    )
    states["relative_restriction_output_mismatch"] = (
        high["restriction"] & low["output"]
    )
    states["relative_post_nasp_phenotype"] = high["post"]
    states["relative_post_without_nasp_evidence"] = (
        high["post"] & low["competence"] & low["output"]
    )

    competence = axes["relative_competence"]
    output = axes["relative_output"]
    states["output_minus_competence_gap"] = output - competence
    states["restriction_minus_output_gap"] = (
        axes["relative_restriction"] - output
    )
    states["feedback_minus_output_gap"] = axes["relative_feedback"] - output

    nasp_evidence = pd.concat([competence, output], axis=1).max(axis=1)
    nasp_evidence = nasp_evidence.where(competence.notna() & output.notna())
    states["post_minus_nasp_gap"] = axes["relative_post"] - nasp_evidence
    states["high_percentile_threshold"] = high_percentile
    states["low_percentile_threshold"] = low_percentile

    return states


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    """Raise an actionable error when required frame columns are absent."""
    if missing := [column for column in columns if column not in frame.columns]:
        raise KeyError(f"{frame_name} missing required columns: {missing}")


def _correlation_values(result: object) -> tuple[float, float]:
    """Return a coefficient and p-value from a SciPy test result."""
    values = np.asarray(result, dtype=float)
    return float(values[0]), float(values[1])


def _single_label(frame: pd.DataFrame, column: str) -> str:
    """Return one frame-wide analysis label, rejecting mixed inputs."""
    if column not in frame.columns or frame.empty:
        return ""
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"unit_frame contains multiple {column} values")
    return str(values[0])


def _count_strata(
    paired: pd.DataFrame,
    strata_columns: Sequence[str],
) -> int:
    """Count represented strata in a complete-pair wide frame."""
    if paired.empty:
        return 0
    if not strata_columns:
        return 1
    return int(
        paired.index.to_frame(index=False)[list(strata_columns)]
        .drop_duplicates()
        .shape[0]
    )


def _percentile_rank(values: pd.Series) -> pd.Series:
    """Rank finite values from zero to one with neutral tied groups."""
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna()
    n_values = int(valid.sum())
    if n_values == 0:
        return result
    if n_values == 1:
        result.loc[valid] = 0.5
        return result
    ranks = numeric.loc[valid].rank(method="average")
    result.loc[valid] = (ranks - 1.0) / (n_values - 1.0)
    return result


def _threshold_flag(
    values: pd.Series,
    threshold: float,
    high: bool,
) -> pd.Series:
    """Return a nullable high- or low-threshold flag."""
    flag = values.ge(threshold) if high else values.le(threshold)
    return flag.astype("boolean").mask(values.isna(), pd.NA)
