"""Explicit-panel scoring and bounded donor aggregation for one population."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]
from nasp_compendium.types import GeneModule  # type: ignore[import]

from nasp_atlas.single_cell.associations import FeatureSpec
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import aggregate_feature_frame_by_keys
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import inverse_module_score_name
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import positive_module_score_name
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_modules


__all__ = ["CohortScores", "aggregate_module_scores", "score_prepared_cohort"]


@dataclass(frozen=True)
class CohortScores:
    """Scores and measurement eligibility, including each declared module."""

    scores: pd.DataFrame
    eligibility: pd.DataFrame
    arm_diagnostics: pd.DataFrame


def score_prepared_cohort(
    prepared: ad.AnnData,
    modules: Sequence[GeneModule],
    coverage: pd.DataFrame,
    *,
    scorer: ScorerName,
    random_state: int = 42,
    aucell_chunk_size: int = 500,
    aucell_num_workers: int = 1,
) -> CohortScores:
    """Score eligible frozen definitions over the complete input population.

    X must already contain prepared log expression on the shared background.
    Calls existing scorers without changing their historical behavior. Missing
    coverage and degenerate arms stay visible in eligibility; unexpected
    scoring errors propagate. Expression is shared, metadata is copied, and
    inputs are not mutated. No donor chunks are scored independently.
    """
    if scorer not in ("scanpy", "aucell"):
        raise ValueError(f"Unsupported scorer: {scorer}")
    if not isinstance(prepared.obs, pd.DataFrame) or not isinstance(
        prepared.var, pd.DataFrame
    ):
        raise TypeError("Prepared cohort requires in-memory obs and var")
    if not prepared.n_obs:
        raise ValueError("Cannot score an empty population")
    if not prepared.var["feature_name"].is_unique:
        raise ValueError("Resolve duplicate scoring labels before scoring")
    module_ids = [module.module_id for module in modules]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("Module IDs must be unique within a scoring request")
    eligible, eligibility = _coverage_eligibility(modules, coverage, scorer)
    working = ad.AnnData(
        X=prepared.X, obs=prepared.obs.copy(), var=prepared.var.copy()
    )
    scores = pd.DataFrame(index=prepared.obs_names)
    if eligible and scorer == "scanpy":
        score_scanpy_modules(
            working,
            [module.module_id for module in eligible],
            gene_modules=eligible,
            expression_layer=None,
            random_state=random_state,
        )
        columns = []
        for module in eligible:
            columns.extend(_score_columns(module, scorer))
        scored_obs = cast(pd.DataFrame, working.obs)
        scores = scored_obs[list(dict.fromkeys(columns))].copy()
    elif eligible:
        _, scores, _ = score_aucell_modules(
            working,
            [module.module_id for module in eligible],
            gene_modules=eligible,
            expression_layer=None,
            random_state=random_state,
            chunk_size=aucell_chunk_size,
            num_workers=aucell_num_workers,
        )

    if (
        not scores.index.is_unique
        or len(scores) != prepared.n_obs
        or not prepared.obs_names.isin(scores.index).all()
    ):
        raise ValueError("Scorer did not return exactly the selected cells")
    # AUCell sorts identities within blocks. Restore input observation order.
    scores = scores.reindex(prepared.obs_names)
    scores.index = prepared.obs_names.copy().rename("obs_name")
    arms = _arm_diagnostics(prepared, scores, eligible, scorer)
    for row in eligibility:
        if row["status"] != "ok":
            continue
        module_arms = arms.loc[arms.module_id.eq(row["module_id"])]
        failures = module_arms.loc[module_arms.status.ne("ok"), "status"]
        if not failures.empty:
            row["status"] = ";".join(sorted(set(failures)))
    return CohortScores(scores, pd.DataFrame(eligibility), arms)


def aggregate_module_scores(
    obs: pd.DataFrame,
    scores: pd.DataFrame,
    modules: Sequence[GeneModule],
    *,
    scorer: ScorerName,
    scoring_context_id: str,
    minimum_cells: int = 10,
) -> pd.DataFrame:
    """Aggregate one module at a time, retaining finite/total support.

    These are cell-weighted means within people; downstream paired inference
    weights complete people equally. Missing score columns represent unscored
    modules and yield no donor rows; the hypothesis registry must retain them.
    Partial finite-score loss makes that unit ineligible, never a silent drop.
    """
    if not obs.index.is_unique or not scores.index.is_unique:
        raise ValueError("Observation and score identities must be unique")
    if len(obs) != len(scores) or not obs.index.isin(scores.index).all():
        raise ValueError(
            "Scores must identify exactly the selected observations"
        )
    if minimum_cells < 1 or not scoring_context_id:
        raise ValueError(
            "Require positive cell support and a scoring context ID"
        )
    unit_keys = ["cohort_id", "donor_id", "tissue", "population", "modality"]
    if obs[unit_keys].isna().any(axis=None):
        raise ValueError("Donor aggregation identities must not be missing")
    metadata = obs[unit_keys].copy()
    frames = []
    for module in modules:
        feature = FeatureSpec(
            feature_type="module_score",
            feature_id=module_score_name(module, scorer=scorer),
            feature_label=module.module_id,
            source="scores",
        )
        if feature.feature_id not in scores:
            continue
        cell_frame = metadata.assign(
            feature_type=feature.feature_type,
            feature_id=feature.feature_id,
            feature_label=feature.feature_label,
            feature_value=scores.loc[obs.index, feature.feature_id].to_numpy(),
        )
        donor = aggregate_feature_frame_by_keys(
            cell_frame,
            unit_keys=unit_keys,
            statistical_unit="donor_context",
            aggregation="mean",
            schema=ObsSchema(
                study_key="cohort_id",
                tissue_key="tissue",
                cell_type_key="population",
            ),
        )
        donor["eligible"] = donor.n_cells.ge(minimum_cells) & donor.n_cells.eq(
            donor.n_cells_total
        )
        donor["eligibility_reason"] = np.select(
            [
                donor.n_cells.ne(donor.n_cells_total),
                donor.n_cells.lt(minimum_cells),
            ],
            ["nonfinite_cell_scores", "insufficient_cells"],
            default="eligible",
        )
        frames.append(
            donor.assign(
                module_id=module.module_id,
                scorer=scorer,
                scoring_context_id=scoring_context_id,
            )
        )
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(
        columns=[
            *unit_keys,
            "feature_type",
            "feature_id",
            "feature_label",
            "feature_value",
            "n_cells",
            "n_cells_total",
            "eligible",
            "eligibility_reason",
            "module_id",
            "scorer",
            "scoring_context_id",
        ]
    )


def _coverage_eligibility(
    modules: Sequence[GeneModule], coverage: pd.DataFrame, scorer: ScorerName
) -> tuple[list[GeneModule], list[dict[str, str]]]:
    """Check every frozen arm member before invoking a scorer."""
    eligible = []
    records = []
    for module in modules:
        rows = coverage.loc[
            coverage.module_id.eq(module.module_id) & coverage.scored
        ]
        required = set(module.positive_genes + module.inverse_genes)
        represented = set(rows.gene)
        if required != represented or rows.duplicated(["gene", "arm"]).any():
            raise ValueError(
                f"Incomplete coverage audit for {module.module_id}"
            )
        status = (
            "ok"
            if required and rows.status.eq("matched").all()
            else "incomplete_signature"
        )
        records.append(
            {"module_id": module.module_id, "scorer": scorer, "status": status}
        )
        if status == "ok":
            eligible.append(module)
    return eligible, records


def _score_columns(module: GeneModule, scorer: ScorerName) -> list[str]:
    """Name the components and final score actually produced for a module."""
    columns = [module_score_name(module, scorer=scorer)]
    if module.positive_genes:
        columns.append(positive_module_score_name(module, scorer=scorer))
    inverse = inverse_module_score_name(module, scorer=scorer)
    if inverse is not None:
        columns.append(inverse)
    return columns


def _arm_diagnostics(
    prepared: ad.AnnData,
    scores: pd.DataFrame,
    modules: Sequence[GeneModule],
    scorer: ScorerName,
) -> pd.DataFrame:
    """Identify undetected/nonfinite arms and lost signed-score information."""
    if not sp.issparse(prepared.X):
        raise TypeError("Prepared expression must remain sparse")
    matrix = sp.csr_matrix(prepared.X)
    symbols = pd.Index(prepared.var["feature_name"])
    rows = []
    for module in modules:
        arms = (
            (
                "positive",
                module.positive_genes,
                positive_module_score_name(module, scorer=scorer),
            ),
            (
                "inverse",
                module.inverse_genes,
                inverse_module_score_name(module, scorer=scorer),
            ),
        )
        signed = bool(module.positive_genes and module.inverse_genes)
        for arm, genes, column in arms:
            if not genes or column is None:
                continue
            positions = symbols.get_indexer(pd.Index(genes))
            if (positions < 0).any():
                raise ValueError(
                    "Coverage audit disagrees with scoring annotation"
                )
            values = scores[column].to_numpy(dtype=float)
            finite = bool(np.isfinite(values).all())
            deviation = float(values.std(ddof=0)) if finite else np.nan
            detected = int((matrix[:, positions] > 0).getnnz())
            if not finite:
                status = "nonfinite_arm"
            elif detected == 0:
                status = "undetected_arm"
            elif signed and np.isclose(deviation, 0.0, atol=1e-8, rtol=1e-5):
                status = "degenerate_arm"
            else:
                status = "ok"
            rows.append(
                {
                    "module_id": module.module_id,
                    "scorer": scorer,
                    "arm": arm,
                    "n_genes": len(genes),
                    "n_detected_entries": detected,
                    "score_sd": deviation,
                    "zero_sd_atol": 1e-8,
                    "status": status,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "module_id",
            "scorer",
            "arm",
            "n_genes",
            "n_detected_entries",
            "score_sd",
            "zero_sd_atol",
            "status",
        ],
    )
