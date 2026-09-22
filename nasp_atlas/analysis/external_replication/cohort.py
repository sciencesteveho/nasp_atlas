"""Metadata-first cohort selection and explicit sparse count preparation."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad  # type: ignore[import]
import h5py  # type: ignore[import]
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]
from anndata.io import read_elem  # type: ignore[import]

from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.single_cell.io import read_h5ad_rows


__all__ = [
    "CohortIntake",
    "prepare_cohort_counts",
    "read_cohort_metadata",
    "select_cohort",
]


@dataclass(frozen=True)
class CohortIntake:
    """Exact selected cells, donor support and sequential exclusion counts."""

    selected_obs: pd.DataFrame
    donor_support: pd.DataFrame
    selection_audit: pd.DataFrame
    preparation_support: pd.DataFrame


def read_cohort_metadata(spec: CohortSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read obs and the declared source's var without expression arrays."""
    with h5py.File(spec.input_path, "r") as handle:
        if spec.counts_source not in handle:
            raise KeyError(f"Missing {spec.counts_source} in {spec.input_path}")
        var_path = "raw/var" if spec.counts_source == "raw/X" else "var"
        obs_group, var_group = handle["obs"], handle[var_path]
        if not isinstance(obs_group, h5py.Group) or not isinstance(
            var_group, h5py.Group
        ):
            raise TypeError("AnnData metadata must be encoded DataFrame groups")
        obs = read_elem(obs_group)
        var = read_elem(var_group)
    if not isinstance(obs, pd.DataFrame) or not isinstance(var, pd.DataFrame):
        raise TypeError("AnnData obs and var must be DataFrames")
    if not obs.index.is_unique or not var.index.is_unique:
        raise ValueError(
            "Source observation and feature indices must be unique"
        )
    return obs, var


def select_cohort(obs: pd.DataFrame, spec: CohortSpec) -> CohortIntake:
    """Select complete donors from metadata and retain every exclusion stage.

    Original selected columns are preserved with a source_ prefix. Canonical
    columns are explicit mappings/constants. Libraries never become people.
    Population memberships must be disjoint. The function does not modify obs.
    """
    _validate_metadata(obs, spec)
    keep = pd.Series(True, index=obs.index)
    audit: list[dict[str, object]] = [
        {"step": "source", "n_retained": len(obs), "n_excluded": 0}
    ]
    for column, values in spec.filters:
        before = int(keep.sum())
        keep &= obs[column].isin(values)
        audit.append(
            {
                "step": f"filter:{column}",
                "n_retained": int(keep.sum()),
                "n_excluded": before - int(keep.sum()),
            }
        )

    selected = obs.loc[keep]
    population = pd.Series(pd.NA, index=selected.index, dtype="string")
    for definition in spec.populations:
        member = selected[definition.column].isin(definition.labels)
        if (member & population.notna()).any():
            raise ValueError(
                f"Overlapping population mapping: {definition.name}"
            )
        population.loc[member] = definition.name
    included = population.notna()
    audit.append(
        {
            "step": "mapped_population",
            "n_retained": int(included.sum()),
            "n_excluded": int((~included).sum()),
        }
    )
    canonical = _canonical_obs(selected.loc[included], spec)
    canonical["population"] = population.loc[included]
    comparison = spec.comparison
    in_context = canonical[comparison.level_key].isin(
        [comparison.target, comparison.reference]
    )
    audit.append(
        {
            "step": "compared_contexts",
            "n_retained": int(in_context.sum()),
            "n_excluded": int((~in_context).sum()),
        }
    )
    canonical = canonical.loc[in_context].copy()
    other_axis = "population" if comparison.level_key == "tissue" else "tissue"
    if canonical[other_axis].nunique() > 1:
        raise ValueError(f"This comparison must fix exactly one {other_axis}")
    if canonical["modality"].nunique() > 1:
        raise ValueError("One scoring population cannot mix cells and nuclei")

    support = _paired_support(canonical, spec)
    eligible = support.loc[support["eligible"], "donor_id"]
    complete = canonical["donor_id"].isin(eligible)
    audit.append(
        {
            "step": "complete_donors",
            "n_retained": int(complete.sum()),
            "n_excluded": int((~complete).sum()),
        }
    )
    preparation_keys = [
        "cohort_id",
        "donor_id",
        "tissue",
        "population",
        "assay",
        "modality",
        *(f"source_{key}" for key in spec.library_keys),
    ]
    preparations = (
        canonical.groupby(preparation_keys, observed=True, dropna=False)
        .size()
        .reset_index(name="n_cells")
    )
    return CohortIntake(
        selected_obs=canonical.loc[complete].copy(),
        donor_support=support,
        selection_audit=pd.DataFrame(audit),
        preparation_support=preparations,
    )


def prepare_cohort_counts(
    spec: CohortSpec,
    selected_obs: pd.DataFrame,
    *,
    target_sum: float = 10_000.0,
) -> ad.AnnData:
    """Read declared counts and normalize once on their full gene universe.

    Returned X is sparse natural-log expression; layers/counts preserves the
    exact selected count matrix. No shared-gene subset has yet been applied.
    Zero-library, nonfinite, negative and fractional inputs fail actionably.
    This V1 contract requires nonnegative integer-valued counts, even when
    stored as floating point. Source matrices and selected_obs are unchanged.
    """
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("Normalization target_sum must be positive and finite")
    layer = spec.counts_source.removeprefix("layers/")
    source = read_h5ad_rows(
        spec.input_path,
        selected_obs.index.tolist(),
        read_x=spec.counts_source == "X",
        read_raw=spec.counts_source == "raw/X",
        layer_keys=(layer,) if spec.counts_source.startswith("layers/") else (),
    )
    if spec.counts_source == "raw/X":
        if source.raw is None:
            raise ValueError("Requested raw counts are unavailable")
        matrix, var = source.raw.X, source.raw.var
    elif spec.counts_source == "X":
        matrix, var = source.X, source.var
    else:
        matrix, var = source.layers[layer], source.var
    if not isinstance(var, pd.DataFrame):
        raise TypeError("Selective reading must return in-memory var")
    if not sp.issparse(matrix):
        raise TypeError("This cohort workflow requires sparse count storage")
    counts = sp.csr_matrix(matrix, copy=True)
    values = counts.data
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(
            f"Invalid counts in declared source {spec.counts_source}"
        )
    totals = np.asarray(counts.sum(axis=1), dtype=float).ravel()
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError(
            "Selected cells have nonfinite or zero count libraries"
        )

    expression = (
        counts.astype(np.float64)
        .multiply((target_sum / totals)[:, None])
        .tocsr()
    )
    expression.data = np.log1p(expression.data)
    prepared = ad.AnnData(
        X=expression,
        obs=selected_obs.copy(),
        var=var.copy(),
        layers={"counts": counts},
    )
    prepared.obs["normalization_total_counts"] = totals
    prepared.uns["expression_preparation"] = {
        "counts_source": spec.counts_source,
        "normalization": "library_size_log1p",
        "target_sum": target_sum,
        "normalization_n_genes": len(var),
    }
    return prepared


def _validate_metadata(obs: pd.DataFrame, spec: CohortSpec) -> None:
    """Validate requested metadata before filtering can hide identity loss."""
    if not obs.index.is_unique or np.asarray(obs.index.isna()).any():
        raise ValueError("Source cell identities must be unique and non-null")
    required = {
        *(source for _, source in spec.obs_columns),
        *(key for key, _ in spec.filters),
        *(definition.column for definition in spec.populations),
        *spec.library_keys,
    }
    if missing := required.difference(obs.columns):
        raise KeyError(f"Missing source metadata: {sorted(missing)}")
    donor_key = dict(spec.obs_columns)["donor_id"]
    if (
        obs[donor_key].isna().any()
        or obs[donor_key].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Source donor identity must not be missing or empty")


def _canonical_obs(obs: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """Preserve source annotations and validate canonical donor metadata."""
    frame = obs.add_prefix("source_").copy()
    for key, source in spec.obs_columns:
        frame[key] = obs[source].astype("string")
    for key, value in spec.obs_constants:
        frame[key] = value
    frame["cohort_id"] = spec.cohort_id
    identities = ["donor_id", "tissue", "assay", "modality"]
    if frame[identities].isna().any(axis=None):
        raise ValueError(
            "Selected canonical identity/assay fields contain nulls"
        )
    if missing := set(spec.donor_metadata).difference(frame.columns):
        raise KeyError(f"Unknown canonical donor metadata: {sorted(missing)}")
    if spec.donor_metadata:
        variation = frame.groupby("donor_id", observed=True)[
            list(spec.donor_metadata)
        ].nunique(dropna=False)
        if variation.gt(1).any(axis=None):
            raise ValueError("Donor-level metadata varies within a person")
    return frame


def _paired_support(obs: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """Count available contexts without discarding unsupported donors."""
    comparison = spec.comparison
    counts = (
        obs.groupby(["donor_id", comparison.level_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(
        columns=[comparison.target, comparison.reference], fill_value=0
    )
    counts.columns = ["n_target_cells", "n_reference_cells"]
    counts["eligible"] = counts.ge(comparison.minimum_cells).all(axis=1)
    counts["reason"] = np.where(
        counts["eligible"], "eligible", "insufficient_context_cells"
    )
    return counts.reset_index().assign(cohort_id=spec.cohort_id)
