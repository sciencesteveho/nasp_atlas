"""Metadata-first selection and count preparation for a condition cohort."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd

from nasp_atlas.analysis.condition_cohorts.specification import (
    CONDITION_LABEL_SEPARATOR,
)
from nasp_atlas.analysis.condition_cohorts.specification import (
    ConditionCohortSpec,
)
from nasp_atlas.analysis.stage_artifacts import file_identity
from nasp_atlas.analysis.stage_artifacts import publish_stage_manifest
from nasp_atlas.analysis.stage_artifacts import read_stage_manifest
from nasp_atlas.analysis.stage_artifacts import validate_stage
from nasp_atlas.single_cell.count_normalization import log_normalize_counts
from nasp_atlas.single_cell.io import read_h5ad_metadata
from nasp_atlas.single_cell.io import read_h5ad_rows
from nasp_atlas.single_cell.utils import normalize_h5ad_string_storage


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionCohortIntake:
    """Selected cells, donor and sample design, and every exclusion.

    Attributes:
      selected_obs: Canonical metadata of the selected cells, indexed by the
        source observation names.
      donors: One row per selected donor with arms, samples and covariates.
      samples: One row per selected sample with its donor and arm.
      selection_audit: Cells and donors retained after each selection step.
      condition_crosswalk: Every source condition label, its arm (or none)
        and its support before condition mapping.
      excluded_donors: Donors removed for a design reason, with the reason.
    """

    selected_obs: pd.DataFrame
    donors: pd.DataFrame
    samples: pd.DataFrame
    selection_audit: pd.DataFrame
    condition_crosswalk: pd.DataFrame
    excluded_donors: pd.DataFrame


@dataclass(frozen=True)
class PreparedConditionCohort:
    """A published intake stage: design tables and prepared expression.

    Attributes:
      intake: Selection results written by the stage.
      h5ad_path: Prepared h5ad with log1p CP10k expression in X.
      manifest: Completion manifest of the stage.
    """

    intake: ConditionCohortIntake
    h5ad_path: Path
    manifest: dict[str, object]


def fetch_condition_cohort_input(
    spec: ConditionCohortSpec,
    *,
    data_dir: Path,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> Path:
    """Download the spec's input into `data_dir` and verify its checksum.

    An existing file is kept when its SHA-256 matches and rejected when it
    does not. A download streams into a `.partial` file, is hashed while
    writing, and is renamed into place only when the checksum matches, so
    an interrupted or wrong download never looks complete. An internal
    input (no `source_url`) is only verified.
    """
    target = data_dir / spec.input_filename
    if target.exists():
        if file_identity(target)["sha256"] != spec.input_sha256:
            raise ValueError(
                f"{target} exists with a different SHA-256; move it aside "
                "before fetching"
            )
        logger.info("Input already present and verified: %s", target)
        return target

    if spec.source_url is None:
        raise FileNotFoundError(
            f"{target} does not exist and the spec declares no source_url; "
            "pass the data directory that holds this internal input"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    digest = hashlib.sha256()
    logger.info("Downloading %s to %s", spec.source_url, target)
    with urllib.request.urlopen(spec.source_url) as response:
        with partial.open("wb") as handle:
            while chunk := response.read(chunk_bytes):
                digest.update(chunk)
                handle.write(chunk)

    if digest.hexdigest() != spec.input_sha256:
        raise ValueError(
            f"Downloaded {spec.source_url} has SHA-256 {digest.hexdigest()}, "
            f"not the declared {spec.input_sha256}; kept as {partial}"
        )

    shutil.move(partial, target)
    return target


def select_condition_cohort(
    obs: pd.DataFrame,
    spec: ConditionCohortSpec,
    *,
    development_donors_per_arm: int | None = None,
    random_state: int = 42,
) -> ConditionCohortIntake:
    """Select cells from metadata and record every exclusion.

    Filters apply in declared order; a None filter value keeps cells missing
    that column. Condition labels outside the declared arms are excluded and
    listed in the crosswalk. A donor whose arm varies across samples is
    excluded unless the spec allows it; a donor missing a covariate is
    excluded with the covariate named. Missing identities, a sample shared by
    several donors, a sample spanning arms and a covariate varying within a
    sample are data errors and raise. `obs` is not modified.

    `development_donors_per_arm` keeps that many whole donors per arm, drawn
    with `random_state`, for fast iteration; results from such a subset are
    for development only and the audit records the step.
    """
    _validate_source_columns(obs, spec)
    keep = pd.Series(True, index=obs.index)
    audit = [_audit_row("source", obs, keep, spec)]
    for column, values in spec.filters:
        listed = [value for value in values if value is not None]
        matched = obs[column].isin(listed)
        if None in values:
            matched |= obs[column].isna()
        keep &= matched
        audit.append(_audit_row(f"filter:{column}", obs, keep, spec))

    labels = _condition_labels(obs, spec.condition.columns)
    arms = labels.map(spec.condition.arm_by_label())
    crosswalk = _condition_crosswalk(obs.loc[keep], labels, arms, spec)
    keep &= arms.notna()
    audit.append(_audit_row("mapped_condition", obs, keep, spec))

    selected = _canonical_obs(obs.loc[keep], spec)
    selected["source_condition"] = labels.loc[keep]
    selected["condition"] = arms.loc[keep]
    _validate_sample_design(selected, spec)

    excluded = _design_exclusions(selected, spec)
    selected, exclusion_audit = _exclude_donors(selected, excluded)
    audit.extend(exclusion_audit)

    if development_donors_per_arm is not None:
        eligible = len(selected)
        selected = _development_subset(
            selected,
            per_arm=development_donors_per_arm,
            random_state=random_state,
        )
        audit.append(
            {
                "step": "development_subset",
                "n_cells_retained": len(selected),
                "n_cells_excluded": eligible - len(selected),
                "n_donors_retained": int(selected["donor_id"].nunique()),
            }
        )

    return ConditionCohortIntake(
        selected_obs=selected,
        donors=_donor_table(selected, spec),
        samples=_sample_table(selected, spec),
        selection_audit=pd.DataFrame(audit),
        condition_crosswalk=crosswalk,
        excluded_donors=excluded,
    )


def prepare_condition_cohort(
    spec: ConditionCohortSpec,
    *,
    input_path: Path,
    output_dir: Path,
    target_sum: float = 10_000.0,
    development_donors_per_arm: int | None = None,
    random_state: int = 42,
) -> PreparedConditionCohort:
    """Publish selected cells as log1p CP10k expression plus design tables.

    `input_path` is this host's copy of `spec.input_filename` and must match
    `spec.input_sha256`; the request records the file name and checksum, not
    the host path, so a run moved between hosts stays reusable. Expression is
    normalized once over all selected cells and the full gene universe and
    stored as float32 in X; counts stay in the verified source file. Each
    cell also gets `n_genes_detected` and `total_counts`. An existing output
    directory is reused only when its manifest matches this request and its
    artifacts are unchanged; otherwise choose a new directory. A development
    subset (see `select_condition_cohort`) is part of the request.
    """
    request = {
        **_intake_request(spec, target_sum=target_sum),
        "development_donors_per_arm": development_donors_per_arm,
        "random_state": random_state,
    }

    h5ad_path = output_dir / "prepared.h5ad"
    if output_dir.exists():
        return PreparedConditionCohort(
            intake=_read_intake(output_dir),
            h5ad_path=h5ad_path,
            manifest=validate_stage(output_dir, request, stage="intake"),
        )

    source = file_identity(input_path)
    if source["sha256"] != spec.input_sha256:
        raise ValueError(
            f"{input_path} has SHA-256 {source['sha256']}, not the declared "
            f"{spec.input_sha256}. Restore it from "
            f"{spec.source_url or 'its internal source'} or update the spec "
            "deliberately."
        )

    obs, var = read_h5ad_metadata(
        input_path, use_raw_var=spec.counts_source == "raw/X"
    )
    intake = select_condition_cohort(
        obs,
        spec,
        development_donors_per_arm=development_donors_per_arm,
        random_state=random_state,
    )
    logger.info(
        "Selected %d cells from %d donors",
        len(intake.selected_obs),
        intake.donors.shape[0],
    )

    prepared = _prepared_expression(
        spec, intake, input_path=input_path, var=var, target_sum=target_sum
    )

    _publish_intake(output_dir, request, intake, prepared, h5ad_path=h5ad_path)
    return PreparedConditionCohort(
        intake=intake,
        h5ad_path=h5ad_path,
        manifest=read_stage_manifest(
            output_dir / "intake_manifest.json", stage="intake"
        ),
    )


def _publish_intake(
    output_dir: Path,
    request: dict[str, object],
    intake: ConditionCohortIntake,
    prepared: ad.AnnData,
    *,
    h5ad_path: Path,
) -> None:
    """Write the design tables and prepared h5ad, then the manifest last."""
    output_dir.mkdir(parents=True, exist_ok=False)
    tables = {
        "selection_audit.csv": intake.selection_audit,
        "condition_crosswalk.csv": intake.condition_crosswalk,
        "donors.csv": intake.donors,
        "samples.csv": intake.samples,
        "excluded_donors.csv": intake.excluded_donors,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)

    prepared.obs = _object_categoricals(prepared.obs)
    normalize_h5ad_string_storage(prepared)
    prepared.write_h5ad(h5ad_path, compression="gzip")

    publish_stage_manifest(
        output_dir,
        request,
        [*tables, h5ad_path.name],
        status="completed_intake",
        manifest_name="intake_manifest.json",
    )


def _validate_source_columns(
    obs: pd.DataFrame,
    spec: ConditionCohortSpec,
) -> None:
    """Fail before filtering when a declared source column is absent."""
    if not obs.index.is_unique:
        raise ValueError("Source observation names must be unique")

    required = {
        *(source for _, source in spec.obs_columns),
        *(column for column, _ in spec.filters),
        *spec.sample_columns,
        *spec.condition.columns,
    }
    if missing := sorted(required.difference(obs.columns)):
        raise KeyError(f"Missing source obs columns: {missing}")


def _audit_row(
    step: str,
    obs: pd.DataFrame,
    keep: pd.Series,
    spec: ConditionCohortSpec,
) -> dict[str, object]:
    """Count cells and donors retained after one selection step."""
    donors = obs.loc[keep, spec.source_column("donor_id")]
    return {
        "step": step,
        "n_cells_retained": int(keep.sum()),
        "n_cells_excluded": int((~keep).sum()),
        "n_donors_retained": int(donors.nunique()),
    }


def _condition_labels(
    obs: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.Series:
    """Join each cell's condition column values into one source label.

    A missing value in any column leaves the label missing, so the cell maps
    to no arm and is counted in the crosswalk rather than guessed.
    """
    labels = obs[columns[0]].astype("string")
    for column in columns[1:]:
        labels = (
            labels + CONDITION_LABEL_SEPARATOR + obs[column].astype("string")
        )
    return labels


def _condition_crosswalk(
    filtered: pd.DataFrame,
    labels: pd.Series,
    arms: pd.Series,
    spec: ConditionCohortSpec,
) -> pd.DataFrame:
    """List every filtered source label, its arm and its support."""
    frame = pd.DataFrame(
        {
            "source_label": labels.loc[filtered.index],
            "condition": arms.loc[filtered.index],
            "donor_id": filtered[spec.source_column("donor_id")].astype(str),
        }
    )

    crosswalk = (
        frame.groupby(["source_label", "condition"], dropna=False)
        .agg(n_cells=("donor_id", "size"), n_donors=("donor_id", "nunique"))
        .reset_index()
    )

    crosswalk["source_column"] = CONDITION_LABEL_SEPARATOR.join(
        spec.condition.columns
    )
    crosswalk["status"] = np.where(
        crosswalk["condition"].isna(), "not_in_spec", "mapped"
    )

    for label in spec.condition.arm_by_label().keys() - set(
        crosswalk["source_label"].dropna()
    ):
        logger.warning("Declared condition label %r has no cells", label)
    return crosswalk


def _canonical_obs(
    obs: pd.DataFrame,
    spec: ConditionCohortSpec,
) -> pd.DataFrame:
    """Copy declared columns under canonical names; parse covariates."""
    frame = pd.DataFrame(index=obs.index)
    for canonical, source in spec.obs_columns:
        frame[canonical] = obs[source].astype("string")
    for canonical, value in spec.obs_constants:
        frame[canonical] = value
    frame["sample_id"] = (
        obs[list(spec.sample_columns)].astype("string").agg("|".join, axis=1)
    )

    identities = ["donor_id", "sample_id", "cell_type", "assay", "tissue"]
    if frame[identities].isna().any(axis=None):
        missing = frame[identities].isna().sum()
        raise ValueError(
            f"Selected cells lack identity metadata: {missing[missing > 0]}"
        )

    for covariate in spec.covariates:
        frame[covariate.name] = _parse_values(
            frame[covariate.column], covariate.kind
        )
    for slope in spec.continuous_contrasts:
        frame[f"{slope.estimand}_measure"] = _parse_values(
            frame[slope.column], slope.kind
        )

    return frame


def _parse_values(values: pd.Series, kind: str) -> pd.Series:
    """Convert labels to their model scale; unparsable values are NA."""
    if kind == "categorical":
        return values.astype("string")
    if kind == "numeric":
        return pd.to_numeric(values, errors="coerce").astype(float)
    if kind == "development_stage":
        return (
            values.astype(str)
            .str.extract(r"^(\d+)-year-old stage$")[0]
            .astype(float)
        )
    if kind == "development_stage_top_coded":
        return (
            values.astype(str)
            .str.extract(r"^(\d+)(?:-year-old| year-old and over) stage$")[0]
            .astype(float)
        )
    if kind != "binned":
        raise ValueError(f"Unknown value kind {kind!r}")

    parsed = (
        values.astype(str)
        .str.extract(r"^\s*(\d+)\s*-\s*(\d+)(?:\s+\S.*)?$")
        .astype(float)
    )
    # Integer bins such as 60-69 span [60, 70), so the midpoint is 65.
    return (parsed[0] + parsed[1] + 1.0) / 2.0


def _validate_sample_design(
    selected: pd.DataFrame,
    spec: ConditionCohortSpec,
) -> None:
    """Require samples nested in donors with one arm and fixed covariates."""
    per_sample = selected.groupby("sample_id", observed=True).agg(
        n_donors=("donor_id", "nunique"),
        n_arms=("condition", "nunique"),
    )

    if (per_sample["n_donors"] > 1).any():
        examples = per_sample.index[per_sample["n_donors"] > 1][:3].tolist()
        raise ValueError(
            f"Samples belong to several donors, e.g. {examples}; use "
            "sample_columns that nest within donors"
        )

    if (per_sample["n_arms"] > 1).any():
        examples = per_sample.index[per_sample["n_arms"] > 1][:3].tolist()
        raise ValueError(
            f"Samples span several condition arms, e.g. {examples}"
        )

    # Units carry these sample-level values, so each must be constant.
    names = [
        "assay",
        "tissue",
        *(covariate.name for covariate in spec.covariates),
        *(f"{slope.estimand}_measure" for slope in spec.continuous_contrasts),
    ]
    varying = (
        selected.groupby("sample_id", observed=True)[names]
        .nunique(dropna=False)
        .gt(1)
        .any()
    )
    if varying.any():
        raise ValueError(
            "Values vary within a sample: "
            f"{varying.index[varying].tolist()}. Assay, tissue, covariates "
            "and slope measurements must be sample- or donor-level."
        )


def _design_exclusions(
    selected: pd.DataFrame,
    spec: ConditionCohortSpec,
) -> pd.DataFrame:
    """Return donors that cannot enter the declared design, with reasons."""
    reasons = []
    if not spec.condition.varies_within_donor:
        arms = selected.groupby("donor_id", observed=True)["condition"]
        varying = arms.nunique()
        reasons.append(
            pd.DataFrame(
                {
                    "donor_id": varying.index[varying > 1],
                    "reason": "condition_varies_within_donor",
                }
            )
        )

    for covariate in spec.covariates:
        missing = selected.loc[
            selected[covariate.name].isna(), "donor_id"
        ].unique()
        reasons.append(
            pd.DataFrame(
                {
                    "donor_id": missing,
                    "reason": f"missing_covariate:{covariate.name}",
                }
            )
        )

    excluded = pd.concat(reasons, ignore_index=True)
    return (
        excluded.astype({"donor_id": "string"})
        .drop_duplicates("donor_id")
        .sort_values(["reason", "donor_id"], kind="stable")
        .reset_index(drop=True)
    )


def _exclude_donors(
    selected: pd.DataFrame,
    excluded: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove excluded donors one reason at a time, auditing each step."""
    retained = pd.Series(True, index=selected.index)
    audit = []
    for reason, donors in excluded.groupby("reason", sort=True)["donor_id"]:
        removed = retained & selected["donor_id"].isin(donors)
        retained &= ~removed
        audit.append(
            {
                "step": f"exclude:{reason}",
                "n_cells_retained": int(retained.sum()),
                "n_cells_excluded": int(removed.sum()),
                "n_donors_retained": int(
                    selected.loc[retained, "donor_id"].nunique()
                ),
            }
        )

    return selected.loc[retained].copy(), audit


def _development_subset(
    selected: pd.DataFrame,
    *,
    per_arm: int,
    random_state: int,
) -> pd.DataFrame:
    """Keep up to `per_arm` whole donors from each arm, reproducibly."""
    if per_arm < 1:
        raise ValueError("development_donors_per_arm must be positive")

    rng = np.random.default_rng(random_state)
    chosen: set[str] = set()
    for _, donors in selected.groupby("condition", observed=True)["donor_id"]:
        candidates = np.sort(donors.unique().astype(str))
        chosen.update(rng.permutation(candidates)[:per_arm].tolist())
    return selected.loc[selected["donor_id"].isin(chosen)].copy()


def _donor_table(
    selected: pd.DataFrame,
    spec: ConditionCohortSpec,
) -> pd.DataFrame:
    """Summarize each donor's arms, samples, cells and design values."""
    names = [
        *(covariate.name for covariate in spec.covariates),
        *(f"{slope.estimand}_measure" for slope in spec.continuous_contrasts),
    ]

    grouped = selected.groupby("donor_id", observed=True)
    donors = grouped.agg(
        conditions=("condition", lambda s: ";".join(sorted(s.unique()))),
        n_samples=("sample_id", "nunique"),
        n_cells=("sample_id", "size"),
    )
    for name in names:
        donors[name] = grouped[name].agg(
            lambda values: (
                values.iloc[0]
                if values.nunique(dropna=False) == 1
                else ";".join(sorted(values.astype(str).unique()))
            )
        )

    return donors.reset_index()


def _sample_table(
    selected: pd.DataFrame,
    spec: ConditionCohortSpec,
) -> pd.DataFrame:
    """Summarize each sample's donor, arm, source label and design values."""
    names = [
        *(covariate.name for covariate in spec.covariates),
        *(f"{slope.estimand}_measure" for slope in spec.continuous_contrasts),
    ]

    return (
        selected.groupby("sample_id", observed=True)
        .agg(
            donor_id=("donor_id", "first"),
            condition=("condition", "first"),
            source_condition=("source_condition", "first"),
            n_cells=("donor_id", "size"),
            **{name: (name, "first") for name in names},
        )
        .reset_index()
    )


def _prepared_expression(
    spec: ConditionCohortSpec,
    intake: ConditionCohortIntake,
    *,
    input_path: Path,
    var: pd.DataFrame,
    target_sum: float,
) -> ad.AnnData:
    """Read selected counts, normalize once and attach per-cell depth."""
    layer = spec.counts_source.removeprefix("layers/")
    source = read_h5ad_rows(
        input_path,
        intake.selected_obs.index.tolist(),
        read_x=spec.counts_source == "X",
        read_raw=spec.counts_source == "raw/X",
        layer_keys=(layer,) if spec.counts_source.startswith("layers/") else (),
        obsm_keys=("X_umap",),
    )
    if spec.counts_source == "raw/X":
        if source.raw is None:
            raise ValueError(f"No raw counts in {input_path}")
        matrix = source.raw.X
    elif spec.counts_source == "X":
        matrix = source.X
    else:
        matrix = source.layers[layer]

    normalized = log_normalize_counts(
        matrix,
        target_sum=target_sum,
        source=f"{input_path}:{spec.counts_source}",
        dtype=np.float32,
    )

    obs = intake.selected_obs.copy()
    obs["n_genes_detected"] = normalized.counts.getnnz(axis=1)
    obs["total_counts"] = normalized.totals

    prepared = ad.AnnData(X=normalized.expression, obs=obs, var=var.copy())
    prepared.obsm["X_umap"] = source.obsm["X_umap"]
    prepared.uns["expression_preparation"] = {
        "counts_source": spec.counts_source,
        "normalization": "library_size_log1p",
        "target_sum": target_sum,
        "normalization_n_genes": int(var.shape[0]),
    }

    return prepared


def _object_categoricals(obs: object) -> pd.DataFrame:
    """Store text columns as categoricals with object-dtype categories.

    anndata would otherwise convert them while writing and, under pandas 3,
    produce Arrow-backed categories that h5ad cannot store.
    """
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Prepared AnnData must have in-memory obs")

    frame = obs.copy()
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_string_dtype(values.dtype) and not isinstance(
            values.dtype, pd.CategoricalDtype
        ):
            text = values.astype(object)
            categories = pd.Index(sorted(text.dropna().unique()), dtype=object)
            frame[column] = pd.Categorical(text, categories=categories)
    return frame


def _intake_request(
    spec: ConditionCohortSpec,
    *,
    target_sum: float,
) -> dict[str, object]:
    """Describe only what determines the prepared cells and values.

    Planned contrasts, cell states, IFN strata, source notes and the download
    URL do not change the prepared data, so editing them does not invalidate
    an intake. The input is identified by file name and checksum.
    """
    record = {
        key: value
        for key, value in asdict(spec).items()
        if key
        not in {
            "contrasts",
            "cell_state",
            "ifn_strata",
            "state_contrasts",
            "source_notes",
            "source_url",
        }
    }
    return json.loads(
        json.dumps(
            {"spec": record, "target_sum": target_sum, "dtype": "float32"},
            default=str,
        )
    )


def _read_intake(output_dir: Path) -> ConditionCohortIntake:
    """Read a validated intake's tables and selected-cell metadata."""
    return ConditionCohortIntake(
        selected_obs=read_h5ad_metadata(output_dir / "prepared.h5ad")[0],
        donors=pd.read_csv(output_dir / "donors.csv"),
        samples=pd.read_csv(output_dir / "samples.csv"),
        selection_audit=pd.read_csv(output_dir / "selection_audit.csv"),
        condition_crosswalk=pd.read_csv(output_dir / "condition_crosswalk.csv"),
        excluded_donors=pd.read_csv(output_dir / "excluded_donors.csv"),
    )
