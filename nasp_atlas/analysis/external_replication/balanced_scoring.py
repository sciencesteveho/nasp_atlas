"""Rescore balanced TS and external populations as sensitivity measurements."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from nasp_compendium.types import GeneModule

from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import sampling
from nasp_atlas.analysis.external_replication import sensitivity
from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.provenance import scoring_request
from nasp_atlas.analysis.external_replication.registered_inputs import (
    RegisteredPrimary,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    validate_analysis_artifacts,
)
from nasp_atlas.analysis.external_replication.sampling import (
    select_balanced_cells,
)
from nasp_atlas.analysis.external_replication.scoring import (
    aggregate_module_scores,
)
from nasp_atlas.analysis.external_replication.sensitivity import (
    matched_donor_effects,
)
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell.associations import PairedContrastResult
from nasp_atlas.single_cell.associations import paired
from nasp_atlas.single_cell.module_scoring import ScorerName


__all__ = ["rescore_balanced_comparison", "validate_balanced_artifacts"]


def rescore_balanced_comparison(
    prepared: RegisteredPrimary,
    *,
    cohort_key: str,
    primary_analysis_dir: Path,
    primary_score_dir: Path,
    output_dir: Path,
) -> Path:
    """Rescore both cohorts at the metadata-frozen cap, retaining all people.

    Each cohort keeps its own calibration and effect scale. Results are
    sensitivity estimates, never a new primary decision or TS eligibility
    update. Completed child scoring stages survive interrupted analysis and
    are reused through their scoring dependency checks. One writer owns the
    parent stage; partial output never carries a completion manifest.
    """
    primary = validate_analysis_artifacts(primary_analysis_dir)
    settings = prepared.registration.specification
    cohorts = settings["cohorts"]
    discovery = settings[f"{cohort_key}_discovery_score_manifest"]
    if not isinstance(cohorts, dict) or not isinstance(discovery, dict):
        raise ValueError("Missing frozen cohort or discovery scoring request")
    primary_request = primary["request"]
    if (
        not isinstance(primary_request, dict)
        or primary_request["registration_identity"]
        != prepared.registration.identity
        or primary_request["cohort_id"] != prepared.spec.cohort_id
    ):
        raise ValueError(
            "Balanced and primary analyses must share a registration"
        )
    primary_identity = file_identity(primary_score_dir / "stage_manifest.json")
    if primary_identity != primary_request["score_manifest"]:
        raise ValueError("Balanced baseline is not the primary score artifact")
    reference_score_dir = Path(discovery["path"]).parent
    if file_identity(reference_score_dir / "stage_manifest.json") != discovery:
        raise ValueError("Changed registered discovery score manifest")
    cap = cohorts[cohort_key]["balanced_cells_per_donor_context"]
    populations = (
        (
            "external",
            prepared.spec,
            prepared.intake.selected_obs,
            primary_score_dir,
        ),
        (
            "reference",
            prepared.reference_spec,
            prepared.reference_intake.selected_obs,
            reference_score_dir,
        ),
    )
    balanced_cells = {
        role: select_balanced_cells(
            obs,
            spec.comparison,
            cells_per_context=cap,
            random_state=prepared.random_state,
        )
        for role, spec, obs, _ in populations
    }
    score_contexts = {}
    for role, spec, _, _ in populations:
        features = (
            prepared.alignment.reference_features
            if role == "reference"
            else prepared.alignment.external_features
        )
        expected_scores = scoring_request(
            spec,
            balanced_cells[role],
            features,
            prepared.alignment.shared_gene_ids,
            prepared.modules,
            prepared.alignment.coverage,
            panel_path=prepared.panel_path,
            target_sum=prepared.target_sum,
            random_state=prepared.random_state,
            aucell_chunk_size=prepared.aucell_chunk_size,
            aucell_num_workers=prepared.aucell_num_workers,
        )
        score_contexts[role] = request_identity(expected_scores)
    request = {
        "registration_identity": prepared.registration.identity,
        "cells_per_context": cap,
        "random_state": prepared.random_state,
        "scoring_contexts": score_contexts,
        "primary_score_manifest": primary_identity,
        "reference_score_manifest": discovery,
        "primary_analysis_manifest": file_identity(
            primary_analysis_dir / "analysis_manifest.json"
        ),
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (sampling, sensitivity, contrasts, paired)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    if (output_dir / "balanced_manifest.json").exists():
        validate_balanced_artifacts(output_dir, request=request)
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "balanced.lock"
    with lock_path.open("x") as lock:
        lock.write(json.dumps({"pid": os.getpid()}))
    try:
        artifacts = []
        for role, spec, obs, baseline_dir in populations:
            artifacts.extend(
                _rescore_balanced_cohort(
                    prepared,
                    spec,
                    obs,
                    balanced_cells[role],
                    baseline_dir=baseline_dir,
                    output_dir=output_dir,
                    role=role,
                    cells_per_context=cap,
                )
            )
        record = {
            "status": "completed_balanced_sensitivity",
            "request": request,
            "identity": request_identity(request),
            "artifacts": {
                str(path.relative_to(output_dir)): file_identity(path)
                for path in artifacts
            },
        }
        temporary = output_dir / "balanced_manifest.json.partial"
        temporary.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(output_dir / "balanced_manifest.json")
    finally:
        lock_path.unlink()
    return output_dir


def validate_balanced_artifacts(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Require intact TS/external scores and paired sensitivity tables."""
    record = read_stage_manifest(
        output_dir / "balanced_manifest.json", stage="balanced sensitivity"
    )
    if record["status"] != "completed_balanced_sensitivity":
        raise ValueError(f"Incomplete balanced sensitivity: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(f"Incompatible balance request: {output_dir}")
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed balance request: {output_dir}")
    for name, identity in record["artifacts"].items():
        if file_identity(output_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed balanced artifact: {output_dir / name}")
    for role in ("external", "reference"):
        scores = validate_scoring_artifacts(output_dir / f"{role}_scores")
        if (
            scores["scoring_context_id"]
            != record["request"]["scoring_contexts"][role]
        ):
            raise ValueError(f"Balanced score context differs: {role}")
    return record


def _rescore_balanced_cohort(
    prepared: RegisteredPrimary,
    spec: CohortSpec,
    obs: pd.DataFrame,
    balanced: pd.DataFrame,
    *,
    baseline_dir: Path,
    output_dir: Path,
    role: str,
    cells_per_context: int,
) -> list[Path]:
    """Sample one population, rescore it and compare matched-person effects."""
    score_dir = score_cohort(
        spec,
        balanced,
        prepared.alignment,
        prepared.modules,
        reference=role == "reference",
        panel_path=prepared.panel_path,
        output_dir=output_dir / f"{role}_scores",
        target_sum=prepared.target_sum,
        random_state=prepared.random_state,
        aucell_chunk_size=prepared.aucell_chunk_size,
        aucell_num_workers=prepared.aucell_num_workers,
    )
    donors, effects = _paired_scores(
        score_dir,
        balanced,
        spec,
        prepared.modules,
        alpha=prepared.alpha,
    )
    _, baseline = _paired_scores(
        baseline_dir,
        obs,
        spec,
        prepared.modules,
        alpha=prepared.alpha,
    )
    score_manifest = validate_scoring_artifacts(score_dir)
    matched = matched_donor_effects(
        baseline.donor_differences, effects.donor_differences
    )
    estimates = effects.estimates.merge(
        matched, on=["module_id", "scorer"], how="left", validate="one_to_one"
    ).assign(
        cohort_id=spec.cohort_id,
        cohort_role=role,
        variant_id="balanced",
        cells_per_context=cells_per_context,
        scoring_context_id=str(score_manifest["scoring_context_id"]),
        modality=balanced.modality.unique().item(),
    )
    tables = {
        f"{role}_selected_cells.csv.gz": balanced.rename_axis(
            "obs_name"
        ).reset_index(),
        f"{role}_donor_scores.csv.gz": donors,
        f"{role}_contrasts.csv": estimates,
        f"{role}_donor_differences.csv": effects.donor_differences,
    }
    artifacts = [score_dir / "stage_manifest.json"]
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False)
        artifacts.append(path)
    return artifacts


def _paired_scores(
    score_dir: Path,
    selected: pd.DataFrame,
    spec: CohortSpec,
    modules: Sequence[GeneModule],
    *,
    alpha: float,
) -> tuple[pd.DataFrame, PairedContrastResult]:
    """Read intact scores, preserve exact cells and estimate donor effects."""
    donors, eligibility = _donor_scores(score_dir, selected, spec, modules)
    return donors, estimate_module_contrasts(
        donors, eligibility, spec.comparison, alpha=alpha
    )


def _donor_scores(
    score_dir: Path,
    selected: pd.DataFrame,
    spec: CohortSpec,
    modules: Sequence[GeneModule],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate each person's module scores and their measurement eligibility.

    The aggregation depends on the scored cells and the support floor, not on
    which contexts a later comparison names, so one aggregation serves every
    comparison over the same cells.
    """
    manifest = validate_scoring_artifacts(score_dir)
    scores = (
        pd.read_csv(score_dir / "scores.csv.gz", index_col="obs_name")
        if manifest["has_scores"]
        else pd.DataFrame(index=selected.index)
    )
    eligibility = pd.read_csv(score_dir / "score_eligibility.csv")
    scorers: tuple[ScorerName, ...] = ("scanpy", "aucell")
    donors = pd.concat(
        [
            aggregate_module_scores(
                selected,
                scores,
                modules,
                scorer=scorer,
                scoring_context_id=str(manifest["scoring_context_id"]),
                minimum_cells=spec.comparison.minimum_cells,
            )
            for scorer in scorers
        ],
        ignore_index=True,
    )
    return donors, eligibility
