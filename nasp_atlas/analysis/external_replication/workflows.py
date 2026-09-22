"""Restartable preparation and scoring of one declared population."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import resource
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp

import anndata as ad
import pandas as pd
from nasp_compendium.types import GeneModule

from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import replication
from nasp_atlas.analysis.external_replication import scoring
from nasp_atlas.analysis.external_replication.cohort import (
    prepare_cohort_counts,
)
from nasp_atlas.analysis.external_replication.contrasts import (
    estimate_module_contrasts,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    FeatureAlignment,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    apply_feature_alignment,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.provenance import scoring_request
from nasp_atlas.analysis.external_replication.registered_inputs import (
    RegisteredPrimary,
)
from nasp_atlas.analysis.external_replication.replication import (
    classify_registered_contrasts,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    write_analysis_artifacts,
)
from nasp_atlas.analysis.external_replication.scoring import CohortScores
from nasp_atlas.analysis.external_replication.scoring import (
    aggregate_module_scores,
)
from nasp_atlas.analysis.external_replication.scoring import (
    score_prepared_cohort,
)
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.analysis.score_checkpoint import score_checkpoint_matches
from nasp_atlas.analysis.score_checkpoint import write_score_checkpoint
from nasp_atlas.single_cell.associations import paired
from nasp_atlas.single_cell.module_scoring import ScorerName


logger = logging.getLogger(__name__)

__all__ = [
    "analyze_primary_cohort",
    "score_cohort",
    "validate_scoring_artifacts",
]


def analyze_primary_cohort(
    prepared: RegisteredPrimary,
    *,
    score_dir: Path,
    output_dir: Path,
) -> Path:
    """Aggregate saved scores and publish the full registered primary family.

    The prepared request must have passed frozen-input checks. This stage never
    rescales scores or changes the discovery question. Numerical dependencies
    include selected metadata and inferential code, independently of rendering.
    """
    selected = prepared.intake.selected_obs
    expected_scores = scoring_request(
        prepared.spec,
        selected,
        prepared.alignment.external_features,
        prepared.alignment.shared_gene_ids,
        prepared.modules,
        prepared.alignment.coverage,
        panel_path=prepared.panel_path,
        target_sum=prepared.target_sum,
        random_state=prepared.random_state,
        aucell_chunk_size=prepared.aucell_chunk_size,
        aucell_num_workers=prepared.aucell_num_workers,
    )
    manifest = validate_scoring_artifacts(score_dir, request=expected_scores)
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
                prepared.modules,
                scorer=scorer,
                scoring_context_id=str(manifest["scoring_context_id"]),
                minimum_cells=prepared.spec.comparison.minimum_cells,
            )
            for scorer in scorers
        ],
        ignore_index=True,
    )
    paired_result = estimate_module_contrasts(
        donors, eligibility, prepared.spec.comparison, alpha=prepared.alpha
    )
    classified = classify_registered_contrasts(
        prepared.hypotheses, paired_result.estimates, alpha=prepared.alpha
    ).assign(
        modality=selected.modality.unique().item(),
        scoring_context_id=str(manifest["scoring_context_id"]),
        variant_id="baseline",
        minimum_cells=prepared.spec.comparison.minimum_cells,
    )
    tables = {
        "selected_cells.csv.gz": selected.rename_axis("obs_name").reset_index(),
        "donor_support.csv": prepared.intake.donor_support,
        "preparation_support.csv": prepared.intake.preparation_support,
        "cohort_audit.csv": prepared.intake.selection_audit,
        "module_coverage.csv": prepared.alignment.coverage,
        "feature_crosswalk.csv": prepared.alignment.external_features,
        "donor_scores.csv.gz": donors,
        "module_contrasts.csv": paired_result.estimates,
        "donor_differences.csv": paired_result.donor_differences,
        "replication_contrasts.csv": classified,
    }
    dependencies = (contrasts, replication, scoring, paired)
    request = {
        "cohort_id": prepared.spec.cohort_id,
        "registration_identity": prepared.registration.identity,
        "score_manifest": file_identity(score_dir / "stage_manifest.json"),
        "selected_metadata_sha256": hashlib.sha256(
            selected.to_csv().encode()
        ).hexdigest(),
        "comparison": asdict(prepared.spec.comparison),
        "alpha": prepared.alpha,
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in dependencies
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    return write_analysis_artifacts(tables, request, output_dir=output_dir)


def score_cohort(
    spec: CohortSpec,
    selected_obs: pd.DataFrame,
    alignment: FeatureAlignment,
    modules: Sequence[GeneModule],
    *,
    reference: bool,
    panel_path: Path,
    output_dir: Path,
    target_sum: float = 10_000.0,
    random_state: int = 42,
    aucell_chunk_size: int = 500,
    aucell_num_workers: int = 1,
) -> Path:
    """Publish or validate scores with explicit scientific dependencies.

    This stage performs no hypothesis selection or inference. Its caller must
    complete discovery/registration before using it on external populations.
    A different request or corrupt completed output requires a new directory;
    completed results are never overwritten. Failed attempts remain separately
    identifiable and can be retried at the same output path.
    """
    features = (
        alignment.reference_features
        if reference
        else alignment.external_features
    )
    request = scoring_request(
        spec,
        selected_obs,
        features,
        alignment.shared_gene_ids,
        modules,
        alignment.coverage,
        panel_path=panel_path,
        target_sum=target_sum,
        random_state=random_state,
        aucell_chunk_size=aucell_chunk_size,
        aucell_num_workers=aucell_num_workers,
    )
    with _scoring_attempt(output_dir) as attempt:
        if output_dir.exists():
            validate_scoring_artifacts(output_dir, request=request)
            logger.info("Reusing validated scores: %s", output_dir)
            return output_dir

        started = time.perf_counter()
        prepared = prepare_cohort_counts(
            spec, selected_obs, target_sum=target_sum
        )
        aligned = apply_feature_alignment(
            prepared, features, alignment.shared_gene_ids
        )
        del prepared
        scored = _score_methods(
            aligned,
            modules,
            alignment.coverage,
            random_state=random_state,
            aucell_chunk_size=aucell_chunk_size,
            aucell_num_workers=aucell_num_workers,
        )
        resources = {
            "wall_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "memory_scope": "process lifetime including preceding work",
        }
        _write_scoring_artifacts(attempt, scored, request, resources)
        attempt.replace(output_dir)
    return output_dir


def validate_scoring_artifacts(
    output_dir: Path, *, request: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Require completion, compatible dependencies, hashes and exact cell IDs.

    Without a request this validates stored provenance for a downstream
    consumer, not compatibility with changed source files.
    """
    manifest_path = output_dir / "stage_manifest.json"
    try:
        record = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Missing or invalid score manifest: {manifest_path}"
        ) from error
    if record.get("status") != "completed" or not record.get("artifacts"):
        raise ValueError(f"Incomplete scoring stage: {output_dir}")
    saved_request = record["request"]
    if request is not None and saved_request != dict(request):
        raise ValueError(
            f"Incompatible scores in {output_dir}; use a new output directory"
        )
    if record["scoring_context_id"] != request_identity(saved_request):
        raise ValueError(f"Changed scoring request in {manifest_path}")
    for name, identity in record["artifacts"].items():
        path = output_dir / name
        if (
            not path.is_file()
            or file_identity(path)["sha256"] != identity["sha256"]
        ):
            raise ValueError(f"Missing or changed scoring artifact: {path}")
    eligibility = pd.read_csv(output_dir / "score_eligibility.csv")
    if eligibility.duplicated(["module_id", "scorer"]).any():
        raise ValueError("Duplicate measurement eligibility records")
    if record["has_scores"]:
        score_path = output_dir / "scores.csv.gz"
        if not score_checkpoint_matches(score_path, saved_request):
            raise ValueError(f"Invalid shared score checkpoint: {score_path}")
        cells = pd.read_csv(score_path, usecols=["obs_name"], dtype=str)
        if cells.obs_name.tolist() != saved_request["selected_cells"]:
            raise ValueError(
                "Score observations do not match the frozen population"
            )
    return record


def _score_methods(
    prepared: ad.AnnData,
    modules: Sequence[GeneModule],
    coverage: pd.DataFrame,
    *,
    random_state: int,
    aucell_chunk_size: int,
    aucell_num_workers: int,
) -> CohortScores:
    """Collect both methods without independently standardizing cell blocks."""
    results = []
    scorers: tuple[ScorerName, ...] = ("scanpy", "aucell")
    for scorer in scorers:
        logger.info("Scoring %s on %d cells", scorer, prepared.n_obs)
        results.append(
            score_prepared_cohort(
                prepared,
                modules,
                coverage,
                scorer=scorer,
                random_state=random_state,
                aucell_chunk_size=aucell_chunk_size,
                aucell_num_workers=aucell_num_workers,
            )
        )
    return CohortScores(
        scores=pd.concat([result.scores for result in results], axis=1),
        eligibility=pd.concat(
            [result.eligibility for result in results], ignore_index=True
        ),
        arm_diagnostics=pd.concat(
            [result.arm_diagnostics for result in results], ignore_index=True
        ),
    )


def _write_scoring_artifacts(
    attempt: Path,
    result: CohortScores,
    request: Mapping[str, object],
    resources: Mapping[str, object],
) -> None:
    """Write score products and publish their completion record last."""
    context_id = request_identity(request)
    has_scores = bool(len(result.scores.columns))
    result.eligibility.to_csv(attempt / "score_eligibility.csv", index=False)
    result.arm_diagnostics.to_csv(attempt / "arm_diagnostics.csv", index=False)
    if has_scores:
        score_path = attempt / "scores.csv.gz"
        result.scores.assign(
            scoring_scorers="scanpy,aucell", scoring_context_id=context_id
        ).to_csv(score_path, index=True, index_label="obs_name")
        write_score_checkpoint(score_path, request)
    artifacts = {
        path.name: {
            "sha256": file_identity(path)["sha256"],
            "bytes": path.stat().st_size,
        }
        for path in sorted(attempt.iterdir())
        if path.is_file()
    }
    record = {
        "status": "completed",
        "stage": "score",
        "request": dict(request),
        "scoring_context_id": context_id,
        "has_scores": has_scores,
        "resources": dict(resources),
        "artifacts": artifacts,
    }
    (attempt / "stage_manifest.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


@contextmanager
def _scoring_attempt(output_dir: Path) -> Iterator[Path]:
    """Own one writer and retain failed attempts outside completed output."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.with_name(output_dir.name + ".lock")
    try:
        handle = lock.open("x")
    except FileExistsError as error:
        raise RuntimeError(f"Scoring writer lock exists: {lock}") from error
    attempt: Path | None = None
    try:
        with handle:
            handle.write(
                json.dumps({"pid": os.getpid(), "output": str(output_dir)})
            )
        attempt = Path(
            mkdtemp(
                prefix=f".{output_dir.name}.attempt-", dir=output_dir.parent
            )
        )
        yield attempt
    finally:
        if attempt is not None and attempt.exists():
            error = sys.exc_info()[1]
            if error is None:
                attempt.rmdir()
            else:
                (attempt / "failure.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "error": str(error),
                        },
                        indent=2,
                    )
                    + "\n"
                )
        lock.unlink(missing_ok=True)
