"""Publish gene diagnostics and actual removal effects on the primary cells."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import gene_effects
from nasp_atlas.analysis.external_replication import sensitivity
from nasp_atlas.analysis.external_replication.cohort import (
    prepare_cohort_counts,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    apply_feature_alignment,
)
from nasp_atlas.analysis.external_replication.gene_effects import (
    estimate_gene_removal_effects,
)
from nasp_atlas.analysis.external_replication.gene_effects import (
    paired_gene_expression,
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
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell import context_summary
from nasp_atlas.single_cell import gene_diagnostics
from nasp_atlas.single_cell import gene_sensitivity
from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.associations import paired
from nasp_atlas.single_cell.gene_diagnostics import module_gene_diagnostics
from nasp_atlas.single_cell.gene_sensitivity import gene_removal_sensitivity
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import module_score_name


__all__ = ["analyze_gene_sensitivity", "validate_gene_sensitivity"]


def analyze_gene_sensitivity(
    prepared: RegisteredPrimary,
    *,
    primary_analysis_dir: Path,
    primary_score_dir: Path,
    output_dir: Path,
) -> Path:
    """Reuse primary scores; rescore one removal per arm on identical cells.

    The existing expression-based driver rule requires three diagnostic donors;
    paired effect inference retains the registered external support threshold.
    These data-selected variants are descriptive robustness checks. Unsupported
    parents and lost/degenerate arms remain explicit unavailable rows.
    """
    selected = prepared.intake.selected_obs
    score_request = scoring_request(
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
    score_manifest = validate_scoring_artifacts(
        primary_score_dir, request=score_request
    )
    analysis = validate_analysis_artifacts(primary_analysis_dir)
    primary_request = analysis["request"]
    if not isinstance(primary_request, dict) or (
        primary_request["registration_identity"]
        != prepared.registration.identity
        or primary_request["cohort_id"] != prepared.spec.cohort_id
        or primary_request["score_manifest"]
        != file_identity(primary_score_dir / "stage_manifest.json")
    ):
        raise ValueError(
            "Gene sensitivity must use the registered primary measurement"
        )
    request = {
        "registration_identity": prepared.registration.identity,
        "primary_analysis": file_identity(
            primary_analysis_dir / "analysis_manifest.json"
        ),
        "baseline_scoring_context": score_manifest["scoring_context_id"],
        "diagnostic_minimum_donors": 3,
        "dominant_genes_per_arm": 1,
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (
                gene_effects,
                gene_diagnostics,
                gene_sensitivity,
                context_summary,
                contrasts,
                sensitivity,
                paired,
            )
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    if output_dir.exists():
        validate_gene_sensitivity(output_dir, request=request)
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f"{output_dir.name}.lock")
    with lock_path.open("x") as lock:
        lock.write(json.dumps({"pid": os.getpid()}))
    staging = Path(
        mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        tables = _gene_tables(
            prepared,
            primary_score_dir,
            primary_analysis_dir,
            request_identity(request),
        )
        for name, frame in tables.items():
            frame.to_csv(staging / name, index=False)
        manifest = {
            "status": "completed_gene_sensitivity",
            "request": request,
            "identity": request_identity(request),
            "artifacts": {
                name: {
                    **file_identity(staging / name),
                    "path": str((output_dir / name).resolve()),
                }
                for name in tables
            },
        }
        (staging / "gene_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        staging.rename(output_dir)
    finally:
        lock_path.unlink()
    return output_dir


def validate_gene_sensitivity(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Require complete, intact diagnostics and removal effect artifacts."""
    record = read_stage_manifest(
        output_dir / "gene_manifest.json", stage="gene sensitivity"
    )
    if record["status"] != "completed_gene_sensitivity":
        raise ValueError(f"Incomplete gene sensitivity: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(
            f"Incompatible gene sensitivity; use a new directory: {output_dir}"
        )
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed gene sensitivity request: {output_dir}")
    for name, identity in record["artifacts"].items():
        if file_identity(output_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed gene artifact: {output_dir / name}")
    return record


def _gene_tables(
    prepared: RegisteredPrimary,
    score_dir: Path,
    analysis_dir: Path,
    context_id: str,
) -> dict[str, pd.DataFrame]:
    """Prepare expression once and collect both scorers' gene diagnostics."""
    registered_modules = set(prepared.hypotheses.module_id)
    modules = tuple(
        module
        for module in prepared.modules
        if module.module_id in registered_modules
    )
    expression = prepare_cohort_counts(
        prepared.spec,
        prepared.intake.selected_obs,
        target_sum=prepared.target_sum,
    )
    expression = apply_feature_alignment(
        expression,
        prepared.alignment.external_features,
        prepared.alignment.shared_gene_ids,
    )
    scores = (
        pd.read_csv(score_dir / "scores.csv.gz", index_col="obs_name")
        if (score_dir / "scores.csv.gz").exists()
        else pd.DataFrame(index=expression.obs_names)
    )
    eligibility = pd.read_csv(score_dir / "score_eligibility.csv")
    baseline = pd.read_csv(analysis_dir / "donor_differences.csv")
    schema = ObsSchema(
        study_key="cohort_id", tissue_key="tissue", cell_type_key="population"
    )
    tables: dict[str, pd.DataFrame] = {}
    scorers: tuple[ScorerName, ...] = ("scanpy", "aucell")
    for scorer in scorers:
        baseline_scores = scores.reindex(
            columns=[
                module_score_name(module, scorer=scorer) for module in modules
            ]
        )
        diagnostics = module_gene_diagnostics(
            expression,
            baseline_scores,
            modules,
            schema=schema,
            scorer=scorer,
            minimum_cells=prepared.spec.comparison.minimum_cells,
            minimum_donors=3,
        )
        available = set(
            eligibility.loc[
                eligibility.scorer.eq(scorer) & eligibility.status.eq("ok"),
                "module_id",
            ]
        )
        scored_modules = [
            module for module in modules if module.module_id in available
        ]
        removal = gene_removal_sensitivity(
            expression,
            baseline_scores,
            scored_modules,
            diagnostics,
            schema=schema,
            scorer=scorer,
            dominant_per_arm=True,
            minimum_cells=prepared.spec.comparison.minimum_cells,
            minimum_donors=3,
            random_state=prepared.random_state,
            aucell_chunk_size=prepared.aucell_chunk_size,
            aucell_num_workers=prepared.aucell_num_workers,
        )
        unavailable = [
            {
                "variant_id": f"unavailable:{module.module_id}:{arm}",
                "module_id": module.module_id,
                "mode": "dominant_gene",
                "selection_arm": arm,
                "removed_genes": "",
                "positive_genes": "",
                "inverse_genes": "",
                "status": "baseline_unavailable",
            }
            for module in modules
            if module.module_id not in available
            for arm, members in (
                ("positive", module.positive_genes),
                ("inverse", module.inverse_genes),
            )
            if members
        ]
        if unavailable:
            removal = replace(
                removal,
                variants=pd.concat(
                    [removal.variants, pd.DataFrame(unavailable)],
                    ignore_index=True,
                ),
            )
        effects, arms, donors = estimate_gene_removal_effects(
            removal,
            expression,
            modules,
            prepared.spec.comparison,
            baseline,
            scorer=scorer,
            scoring_context_id=context_id,
            alpha=prepared.alpha,
        )
        outputs = {
            "gene_donor_expression.csv.gz": diagnostics.donor_expression,
            "gene_context_summary.csv": diagnostics.context_summary,
            "gene_paired_differences.csv": paired_gene_expression(
                diagnostics.donor_expression, prepared.spec.comparison
            ),
            "removal_variants.csv": removal.variants,
            "removal_arm_diagnostics.csv": arms,
            "removal_contrasts.csv": effects.estimates,
            "removal_donor_differences.csv": effects.donor_differences,
            "removal_donor_scores.csv.gz": donors,
            "removal_cell_scores.csv.gz": removal.cell_scores.rename_axis(
                "obs_name"
            ).reset_index(),
        }
        tables.update(
            {
                f"{scorer}_{name}": frame.assign(
                    scorer=scorer, scoring_context_id=context_id
                )
                for name, frame in outputs.items()
            }
        )
    return tables
