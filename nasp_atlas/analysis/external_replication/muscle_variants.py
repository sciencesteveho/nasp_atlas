"""Muscle released-QC and nucleus rescoring sensitivities."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.external_replication import balanced_scoring
from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import sensitivity
from nasp_atlas.analysis.external_replication.balanced_scoring import (
    _paired_scores,
)
from nasp_atlas.analysis.external_replication.cohort import read_cohort_metadata
from nasp_atlas.analysis.external_replication.cohort import select_cohort
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registered_inputs import (
    RegisteredPrimary,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    publish_stage_tables,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    validate_analysis_artifacts,
)
from nasp_atlas.analysis.external_replication.sensitivity import (
    matched_donor_effects,
)
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell.associations import paired


__all__ = ["rescore_muscle_variants", "validate_muscle_variants"]


def rescore_muscle_variants(
    prepared: RegisteredPrimary,
    *,
    primary_analysis_dir: Path,
    output_dir: Path,
) -> Path:
    """Rescore the as-released cells and the nuclei, each on its own cells.

    "as_released" removes only the released doublet-flag filter, so donor
    membership can change; it equals the published Scrublet-threshold cell set
    for cells and is reported once. "nuclei" replaces the cell modality with
    nuclei under the same population and flag rules. Both are changed
    measurements scored in their own population, compared with the primary on
    the donors they share, and never extra independent people.
    """
    settings = prepared.registration.specification["sensitivities"]
    if (
        not isinstance(settings, dict)
        or settings.get("muscle_QC")
        != ["as_released", "scrublet_score_at_most_0.4"]
        or not str(settings.get("muscle_modality", "")).startswith(
            "nuclei scored separately"
        )
    ):
        raise ValueError(
            "Muscle QC and nuclei sensitivities are not registered"
        )
    primary = validate_analysis_artifacts(primary_analysis_dir)
    original = primary["request"]
    if (
        not isinstance(original, dict)
        or original["registration_identity"] != prepared.registration.identity
        or original["cohort_id"] != prepared.spec.cohort_id
    ):
        raise ValueError("Muscle variants and primary registration differ")

    variant_specs = {
        "as_released": _without_filter(prepared.spec, "is_doublet"),
        "nuclei": _with_filter(prepared.spec, "batch", ("nuclei",)),
    }
    selections = {}
    for variant_id, spec in variant_specs.items():
        metadata, _ = read_cohort_metadata(spec)
        intake = select_cohort(metadata, spec)
        score_dir = score_cohort(
            spec,
            intake.selected_obs,
            prepared.alignment,
            prepared.modules,
            reference=False,
            panel_path=prepared.panel_path,
            output_dir=output_dir / variant_id / "scores",
            target_sum=prepared.target_sum,
            random_state=prepared.random_state,
            aucell_chunk_size=prepared.aucell_chunk_size,
            aucell_num_workers=prepared.aucell_num_workers,
        )
        selections[variant_id] = (spec, intake, score_dir)

    request = {
        "registration_identity": prepared.registration.identity,
        "primary_analysis": file_identity(
            primary_analysis_dir / "analysis_manifest.json"
        ),
        "score_manifests": {
            variant_id: file_identity(score_dir / "stage_manifest.json")
            for variant_id, (_, _, score_dir) in selections.items()
        },
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (balanced_scoring, contrasts, sensitivity, paired)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    analysis_dir = output_dir / "analysis"
    if analysis_dir.exists():
        validate_muscle_variants(output_dir, request=request)
        return output_dir

    baseline = pd.read_csv(primary_analysis_dir / "donor_differences.csv")
    contrast_tables = []
    tables: dict[str, pd.DataFrame] = {}
    for variant_id, (spec, intake, score_dir) in selections.items():
        donors, effects = _paired_scores(
            score_dir,
            intake.selected_obs,
            spec,
            prepared.modules,
            alpha=prepared.alpha,
        )
        matched = matched_donor_effects(baseline, effects.donor_differences)
        contrast_tables.append(
            effects.estimates.merge(
                matched,
                on=["module_id", "scorer"],
                how="left",
                validate="one_to_one",
            ).assign(
                variant_id=variant_id,
                variant_kind="released_selection",
                cohort_id=spec.cohort_id,
                modality=intake.selected_obs.modality.unique().item(),
                scoring_context_id=str(
                    validate_scoring_artifacts(score_dir)["scoring_context_id"]
                ),
            )
        )
        tables |= {
            f"{variant_id}_donor_differences.csv": effects.donor_differences,
            f"{variant_id}_donor_scores.csv.gz": donors,
            f"{variant_id}_donor_support.csv": intake.donor_support,
            f"{variant_id}_selection_audit.csv": intake.selection_audit,
        }
    tables["contrasts.csv"] = pd.concat(contrast_tables, ignore_index=True)
    publish_stage_tables(
        analysis_dir,
        request,
        tables,
        status="completed_muscle_variants",
        manifest_name="muscle_variants_manifest.json",
    )
    return output_dir


def validate_muscle_variants(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Require intact scores and complete muscle variant artifacts."""
    analysis_dir = output_dir / "analysis"
    record = read_stage_manifest(
        analysis_dir / "muscle_variants_manifest.json", stage="muscle variants"
    )
    if record["status"] != "completed_muscle_variants":
        raise ValueError(f"Incomplete muscle variants: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(f"Incompatible muscle variants: {output_dir}")
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed muscle variants request: {output_dir}")
    for variant_id, expected in record["request"]["score_manifests"].items():
        score_manifest = (
            output_dir / variant_id / "scores" / "stage_manifest.json"
        )
        if file_identity(score_manifest) != expected:
            raise ValueError(f"Changed {variant_id} scores: {score_manifest}")
        validate_scoring_artifacts(score_manifest.parent)
    for name, identity in record["artifacts"].items():
        if file_identity(analysis_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed muscle variant artifact: {name}")
    return record


def _without_filter(spec: CohortSpec, column: str) -> CohortSpec:
    """Return the specification with one source-column filter removed."""
    if not any(key == column for key, _ in spec.filters):
        raise ValueError(f"Primary selection has no {column} filter to relax")
    return replace(
        spec,
        filters=tuple(
            (key, values) for key, values in spec.filters if key != column
        ),
    )


def _with_filter(
    spec: CohortSpec, column: str, values: tuple[str | bool, ...]
) -> CohortSpec:
    """Return the specification with one existing filter's values replaced."""
    if not any(key == column for key, _ in spec.filters):
        raise ValueError(f"Primary selection has no {column} filter to replace")
    return replace(
        spec,
        filters=tuple(
            (key, values if key == column else existing)
            for key, existing in spec.filters
        ),
    )
