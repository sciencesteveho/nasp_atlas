"""The frozen population extensions: Wells nonclassical, muscle subtypes."""

from __future__ import annotations

import inspect
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.external_replication import balanced_scoring
from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import replication
from nasp_atlas.analysis.external_replication.balanced_scoring import (
    _donor_scores,
)
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
from nasp_atlas.analysis.external_replication.registration import (
    registered_questions,
)
from nasp_atlas.analysis.external_replication.replication import (
    classify_registered_contrasts,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    publish_stage_tables,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.specification import (
    PopulationSpec,
)
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell.associations import paired


__all__ = [
    "analyze_muscle_extension",
    "analyze_wells_extension",
    "validate_extension_artifacts",
]


def analyze_wells_extension(
    primary: RegisteredPrimary, *, output_dir: Path
) -> Path:
    """Score the registered extension population and test its own Holm family.

    The primary preparation resolves source and technical gates. The extension
    changes only the population mapping; both its spec and exact selected cells
    must match the pre-score freeze. No TS sign or primary success is required.
    """
    cohorts = primary.registration.specification["cohorts"]
    if not isinstance(cohorts, dict):
        raise ValueError("Missing frozen cohort records")
    cohort = cohorts["wells"]
    frozen = cohort["extension_spec"]
    populations = tuple(
        PopulationSpec(
            name=row["name"], column=row["column"], labels=tuple(row["labels"])
        )
        for row in frozen["populations"]
    )
    spec = replace(primary.spec, populations=populations)
    supplied = asdict(spec)
    supplied["input_path"] = str(spec.input_path)
    if request_identity(supplied) != request_identity(frozen):
        raise ValueError("Wells extension differs from the frozen selection")
    metadata, _ = read_cohort_metadata(spec)
    intake = select_cohort(metadata, spec)
    selected_identity = request_identity(
        {"cells": intake.selected_obs.index.astype(str).tolist()}
    )
    if selected_identity != cohort["extension_selected_cells_sha256"]:
        raise ValueError("Wells extension cells differ from the registration")
    questions = primary.registration.hypotheses.loc[
        primary.registration.hypotheses.cohort_id.eq(spec.cohort_id)
        & primary.registration.hypotheses.analysis_role.eq("extension")
    ]
    if questions.empty or questions.family_id.nunique() != 1:
        raise ValueError("Require the registered Wells extension family")
    module_ids = set(questions.module_id)
    modules = tuple(
        module for module in primary.modules if module.module_id in module_ids
    )
    if {module.module_id for module in modules} != module_ids:
        raise ValueError("Extension requires the same curated primary modules")
    score_dir = score_cohort(
        spec,
        intake.selected_obs,
        primary.alignment,
        modules,
        reference=False,
        panel_path=primary.panel_path,
        output_dir=output_dir / "scores",
        target_sum=primary.target_sum,
        random_state=primary.random_state,
        aucell_chunk_size=primary.aucell_chunk_size,
        aucell_num_workers=primary.aucell_num_workers,
    )
    request = {
        "registration_identity": primary.registration.identity,
        "cohort_id": spec.cohort_id,
        "analysis_role": "extension",
        "score_manifest": file_identity(score_dir / "stage_manifest.json"),
        "comparison": asdict(spec.comparison),
        "alpha": primary.alpha,
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (balanced_scoring, contrasts, replication, paired)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    analysis_dir = output_dir / "analysis"
    if analysis_dir.exists():
        validate_extension_artifacts(output_dir, request=request)
        return output_dir
    donors, effects = _paired_scores(
        score_dir, intake.selected_obs, spec, modules, alpha=primary.alpha
    )
    score_record = validate_scoring_artifacts(score_dir)
    classified = classify_registered_contrasts(
        questions, effects.estimates, alpha=primary.alpha
    ).assign(
        scoring_context_id=str(score_record["scoring_context_id"]),
        modality=intake.selected_obs.modality.unique().item(),
        population=intake.selected_obs.population.unique().item(),
    )
    tables = {
        "extension_contrasts.csv": classified,
        "donor_differences.csv": effects.donor_differences,
        "donor_scores.csv.gz": donors,
        "selected_cells.csv.gz": intake.selected_obs.rename_axis(
            "obs_name"
        ).reset_index(),
        "donor_support.csv": intake.donor_support,
        "preparation_support.csv": intake.preparation_support,
        "cohort_audit.csv": intake.selection_audit,
    }
    publish_stage_tables(
        analysis_dir,
        request,
        tables,
        status="completed_extension_analysis",
        manifest_name="extension_manifest.json",
    )
    return output_dir


def analyze_muscle_extension(
    primary: RegisteredPrimary,
    *,
    cohort_key: str,
    primary_score_dir: Path,
    output_dir: Path,
) -> Path:
    """Test each registered vascular subtype against CD4 T cells.

    No cell is rescored: subtypes relabel the primary scored vascular cells,
    so every test uses the primary scoring population and the primary donor
    universe. The subtype contrasts form one separate Holm family, AUCell stays
    a sensitivity, and unsupported subtype/module rows remain registered.
    Subtype frequencies are descriptive: a broad mean can reflect composition.
    """
    cohorts = primary.registration.specification["cohorts"]
    if not isinstance(cohorts, dict):
        raise ValueError("Missing frozen cohort records")
    frozen = cohorts[cohort_key]["extension"]
    questions = registered_questions(
        primary.registration, cohort_key, analysis_role="extension"
    )
    subtypes = tuple(frozen["subtypes"])
    if questions.family_id.nunique() != 1 or set(
        questions.target_context
    ) != set(subtypes):
        raise ValueError("Require the registered subtype extension family")
    module_ids = set(questions.module_id)
    modules = tuple(
        module for module in primary.modules if module.module_id in module_ids
    )
    if {module.module_id for module in modules} != module_ids:
        raise ValueError("Extension requires the same curated primary modules")

    spec = primary.spec
    selected = primary.intake.selected_obs
    is_vascular = selected.population.eq(spec.comparison.target)
    labels = selected[f"source_{frozen['source_column']}"].astype(str)
    unexpected = set(labels[is_vascular]) - set(subtypes)
    if unexpected:
        raise ValueError(
            f"Vascular cells have unregistered subtypes: {unexpected}"
        )
    relabeled = selected.assign(
        population=labels.where(is_vascular, selected.population.astype(str))
    )
    score_record = validate_scoring_artifacts(primary_score_dir)
    request = {
        "registration_identity": primary.registration.identity,
        "cohort_id": spec.cohort_id,
        "analysis_role": "extension",
        "score_manifest": file_identity(
            primary_score_dir / "stage_manifest.json"
        ),
        "comparison": asdict(spec.comparison),
        "subtypes": list(subtypes),
        "alpha": primary.alpha,
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (balanced_scoring, contrasts, replication, paired)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    analysis_dir = output_dir / "analysis"
    if analysis_dir.exists():
        validate_extension_artifacts(output_dir, request=request)
        return output_dir

    donors, eligibility = _donor_scores(
        primary_score_dir, relabeled, spec, modules
    )
    estimates = []
    differences = []
    for subtype in subtypes:
        effects = contrasts.estimate_module_contrasts(
            donors,
            eligibility,
            replace(spec.comparison, target=subtype),
            alpha=primary.alpha,
        )
        estimates.append(effects.estimates)
        differences.append(effects.donor_differences)
    classified = classify_registered_contrasts(
        questions, pd.concat(estimates, ignore_index=True), alpha=primary.alpha
    ).assign(
        scoring_context_id=str(score_record["scoring_context_id"]),
        modality=selected.modality.unique().item(),
    )

    subtype_cells = (
        relabeled.loc[is_vascular]
        .groupby(["donor_id", "population"])
        .size()
        .rename("n_cells")
        .reset_index()
        .rename(columns={"population": "subtype"})
    )
    subtype_cells["fraction_of_vascular_cells"] = subtype_cells.n_cells / (
        subtype_cells.groupby("donor_id").n_cells.transform("sum")
    )
    tables = {
        "extension_contrasts.csv": classified,
        "donor_differences.csv": pd.concat(differences, ignore_index=True),
        "donor_scores.csv.gz": donors,
        "subtype_frequencies.csv": subtype_cells,
        "selected_cells.csv.gz": relabeled.rename_axis(
            "obs_name"
        ).reset_index(),
        "donor_support.csv": primary.intake.donor_support,
    }
    publish_stage_tables(
        analysis_dir,
        request,
        tables,
        status="completed_extension_analysis",
        manifest_name="extension_manifest.json",
    )
    return output_dir


def validate_extension_artifacts(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Require intact scores and the complete registered extension result."""
    analysis_dir = output_dir / "analysis"
    record = read_stage_manifest(
        analysis_dir / "extension_manifest.json", stage="extension"
    )
    if record["status"] != "completed_extension_analysis":
        raise ValueError(f"Incomplete extension: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(f"Incompatible extension: {output_dir}")
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed extension request: {output_dir}")
    expected = record["request"]["score_manifest"]
    score_manifest = Path(expected["path"])
    if file_identity(score_manifest) != expected:
        raise ValueError(f"Changed extension scores: {score_manifest}")
    validate_scoring_artifacts(score_manifest.parent)
    for name, identity in record["artifacts"].items():
        if file_identity(analysis_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed extension artifact: {name}")
    return record
