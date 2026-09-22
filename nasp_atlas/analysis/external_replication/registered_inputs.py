"""Resolve the frozen primary request before reading expression or scoring."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from nasp_compendium import GeneModules
from nasp_compendium.types import GeneModule

from nasp_atlas.analysis.external_replication.cohort import CohortIntake
from nasp_atlas.analysis.external_replication.cohort import read_cohort_metadata
from nasp_atlas.analysis.external_replication.cohort import select_cohort
from nasp_atlas.analysis.external_replication.feature_mapping import (
    FeatureAlignment,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    align_cohort_features,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    panel_aliases,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registration import Registration
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)
from nasp_atlas.analysis.external_replication.registration import (
    registered_questions,
)
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.analysis.external_replication.specification import (
    read_cohort_spec,
)


__all__ = ["RegisteredPrimary", "prepare_registered_primary"]


@dataclass(frozen=True, kw_only=True)
class RegisteredPrimary:
    """Verified inputs that travel through scoring and donor inference."""

    registration: Registration
    spec: CohortSpec
    intake: CohortIntake
    reference_spec: CohortSpec
    reference_intake: CohortIntake
    alignment: FeatureAlignment
    modules: tuple[GeneModule, ...]
    panel_path: Path
    hypotheses: pd.DataFrame
    target_sum: float
    random_state: int
    aucell_chunk_size: int
    aucell_num_workers: int
    alpha: float


@dataclass(frozen=True, kw_only=True)
class _FrozenCohortInputs:
    """Verified frozen files and settings before cohort-specific gates."""

    registration: Registration
    cohort: dict[str, Any]
    spec: CohortSpec
    reference: CohortSpec
    settings: dict[str, Any]
    inference: dict[str, Any]


def prepare_registered_primary(
    *,
    registration_dir: Path,
    cohort_key: str,
    spec_path: Path,
    reference_path: Path,
    panel_path: Path,
) -> RegisteredPrimary:
    """Verify the freeze, source gates, exact cells and curated definitions.

    This reads metadata and hashes source files but never reads expression
    matrices or computes effects. Pending source/discovery gates stop primary
    execution. Resolved unavailable hypotheses remain in the result family.
    """
    frozen = _read_frozen_cohort_inputs(
        registration_dir=registration_dir,
        cohort_key=cohort_key,
        spec_path=spec_path,
        reference_path=reference_path,
        panel_path=panel_path,
    )
    registration = frozen.registration
    cohort = frozen.cohort
    if cohort["source_gate"] != "resolved":
        raise ValueError(f"Unresolved source gate for {cohort_key}")
    spec = frozen.spec
    reference = frozen.reference

    hypotheses = registered_questions(
        registration, cohort_key, analysis_role="primary"
    )
    if (
        hypotheses.empty
        or not hypotheses.eligibility_status.isin(
            ["eligible", "unavailable"]
        ).all()
    ):
        raise ValueError(
            f"Resolve discovery eligibility before scoring {cohort_key}"
        )
    for key in (
        "source_audit",
        f"{cohort_key}_discovery_contrasts",
        f"{cohort_key}_discovery_score_manifest",
    ):
        artifact = registration.specification[key]
        if not isinstance(artifact, dict):
            raise ValueError(f"Missing registered evidence: {key}")
        if (
            file_identity(Path(artifact["path"]))["sha256"]
            != artifact["sha256"]
        ):
            raise ValueError(f"Registered evidence has changed: {key}")
    obs, features = read_cohort_metadata(spec)
    reference_obs, reference_features = read_cohort_metadata(reference)
    intake = select_cohort(obs, spec)
    reference_intake = select_cohort(reference_obs, reference)
    for selected, key in (
        (intake.selected_obs, "selected_cells_sha256"),
        (reference_intake.selected_obs, "reference_selected_cells_sha256"),
    ):
        identity = request_identity(
            {"cells": selected.index.astype(str).tolist()}
        )
        if identity != cohort[key]:
            raise ValueError(f"Selected cells differ from registration: {key}")

    catalog = GeneModules(panel_path=panel_path)
    module_ids = [*hypotheses.module_id, *cohort.get("descriptive_modules", [])]
    modules = tuple(
        catalog.get_module(module_id, output="symbols")
        for module_id in dict.fromkeys(module_ids)
    )
    alignment = align_cohort_features(
        reference_features,
        features,
        modules,
        reference_id_column=reference.gene_id_column,
        reference_symbol_column=reference.gene_symbol_column,
        external_id_column=spec.gene_id_column,
        external_symbol_column=spec.gene_symbol_column,
        aliases=panel_aliases(catalog.panel),
    )
    return RegisteredPrimary(
        registration=registration,
        spec=spec,
        intake=intake,
        reference_spec=reference,
        reference_intake=reference_intake,
        alignment=alignment,
        modules=modules,
        panel_path=panel_path,
        hypotheses=hypotheses,
        target_sum=frozen.settings["target_sum"],
        random_state=frozen.settings["random_state"],
        aucell_chunk_size=frozen.settings["aucell_chunk_size"],
        aucell_num_workers=frozen.settings["aucell_num_workers"],
        alpha=frozen.inference["alpha"],
    )


def _read_frozen_cohort_inputs(
    *,
    registration_dir: Path,
    cohort_key: str,
    spec_path: Path,
    reference_path: Path,
    panel_path: Path,
) -> _FrozenCohortInputs:
    """Verify immutable files without requiring a resolved scientific gate."""
    registration = read_registration(registration_dir)
    cohorts = registration.specification["cohorts"]
    settings = registration.specification["scoring"]
    inference = registration.specification["inference"]
    if (
        not isinstance(cohorts, dict)
        or not isinstance(settings, dict)
        or not isinstance(inference, dict)
    ):
        raise ValueError(
            "Registration requires cohort, scoring and inference records"
        )
    cohort = cohorts[cohort_key]
    if not isinstance(cohort, dict):
        raise ValueError(f"Invalid frozen cohort record: {cohort_key}")
    spec = read_cohort_spec(spec_path)
    reference = read_cohort_spec(reference_path)
    for supplied, expected in (
        (spec, cohort["cohort_spec"]),
        (reference, cohort["reference_spec"]),
    ):
        record = asdict(supplied)
        record["input_path"] = str(supplied.input_path)
        if request_identity(record) != request_identity(expected):
            raise ValueError(
                f"Specification differs from freeze: {supplied.cohort_id}"
            )
    for path, expected in (
        (spec.input_path, cohort["source"]),
        (reference.input_path, cohort["reference_source"]),
        (panel_path, registration.specification["panel"]),
    ):
        if (
            not isinstance(expected, dict)
            or file_identity(path)["sha256"] != expected["sha256"]
        ):
            raise ValueError(f"Source differs from frozen registration: {path}")
    compendium = json.loads(
        importlib.metadata.distribution("nasp_compendium").read_text(
            "direct_url.json"
        )
        or "null"
    )
    if compendium != registration.specification["compendium_install"]:
        raise ValueError("Installed compendium differs from the frozen source")
    return _FrozenCohortInputs(
        registration=registration,
        cohort=cohort,
        spec=spec,
        reference=reference,
        settings=settings,
        inference=inference,
    )
