"""Rescore registered modules without the genes that carry their contrast."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from nasp_compendium import GeneModule

from nasp_atlas.analysis.external_replication import balanced_scoring
from nasp_atlas.analysis.external_replication import contrasts
from nasp_atlas.analysis.external_replication import sensitivity
from nasp_atlas.analysis.external_replication.balanced_scoring import (
    _donor_scores,
)
from nasp_atlas.analysis.external_replication.feature_mapping import (
    FeatureAlignment,
)
from nasp_atlas.analysis.external_replication.gene_workflows import (
    validate_gene_sensitivity,
)
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
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.external_replication.workflows import (
    validate_scoring_artifacts,
)
from nasp_atlas.single_cell.associations import paired


__all__ = [
    "rescore_without_contrast_drivers",
    "select_contrast_drivers",
    "validate_contrast_driver_removal",
]


def select_contrast_drivers(
    paired_genes: pd.DataFrame, *, drivers_per_module: int = 2
) -> dict[str, tuple[str, ...]]:
    """Select scored genes by absolute mean paired expression difference.

    This is a post-hoc expression diagnostic, not a decomposition of a module
    score. Arm size, control subtraction, signed-arm standardization and AUCell
    ranks prevent interpreting these differences as score contributions.
    Genes moving either way can be selected; context-dependent members cannot.
    Ties are resolved by gene name. Missing scored-gene differences are errors,
    not zero contributions or an invitation to select from fewer donors.

    Args:
      paired_genes: Per-gene paired differences with module_id, gene, arm and
        mean_expression_difference. Gene differences do not depend on the
        scorer, so either scorer's saved table gives the same selection.
      drivers_per_module: How many genes to name per module.

    Returns:
      Each module id mapped to its selected genes, strongest first.
    """
    if drivers_per_module < 1:
        raise ValueError("Select at least one contrast driver per module")

    if not paired_genes.arm.isin(
        ["positive", "inverse", "context_dependent"]
    ).all():
        raise ValueError("Unknown gene arm in paired expression diagnostics")
    scored = paired_genes.loc[paired_genes.arm.isin(["positive", "inverse"])]
    if not np.isfinite(scored.mean_expression_difference).all():
        raise ValueError("Scored genes require finite paired differences")
    differences = scored.groupby(
        ["module_id", "gene"], as_index=False
    ).mean_expression_difference.mean()
    drivers = {}
    for module_id, frame in differences.groupby("module_id"):
        ranked = frame.assign(
            magnitude=frame.mean_expression_difference.abs()
        ).sort_values(["magnitude", "gene"], ascending=[False, True])
        selected = ranked.loc[ranked.magnitude.gt(0.0)]
        drivers[str(module_id)] = tuple(
            selected.gene.head(drivers_per_module).astype(str)
        )
    return drivers


def rescore_without_contrast_drivers(
    prepared: RegisteredPrimary,
    *,
    primary_analysis_dir: Path,
    gene_dir: Path,
    output_dir: Path,
    drivers_per_module: int = 2,
) -> Path:
    """Re-estimate every registered module without its contrast drivers.

    This changes the measurement, not the population: the same cells and the
    same people are rescored under a reduced module definition. It is a
    descriptive check on how few genes a contrast rests on, and it cannot
    redefine a frozen primary decision. An arm emptied by removal becomes
    unscorable rather than zero, and each estimate is also compared with the
    primary on the donors the two analyses share.
    """
    primary = validate_analysis_artifacts(primary_analysis_dir)
    original = primary["request"]
    if (
        not isinstance(original, dict)
        or original["registration_identity"] != prepared.registration.identity
        or original["cohort_id"] != prepared.spec.cohort_id
    ):
        raise ValueError("Driver removal and primary registration differ")
    gene_manifest = validate_gene_sensitivity(gene_dir)
    gene_request = gene_manifest["request"]
    primary_identity = file_identity(
        primary_analysis_dir / "analysis_manifest.json"
    )
    if (
        not isinstance(gene_request, dict)
        or gene_request["registration_identity"]
        != prepared.registration.identity
        or gene_request["primary_analysis"] != primary_identity
    ):
        raise ValueError("Driver diagnostics must use this primary analysis")

    paired_genes = pd.read_csv(gene_dir / "scanpy_gene_paired_differences.csv")
    drivers = select_contrast_drivers(
        paired_genes, drivers_per_module=drivers_per_module
    )
    reduced = _reduced_modules(prepared.modules, drivers)
    donors, eligibility, score_manifest = _driver_scores(
        prepared,
        reduced,
        drivers,
        primary_analysis_dir=primary_analysis_dir,
        output_dir=output_dir,
    )

    request = {
        "registration_identity": prepared.registration.identity,
        "primary_analysis": primary_identity,
        "gene_manifest": file_identity(gene_dir / "gene_manifest.json"),
        "score_manifest": score_manifest,
        "analysis_role": "post_hoc_measurement_sensitivity",
        "selection_rule": "absolute_mean_paired_expression_difference",
        "removed_genes": {
            module_id: list(genes) for module_id, genes in drivers.items()
        },
        "drivers_per_module": drivers_per_module,
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
        validate_contrast_driver_removal(output_dir, request=request)
        return output_dir

    effects = contrasts.estimate_module_contrasts(
        donors,
        eligibility,
        prepared.spec.comparison,
        alpha=prepared.alpha,
    )
    baseline = pd.read_csv(primary_analysis_dir / "donor_differences.csv")
    matched = matched_donor_effects(baseline, effects.donor_differences)
    estimates = effects.estimates.merge(
        matched, on=["module_id", "scorer"], how="left", validate="one_to_one"
    ).assign(
        variant_id="contrast_driver_removal",
        variant_kind="module_definition",
        analysis_role="post_hoc_measurement_sensitivity",
        cohort_id=prepared.spec.cohort_id,
        cohort_spec_id=prepared.spec.cohort_id,
        removed_genes=lambda frame: frame.module_id.map(
            {module_id: ";".join(genes) for module_id, genes in drivers.items()}
        ).fillna("none_selected"),
        scoring_context_id=(
            donors.scoring_context_id.iloc[0] if not donors.empty else np.nan
        ),
    )
    estimates["n_matched_donors"] = estimates.n_matched_donors.fillna(0).astype(
        int
    )
    estimates["matched_donors"] = estimates.matched_donors.fillna("[]")
    publish_stage_tables(
        analysis_dir,
        request,
        {
            "contrasts.csv": estimates,
            "donor_differences.csv": effects.donor_differences,
            "donor_scores.csv.gz": donors,
        },
        status="completed_contrast_driver_removal",
        manifest_name="contrast_driver_manifest.json",
    )
    return output_dir


def validate_contrast_driver_removal(
    output_dir: Path, *, request: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Require intact scores and complete driver-removal artifacts."""
    analysis_dir = output_dir / "analysis"
    record = read_stage_manifest(
        analysis_dir / "contrast_driver_manifest.json",
        stage="contrast driver removal",
    )
    if record["status"] != "completed_contrast_driver_removal":
        raise ValueError(f"Incomplete contrast driver removal: {output_dir}")
    if request is not None and record["request"] != dict(request):
        raise ValueError(
            f"Incompatible contrast driver removal; use a new directory: "
            f"{output_dir}"
        )
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed driver removal request: {output_dir}")
    expected_scores = record["request"]["score_manifest"]
    if expected_scores is not None:
        score_path = output_dir / "scores" / "stage_manifest.json"
        if file_identity(score_path) != expected_scores:
            raise ValueError(f"Changed driver scores: {score_path}")
        validate_scoring_artifacts(score_path.parent)
    for name, identity in record["artifacts"].items():
        if file_identity(analysis_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed driver removal artifact: {name}")
    return record


def _driver_scores(
    prepared: RegisteredPrimary,
    reduced: Sequence[GeneModule],
    drivers: Mapping[str, Sequence[str]],
    *,
    primary_analysis_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str | int] | None]:
    """Score retained definitions and preserve emptied arms as unavailable."""
    lost = {
        original.module_id
        for original, variant in zip(prepared.modules, reduced, strict=True)
        if (original.positive_genes and not variant.positive_genes)
        or (original.inverse_genes and not variant.inverse_genes)
    }
    scorable = [module for module in reduced if module.module_id not in lost]
    unavailable = pd.DataFrame(
        [
            {"module_id": module, "scorer": scorer, "status": "unscorable_arm"}
            for module in sorted(lost)
            for scorer in ("scanpy", "aucell")
        ],
        columns=["module_id", "scorer", "status"],
    )
    if not scorable:
        donors = pd.read_csv(
            primary_analysis_dir / "donor_scores.csv.gz", nrows=0
        )
        return donors, unavailable, None

    score_dir = score_cohort(
        prepared.spec,
        prepared.intake.selected_obs,
        _reduced_alignment(prepared.alignment, drivers),
        scorable,
        reference=False,
        panel_path=prepared.panel_path,
        output_dir=output_dir / "scores",
        target_sum=prepared.target_sum,
        random_state=prepared.random_state,
        aucell_chunk_size=prepared.aucell_chunk_size,
        aucell_num_workers=prepared.aucell_num_workers,
    )
    donors, eligibility = _donor_scores(
        score_dir, prepared.intake.selected_obs, prepared.spec, scorable
    )
    return (
        donors,
        pd.concat([eligibility, unavailable], ignore_index=True),
        file_identity(score_dir / "stage_manifest.json"),
    )


def _reduced_alignment(
    alignment: FeatureAlignment, drivers: Mapping[str, Sequence[str]]
) -> FeatureAlignment:
    """Drop the removed members from the module coverage audit.

    The scorer checks each module's declared arms against this audit, so a
    reduced definition needs a matching audit or its own members look like a
    coverage failure. The shared background is deliberately untouched: removal
    changes the signature, not the expression universe it is scored against.
    """
    removed = {
        (module_id, gene)
        for module_id, genes in drivers.items()
        for gene in genes
    }
    keys = pd.MultiIndex.from_arrays(
        [alignment.coverage.module_id, alignment.coverage.gene]
    )
    retained = alignment.coverage.loc[~keys.isin(removed)]
    return dataclasses.replace(alignment, coverage=retained)


def _reduced_modules(
    modules: Sequence[GeneModule], drivers: Mapping[str, Sequence[str]]
) -> tuple[GeneModule, ...]:
    """Drop each module's selected drivers from its signed arms.

    A scored module the gene stage did not cover, such as a cohort's
    descriptive-only module, has no drivers to remove and passes through
    unchanged. It is still scored so the eligibility table stays consistent,
    and its row is labelled so an unchanged estimate cannot be misread as
    removal having no effect.
    """
    reduced = []
    for module in modules:
        removed = set(drivers.get(module.module_id, ()))
        if not removed:
            reduced.append(module)
            continue
        reduced.append(
            dataclasses.replace(
                module,
                positive_genes=tuple(
                    gene
                    for gene in module.positive_genes
                    if gene not in removed
                ),
                inverse_genes=tuple(
                    gene for gene in module.inverse_genes if gene not in removed
                ),
            )
        )
    return tuple(reduced)
