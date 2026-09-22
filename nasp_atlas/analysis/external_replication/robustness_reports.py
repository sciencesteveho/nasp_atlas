"""Publish a robustness report from completed numerical artifacts."""

from __future__ import annotations

import inspect
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nasp_atlas.analysis.external_replication import reporting
from nasp_atlas.analysis.external_replication import robustness_tables
from nasp_atlas.analysis.external_replication.balanced_scoring import (
    validate_balanced_artifacts,
)
from nasp_atlas.analysis.external_replication.chemistry_workflows import (
    validate_chemistry_artifacts,
)
from nasp_atlas.analysis.external_replication.gene_workflows import (
    validate_gene_sensitivity,
)
from nasp_atlas.analysis.external_replication.muscle_variants import (
    validate_muscle_variants,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registration import Registration
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)
from nasp_atlas.analysis.external_replication.reporting import (
    read_report_limitations,
)
from nasp_atlas.analysis.external_replication.reporting import (
    render_replication_report,
)
from nasp_atlas.analysis.external_replication.reporting import (
    render_robustness_section,
)
from nasp_atlas.analysis.external_replication.reports import (
    validate_primary_report,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.robustness_tables import (
    gene_support_summary,
)
from nasp_atlas.analysis.external_replication.robustness_tables import (
    sensitivity_display_table,
)
from nasp_atlas.analysis.external_replication.sensitivity_workflows import (
    validate_fixed_sensitivity,
)
from nasp_atlas.single_cell.visualization import replication as plots


__all__ = ["publish_robustness_report", "validate_robustness_report"]


def publish_robustness_report(
    *,
    primary_report_dir: Path,
    fixed_dir: Path,
    balanced_dir: Path,
    gene_dir: Path,
    variants_dir: Path,
    registration_dir: Path,
    cohort_key: str,
    output_dir: Path,
    title: str,
    reproduction_command: str,
    limitations_path: Path | None = None,
) -> Path:
    """Retain primary figures and add checks without opening expression.

    Every numerical input must refer to the same primary analysis.
    `variants_dir` holds the cohort's rescored selection variants: pooled
    chemistry for Wells, released QC and nuclei for muscle. This report
    completes the robustness checks, while explicitly leaving the population
    extension incomplete. No sensitivity can change a frozen primary
    classification.
    """
    primary = validate_primary_report(primary_report_dir)
    stages = [
        (
            "fixed",
            fixed_dir / "sensitivity_manifest.json",
            validate_fixed_sensitivity(fixed_dir),
            "primary_analysis",
        ),
        (
            "balanced",
            balanced_dir / "balanced_manifest.json",
            validate_balanced_artifacts(balanced_dir),
            "primary_analysis_manifest",
        ),
        (
            "gene",
            gene_dir / "gene_manifest.json",
            validate_gene_sensitivity(gene_dir),
            "primary_analysis",
        ),
        (
            "variants",
            *_validate_variants(variants_dir),
            "primary_analysis",
        ),
    ]
    registration = read_registration(registration_dir)
    primary_request = primary["request"]
    if (
        not isinstance(primary_request, dict)
        or primary_request["registration_identity"] != registration.identity
    ):
        raise ValueError("Report inputs use different registrations")
    for name, _, stage, primary_key in stages:
        stage_request = stage["request"]
        if (
            not isinstance(stage_request, dict)
            or stage_request[primary_key]
            != primary_request["analysis_manifest"]
        ):
            raise ValueError(
                f"{name} sensitivity refers to another primary analysis"
            )
    request = {
        "primary_report": file_identity(
            primary_report_dir / "report_manifest.json"
        ),
        "registration_identity": registration.identity,
        "stages": {name: file_identity(path) for name, path, _, _ in stages},
        "cohort_key": cohort_key,
        "title": title,
        "reproduction_command": reproduction_command,
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (reporting, robustness_tables, plots)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    if limitations_path is not None:
        request["limitations"] = file_identity(limitations_path)
    if output_dir.exists():
        validate_robustness_report(output_dir, request=request)
        return output_dir

    limitations = read_report_limitations(
        registration.specification["limitations"], limitations_path
    )

    results = pd.read_csv(
        primary_report_dir / "figures" / "replication_displayed.csv"
    )
    if set(results.cohort_id) != {cohort_key}:
        raise ValueError("Report cohort differs from the saved primary results")
    discovery = pd.read_csv(
        primary_report_dir / "figures" / "discovery_displayed.csv"
    )
    fixed = pd.read_csv(fixed_dir / "sensitivity_contrasts.csv")
    balanced = pd.read_csv(balanced_dir / "external_contrasts.csv")
    removals = pd.concat(
        [
            pd.read_csv(gene_dir / f"{scorer}_removal_contrasts.csv")
            for scorer in ("scanpy", "aucell")
        ],
        ignore_index=True,
    )
    variants = pd.read_csv(variants_dir / "analysis" / "contrasts.csv")
    genes = pd.read_csv(gene_dir / "scanpy_gene_paired_differences.csv")
    gene_summary = gene_support_summary(genes)
    effects = sensitivity_display_table(fixed, balanced, removals, variants)
    modules = results.module_id.drop_duplicates().tolist()
    target = str(results.target_context.unique().item())
    reference = str(results.reference_context.unique().item())
    output_dir.mkdir(parents=True, exist_ok=False)
    figures = _publish_robustness_figures(
        primary_report_dir,
        output_dir,
        gene_summary,
        genes,
        effects,
        modules=modules,
        target=target,
        reference=reference,
        title=title,
    )
    pd.read_csv(balanced_dir / "reference_contrasts.csv").to_csv(
        output_dir / "balanced_TS_contrasts.csv", index=False
    )
    fixed.loc[fixed.variant_kind.eq("descriptive_stratum")].to_csv(
        output_dir / "technical_strata.csv", index=False
    )
    report = render_replication_report(
        results,
        discovery,
        title=title,
        registration_identity=registration.identity,
        report_scope=(
            "**Primary and required robustness stages completed.** The "
            "registered population extension (M5) and final cohort report "
            "remain incomplete. Primary classifications are unchanged by "
            "sensitivities."
        ),
        limitations=[
            *limitations,
            "Pairing does not separate tissue biology from collection, "
            "dissociation, handling or recovery differences.",
            "Intervals condition on fitted controls and annotations. Balanced "
            "and gene-removal rescoring change the measurement; their effect "
            "differences are not controlled biological perturbations.",
            "Gene candidates were selected from these data by the documented "
            "expression heuristic, not by independent validation or causal "
            "attribution.",
        ],
        figures=figures,
    )
    robustness_section = render_robustness_section(
        fixed,
        balanced,
        removals,
        variants,
        registered_notes=_unrescored_check_notes(registration, variants),
    )
    (output_dir / "robustness.md").write_text(robustness_section)
    report += "\n## Robustness interpretation\n\n" + robustness_section
    report += "\n## Numerical provenance\n\n" + "\n".join(
        f"- {name}: `{path.resolve()}`" for name, path, _, _ in stages
    )
    report += (
        "\n\n## Reproduce this rendering\n\n```sh\n"
        + reproduction_command
        + "\n```\n"
    )
    (output_dir / "report.md").write_text(report)
    record = {
        "status": "completed_primary_and_robustness_report",
        "request": request,
        "identity": request_identity(request),
        "artifacts": {
            str(path.relative_to(output_dir)): file_identity(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        },
    }
    partial = output_dir / "report_manifest.json.partial"
    partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    partial.replace(output_dir / "report_manifest.json")
    return output_dir


def validate_robustness_report(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Reject changed or incomplete M4 reports; M5 remains separate."""
    record = read_stage_manifest(
        output_dir / "report_manifest.json", stage="robustness report"
    )
    if record["status"] != "completed_primary_and_robustness_report":
        raise ValueError(f"Incomplete robustness report: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(
            f"Incompatible report; use a new directory: {output_dir}"
        )
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed report request: {output_dir}")
    for name, identity in record["artifacts"].items():
        if file_identity(output_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed report artifact: {name}")
    return record


def _validate_variants(
    variants_dir: Path,
) -> tuple[Path, dict[str, object]]:
    """Validate the cohort's rescored-variants stage and locate its manifest."""
    analysis_dir = variants_dir / "analysis"
    if (analysis_dir / "chemistry_manifest.json").exists():
        return (
            analysis_dir / "chemistry_manifest.json",
            validate_chemistry_artifacts(variants_dir),
        )
    if (analysis_dir / "muscle_variants_manifest.json").exists():
        return (
            analysis_dir / "muscle_variants_manifest.json",
            validate_muscle_variants(variants_dir),
        )
    raise ValueError(f"No completed rescored-variants stage in {variants_dir}")


def _unrescored_check_notes(
    registration: Registration,
    variants: pd.DataFrame,
    *,
    equivalent_checks: Mapping[str, str] | None = None,
) -> list[str]:
    """Account for registered released-QC checks without a rescored variant.

    A registered check that produced no variant of its own must still be
    reported, or a reviewer cross-checking the registry finds a registered
    check with no result. Only a check whose equivalence to a rescored variant
    was established in the source audit can be reported that way, so an
    unrescored check without a reviewed reason is an error rather than a note.
    """
    reasons = (
        {
            "scrublet_score_at_most_0.4": (
                "no cell-modality observation exceeds a Scrublet score of "
                "0.4, so it selects a cell set identical to as_released"
            )
        }
        if equivalent_checks is None
        else equivalent_checks
    )
    rescored = set(variants.variant_id)
    if "as_released" not in rescored:
        return []

    settings = registration.specification["sensitivities"]
    if not isinstance(settings, dict) or "muscle_QC" not in settings:
        raise ValueError(
            "A rescored as_released variant requires a registered muscle_QC "
            "check list; this registration has none"
        )
    unrescored = [
        check for check in settings["muscle_QC"] if check not in rescored
    ]
    unexplained = [check for check in unrescored if check not in reasons]
    if unexplained:
        raise ValueError(
            f"Registered checks with no rescored variant and no reviewed "
            f"equivalence: {', '.join(unexplained)}. Rescore each one or "
            "record why it repeats an existing variant."
        )
    return [
        f"- {check}: registered beside as_released, but {reasons[check]} and "
        "is reported once there rather than as separate evidence."
        for check in unrescored
    ]


def _publish_robustness_figures(
    primary_report_dir: Path,
    output_dir: Path,
    gene_summary: pd.DataFrame,
    genes: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    modules: list[str],
    target: str,
    reference: str,
    title: str,
) -> dict[str, str]:
    """Export the two new figures and their exact displayed data tables."""
    figure_dir = output_dir / "figures"
    shutil.copytree(primary_report_dir / "figures", figure_dir)
    figures = {
        "1 Cohort support": "figures/1_cohort_support.png",
        "2 Registered results": "figures/2_replication_matrix.png",
        "3 Donor effects": "figures/3_donor_contrasts.png",
    }
    additional = [
        (
            "4_gene_support",
            plots.plot_gene_support(
                gene_summary,
                genes,
                module_ids=modules,
                target=target,
                reference=reference,
            )[0],
        ),
        (
            "5_sensitivity_effects",
            plots.plot_sensitivity_effects(
                effects,
                module_ids=modules,
                title=f"{title}: {target} minus {reference}",
            )[0],
        ),
    ]
    for name, figure in additional:
        try:
            for suffix in ("png", "pdf"):
                figure.savefig(figure_dir / f"{name}.{suffix}", dpi=300)
            figures[name.replace("_", " ")] = f"figures/{name}.png"
        finally:
            plt.close(figure)
    gene_summary.to_csv(figure_dir / "gene_support_displayed.csv", index=False)
    genes.to_csv(figure_dir / "gene_donor_points_displayed.csv", index=False)
    effects.to_csv(
        figure_dir / "sensitivity_effects_displayed.csv", index=False
    )
    return figures
