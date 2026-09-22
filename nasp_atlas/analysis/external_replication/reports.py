"""Rebuild primary reports and figures from validated saved numerical tables."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nasp_atlas.analysis.external_replication import reporting
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)
from nasp_atlas.analysis.external_replication.reporting import (
    read_report_limitations,
)
from nasp_atlas.analysis.external_replication.reporting import (
    render_replication_report,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    read_stage_manifest,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    validate_analysis_artifacts,
)
from nasp_atlas.single_cell.visualization import replication as plots


__all__ = ["publish_primary_report", "validate_primary_report"]


def publish_primary_report(
    *,
    analysis_dir: Path,
    registration_dir: Path,
    cohort_key: str,
    output_dir: Path,
    title: str,
    reproduction_command: str,
    limitations_path: Path | None = None,
) -> Path:
    """Publish the first three figures and report without opening expression.

    This is a primary-only report, explicitly incomplete for robustness and
    extension. Source files need not be mounted: numerical artifacts and the
    frozen discovery reference must still pass integrity checks.
    """
    analysis = validate_analysis_artifacts(analysis_dir)
    registration = read_registration(registration_dir)
    saved_request = analysis["request"]
    if (
        not isinstance(saved_request, dict)
        or saved_request["registration_identity"] != registration.identity
    ):
        raise ValueError("Analysis and report registrations differ")
    cohorts = registration.specification["cohorts"]
    if not isinstance(cohorts, dict):
        raise ValueError("Missing registered cohort specifications")
    if (
        saved_request["cohort_id"]
        != cohorts[cohort_key]["cohort_spec"]["cohort_id"]
    ):
        raise ValueError("Analysis belongs to a different registered cohort")
    discovery_record = registration.specification[
        f"{cohort_key}_discovery_contrasts"
    ]
    if not isinstance(discovery_record, dict):
        raise ValueError("Register the discovery artifact before reporting")
    discovery_path = Path(discovery_record["path"])
    if file_identity(discovery_path)["sha256"] != discovery_record["sha256"]:
        raise ValueError(f"Changed discovery reference: {discovery_path}")
    request = {
        "analysis_manifest": file_identity(
            analysis_dir / "analysis_manifest.json"
        ),
        "discovery": discovery_record,
        "registration_identity": registration.identity,
        "title": title,
        "reproduction_command": reproduction_command,
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (reporting, plots)
        },
        "publication": file_identity(Path(__file__))["sha256"],
    }
    if limitations_path is not None:
        request["limitations"] = file_identity(limitations_path)
    if output_dir.exists():
        validate_primary_report(output_dir, request=request)
        return output_dir

    results = pd.read_csv(analysis_dir / "replication_contrasts.csv")
    discovery = pd.read_csv(discovery_path)
    modules = results.module_id.drop_duplicates().tolist()
    discovery = discovery.loc[discovery.module_id.isin(modules)]
    differences = pd.read_csv(analysis_dir / "donor_differences.csv")
    support = pd.read_csv(analysis_dir / "donor_support.csv")
    selected = pd.read_csv(analysis_dir / "selected_cells.csv.gz")
    sensitivities = registration.specification["sensitivities"]
    if not isinstance(sensitivities, dict):
        raise ValueError("Missing registered participant overlap")
    linked = sensitivities["cross_release_linked_ids"]
    support = _annotate_support(support, selected, linked_ids=linked)
    comparison = saved_request["comparison"]

    output_dir.mkdir(parents=True, exist_ok=False)
    figures = _publish_primary_figures(
        output_dir,
        support,
        results,
        discovery,
        differences,
        target=comparison["target"],
        reference=comparison["reference"],
        title=title,
    )
    limitations = read_report_limitations(
        registration.specification["limitations"], limitations_path
    )
    report = render_replication_report(
        results,
        discovery,
        title=title,
        registration_identity=registration.identity,
        report_scope=(
            "**Preliminary primary report.** Robustness, gene diagnostics, "
            "the registered population extension and final cohort review "
            "are not yet complete. Primary results remain conditional on "
            "those checks."
        ),
        limitations=[
            *limitations,
            "Pairing does not separate tissue biology from collection, "
            "dissociation, handling or recovery differences.",
            "Inference conditions on fitted score controls and annotations; "
            "balanced rescoring has not yet assessed this dependence.",
        ],
        figures=figures,
    )
    report += (
        "\n## Reproduce this stage\n\n```sh\n"
        + reproduction_command
        + "\n```\n"
    )
    (output_dir / "report.md").write_text(report)
    record = {
        "status": "completed_primary_report_only",
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


def validate_primary_report(
    output_dir: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    """Reject partial, changed or incompatible primary report output."""
    record = read_stage_manifest(
        output_dir / "report_manifest.json", stage="primary report"
    )
    if record["status"] != "completed_primary_report_only":
        raise ValueError(f"Incomplete primary report: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(
            f"Incompatible report: {output_dir}; use a new directory"
        )
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed report request: {output_dir}")
    for name, identity in record["artifacts"].items():
        if file_identity(output_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed report artifact: {output_dir / name}")
    return record


def _annotate_support(
    support: pd.DataFrame, selected: pd.DataFrame, *, linked_ids: list[str]
) -> pd.DataFrame:
    """Attach observed preparation strata without pretending they are people."""
    annotated = support.copy()
    annotations = []
    for donor in support.donor_id:
        cells = selected.loc[selected.donor_id.eq(donor)]
        labels = []
        for column in ("assay", "institute", "modality"):
            if column in cells and not cells.empty:
                labels.append("/".join(sorted(cells[column].dropna().unique())))
        if donor in linked_ids:
            labels.append("linked across releases")
        annotations.append(" | ".join(labels) or "not scored")
    annotated["annotation"] = annotations
    return annotated


def _publish_primary_figures(
    output_dir: Path,
    support: pd.DataFrame,
    results: pd.DataFrame,
    discovery: pd.DataFrame,
    differences: pd.DataFrame,
    *,
    target: str,
    reference: str,
    title: str,
) -> dict[str, str]:
    """Render and export primary figures with the exact displayed tables."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    figures = [
        (
            "1_cohort_support",
            plots.plot_cohort_support(
                support,
                target=target,
                reference=reference,
                title=title,
            )[0],
        ),
        (
            "2_replication_matrix",
            plots.plot_replication_matrix(
                results,
                discovery,
                title="All frozen primary questions",
            )[0],
        ),
        (
            "3_donor_contrasts",
            plots.plot_donor_contrasts(
                results,
                differences,
                module_ids=results.module_id.drop_duplicates().tolist(),
                title=f"{title}: {target} minus {reference}",
            )[0],
        ),
    ]
    paths = {}
    for name, figure in figures:
        try:
            for suffix in ("png", "pdf"):
                figure.savefig(figure_dir / f"{name}.{suffix}", dpi=300)
            paths[name.replace("_", " ")] = f"figures/{name}.png"
        finally:
            plt.close(figure)
    support.to_csv(figure_dir / "support_displayed.csv", index=False)
    results.to_csv(figure_dir / "replication_displayed.csv", index=False)
    discovery.to_csv(figure_dir / "discovery_displayed.csv", index=False)
    differences.to_csv(
        figure_dir / "donor_differences_displayed.csv", index=False
    )
    return paths
