"""Publish a cohort's complete primary, robustness and extension report."""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nasp_atlas.analysis.external_replication import reporting
from nasp_atlas.analysis.external_replication.extension_workflows import (
    validate_extension_artifacts,
)
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
from nasp_atlas.analysis.external_replication.robustness_reports import (
    validate_robustness_report,
)
from nasp_atlas.single_cell.visualization import replication as plots


__all__ = ["publish_final_report", "validate_final_report"]


def publish_final_report(
    *,
    robustness_report_dir: Path,
    extension_dir: Path,
    registration_dir: Path,
    cohort_key: str,
    output_dir: Path,
    title: str,
    reproduction_command: str,
    limitations_path: Path | None = None,
) -> Path:
    """Combine validated M0-M5 results without recomputing any inference."""
    robustness = validate_robustness_report(robustness_report_dir)
    extension = validate_extension_artifacts(extension_dir)
    registration = read_registration(registration_dir)
    for name, stage in (("robustness", robustness), ("extension", extension)):
        request = stage["request"]
        if (
            not isinstance(request, dict)
            or request["registration_identity"] != registration.identity
        ):
            raise ValueError(f"{name} report input uses another registration")
    request = {
        "registration_identity": registration.identity,
        "robustness_report": file_identity(
            robustness_report_dir / "report_manifest.json"
        ),
        "extension": file_identity(
            extension_dir / "analysis" / "extension_manifest.json"
        ),
        "title": title,
        "reproduction_command": reproduction_command,
        "implementation": {
            module.__name__: file_identity(Path(inspect.getfile(module)))[
                "sha256"
            ]
            for module in (reporting, plots)
        },
        "workflow": file_identity(Path(__file__))["sha256"],
    }
    if limitations_path is not None:
        request["limitations"] = file_identity(limitations_path)
    if output_dir.exists():
        validate_final_report(
            output_dir, cohort_key=cohort_key, request=request
        )
        return output_dir

    primary = pd.read_csv(
        robustness_report_dir / "figures" / "replication_displayed.csv"
    )
    extension_results = pd.read_csv(
        extension_dir / "analysis" / "extension_contrasts.csv"
    )
    results = pd.concat([primary, extension_results], ignore_index=True)
    discovery = pd.read_csv(
        robustness_report_dir / "figures" / "discovery_displayed.csv"
    )
    differences = pd.read_csv(
        extension_dir / "analysis" / "donor_differences.csv"
    )
    modules = extension_results.module_id.drop_duplicates().tolist()
    extension_request = extension["request"]
    if not isinstance(extension_request, dict):
        raise ValueError("Extension manifest is missing its request")
    comparison = extension_request["comparison"]
    if not isinstance(comparison, dict):
        raise ValueError("Extension manifest is missing its comparison")

    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir = output_dir / "figures"
    shutil.copytree(robustness_report_dir / "figures", figure_dir)
    for name in ("technical_strata.csv", "balanced_TS_contrasts.csv"):
        shutil.copy2(robustness_report_dir / name, output_dir / name)
    subtypes = extension_request.get("subtypes")
    if subtypes is None:
        figure = plots.plot_donor_contrasts(
            extension_results,
            differences,
            module_ids=modules,
            title=(
                f"Nonclassical monocytes: {comparison['target']} minus "
                f"{comparison['reference']}"
            ),
        )[0]
        extension_note = (
            "The extension tests a different population and is not a "
            "population interaction test."
        )
    else:
        frequencies = pd.read_csv(
            extension_dir / "analysis" / "subtype_frequencies.csv"
        )
        figure = plots.plot_subtype_extension(
            extension_results,
            differences,
            frequencies,
            module_ids=modules,
            subtypes=subtypes,
            title=f"Vascular subtypes minus {comparison['reference']}",
        )[0]
        frequencies.to_csv(
            figure_dir / "subtype_frequencies_displayed.csv", index=False
        )
        extension_note = (
            "Each subtype is tested against the same reference; different "
            "p-values do not show that subtypes differ, and composition can "
            "explain a broad mean."
        )
    try:
        for suffix in ("png", "pdf"):
            figure.savefig(
                figure_dir / f"6_population_extension.{suffix}", dpi=300
            )
    finally:
        plt.close(figure)
    extension_results.to_csv(
        figure_dir / "extension_contrasts_displayed.csv", index=False
    )
    differences.to_csv(
        figure_dir / "extension_donor_differences_displayed.csv", index=False
    )

    figures = {
        "1 Cohort support": "figures/1_cohort_support.png",
        "2 Registered results": "figures/2_replication_matrix.png",
        "3 Donor effects": "figures/3_donor_contrasts.png",
        "4 Gene support": "figures/4_gene_support.png",
        "5 Sensitivity effects": "figures/5_sensitivity_effects.png",
        "6 Registered population extension": (
            "figures/6_population_extension.png"
        ),
    }
    limitations = read_report_limitations(
        registration.specification["limitations"], limitations_path
    )
    report = render_replication_report(
        results,
        discovery,
        title=title,
        registration_identity=registration.identity,
        report_scope=(
            f"**Complete {cohort_key} report.** Primary decisions remain "
            "frozen; the registered extension is a separate association "
            "family and does not strengthen or rescue replication."
        ),
        limitations=[
            *limitations,
            "Pairing does not separate tissue biology from collection, "
            "dissociation, handling or recovery differences.",
            extension_note,
        ],
        figures=figures,
    )
    report += (
        "\n## Robustness interpretation\n\n"
        + (robustness_report_dir / "robustness.md").read_text()
    )
    report += (
        "\n## Reproduce this rendering\n\n```sh\n"
        + reproduction_command
        + "\n```\n"
    )
    (output_dir / "report.md").write_text(report)
    record = {
        "status": f"completed_{cohort_key}_primary_robustness_extension_report",
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


def validate_final_report(
    output_dir: Path,
    *,
    cohort_key: str,
    request: dict[str, object] | None = None,
) -> dict[str, object]:
    """Require the intact completed cohort report and figures."""
    record = read_stage_manifest(
        output_dir / "report_manifest.json", stage=f"final {cohort_key} report"
    )
    if (
        record["status"]
        != f"completed_{cohort_key}_primary_robustness_extension_report"
    ):
        raise ValueError(f"Incomplete final {cohort_key} report: {output_dir}")
    if request is not None and record["request"] != request:
        raise ValueError(
            f"Incompatible final {cohort_key} report: {output_dir}"
        )
    if record["identity"] != request_identity(record["request"]):
        raise ValueError(f"Changed final {cohort_key} request: {output_dir}")
    for name, identity in record["artifacts"].items():
        if file_identity(output_dir / name)["sha256"] != identity["sha256"]:
            raise ValueError(f"Changed final {cohort_key} artifact: {name}")
    return record
