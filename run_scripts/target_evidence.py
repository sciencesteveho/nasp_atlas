"""Summarize sensor target evidence across completed condition-cohort runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.analysis.condition_cohorts.evidence_report import (
    render_evidence_report,
)
from nasp_atlas.analysis.condition_cohorts.target_evidence import (
    evidence_matrix,
)
from nasp_atlas.analysis.condition_cohorts.target_evidence import (
    summarize_target_evidence,
)
from nasp_atlas.analysis.stage_artifacts import file_identity
from nasp_atlas.single_cell.visualization.target_evidence import (
    TargetEvidencePlotter,
)


logger = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Parse target-evidence command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine completed condition-cohort runs into per-sensor "
            "evidence, drug-discovery, siRNA and CRISPRi tiers, an evidence "
            "figure and a Markdown report."
        ),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories, <output-path>/<cohort_id>/<population_id>.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory for the evidence tables, figure and report.",
    )
    parser.add_argument(
        "--annotation-panel",
        type=Path,
        default=None,
        help=(
            "Compendium marker_genes.tsv with essentiality and protein "
            "columns; defaults to the installed compendium panel."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sirna-detection", type=float, default=0.10)
    parser.add_argument("--crispri-detection", type=float, default=0.25)
    parser.add_argument("--donor-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-donors", type=int, default=3)
    parser.add_argument(
        "--visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the evidence-matrix figure.",
    )
    return parser.parse_args()


def _write_evidence(args: argparse.Namespace) -> None:
    """Build the evidence tables and write them with their provenance."""
    panel = Path(args.annotation_panel or GeneModules.default_panel_path())
    evidence = summarize_target_evidence(
        args.runs,
        annotation_panel=panel,
        alpha=args.alpha,
        sirna_detection=args.sirna_detection,
        crispri_detection=args.crispri_detection,
        donor_fraction=args.donor_fraction,
        minimum_donors=args.minimum_donors,
    )

    output = args.output_path
    output.mkdir(parents=True, exist_ok=True)
    evidence.long.to_csv(output / "target_evidence_long.csv", index=False)
    evidence.summary.to_csv(output / "target_evidence_summary.csv", index=False)
    evidence.sensors.to_csv(output / "target_evidence_sensors.csv", index=False)

    if args.visualizations:
        TargetEvidencePlotter(output).plot_evidence_matrix(
            evidence_matrix(evidence)
        )

    identity = file_identity(panel)
    settings = {
        "annotation_panel": identity["path"],
        "annotation_panel_sha256": identity["sha256"],
        "alpha": args.alpha,
        "sirna_detection": args.sirna_detection,
        "crispri_detection": args.crispri_detection,
        "donor_fraction": args.donor_fraction,
        "minimum_donors": args.minimum_donors,
    }
    (output / "target_evidence_provenance.txt").write_text(
        "\n".join(
            [
                f"runs: {', '.join(str(run) for run in args.runs)}",
                *(f"{name}: {value}" for name, value in settings.items()),
            ]
        )
        + "\n"
    )

    (output / "report.md").write_text(
        render_evidence_report(evidence, run_dirs=args.runs, settings=settings)
    )

    logger.info(
        "Tiers: %s",
        evidence.summary["drug_discovery_tier"].value_counts().to_dict(),
    )


def main() -> None:
    """Summarize target evidence across cohort runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _write_evidence(_parse_arguments())


if __name__ == "__main__":
    main()
