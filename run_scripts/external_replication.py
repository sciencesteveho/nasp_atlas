"""Execute frozen external replication stages; Wells before muscle."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

import matplotlib

from nasp_atlas.analysis.external_replication.balanced_scoring import (
    rescore_balanced_comparison,
)
from nasp_atlas.analysis.external_replication.chemistry_workflows import (
    rescore_pooled_chemistry,
)
from nasp_atlas.analysis.external_replication.contrast_drivers import (
    rescore_without_contrast_drivers,
)
from nasp_atlas.analysis.external_replication.discovery import (
    finalize_cohort_registration,
)
from nasp_atlas.analysis.external_replication.discovery import (
    reconstruct_discovery_reference,
)
from nasp_atlas.analysis.external_replication.extension_workflows import (
    analyze_muscle_extension,
)
from nasp_atlas.analysis.external_replication.extension_workflows import (
    analyze_wells_extension,
)
from nasp_atlas.analysis.external_replication.final_reports import (
    publish_final_report,
)
from nasp_atlas.analysis.external_replication.gene_workflows import (
    analyze_gene_sensitivity,
)
from nasp_atlas.analysis.external_replication.muscle_variants import (
    rescore_muscle_variants,
)
from nasp_atlas.analysis.external_replication.registered_inputs import (
    prepare_registered_primary,
)
from nasp_atlas.analysis.external_replication.registration import Registration
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)
from nasp_atlas.analysis.external_replication.reports import (
    publish_primary_report,
)
from nasp_atlas.analysis.external_replication.robustness_reports import (
    publish_robustness_report,
)
from nasp_atlas.analysis.external_replication.sensitivity_workflows import (
    analyze_fixed_score_sensitivity,
)
from nasp_atlas.analysis.external_replication.workflows import (
    analyze_primary_cohort,
)
from nasp_atlas.analysis.external_replication.workflows import score_cohort
from nasp_atlas.analysis.sensor_reports import analyze_replication_sensors


logger = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Expose implemented stages and their actual input dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "discovery",
            "finalize-registration",
            "score",
            "analyze",
            "report",
            "primary",
            "fixed-sensitivity",
            "balanced",
            "gene-sensitivity",
            "sensor-report",
            "pooled-chemistry",
            "muscle-variants",
            "contrast-drivers",
            "robustness-report",
            "extension",
            "final-report",
        ),
        required=True,
        help="primary executes scoring, inference and the preliminary report",
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument(
        "--cohort", required=True, help="Key in frozen cohorts record"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument(
        "--report-output", type=Path, help="New directory for changed rendering"
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        help="New directory for changed inference code",
    )
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        help="New directory for changed sensitivity code",
    )
    parser.add_argument("--balanced-output", type=Path)
    parser.add_argument("--gene-output", type=Path)
    parser.add_argument(
        "--sensor-output",
        type=Path,
        help="New exploratory individual-sensor report directory",
    )
    parser.add_argument("--chemistry-output", type=Path)
    parser.add_argument("--extension-output", type=Path)
    parser.add_argument("--variants-output", type=Path)
    parser.add_argument(
        "--driver-output",
        type=Path,
        help="New directory for contrast-driver removal",
    )
    parser.add_argument("--discovery-output", type=Path)
    parser.add_argument("--source-gate", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--primary-report", type=Path)
    parser.add_argument("--robustness-report", type=Path)
    parser.add_argument(
        "--limitations",
        type=Path,
        help="Reviewed YAML limitations replacing the registered list",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata/request; never score or write",
    )
    args = parser.parse_args()
    if args.stage == "robustness-report" and args.primary_report is None:
        parser.error("--primary-report is required for robustness-report")
    if args.stage == "discovery" and args.source_gate is None:
        parser.error("--source-gate is required for discovery")
    if args.stage == "finalize-registration" and any(
        value is None for value in (args.discovery_output, args.source_audit)
    ):
        parser.error(
            "--discovery-output and --source-audit are required for "
            "finalize-registration"
        )
    if args.stage not in (
        "report",
        "fixed-sensitivity",
        "robustness-report",
        "final-report",
        "finalize-registration",
    ) and any(
        value is None for value in (args.spec, args.reference, args.panel)
    ):
        parser.error(
            "--spec, --reference and --panel are required for this stage"
        )
    return args


def _execute(args: argparse.Namespace) -> None:
    """Dispatch flat stages while keeping scoring and reporting independent."""
    output = args.output.resolve()
    if args.stage == "discovery":
        reconstruct_discovery_reference(
            registration_dir=args.registration,
            cohort_key=args.cohort,
            spec_path=args.spec,
            reference_path=args.reference,
            panel_path=args.panel,
            source_gate_path=args.source_gate,
            output_dir=output,
            dry_run=args.dry_run,
        )
        logger.info(
            "%s: %s",
            "Validated harmonized discovery request"
            if args.dry_run
            else "Harmonized discovery reference",
            output,
        )
        return
    if args.stage == "finalize-registration":
        if args.dry_run:
            logger.info(
                "Would validate %s and freeze %s",
                args.discovery_output,
                output,
            )
            return
        finalize_cohort_registration(
            base_registration_dir=args.registration,
            cohort_key=args.cohort,
            discovery_dir=args.discovery_output.resolve(),
            source_audit_path=args.source_audit.resolve(),
            output_dir=output,
        )
        logger.info("Resolved registration: %s", output)
        return
    score_dir = output / "primary_scores"
    analysis_dir = (
        args.analysis_output or output / "primary_analysis"
    ).resolve()
    report_dir = (args.report_output or output / "primary_report").resolve()
    if args.stage == "robustness-report":
        _publish_robustness(args)
        return
    if args.stage == "final-report":
        _publish_final_report(args)
        return
    if args.stage == "fixed-sensitivity":
        sensitivity_dir = (
            args.sensitivity_output or output / "fixed_sensitivity"
        ).resolve()
        if args.dry_run:
            logger.info(
                "Would validate %s and publish %s",
                analysis_dir,
                sensitivity_dir,
            )
            return
        analyze_fixed_score_sensitivity(
            analysis_dir=analysis_dir,
            score_dir=score_dir,
            registration_dir=args.registration,
            cohort_key=args.cohort,
            output_dir=sensitivity_dir,
        )
        logger.info("Fixed-score sensitivities: %s", sensitivity_dir)
        return
    if args.stage != "report":
        prepared = prepare_registered_primary(
            registration_dir=args.registration,
            cohort_key=args.cohort,
            spec_path=args.spec,
            reference_path=args.reference,
            panel_path=args.panel,
        )
        logger.info(
            "%s: %d selected cells, %d paired people; registration %s",
            prepared.spec.cohort_id,
            len(prepared.intake.selected_obs),
            prepared.intake.selected_obs.donor_id.nunique(),
            prepared.registration.identity,
        )
        if args.dry_run or args.stage == "audit":
            if args.stage in (
                "analyze",
                "primary",
                "balanced",
                "gene-sensitivity",
                "muscle-variants",
                "pooled-chemistry",
                "contrast-drivers",
            ):
                logger.info(
                    "Validated request; intended output: %s; analysis: %s",
                    output,
                    analysis_dir,
                )
            else:
                logger.info("Validated request; intended output: %s", output)
            return
        if args.stage == "balanced":
            balanced_dir = (
                args.balanced_output or output / "balanced"
            ).resolve()
            rescore_balanced_comparison(
                prepared,
                cohort_key=args.cohort,
                primary_analysis_dir=analysis_dir,
                primary_score_dir=score_dir,
                output_dir=balanced_dir,
            )
            logger.info("Balanced sensitivity: %s", balanced_dir)
            return
        if args.stage == "sensor-report":
            sensor_dir = (
                args.sensor_output or output / "sensor_report"
            ).resolve()
            analyze_replication_sensors(
                prepared,
                primary_analysis_dir=analysis_dir,
                gene_dir=(
                    args.gene_output or output / "gene_sensitivity"
                ).resolve(),
                output_dir=sensor_dir,
            )
            logger.info("Exploratory sensor report: %s", sensor_dir)
            return
        if args.stage == "pooled-chemistry":
            chemistry_dir = (
                args.chemistry_output or output / "pooled_chemistry"
            ).resolve()
            rescore_pooled_chemistry(
                prepared,
                primary_analysis_dir=analysis_dir,
                output_dir=chemistry_dir,
            )
            logger.info("Pooled chemistry sensitivity: %s", chemistry_dir)
            return
        if args.stage == "muscle-variants":
            variants_dir = (
                args.variants_output or output / "muscle_variants"
            ).resolve()
            rescore_muscle_variants(
                prepared,
                primary_analysis_dir=analysis_dir,
                output_dir=variants_dir,
            )
            logger.info("Muscle QC and nuclei sensitivities: %s", variants_dir)
            return
        if args.stage == "contrast-drivers":
            driver_dir = (
                args.driver_output or output / "contrast_drivers"
            ).resolve()
            rescore_without_contrast_drivers(
                prepared,
                primary_analysis_dir=analysis_dir,
                gene_dir=(
                    args.gene_output or output / "gene_sensitivity"
                ).resolve(),
                output_dir=driver_dir,
            )
            logger.info("Contrast-driver removal: %s", driver_dir)
            return
        if args.stage == "gene-sensitivity":
            gene_dir = (
                args.gene_output or output / "gene_sensitivity"
            ).resolve()
            analyze_gene_sensitivity(
                prepared,
                primary_analysis_dir=analysis_dir,
                primary_score_dir=score_dir,
                output_dir=gene_dir,
            )
            logger.info("Gene diagnostics/removal: %s", gene_dir)
            return
        if args.stage == "extension":
            extension_dir = (
                args.extension_output or output / "extension"
            ).resolve()
            design = _registered_cohort_design(
                prepared.registration, args.cohort
            )
            if design == "scored_extension_population":
                analyze_wells_extension(prepared, output_dir=extension_dir)
            else:
                analyze_muscle_extension(
                    prepared,
                    cohort_key=args.cohort,
                    primary_score_dir=score_dir,
                    output_dir=extension_dir,
                )
            logger.info("Registered extension: %s", extension_dir)
            return
        if args.stage in ("score", "primary"):
            score_cohort(
                prepared.spec,
                prepared.intake.selected_obs,
                prepared.alignment,
                prepared.modules,
                reference=False,
                panel_path=prepared.panel_path,
                output_dir=score_dir,
                target_sum=prepared.target_sum,
                random_state=prepared.random_state,
                aucell_chunk_size=prepared.aucell_chunk_size,
                aucell_num_workers=prepared.aucell_num_workers,
            )
        if args.stage in ("analyze", "primary"):
            analyze_primary_cohort(
                prepared, score_dir=score_dir, output_dir=analysis_dir
            )
    if args.stage in ("report", "primary"):
        if args.dry_run:
            logger.info(
                "Would validate saved analysis %s and render %s",
                analysis_dir,
                report_dir,
            )
            return
        command = shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stage",
                "report",
                "--registration",
                str(args.registration.resolve()),
                "--cohort",
                args.cohort,
                "--output",
                str(output),
                "--analysis-output",
                str(analysis_dir),
                "--report-output",
                str(report_dir),
                *(
                    ["--limitations", str(args.limitations.resolve())]
                    if args.limitations
                    else []
                ),
            ]
        )
        publish_primary_report(
            analysis_dir=analysis_dir,
            registration_dir=args.registration,
            cohort_key=args.cohort,
            output_dir=report_dir,
            title=f"{args.cohort.capitalize()} primary replication",
            reproduction_command=(
                "MPLBACKEND=Agg NUMBA_CACHE_DIR=/tmp/nasp_external_numba "
                "MPLCONFIGDIR=/tmp/nasp_external_matplotlib PYTHONPATH=. "
                + command
            ),
            limitations_path=args.limitations,
        )
        logger.info("Primary report: %s", report_dir / "report.md")


def _registered_cohort_design(
    registration: Registration, cohort_key: str
) -> str:
    """Identify how a cohort's registered extension and variants are produced.

    A cohort that registers `extension_spec` scores a separate population for
    its extension and rescores pooled chemistries; one that registers
    `extension` relabels the already scored primary cells and rescores
    released-QC and modality variants. The frozen record decides, so a new
    cohort fails here instead of silently taking another cohort's path.
    """
    cohorts = registration.specification["cohorts"]
    if not isinstance(cohorts, dict) or cohort_key not in cohorts:
        raise ValueError(
            f"Cohort {cohort_key} has no frozen record in this registration; "
            f"registered cohorts are {', '.join(sorted(cohorts))}"
            if isinstance(cohorts, dict)
            else "The registration has no frozen cohort records"
        )
    record = cohorts[cohort_key]
    if "extension_spec" in record:
        return "scored_extension_population"
    if "extension" in record:
        return "relabelled_primary_extension"
    raise ValueError(
        f"Cohort {cohort_key} registers no population extension: its frozen "
        "record has neither extension_spec (a separately scored population) "
        "nor extension (a relabelling of the primary scores)"
    )


def _default_variants_directory(args: argparse.Namespace) -> str:
    """Name the rescored-variants directory this cohort's design produces."""
    design = _registered_cohort_design(
        read_registration(args.registration), args.cohort
    )
    if design == "scored_extension_population":
        return "pooled_chemistry"
    return "muscle_variants"


def _publish_robustness(args: argparse.Namespace) -> None:
    """Resolve saved stage locations and record this rendering command."""
    output = args.output.resolve()
    report_dir = (args.report_output or output / "robustness_report").resolve()
    if args.dry_run:
        logger.info(
            "Would validate saved sensitivity stages and render %s", report_dir
        )
        return
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    publish_robustness_report(
        primary_report_dir=args.primary_report.resolve(),
        fixed_dir=(
            args.sensitivity_output or output / "fixed_sensitivity"
        ).resolve(),
        balanced_dir=(args.balanced_output or output / "balanced").resolve(),
        gene_dir=(args.gene_output or output / "gene_sensitivity").resolve(),
        variants_dir=(
            args.variants_output
            or args.chemistry_output
            or output / _default_variants_directory(args)
        ).resolve(),
        registration_dir=args.registration,
        output_dir=report_dir,
        cohort_key=args.cohort,
        title=f"{args.cohort.capitalize()} primary replication and robustness",
        reproduction_command=(
            "MPLBACKEND=Agg NUMBA_CACHE_DIR=/tmp/nasp_external_numba "
            "MPLCONFIGDIR=/tmp/nasp_external_matplotlib PYTHONPATH=. " + command
        ),
        limitations_path=args.limitations,
    )
    logger.info("Robustness report: %s", report_dir / "report.md")


def _publish_final_report(args: argparse.Namespace) -> None:
    """Render the complete cohort report from saved M0-M5 artifacts."""
    output = args.output.resolve()
    report_dir = (args.report_output or output / "final_report").resolve()
    robustness_dir = (
        args.robustness_report or output / "robustness_report_v2"
    ).resolve()
    if args.dry_run:
        logger.info(
            "Would validate saved M0-M5 artifacts and render %s", report_dir
        )
        return
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    publish_final_report(
        robustness_report_dir=robustness_dir,
        extension_dir=(args.extension_output or output / "extension").resolve(),
        registration_dir=args.registration,
        cohort_key=args.cohort,
        output_dir=report_dir,
        title=f"{args.cohort.capitalize()} external replication",
        reproduction_command=(
            "MPLBACKEND=Agg NUMBA_CACHE_DIR=/tmp/nasp_external_numba "
            "MPLCONFIGDIR=/tmp/nasp_external_matplotlib PYTHONPATH=. " + command
        ),
        limitations_path=args.limitations,
    )
    logger.info("Final report: %s", report_dir / "report.md")


def main() -> None:
    """Execute the requested stage and propagate technical failures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    matplotlib.use("Agg")
    _execute(_parse_arguments())


if __name__ == "__main__":
    main()
