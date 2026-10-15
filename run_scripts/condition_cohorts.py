"""Prepare and analyze one declared condition-cohort (A/B) population."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nasp_atlas.analysis.condition_cohorts.workflows import (
    audit_condition_cohort,
)
from nasp_atlas.analysis.condition_cohorts.workflows import (
    condition_cohort_analysis,
)


logger = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Parse condition-cohort command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one condition-cohort population from its YAML spec and "
            "publish its stages. Rerunning reuses stages whose request and "
            "artifacts match."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Condition-cohort YAML, e.g. configs/condition_cohorts/*.yaml.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/condition_cohorts"),
        help="Root for <cohort_id>/<population_id> run directories.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding the spec's input_filename (fetch it with "
            "fetch_cohort_input.py); its SHA-256 is verified."
        ),
    )
    parser.add_argument(
        "--minimum-cells",
        type=int,
        default=10,
        help="Cells needed for a donor-sample-cell-type unit.",
    )
    parser.add_argument(
        "--minimum-donors",
        type=int,
        default=3,
        help="Supported donors needed in each compared arm.",
    )
    parser.add_argument(
        "--minimum-state-pairs",
        type=int,
        default=6,
        help="Donors with both states needed for a cell-state contrast.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1_000,
        help="Donor resamples for the composition-within split.",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=10_000.0,
        help="Library size for log1p normalization (CP10k by default).",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        choices=("scanpy", "aucell"),
        default=("scanpy", "aucell"),
        help="Module scorers to run; all arms are scored together.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for Scanpy controls, subsets, bootstraps and plotting.",
    )
    parser.add_argument(
        "--aucell-chunk-size",
        type=int,
        default=1_000,
        help="Cells ranked in one AUCell block.",
    )
    parser.add_argument(
        "--aucell-num-workers",
        type=int,
        default=1,
        help="AUCell worker processes per block.",
    )
    parser.add_argument(
        "--development-donors-per-arm",
        type=int,
        default=None,
        help=(
            "Keep this many whole donors per arm (seeded) for a fast "
            "development run under <population>_dev<N>; not for inference."
        ),
    )
    parser.add_argument(
        "--visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render every stage's figures from its published tables.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Select cells and audit arm support, balance and estimand "
            "availability from metadata only; write nothing."
        ),
    )
    return parser.parse_args()


def _log_audit(args: argparse.Namespace) -> None:
    """Log the selection and design audit without writing outputs."""
    intake, design = audit_condition_cohort(
        args.spec,
        data_dir=args.data_dir,
        minimum_cells=args.minimum_cells,
        minimum_donors=args.minimum_donors,
    )

    logger.info("Selection:\n%s", intake.selection_audit.to_string())
    logger.info(
        "Donors per arm: %s",
        intake.donors["conditions"].value_counts().to_dict(),
    )
    logger.info(
        "Excluded donors: %s",
        intake.excluded_donors.groupby("reason").size().to_dict(),
    )
    logger.info(
        "Supported donors per cell type:\n%s",
        design.arm_support.pivot(
            index="cell_type", columns="condition", values="n_donors"
        ).to_string(),
    )
    logger.info(
        "Estimand availability: %s",
        design.estimand_availability.groupby(["estimand", "status"])
        .size()
        .to_dict(),
    )


def main() -> None:
    """Run or audit one condition-cohort population."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_arguments()
    if args.dry_run:
        _log_audit(args)
        return

    outputs = condition_cohort_analysis(
        spec_path=args.spec,
        data_dir=args.data_dir,
        output_dir=args.output_path,
        minimum_cells=args.minimum_cells,
        minimum_donors=args.minimum_donors,
        minimum_state_pairs=args.minimum_state_pairs,
        target_sum=args.target_sum,
        scorers=args.scorers,
        random_state=args.random_state,
        n_bootstrap=args.n_bootstrap,
        aucell_chunk_size=args.aucell_chunk_size,
        aucell_num_workers=args.aucell_num_workers,
        development_donors_per_arm=args.development_donors_per_arm,
        plot_visualizations=args.visualizations,
    )

    for name, path in outputs.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
