"""Compare a condition-cohort reference arm with a healthy atlas (AB8)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nasp_atlas.analysis.condition_cohorts.workflows import (
    atlas_calibration_analysis,
)


logger = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Parse atlas-calibration command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare a completed condition-cohort run's reference-arm "
            "sensor profiles with matched cells of a healthy atlas."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Condition-cohort YAML of the completed run.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Atlas calibration YAML (cell-type matches).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory holding the atlas h5ad by file name.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/condition_cohorts"),
        help="Root holding <cohort_id>/<population_id> runs.",
    )
    parser.add_argument("--minimum-cells", type=int, default=10)
    parser.add_argument("--minimum-donors", type=int, default=3)
    parser.add_argument(
        "--development-donors-per-arm",
        type=int,
        default=None,
        help="Calibrate a development run instead.",
    )
    parser.add_argument(
        "--visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the calibration figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the reference-arm calibration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_arguments()
    stage = atlas_calibration_analysis(
        spec_path=args.spec,
        calibration_path=args.calibration,
        data_dir=args.data_dir,
        output_dir=args.output_path,
        minimum_cells=args.minimum_cells,
        minimum_donors=args.minimum_donors,
        development_donors_per_arm=args.development_donors_per_arm,
        plot_visualizations=args.visualizations,
    )

    logger.info("atlas_calibration -> %s", stage)


if __name__ == "__main__":
    main()
