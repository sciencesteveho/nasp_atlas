"""Download and verify the input h5ad declared by a condition-cohort spec."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nasp_atlas.analysis.condition_cohorts.intake import (
    fetch_condition_cohort_input,
)
from nasp_atlas.analysis.condition_cohorts.specification import (
    read_condition_cohort_spec,
)


logger = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Parse fetch command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download a condition-cohort spec's source_url into a data "
            "directory under its input_filename and verify its SHA-256."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        nargs="+",
        required=True,
        help="Condition-cohort YAML specs; shared inputs download once.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory that condition_cohorts.py later reads with --data-dir.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch every declared input."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_arguments()
    for spec_path in args.spec:
        path = fetch_condition_cohort_input(
            read_condition_cohort_spec(spec_path), data_dir=args.data_dir
        )
        logger.info("%s -> %s", spec_path, path)


if __name__ == "__main__":
    main()
