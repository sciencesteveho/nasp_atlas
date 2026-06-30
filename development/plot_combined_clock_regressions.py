"""Plot pooled clock regressions from per-tissue clock output tables.

The production clock workflow writes one `clock_{level}_metacells.csv` table
inside each tissue output folder. This module loads those tables from a clock
results root, concatenates predictions across tissues, and writes the same
predicted-vs-chronological-age regression plots used during each single-tissue
clock run.

Example:
    python development/plot_combined_clock_regressions.py \
        results/tabula_sapiens_clocks/10x \
        --output-dir results/tabula_sapiens_clocks/10x/combined_regressions
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.clock import plot_clock_regressions
from nasp_atlas.visualization import _set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)

DEFAULT_LEVELS = ("tissue", "cell_type")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse combined clock-regression plotting arguments.

    Args:
      argv: Optional argument vector. When None, argparse reads from
        ``sys.argv``.

    Returns:
      Namespace containing the clock-results root, output directory, levels to
      combine, chronological-age column, and whether to write combined CSVs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-tissue clock prediction tables and plot pooled "
            "predicted-vs-chronological-age regressions."
        ),
    )
    parser.add_argument(
        "clock_root",
        type=Path,
        help="Root directory containing per-tissue clock output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for combined plots and tables. Defaults to "
            "<clock_root>/combined_regressions."
        ),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help=(
            "Clock aggregation levels to combine. Each level maps to "
            "clock_{level}_metacells.csv files."
        ),
    )
    parser.add_argument(
        "--age-key",
        default="age_years",
        help="Column holding expected chronological age in years.",
    )
    parser.add_argument(
        "--no-combined-csv",
        action="store_true",
        help="Do not write the concatenated clock_{level}_combined.csv tables.",
    )
    return parser.parse_args(argv)


def find_clock_tables(clock_root: str | Path, level: str) -> list[Path]:
    """Return per-tissue clock tables for one aggregation level.

    Args:
      clock_root: Root directory containing clock output folders.
      level: Aggregation level matching `clock_{level}_metacells.csv`.

    Returns:
      Sorted paths for matching per-tissue clock CSV files.
    """
    root = Path(clock_root)
    return sorted(root.rglob(f"clock_{level}_metacells.csv"))


def load_combined_clock_table(
    clock_root: str | Path,
    *,
    level: str,
) -> pd.DataFrame:
    """Load and concatenate per-tissue clock tables for one level.

    Args:
      clock_root: Root directory containing clock output folders.
      level: Aggregation level matching `clock_{level}_metacells.csv`.

    Returns:
      Combined clock prediction table. Empty when no matching tables exist.
      Adds `clock_source_dir` and `clock_source_table` columns to retain the
      source tissue/output folder for each row.
    """
    paths = find_clock_tables(clock_root, level)
    if not paths:
        logger.warning(
            "[combined clocks] no %s tables under %s",
            level,
            clock_root,
        )
        return pd.DataFrame()

    frames = []
    root = Path(clock_root)
    for path in paths:
        table = pd.read_csv(path)
        table["clock_source_dir"] = _relative_parent(path, root)
        table["clock_source_table"] = path.name
        frames.append(table)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "[combined clocks] level=%s | tables=%d | rows=%d",
        level,
        len(paths),
        combined.shape[0],
    )
    return combined


def plot_combined_clock_regressions(
    *,
    clock_root: str | Path,
    output_dir: str | Path | None = None,
    levels: Sequence[str] = DEFAULT_LEVELS,
    age_key: str = "age_years",
    save_combined_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    """Write pooled regression plots for clock outputs across tissues.

    Args:
      clock_root: Root directory containing per-tissue clock output folders.
      output_dir: Directory for combined regression plots. Defaults to
        ``clock_root / "combined_regressions"``.
      levels: Aggregation levels to combine, such as "tissue" or "cell_type".
      age_key: Column holding expected chronological age in years.
      save_combined_csv: Whether to write the concatenated tables used for the
        plots.

    Returns:
      Mapping of each level to the combined table used for plotting. Levels
      with no available tables are omitted.
    """
    root = Path(clock_root)
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else (root / "combined_regressions")
    )
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for level in levels:
        combined = load_combined_clock_table(root, level=level)
        if combined.empty:
            continue
        results[level] = combined

        if save_combined_csv:
            combined.to_csv(
                output_path / f"clock_{level}_combined_metacells.csv",
                index=False,
            )

        plot_clock_regressions(
            combined,
            output_dir=output_path,
            level=f"combined_{level}",
            age_key=age_key,
        )

    return results


def _relative_parent(path: Path, root: Path) -> str:
    """Return a readable source directory label for a clock table.

    Args:
      path: Clock table path.
      root: Clock-results root used for discovery.

    Returns:
      Parent directory path relative to the root when possible.
    """
    try:
        return str(path.parent.relative_to(root))
    except ValueError:
        return str(path.parent)


def main() -> None:
    """Run combined clock-regression plotting from command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _set_matplotlib_publication_parameters()
    args = parse_args()
    plot_combined_clock_regressions(
        clock_root=args.clock_root,
        output_dir=args.output_dir,
        levels=args.levels,
        age_key=args.age_key,
        save_combined_csv=not args.no_combined_csv,
    )


if __name__ == "__main__":
    main()
