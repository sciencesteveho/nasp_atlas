"""Figure summarizing per-sensor target evidence across cohorts."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


class TargetEvidencePlotter(_PlotterBase):
    """Export the sensor-by-evidence matrix behind the assay tiers.

    Example Usage:
      >>> plotter = TargetEvidencePlotter("path/to/output")
      >>> paths = plotter.plot_evidence_matrix(matrix, cell_size=0.13)
    """

    def plot_evidence_matrix(
        self,
        matrix: pd.DataFrame,
        *,
        filename: str = "target_evidence_matrix",
        cell_size: float = 0.13,
        title: str | None = None,
    ) -> list[Path]:
        """Plot each sensor's evidence per cohort and its assay tiers.

        Rows are sensors in tier order. For each cohort, columns show the
        direction of disease modulation, IFN independence, within-sample
        coupling, and siRNA and CRISPRi assayability (in a modulated cell
        type, elsewhere, or not). Shared columns show DepMap common
        essentiality and the three tiers. Sensors a cohort did not measure
        are hatched grey, never shown as absent evidence.

        Args:
          matrix: Rows from `evidence_matrix`: `gene`, `cohort`, `column`,
            `category`.
          filename: Output stem for PNG, PDF and displayed CSV.
          cell_size: Inches per matrix cell.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files.

        Example Usage:
          >>> paths = plotter.plot_evidence_matrix(
          ...     matrix, cell_size=0.15,
          ... )
        """
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be positive and finite")

        colors = _category_colors()
        if unknown := set(matrix["category"]) - set(colors):
            raise ValueError(f"Unknown evidence categories: {sorted(unknown)}")

        genes, columns, categories = _category_grid(matrix)
        image = np.array(
            [[to_rgb(colors[value][0]) for value in row] for row in categories]
        )

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axis = plt.subplots(
                figsize=(
                    2.6 + cell_size * len(columns),
                    1.6 + cell_size * len(genes),
                ),
                layout="constrained",
            )
            try:
                axis.imshow(image, aspect="auto", interpolation="nearest")
                _hatch_unmeasured(axis, categories)

                axis.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
                axis.set_yticks(np.arange(len(genes) + 1) - 0.5, minor=True)
                axis.grid(which="minor", color="white", lw=0.6)
                axis.tick_params(which="both", length=0, pad=1.5)
                axis.set_xticks(
                    range(len(columns)),
                    [column.replace("_", " ") for _, column in columns],
                    rotation=90,
                )
                axis.set_yticks(range(len(genes)), genes)
                for label in axis.get_yticklabels():
                    label.set_fontstyle("italic")
                _label_cohort_blocks(axis, columns)
                axis.spines[:].set_visible(False)

                figure.legend(
                    handles=_legend_handles(categories, colors),
                    loc="outside right center",
                    frameon=False,
                    fontsize=5,
                )
                axis.set_title(
                    "Sensor target evidence across cohorts "
                    "(hypothesis-generating; rules in docs)"
                    if title is None
                    else title,
                    loc="left",
                    pad=18,
                )

                return self._save_with_data(figure, matrix, filename)
            finally:
                plt.close(figure)


def _category_grid(
    matrix: pd.DataFrame,
) -> tuple[list[str], list[tuple[str, str]], np.ndarray]:
    """Arrange matrix categories as genes by (cohort, column) cells.

    Cells absent from `matrix` are "missing".
    """
    genes = list(dict.fromkeys(matrix["gene"]))
    columns = list(
        dict.fromkeys(zip(matrix["cohort"], matrix["column"], strict=True))
    )
    lookup = matrix.set_index(["gene", "cohort", "column"])["category"]
    categories = np.array(
        [
            [
                lookup.get((gene, cohort, column), "missing")
                for cohort, column in columns
            ]
            for gene in genes
        ]
    )
    return genes, columns, categories


def _hatch_unmeasured(axis: Axes, categories: np.ndarray) -> None:
    """Hatch cells of sensors a cohort did not measure or that are missing."""
    hatched_y, hatched_x = np.nonzero(
        np.isin(categories, ["not_measured", "missing"])
    )
    for x, y in zip(hatched_x, hatched_y, strict=True):
        axis.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                fill=False,
                hatch="////",
                edgecolor="#8c8c8c",
                lw=0,
            )
        )


def _label_cohort_blocks(
    axis: Axes,
    columns: list[tuple[str, str]],
) -> None:
    """Name each cohort above its columns and divide adjacent cohorts."""
    cohorts = list(dict.fromkeys(cohort for cohort, _ in columns))
    for cohort in cohorts:
        positions = [
            index for index, (name, _) in enumerate(columns) if name == cohort
        ]
        axis.text(
            (positions[0] + positions[-1]) / 2,
            -0.8,
            "shared" if cohort == "all" else cohort.replace("/", "\n"),
            ha="center",
            va="bottom",
            fontsize=5,
        )
        if positions[0] > 0:
            axis.axvline(positions[0] - 0.5, color="black", lw=0.8)


def _legend_handles(
    categories: np.ndarray,
    colors: dict[str, tuple[str, str]],
) -> list[Patch]:
    """One patch per displayed label, hatched for unmeasured categories."""
    shown = set(categories.ravel())
    handles: dict[str, Patch] = {}
    for value, (color, label) in colors.items():
        if value in shown and label not in handles:
            handles[label] = Patch(
                facecolor=color,
                edgecolor="#8c8c8c",
                hatch="////" if value in ("not_measured", "missing") else None,
                label=label,
            )
    return list(handles.values())


def _category_colors() -> dict[str, tuple[str, str]]:
    """Return the fill colour and legend label of every evidence category."""
    return {
        "up": ("#B2182B", "Modulated up"),
        "down": ("#2166AC", "Modulated down"),
        "mixed": ("#9970AB", "Modulated, mixed direction"),
        "none": ("#F2F2F2", "Not modulated / no / not assayable"),
        "yes": ("#252525", "Yes (IFN-independent, coupled, essential)"),
        "no": ("#F2F2F2", "Not modulated / no / not assayable"),
        "in_modulated": ("#1A9850", "Assayable in a modulated cell type"),
        "elsewhere": ("#A6D96A", "Assayable elsewhere"),
        "priority": ("#1A9850", "Tier: priority"),
        "candidate": ("#A6D96A", "Tier: candidate"),
        "assayable": ("#A6D96A", "Tier: assayable"),
        "context": ("#F2F2F2", "Tier: context"),
        "not_assayable": ("#F2F2F2", "Tier: not assayable"),
        "not_measured": ("#D9D9D9", "Not measured"),
        "missing": ("#D9D9D9", "Annotation missing"),
    }
