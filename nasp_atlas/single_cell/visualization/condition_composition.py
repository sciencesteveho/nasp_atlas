"""Figures for cell-type composition and composition-within splits."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


class CompositionPlotter(_PlotterBase):
    """Export composition contrasts and tissue-shift decompositions.

    Every figure is saved as PNG, vector PDF and the displayed rows as CSV.

    Example Usage:
      >>> plotter = CompositionPlotter("path/to/output")
      >>> paths = plotter.plot_composition_effects(effects, row_height=0.14)
      >>> paths += plotter.plot_decomposition(
      ...     decomposition, estimand="sle_vs_control",
      ...     feature_type="gene_expression",
      ... )
    """

    def plot_composition_effects(
        self,
        effects: pd.DataFrame,
        *,
        alpha: float = 0.05,
        filename: str = "composition_effects",
        panel_width: float = 2.2,
        row_height: float = 0.13,
        title: str | None = None,
    ) -> list[Path]:
        """Plot CLR composition differences per cell type, one panel each.

        Each row is a cell type: the target-minus-reference difference in
        centred log-ratio share with its 95% interval from donor-level
        models; filled markers have BH-FDR below `alpha` within the
        contrast. The text gives mean donor shares (reference to target).
        Rows are ordered by the reference share.

        Args:
          effects: `composition/composition_effects.csv` rows.
          alpha: Adjusted p-value threshold for filled markers.
          filename: Output stem for PNG, PDF and displayed CSV.
          panel_width: Inches per contrast panel.
          row_height: Inches per cell type.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files, or an empty list without estimates.

        Example Usage:
          >>> paths = plotter.plot_composition_effects(
          ...     effects, panel_width=2.5, row_height=0.15,
          ... )
        """
        _validate_positive(panel_width=panel_width, row_height=row_height)
        shown = effects.loc[effects["status"].eq("ok")]
        if shown.empty:
            return []

        estimands = list(dict.fromkeys(shown["estimand"]))
        order = (
            shown.groupby("cell_type")["reference_mean_proportion"]
            .mean()
            .sort_values()
            .index.tolist()
        )

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axes = plt.subplots(
                1,
                len(estimands),
                figsize=(
                    1.2 + panel_width * len(estimands),
                    0.9 + row_height * len(order),
                ),
                sharey=True,
                layout="constrained",
                squeeze=False,
            )
            try:
                for axis, estimand in zip(axes[0], estimands, strict=True):
                    _draw_composition_panel(
                        axis,
                        shown.loc[shown["estimand"].eq(estimand)]
                        .set_index("cell_type")
                        .reindex(order),
                        estimand=estimand,
                        alpha=alpha,
                    )

                axes[0, 0].set_yticks(range(len(order)), order)
                figure.suptitle(
                    "Cell-type composition by arm (donor means, HC3; filled: "
                    f"BH-FDR < {alpha:g})"
                    if title is None
                    else title,
                    x=0.02,
                    ha="left",
                )

                return self._save_with_data(figure, shown, filename)
            finally:
                plt.close(figure)

    def plot_decomposition(
        self,
        decomposition: pd.DataFrame,
        *,
        estimand: str,
        feature_type: str,
        filename: str | None = None,
        width: float = 3.6,
        row_height: float = 0.13,
        title: str | None = None,
    ) -> list[Path]:
        """Plot each feature's tissue-level shift and its two parts.

        Per feature, the target-minus-reference shift of the cell-weighted
        tissue mean (black), its composition part (orange) and its
        within-cell-type part (blue), each with a donor-bootstrap 95%
        percentile interval. The parts sum to the total. Descriptive.

        Args:
          decomposition: `composition/composition_decomposition.csv` rows.
          estimand: Contrast to plot.
          feature_type: "gene_expression", "gene_detection" or
            "module_score".
          filename: Output stem; defaults to "<estimand>_decomposition_..".
          width: Figure width in inches.
          row_height: Inches per feature.
          title: Figure title; an empty string hides it.

        Returns:
          Paths of the saved files, or an empty list without rows.

        Example Usage:
          >>> paths = plotter.plot_decomposition(
          ...     decomposition, estimand="ckd_vs_reference",
          ...     feature_type="gene_expression", row_height=0.12,
          ... )
        """
        _validate_positive(width=width, row_height=row_height)
        rows = decomposition.loc[
            decomposition["estimand"].eq(estimand)
            & decomposition["feature_type"].eq(feature_type)
        ]
        if rows.empty:
            return []

        rows = rows.sort_values("total").reset_index(drop=True)
        parts = (
            ("total", "#000000", 0.0),
            ("composition", "#E69F00", 0.22),
            ("within", "#0072B2", -0.22),
        )

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            figure, axis = plt.subplots(
                figsize=(width, 1.0 + row_height * len(rows)),
                layout="constrained",
            )
            try:
                positions = np.arange(len(rows))
                for part, color, offset in parts:
                    axis.errorbar(
                        rows[part],
                        positions + offset,
                        xerr=[
                            rows[part] - rows[f"{part}_ci_low"],
                            rows[f"{part}_ci_high"] - rows[part],
                        ],
                        fmt="o",
                        ms=2.2,
                        color=color,
                        elinewidth=0.5,
                        label=part,
                    )

                axis.axvline(0, lw=0.4, c="#777777")
                axis.set_yticks(positions, rows["feature_label"].astype(str))
                axis.set_xlabel(f"Tissue-level difference ({feature_type})")
                axis.legend(frameon=False, loc="lower right", fontsize=5)
                axis.spines[["top", "right"]].set_visible(False)
                axis.set_title(
                    f"{estimand}: composition vs within-cell-type parts\n"
                    "(donor bootstrap 95% intervals; descriptive)"
                    if title is None
                    else title,
                    loc="left",
                )

                stem = filename or f"{estimand}_decomposition_{feature_type}"
                return self._save_with_data(figure, rows, stem)
            finally:
                plt.close(figure)


def _validate_positive(**dimensions: float) -> None:
    """Reject non-finite or nonpositive figure geometry."""
    for name, value in dimensions.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")


def _draw_composition_panel(
    axis: Axes,
    rows: pd.DataFrame,
    *,
    estimand: str,
    alpha: float,
) -> None:
    """Draw one contrast's CLR share differences with CIs and mean shares.

    `rows` holds one row per displayed cell type, in plotting order; a dot
    is filled when BH-FDR is below `alpha`.
    """
    positions = np.arange(len(rows))
    significant = rows["pvalue_fdr"].lt(alpha).to_numpy()
    axis.errorbar(
        rows["estimate"],
        positions,
        xerr=[
            rows["estimate"] - rows["ci_low"],
            rows["ci_high"] - rows["estimate"],
        ],
        fmt="none",
        ecolor="#555555",
        elinewidth=0.5,
    )
    axis.scatter(
        rows["estimate"],
        positions,
        s=10,
        facecolors=np.where(significant, "#0072B2", "white"),
        edgecolors="#0072B2",
        linewidths=0.6,
        zorder=3,
    )

    for position, reference, target in zip(
        positions,
        rows["reference_mean_proportion"],
        rows["target_mean_proportion"],
        strict=True,
    ):
        axis.annotate(
            f"{reference:.1%} → {target:.1%}",
            (1.0, float(position)),
            xycoords=("axes fraction", "data"),
            xytext=(2, 0),
            textcoords="offset points",
            va="center",
            fontsize=4.5,
        )

    axis.axvline(0, lw=0.4, c="#777777")
    axis.set_title(estimand, loc="left")
    axis.set_xlabel("CLR share difference")
    axis.spines[["top", "right"]].set_visible(False)
