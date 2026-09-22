"""Compact, descriptive views of donor and gene sensitivity results."""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from nasp_compendium.display import humanize_module_name  # type: ignore[import]

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


class RobustnessPlotter(_PlotterBase):
    """Export bounded diagnostic summaries and their displayed data.

    Example Usage:
      >>> plotter = RobustnessPlotter("results/robustness/figures")
      >>> paths = plotter.plot_tables(tables, max_rows=10, figsize=(3.5, 3.0))
    """

    def plot_tables(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        schema: ObsSchema | None = None,
        minimum_donors: int = 3,
        max_rows: int = 8,
        figsize: tuple[float, float] | None = None,
        label_width: int = 48,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        title_fontsize: float | None = None,
        label_fontsize: float | None = None,
        tick_fontsize: float | None = None,
        show_x_ticks: bool = True,
        show_y_ticks: bool = True,
        show_x_tick_labels: bool = True,
        show_y_tick_labels: bool = True,
        point_color: str = "#333333",
        rescored_color: str = "#0072B2",
        range_color: str = "#777777",
    ) -> list[Path]:
        """Save available summaries as PNG, vector PDF, and displayed CSV.

        ``figsize`` overrides dimensions in inches. By default, figures share
        a 3.5-inch width with height fitted to wrapped rows. Positive
        ``max_rows`` bounds each overview; ``label_width`` wraps labels in
        characters. Rankings
        prioritize sensitivity, not biological activation. Gene diagnostics
        show the largest expression-based driver per module/context/assay,
        prioritizing donor support. Paired contrasts prioritize donor support.
        Missing estimates remain missing; complete results stay in input tables.

        Presentation overrides apply to every figure rendered by one call, so
        pass one table family per call to tune one figure. `title`, `x_label`
        and `y_label` replace each figure's default text when not None; an
        empty string hides it, and the default y label is empty. Font sizes are
        positive points; None keeps the shared publication style. The tick
        flags hide tick marks or tick labels without changing axis limits.
        `point_color` marks baselines, `rescored_color` outlines rescored
        diamonds and `range_color` draws deletion ranges and shift segments.

        Example Usage:
          >>> paths = plotter.plot_tables(
          ...     tables, max_rows=8, figsize=(3.5, 3.0), label_width=44,
          ...     title="Gene removal", show_y_ticks=False,
          ... )
        """
        if max_rows < 1 or label_width < 1:
            raise ValueError("max_rows and label_width must be positive")
        if figsize is not None and (
            len(figsize) != 2
            or any(not np.isfinite(value) or value <= 0 for value in figsize)
        ):
            raise ValueError("figsize must contain two positive finite inches")
        for size in (title_fontsize, label_fontsize, tick_fontsize):
            if size is not None and (not np.isfinite(size) or size <= 0):
                raise ValueError("Font sizes must be positive and finite")

        schema = schema or ObsSchema()
        summaries = _summaries(tables, schema, minimum_donors)
        paths: list[Path] = []

        with plt.rc_context():
            set_matplotlib_publication_parameters()
            for name, (frame, base_title, base_xlabel) in summaries.items():
                if frame.empty:
                    logger.info(
                        "No estimable rows for robustness figure %s", name
                    )
                    continue

                shown = frame.head(max_rows).copy()
                shown["overview_total_rows"] = len(frame)
                lines = sum(
                    len(textwrap.wrap(str(label), label_width))
                    for label in shown.label
                )
                dimensions = figsize or (3.5, max(1.1, 0.65 + 0.115 * lines))

                figure, axis = plt.subplots(figsize=dimensions)
                try:
                    _draw_summary(
                        axis,
                        shown,
                        label_width=label_width,
                        point_color=point_color,
                        rescored_color=rescored_color,
                        range_color=range_color,
                    )
                    axis.set_title(
                        base_title if title is None else title,
                        loc="left",
                        pad=5,
                        fontsize=title_fontsize,
                    )
                    axis.set_xlabel(
                        base_xlabel if x_label is None else x_label,
                        labelpad=3,
                        fontsize=label_fontsize,
                    )
                    if y_label is not None:
                        axis.set_ylabel(y_label, fontsize=label_fontsize)
                    if tick_fontsize is not None:
                        axis.tick_params(labelsize=tick_fontsize)
                    self._set_tick_visibility(
                        axis,
                        show_x_ticks=show_x_ticks,
                        show_y_ticks=show_y_ticks,
                        show_x_tick_labels=show_x_tick_labels,
                        show_y_tick_labels=show_y_tick_labels,
                    )
                    if name == "gene_detection":
                        axis.set_xlim(-0.03, 1.03)
                    elif name == "overlap_coupling":
                        axis.set_xlim(-1.05, 1.05)
                    if name in {
                        "matched_tissue_differences",
                        "overlap_coupling",
                    }:
                        axis.axvline(
                            0, color="#aaaaaa", linewidth=0.4, zorder=0
                        )

                    figure.tight_layout(pad=0.5)
                    for extension in ("png", "pdf"):
                        path = self.output_dir / f"{name}.{extension}"
                        figure.savefig(path, dpi=self.dpi)
                        paths.append(path)
                    data_path = self.output_dir / f"{name}_displayed.csv"
                    shown.to_csv(data_path, index=False)
                    paths.append(data_path)
                finally:
                    plt.close(figure)

        return paths


def _context_labels(frame: pd.DataFrame, contexts: list[str]) -> pd.Series:
    """Label tissue and cell type without collapsing assay context."""
    if frame.empty:
        return pd.Series(index=frame.index, dtype=str)

    return frame[contexts].astype(str).agg(" / ".join, axis=1)


def _summaries(
    tables: Mapping[str, pd.DataFrame],
    schema: ObsSchema,
    minimum_donors: int,
) -> dict[str, tuple[pd.DataFrame, str, str]]:
    """Prepare comparable coordinates without altering analysis tables."""
    summaries: dict[str, tuple[pd.DataFrame, str, str]] = {}
    contexts = [schema.tissue_key, schema.cell_type_key]
    deletion = tables.get("nasp_leave_one_donor_out", pd.DataFrame())
    if not deletion.empty:
        summaries["donor_rank_stability"] = _donor_summary(deletion, contexts)

    paired = tables.get("nasp_matched_tissue_differences", pd.DataFrame())
    if not paired.empty:
        summaries["matched_tissue_differences"] = _paired_summary(
            paired, contexts
        )

    genes = tables.get("nasp_gene_context_diagnostics", pd.DataFrame())
    if not genes.empty:
        summaries["gene_detection"] = _gene_summary(
            genes,
            [
                *contexts,
                *([schema.assay_key] if schema.assay_key in genes else []),
            ],
            minimum_donors,
        )

    changes = tables.get("nasp_gene_removal_context_changes", pd.DataFrame())
    if not changes.empty:
        summaries["gene_removal_ranks"] = _removal_summary(
            changes, contexts, tables["nasp_gene_removal_variants"]
        )

    coupling = tables.get("nasp_overlap_removed_coupling", pd.DataFrame())
    if not coupling.empty:
        summaries["overlap_coupling"] = _coupling_summary(coupling)

    return summaries


def _donor_summary(
    deletion: pd.DataFrame,
    contexts: list[str],
) -> tuple[pd.DataFrame, str, str]:
    """Summarize full-context ranks over estimable donor deletions."""
    deletion = deletion.loc[deletion.ranking_scope.eq("all_contexts")]
    keys = ["feature_label", *contexts]
    frame = (
        deletion.groupby(keys, observed=True)
        .agg(
            point=("context_rank_baseline", "first"),
            low=("context_rank_without_donor", "min"),
            high=("context_rank_without_donor", "max"),
            valid_deletions=("context_rank_without_donor", "count"),
            total_deletions=("status", "size"),
        )
        .reset_index()
    )

    frame["sensitivity"] = (
        frame[["low", "high"]].sub(frame.point, axis=0).abs().max(axis=1)
    )

    frame["label"] = (
        frame.feature_label.map(humanize_module_name)
        + " | "
        + _context_labels(frame, contexts)
        + " | valid deletions "
        + frame.valid_deletions.astype(str)
        + "/"
        + frame.total_deletions.astype(str)
    )
    frame = frame.dropna(subset=["point"]).sort_values(
        "sensitivity", ascending=False, kind="stable"
    )

    return (
        frame,
        "Donor sensitivity: largest rank shifts",
        "Context rank (1 = highest); line: deletion range, not CI",
    )


def _paired_summary(
    paired: pd.DataFrame,
    contexts: list[str],
) -> tuple[pd.DataFrame, str, str]:
    """Prioritize matching-cell-type tissue contrasts by paired support."""
    frame = paired.loc[paired.status.eq("ok")].copy()

    frame["point"] = frame.median_difference

    frame["label"] = (
        frame.feature_label.map(humanize_module_name)
        + " | "
        + frame[contexts[1]].astype(str)
        + " | "
        + frame.tissue.astype(str)
        + " - "
        + frame.reference_tissue.astype(str)
        + " | n="
        + frame.n_paired_donors.astype(str)
    )

    return (
        frame.sort_values("n_paired_donors", ascending=False, kind="stable"),
        "Matched tissues: best-supported donor pairs",
        "Median paired score difference\n(descriptive; module-specific units)",
    )


def _gene_summary(
    genes: pd.DataFrame,
    contexts: list[str],
    minimum_donors: int,
) -> tuple[pd.DataFrame, str, str]:
    """Select context-specific expression drivers without pooling assays."""
    frame = (
        genes.loc[
            genes.n_donors.ge(minimum_donors)
            & genes.arm.ne("context_dependent")
        ]
        .dropna(subset=["driver_magnitude", "median_fraction_detected"])
        .copy()
    )
    keys = ["module_id", *contexts]
    frame = frame.sort_values(
        "driver_magnitude", ascending=False, kind="stable"
    )
    frame = frame.drop_duplicates(keys).sort_values(
        "n_donors", ascending=False, kind="stable"
    )

    frame["point"] = frame.median_fraction_detected

    frame["label"] = (
        frame.module_id.map(humanize_module_name)
        + " | "
        + frame.gene
        + " ("
        + frame.arm
        + ") | "
        + _context_labels(frame, contexts)
        + " | n="
        + frame.n_donors.astype(str)
    )

    return (
        frame,
        "Candidate expression drivers: donor support priority",
        "Median donor fraction detected (not score attribution)",
    )


def _removal_summary(
    changes: pd.DataFrame,
    contexts: list[str],
    variants: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Compare baseline and rescored ranks on supported contexts."""
    frame = changes.merge(
        variants[["variant_id", "removed_genes", "mode", "partner_module"]],
        on="variant_id",
        validate="many_to_one",
    ).dropna(subset=["context_rank_baseline", "context_rank_removed"])

    frame["point"] = frame.context_rank_baseline
    frame["after"] = frame.context_rank_removed

    frame["sensitivity"] = frame.rank_change.abs()

    frame["label"] = (
        frame.module_id.map(humanize_module_name)
        + " | remove "
        + frame.removed_genes
        + " | "
        + _context_labels(frame, contexts)
        + " | n="
        + frame.n_donors_removed.astype(str)
    )

    return (
        frame.sort_values("sensitivity", ascending=False, kind="stable"),
        "Gene removal: largest rank shifts",
        "Context rank; dot: baseline, diamond: rescored\n(descriptive)",
    )


def _coupling_summary(
    coupling: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Compare correlations on the same donor/context observations."""
    frame = (
        coupling.loc[coupling.analysis.eq("within_context_centered")]
        .dropna(subset=["baseline_spearman", "removed_spearman"])
        .copy()
    )

    frame["point"] = frame.baseline_spearman
    frame["after"] = frame.removed_spearman

    frame["sensitivity"] = (frame.after - frame.point).abs()

    frame["label"] = (
        frame.module_a.map(humanize_module_name)
        + " / "
        + frame.module_b.map(humanize_module_name)
        + " | donors="
        + frame.n_donors.astype(str)
        + ", units="
        + frame.n_units.astype(str)
    )

    return (
        frame.sort_values("sensitivity", ascending=False, kind="stable"),
        "Shared-gene removal: largest coupling shifts",
        "Within-context Spearman r\nDot: baseline, diamond: rescored",
    )


def _draw_summary(
    axis: Axes,
    frame: pd.DataFrame,
    *,
    label_width: int,
    point_color: str,
    rescored_color: str,
    range_color: str,
) -> None:
    """Draw point estimates, genuine deletion ranges, and missing ranges."""
    labels = [textwrap.fill(str(label), label_width) for label in frame.label]
    heights = np.array([label.count("\n") + 1 for label in labels])
    positions = np.cumsum(heights) - heights / 2

    if "low" in frame:
        axis.hlines(
            positions, frame.low, frame.high, color=range_color, linewidth=0.6
        )
    if "after" in frame:
        axis.hlines(
            positions,
            frame.point,
            frame.after,
            color=range_color,
            linewidth=0.6,
        )
        axis.scatter(
            frame.after,
            positions,
            marker="D",
            s=9,
            facecolors="white",
            edgecolors=rescored_color,
            linewidths=0.6,
            zorder=3,
        )
    axis.scatter(frame.point, positions, s=7, color=point_color, zorder=4)
    axis.set_yticks(positions, labels)
    axis.set_ylim(float(heights.sum()) + 0.15, -0.15)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.tick_params(axis="x", length=2, pad=2)
    axis.spines[["top", "right", "left"]].set_visible(False)
