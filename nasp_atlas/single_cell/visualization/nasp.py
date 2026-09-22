"""Publication-style visualizations for NASP interpretation tables."""

from __future__ import annotations

import logging
import math
import textwrap
import warnings
from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text  # type: ignore[import]
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from nasp_compendium.display import humanize_module_name  # type: ignore[import]

from nasp_atlas.single_cell.hypothesis_priorities import (
    _hypothesis_priority_label,
)
from nasp_atlas.single_cell.visualization.style import ColorbarStyle
from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


def _darkened_rgba_colors(
    colors: np.ndarray,
    *,
    factor: float,
) -> list[tuple[float, float, float, float]]:
    """Scale RGB channels while preserving each color's alpha channel."""
    return [
        (
            float(color[0]) * factor,
            float(color[1]) * factor,
            float(color[2]) * factor,
            float(color[3]),
        )
        for color in colors
    ]


class NaspPlotter(_PlotterBase):
    """Render mechanistic NASP figures from tidy analysis tables.

    Example Usage:
      >>> plotter = NaspPlotter(output_dir="path/to/output")
      >>> plotter.plot_module_coupling_heatmap(
      ...     coupling,
      ...     filename="module_coupling",
      ... )
    """

    def plot_module_coupling_heatmap(
        self,
        coupling: pd.DataFrame,
        *,
        filename: str,
        statistic: str = "spearman_r",
        fdr_column: str = "spearman_fdr",
        analysis: str | None = "within_context_centered",
        facet_column: str | None = None,
        module_order: Sequence[str] | None = None,
        min_units: int = 3,
        show_fdr: bool = True,
        title: str | None = None,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot symmetric module-coupling matrices with FDR markers.

        Duplicate module pairs are summarized by their median correlation,
        which makes the same method usable for consensus plots after tissue
        result tables have been concatenated. `title` replaces the automatic
        analysis-qualified panel title when provided. `figsize` overrides the
        adaptive total figure dimensions in inches.

        Example Usage:
          >>> plotter.plot_module_coupling_heatmap(
          ...     coupling,
          ...     filename="module_coupling",
          ...     title="Module coupling",
          ...     figsize=(3.08, 2.97),
          ... )
        """
        set_matplotlib_publication_parameters()
        if figsize is not None and (
            len(figsize) != 2
            or any(
                not np.isfinite(dimension) or dimension <= 0.0
                for dimension in figsize
            )
        ):
            raise ValueError(
                "figsize must contain two positive finite dimensions in "
                f"inches; received {figsize}"
            )
        required = ["module_a", "module_b", statistic]
        self._require_columns(coupling, required, table_name="coupling")
        scoped = self._filter_analysis(coupling, analysis=analysis)
        if "skipped" in scoped:
            scoped = scoped.loc[~self._boolean_values(scoped["skipped"])]
        if "n_units" in scoped:
            units = pd.to_numeric(scoped["n_units"], errors="coerce")
            scoped = scoped.loc[units >= min_units]
        scoped = scoped.copy()
        scoped[statistic] = self._numeric_values(scoped[statistic])
        if scoped[statistic].notna().sum() == 0:
            self._warn_empty(filename)
            return

        facets = self._facet_blocks(scoped, facet_column=facet_column)
        fig, axes = self._panel_figure(
            len(facets),
            panel_width=2.8,
            panel_height=2.7,
        )
        if figsize is not None:
            fig.set_size_inches(*figsize)
        image = None
        used_axes: list[Axes] = []
        for ax, (facet_label, block) in zip(axes, facets, strict=False):
            modules = self._module_order(block, requested=module_order)
            matrix, fdr = self._coupling_matrices(
                block,
                modules=modules,
                statistic=statistic,
                fdr_column=fdr_column,
            )
            masked = np.ma.masked_invalid(matrix)
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("#eeeeee")
            image = ax.imshow(
                masked,
                cmap=cmap,
                vmin=-1.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
            self._style_matrix_axis(ax, modules, modules)
            if show_fdr:
                significant_y, significant_x = np.where(
                    np.isfinite(fdr) & (fdr <= 0.05)
                )
                ax.scatter(
                    significant_x,
                    significant_y,
                    s=2.5,
                    c="#222222",
                    linewidths=0,
                )
            panel_title = title
            if panel_title is None:
                panel_title = "Module coupling"
                if not show_fdr:
                    panel_title += " · descriptive"
                if analysis is not None:
                    panel_title += f" · {self._display_label(analysis)}"
            if facet_column is not None:
                panel_title += f"\n{facet_label}"
            ax.set_title(panel_title)
            used_axes.append(ax)
        self._hide_unused_axes(axes[len(facets) :])
        if image is not None:
            resolved_colorbar_style = (
                colorbar_style
                or ColorbarStyle(height="65%", width="4%", pad=0.03)
            ).with_overrides(height=cbar_height, width=cbar_width)
            self._add_embedding_colorbar(
                fig,
                used_axes[-1],
                resolved_colorbar_style,
                mappable=image,
                title="Spearman correlation",
            )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP module-coupling heatmap -> %s",
        )

    def plot_competence_output_state_map(
        self,
        contexts: pd.DataFrame,
        *,
        filename: str,
        label_columns: Sequence[str],
        facet_column: str | None = None,
        title: str | None = "NASP evidence state",
        competence_column: str = "relative_competence",
        output_column: str = "relative_output",
        color_column: str = "output_minus_competence_gap",
        donor_count_column: str = "n_donors",
        min_donors: int = 2,
        max_labels: int = 30,
        adjust_labels: bool = True,
        label_seed: int = 0,
        label_force: tuple[float, float] = (0.5, 0.8),
        label_static_force: tuple[float, float] = (0.1, 0.2),
        label_explode_force: tuple[float, float] = (0.1, 0.5),
        label_expand: tuple[float, float] = (1.05, 1.2),
        label_max_move: tuple[int, int] = (10, 10),
        support_size_range: tuple[float, float] = (12.0, 36.0),
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        figsize: tuple[float, float] | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        title_fontsize: float | None = None,
        label_fontsize: float | None = None,
        tick_fontsize: float | None = None,
        point_label_fontsize: float | None = None,
        show_x_ticks: bool = True,
        show_y_ticks: bool = True,
        show_x_tick_labels: bool = True,
        show_y_tick_labels: bool = True,
        cmap_name: str = "RdBu_r",
        cbar_label: str = "Output - competence",
        corner_labels: tuple[str, str, str] | None = (
            "active-like",
            "responsive-like",
            "competent-like",
        ),
        corner_label_color: str = "#7C7C7C",
    ) -> None:
        """Plot relative competence against output for supported contexts.

        Point color shows the output-minus-competence mismatch and point size
        shows donor support. Axes remain fixed from zero to one so separately
        faceted tissues share the same relative-state geometry. Labels are
        repelled from one another and from scatter points by default; leader
        lines retain their connection to the original context. Set
        `adjust_labels` to False for fixed labels; `label_seed` makes adjusted
        placement reproducible. `label_force` controls label-to-label push,
        while `label_static_force` controls push away from scatter points.
        `label_explode_force`, `label_expand`, and `label_max_move` tune initial
        separation, collision spacing, and the per-iteration movement cap.
        Without `figsize`, each facet scales from the tuned 24-context Liver
        panel according to the number of labels it must accommodate.
        `figsize` overrides the total figure size. Set `title` to `None` to
        show only the facet value as each panel title.

        `x_label` and `y_label` replace the axis labels when not None; an
        empty string hides them. Font sizes are positive points and None keeps
        the shared publication style; `point_label_fontsize` sizes context
        labels. The tick flags hide tick marks or tick labels without changing
        the fixed zero-to-one limits. `cmap_name` recolors the gap on its fixed
        -1 to +1 scale centered on zero; keep a diverging palette.
        `corner_labels` gives the top-right, top-left and bottom-right
        qualitative guides; None hides them.

        Example Usage:
          >>> plotter.plot_competence_output_state_map(
          ...     contexts,
          ...     filename="competence_output",
          ...     label_columns=["cell_type"],
          ...     title=None,
          ...     figsize=(4.0, 4.0),
          ...     point_label_fontsize=4,
          ...     corner_labels=None,
          ... )
        """
        set_matplotlib_publication_parameters()
        for size in (
            title_fontsize,
            label_fontsize,
            tick_fontsize,
            point_label_fontsize,
        ):
            if size is not None and (not np.isfinite(size) or size <= 0):
                raise ValueError("Font sizes must be positive and finite")
        if corner_labels is not None and len(corner_labels) != 3:
            raise ValueError("corner_labels must contain three labels or None")
        required = [
            *label_columns,
            competence_column,
            output_column,
            color_column,
            donor_count_column,
        ]
        self._require_columns(contexts, required, table_name="contexts")
        scoped = contexts.copy()
        scoped[competence_column] = self._numeric_values(
            scoped[competence_column]
        )
        scoped[output_column] = self._numeric_values(scoped[output_column])
        scoped[color_column] = self._numeric_values(scoped[color_column])
        scoped[donor_count_column] = self._numeric_values(
            scoped[donor_count_column]
        )
        scoped = scoped.loc[
            scoped[competence_column].notna()
            & scoped[output_column].notna()
            & (scoped[donor_count_column] >= min_donors)
        ]
        if scoped.empty:
            self._warn_empty(filename)
            return

        facets = self._facet_blocks(scoped, facet_column=facet_column)
        largest_labeled_block = max(
            min(block.shape[0], max_labels) for _, block in facets
        )
        panel_side = 3.2 * max(
            1.0,
            math.sqrt(largest_labeled_block / 24.0),
        )
        fig, axes = self._panel_figure(
            len(facets),
            panel_width=panel_side,
            panel_height=panel_side,
            max_columns=math.ceil(math.sqrt(len(facets))),
        )
        if figsize is not None:
            fig.set_size_inches(*figsize)

        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        cmap = plt.get_cmap(cmap_name)
        used_axes: list[Axes] = []
        for ax, (facet_label, block) in zip(axes, facets, strict=False):
            sizes = self._support_sizes(
                block[donor_count_column],
                minimum_size=support_size_range[0],
                maximum_size=support_size_range[1],
            )
            ax.axline(
                (0, 0),
                slope=1,
                color="#999999",
                lw=0.45,
                ls="--",
                zorder=0,
            )
            # ax.axvline(0.8, color="#dddddd", lw=0.4, zorder=0)
            # ax.axhline(0.8, color="#dddddd", lw=0.4, zorder=0)
            ax.scatter(
                block[competence_column],
                block[output_column],
                c=block[color_column],
                s=sizes,
                cmap=cmap,
                norm=norm,
                edgecolor="#333333",
                linewidth=0.25,
                alpha=0.9,
            )
            labeled = block.iloc[:max_labels]
            labels = self._row_labels(labeled, label_columns)
            texts = [
                ax.text(
                    float(row[competence_column]),
                    float(row[output_column]),
                    label,
                    fontsize=point_label_fontsize,
                )
                for label, (_, row) in zip(
                    labels, labeled.iterrows(), strict=True
                )
            ]
            for label, x, y, ha, va in zip(
                corner_labels or (),
                (0.96, 0.04, 0.96),
                (0.96, 0.96, 0.04),
                ("right", "left", "right"),
                ("top", "top", "bottom"),
                strict=False,
            ):
                ax.text(x, y, label, ha=ha, va=va, color=corner_label_color)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.set_aspect("equal")
            ax.set_xlabel(
                "Relative sensing competence" if x_label is None else x_label,
                fontsize=label_fontsize,
            )
            ax.set_ylabel(
                "Relative pathway output" if y_label is None else y_label,
                fontsize=label_fontsize,
            )
            panel_title = title or ""
            if facet_column is not None:
                separator = " · " if panel_title else ""
                panel_title += f"{separator}{facet_label}"
            ax.set_title(panel_title, fontsize=title_fontsize)
            self._minimal_axis(ax)
            if tick_fontsize is not None:
                ax.tick_params(labelsize=tick_fontsize)
            self._set_tick_visibility(
                ax,
                show_x_ticks=show_x_ticks,
                show_y_ticks=show_y_ticks,
                show_x_tick_labels=show_x_tick_labels,
                show_y_tick_labels=show_y_tick_labels,
            )

            if adjust_labels and texts:
                self._adjust_point_labels(
                    ax,
                    texts,
                    point_x=block[competence_column].to_numpy(dtype=float),
                    point_y=block[output_column].to_numpy(dtype=float),
                    target_x=labeled[competence_column].to_numpy(dtype=float),
                    target_y=labeled[output_column].to_numpy(dtype=float),
                    seed=label_seed,
                    force=label_force,
                    static_force=label_static_force,
                    explode_force=label_explode_force,
                    expand=label_expand,
                    max_move=label_max_move,
                )

            used_axes.append(ax)
        self._hide_unused_axes(axes[len(facets) :])

        resolved_colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.465, width=0.085, pad=0.03)
        ).with_overrides(height=cbar_height, width=cbar_width)
        self._add_embedding_colorbar(
            fig,
            used_axes[-1],
            resolved_colorbar_style,
            mappable=ScalarMappable(norm=norm, cmap=cmap),
            title=cbar_label,
        )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP competence-output state map -> %s",
        )

    def plot_ranked_nasp_hypotheses(
        self,
        priorities: pd.DataFrame,
        *,
        filename: str,
        label_columns: Sequence[str],
        max_per_hypothesis: int = 12,
        panel_height: float = 1.2,
        panel_width: float = 2.0,
        bar_height: float = 0.72,
        label_wrap_width: int | None = None,
        xlabel_wrap_width: int | None = None,
        title: str | None = None,
        bar_color: str | None = None,
        title_fontsize: float | None = None,
        label_fontsize: float | None = None,
        xlabel_fontsize: float | None = None,
        tick_fontsize: float | None = None,
        title_pad: float | None = None,
        xlabel_pad: float | None = None,
        tick_label_pad: float | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        ylabel_fontsize: float | None = None,
        show_x_ticks: bool = True,
        show_y_ticks: bool = True,
        show_x_tick_labels: bool = True,
        show_y_tick_labels: bool = True,
    ) -> None:
        """Plot the highest supported contexts for each NASP hypothesis.

        The method works for a single tissue or an atlas-wide concatenated
        table; callers control whether labels contain cell type alone or both
        tissue and cell type. `panel_height` sets each subplot row's height in
        inches.

        `panel_width` is the width in inches of each panel; `bar_height` is
        the bar thickness in row units. Optional wrapping widths are character
        counts for context labels and formula labels. These controls preserve
        score values, ranking, and the shared zero-to-one axis.
        `title` and `bar_color` override each panel's hypothesis-specific
        default; an empty title hides it. Font sizes and padding are in points;
        None retains the shared publication style. Panel dimensions are
        positive inches; bar height is positive in row units. These categorical
        bars have no colorbar. For separate figures, pass one hypothesis.
        `x_label` replaces each panel's priority-formula label and `y_label`
        adds a context-axis label; empty strings hide them. The tick flags
        hide tick marks or tick labels without changing the zero-to-one axis.

        Example Usage:
          >>> plotter.plot_ranked_nasp_hypotheses(
          ...     priorities,
          ...     filename="hypothesis_priorities",
          ...     label_columns=["tissue", "cell_type"],
          ...     panel_height=1.38,
          ...     panel_width=3.5,
          ...     label_wrap_width=45,
          ...     xlabel_wrap_width=50,
          ...     title_fontsize=10,
          ...     label_fontsize=8,
          ...     tick_label_pad=2,
          ... )
        """
        set_matplotlib_publication_parameters()
        if not np.isfinite(panel_height) or panel_height <= 0.0:
            raise ValueError("panel_height must be a positive finite value")
        for name, value in (
            ("panel_width", panel_width),
            ("bar_height", bar_height),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if max_per_hypothesis < 1:
            raise ValueError("max_per_hypothesis must be at least one")
        for width in (label_wrap_width, xlabel_wrap_width):
            if width is not None and width < 1:
                raise ValueError("Label wrapping widths must be positive")
        for size in (
            title_fontsize,
            label_fontsize,
            xlabel_fontsize,
            ylabel_fontsize,
            tick_fontsize,
        ):
            if size is not None and (not np.isfinite(size) or size <= 0):
                raise ValueError("Font sizes must be positive and finite")
        required = [*label_columns, "hypothesis", "priority_score"]
        self._require_columns(priorities, required, table_name="priorities")
        scoped = priorities.copy()
        scoped["priority_score"] = self._numeric_values(
            scoped["priority_score"]
        )
        scoped = scoped.loc[scoped["priority_score"].notna()]
        if scoped.empty:
            self._warn_empty(filename)
            return

        hypotheses = scoped["hypothesis"].drop_duplicates().astype(str).tolist()
        fig, axes = self._panel_figure(
            len(hypotheses),
            panel_width=panel_width,
            panel_height=panel_height,
            max_columns=2,
        )
        colors = {
            "active_like": "#4c78a8",
            "responsive_like": "#f58518",
            "restricted_buffered": "#54a24b",
            "feedback_dominant": "#b279a2",
            "post_without_nasp": "#e45756",
        }
        for ax, hypothesis in zip(axes, hypotheses, strict=False):
            block = scoped.loc[
                scoped["hypothesis"].astype(str) == hypothesis
            ].nlargest(max_per_hypothesis, "priority_score")
            block = block.sort_values("priority_score", kind="stable")
            labels = self._row_labels(block, label_columns)
            if label_wrap_width is not None:
                labels = [
                    textwrap.fill(label, width=label_wrap_width)
                    for label in labels
                ]
            y = np.arange(block.shape[0])
            ax.barh(
                y,
                block["priority_score"],
                color=bar_color
                if bar_color is not None
                else colors.get(hypothesis, "#4c78a8"),
                height=bar_height,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=label_fontsize)
            ax.set_xlim(0.0, 1.0)
            priority_basis = None
            if "priority_basis" in block:
                observed_basis = (
                    block["priority_basis"]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                )
                if len(observed_basis) == 1:
                    priority_basis = observed_basis.iloc[0]
            xlabel = (
                _hypothesis_priority_label(
                    hypothesis,
                    priority_basis=priority_basis,
                )
                if x_label is None
                else x_label
            )
            if xlabel_wrap_width is not None:
                xlabel = textwrap.fill(xlabel, width=xlabel_wrap_width)
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=xlabel_pad)
            if y_label is not None:
                ax.set_ylabel(y_label, fontsize=ylabel_fontsize)
            ax.set_title(
                self._hypothesis_display_label(hypothesis)
                if title is None
                else title,
                fontsize=title_fontsize,
                pad=title_pad,
            )
            self._minimal_axis(ax)
            if tick_fontsize is not None:
                ax.tick_params(axis="x", labelsize=tick_fontsize)
            if tick_label_pad is not None:
                ax.tick_params(axis="both", pad=tick_label_pad)
            self._set_tick_visibility(
                ax,
                show_x_ticks=show_x_ticks,
                show_y_ticks=show_y_ticks,
                show_x_tick_labels=show_x_tick_labels,
                show_y_tick_labels=show_y_tick_labels,
            )
        self._hide_unused_axes(axes[len(hypotheses) :])
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP ranked hypotheses -> %s",
        )

    def plot_sensor_output_mismatch(
        self,
        coupling: pd.DataFrame,
        *,
        filename: str,
        statistic: str = "spearman_r",
        fdr_column: str = "spearman_fdr",
        analysis: str | None = "within_context_centered",
        facet_column: str | None = None,
        max_genes: int = 40,
        exclude_output_members: bool = True,
        show_fdr: bool = True,
        column_spacing: float = 0.55,
        row_spacing: float = 1.25,
        max_dot_size: float | None = None,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot sensor-gene coupling to canonical output modules.

        Color represents correlation and dot size represents FDR evidence.
        Output-module members are excluded by default to reduce circular
        gene-to-module relationships. `figsize` overrides the total figure
        dimensions; otherwise the size adapts to the displayed genes, output
        modules, and facets while preserving the tuned dot-center spacing.
        `column_spacing` scales the horizontal center-to-center distance, so
        values below 1.0 compact the columns. `row_spacing` independently
        scales the vertical center-to-center distance. `max_dot_size` sets the
        maximum marker area in points squared; the existing scale is retained
        when it is `None`.

        Example Usage:
          >>> plotter.plot_sensor_output_mismatch(
          ...     sensor_output,
          ...     filename="sensor_output_mismatch",
          ...     column_spacing=0.65,
          ...     row_spacing=1.1,
          ...     max_dot_size=24.0,
          ...     figsize=(3.0, 4.0),
          ... )
        """
        set_matplotlib_publication_parameters()
        for parameter_name, spacing in (
            ("column_spacing", column_spacing),
            ("row_spacing", row_spacing),
        ):
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError(
                    f"{parameter_name} must be a positive finite value"
                )

        minimum_dot_size = 7.0 if show_fdr else 10.0
        if max_dot_size is not None and (
            not np.isfinite(max_dot_size) or max_dot_size < minimum_dot_size
        ):
            raise ValueError(
                "max_dot_size must be finite and at least "
                f"{minimum_dot_size:.1f} when show_fdr={show_fdr}"
            )

        required = ["gene", "output_module", statistic]
        self._require_columns(coupling, required, table_name="coupling")

        scoped = self._filter_analysis(coupling, analysis=analysis)
        if exclude_output_members and "gene_in_output_module" in scoped:
            scoped = scoped.loc[
                ~self._boolean_values(scoped["gene_in_output_module"])
            ]
        scoped = scoped.copy()
        scoped[statistic] = self._numeric_values(scoped[statistic])
        if scoped[statistic].notna().sum() == 0:
            self._warn_empty(filename)
            return

        facets = self._facet_blocks(scoped, facet_column=facet_column)
        displayed_gene_count = max(
            min(max_genes, block["gene"].astype(str).nunique())
            for _, block in facets
        )
        output_count = max(
            block["output_module"].astype(str).nunique() for _, block in facets
        )
        fig, axes = self._panel_figure(
            len(facets),
            panel_width=max(0.95, output_count * 0.2375),
            panel_height=max(1.5, displayed_gene_count * 0.08),
        )
        if figsize is not None:
            fig.set_size_inches(*figsize)

        norm = Normalize(vmin=-1.0, vmax=1.0)
        cmap = plt.get_cmap("RdBu_r")
        used_axes: list[Axes] = []
        for ax, (facet_label, block) in zip(axes, facets, strict=False):
            gene_order = (
                block.assign(_abs=block[statistic].abs())
                .groupby("gene", observed=True)["_abs"]
                .max()
                .nlargest(max_genes)
                .index.astype(str)
                .tolist()
            )
            output_order = sorted(block["output_module"].astype(str).unique())

            summary = self._pair_summary(
                block,
                row_column="gene",
                column_column="output_module",
                statistic=statistic,
                fdr_column=fdr_column,
            )
            x_lookup = {
                value: index for index, value in enumerate(output_order)
            }
            y_lookup = {value: index for index, value in enumerate(gene_order)}
            summary = summary.loc[
                summary["gene"].isin(gene_order)
                & summary["output_module"].isin(output_order)
            ]
            x = summary["output_module"].map(x_lookup).to_numpy(dtype=float)
            y = summary["gene"].map(y_lookup).to_numpy(dtype=float)

            resolved_max_dot_size = max_dot_size or 20.0
            sizes = (
                self._fdr_sizes(
                    summary["_fdr"],
                    maximum_size=resolved_max_dot_size,
                )
                if show_fdr
                else 10.0
                + (resolved_max_dot_size - 10.0) * summary["_statistic"].abs()
            )
            statistics = summary["_statistic"].to_numpy(dtype=float)
            edgecolors = _darkened_rgba_colors(
                np.asarray(cmap(norm(statistics))),
                factor=0.75,
            )
            ax.scatter(
                x,
                y,
                c=statistics,
                s=sizes,
                cmap=cmap,
                norm=norm,
                edgecolors=edgecolors,
                linewidth=0.2,
                clip_on=False,
            )
            self._set_vertical_matrix_xticklabels(
                ax,
                [self._module_label(value) for value in output_order],
            )
            ax.set_yticks(range(len(gene_order)))
            ax.set_yticklabels(
                [gene.upper() for gene in gene_order],
                ha="right",
                va="center",
            )
            ax.tick_params(axis="both", which="major", pad=2.0)

            for label in ax.get_yticklabels():
                label.set_fontstyle("italic")

            self._matrix_grid(ax, len(output_order), len(gene_order))
            ax.set_xlim(-0.25, len(output_order) - 0.75)
            ax.set_ylim(len(gene_order) - 0.75, -0.25)

            title = "Sensor-to-output\ncoupling"
            if facet_column is not None:
                title += f" · {facet_label}"

            ax.set_title(title)
            used_axes.append(ax)

        self._hide_unused_axes(axes[len(facets) :])
        if column_spacing != 1.0 or row_spacing != 1.0:
            fig.canvas.draw()
            fig.set_layout_engine(None)
            for ax in used_axes:
                position = ax.get_position()
                scaled_height = position.height * row_spacing
                ax.set_position(
                    (
                        position.x0,
                        position.y1 - scaled_height,
                        position.width * column_spacing,
                        scaled_height,
                    )
                )

        resolved_colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.3, width=0.045, pad=0.5)
        ).with_overrides(height=cbar_height, width=cbar_width)

        self._add_embedding_colorbar(
            fig,
            used_axes[-1],
            resolved_colorbar_style,
            mappable=ScalarMappable(norm=norm, cmap=cmap),
            title="Spearman correlation",
        )

        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP sensor-output mismatch -> %s",
        )

    def plot_age_effect_dotplot(
        self,
        regressions: pd.DataFrame,
        *,
        filename: str,
        analysis_scope: str = "within_tissue_cell_type",
        feature_type: str = "module_score",
        feature_labels: Sequence[str] | None = None,
        slope_column: str = "slope",
        fdr_column: str = "ols_pvalue_fdr",
        max_features: int = 20,
        max_contexts: int = 30,
        column_spacing: float = 0.425,
        row_spacing: float = 0.425,
        tick_label_pad: float = 1.75,
        max_dot_size: float | None = None,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot context-specific age slopes as a significance-sized dot map.

        This is suitable for one tissue or concatenated cross-tissue
        regressions because the y axis is taken from the explicit regression
        stratum labels. Features are selected by strongest FDR evidence, then
        displayed alphabetically. `column_spacing` and `row_spacing` scale
        horizontal and vertical dot-center distances independently.
        `tick_label_pad` controls the gap in points between both axes' tick
        labels and the dot matrix.
        `max_dot_size` sets the maximum marker area in points squared.
        `figsize` overrides adaptive dimensions anchored to the tuned
        20-feature by 20-context Liver layout.

        Example Usage:
          >>> plotter.plot_age_effect_dotplot(
          ...     regressions,
          ...     filename="age_effects",
          ...     column_spacing=0.8,
          ...     row_spacing=1.1,
          ...     tick_label_pad=1.75,
          ...     max_dot_size=24.0,
          ...     figsize=(4.0, 5.0),
          ... )
        """
        set_matplotlib_publication_parameters()
        for parameter_name, spacing in (
            ("column_spacing", column_spacing),
            ("row_spacing", row_spacing),
        ):
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError(
                    f"{parameter_name} must be a positive finite value"
                )
        if not np.isfinite(tick_label_pad) or tick_label_pad < 0.0:
            raise ValueError(
                "tick_label_pad must be a nonnegative finite value"
            )
        if max_dot_size is not None and (
            not np.isfinite(max_dot_size) or max_dot_size < 7.0
        ):
            raise ValueError("max_dot_size must be finite and at least 7.0")

        required = [
            "feature_type",
            "feature_label",
            "stratum",
            "analysis_scope",
            slope_column,
            fdr_column,
        ]
        self._require_columns(regressions, required, table_name="regressions")
        scoped = regressions.loc[
            (regressions["analysis_scope"].astype(str) == analysis_scope)
            & (regressions["feature_type"].astype(str) == feature_type)
        ].copy()
        if feature_labels is not None:
            requested_labels = {str(label) for label in feature_labels}
            scoped = scoped.loc[
                scoped["feature_label"].astype(str).isin(requested_labels)
            ]
        if "skipped" in scoped:
            scoped = scoped.loc[~self._boolean_values(scoped["skipped"])]
        scoped[slope_column] = self._numeric_values(scoped[slope_column])
        scoped[fdr_column] = self._numeric_values(scoped[fdr_column])
        scoped = scoped.loc[scoped[slope_column].notna()]
        if scoped.empty:
            self._warn_empty(filename)
            return

        feature_order = self._top_labels_by_fdr(
            scoped,
            label_column="feature_label",
            fdr_column=fdr_column,
            maximum=max_features,
        )
        feature_order = sorted(
            feature_order,
            key=lambda value: (
                self._module_label(value)
                if feature_type == "module_score"
                else value
            ).casefold(),
        )
        context_order = self._top_labels_by_fdr(
            scoped,
            label_column="stratum",
            fdr_column=fdr_column,
            maximum=max_contexts,
        )
        scoped = scoped.loc[
            scoped["feature_label"].astype(str).isin(feature_order)
            & scoped["stratum"].astype(str).isin(context_order)
        ]
        x_lookup = {value: index for index, value in enumerate(feature_order)}
        y_lookup = {value: index for index, value in enumerate(context_order)}
        x = scoped["feature_label"].astype(str).map(x_lookup)
        y = scoped["stratum"].astype(str).map(y_lookup)
        slope = scoped[slope_column].to_numpy(dtype=float)
        limit = float(np.nanmax(np.abs(slope))) if slope.size else 1.0
        limit = max(limit, np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = plt.get_cmap("RdBu_r")

        width = max(1.5, len(feature_order) * 0.20)
        height = max(2.0, len(context_order) * 0.24)
        fig, ax = plt.subplots(figsize=(width, height))
        if figsize is not None:
            fig.set_size_inches(*figsize)
        resolved_max_dot_size = max_dot_size or 50.0
        edgecolors = _darkened_rgba_colors(
            np.asarray(cmap(norm(slope))),
            factor=0.75,
        )
        ax.scatter(
            x,
            y,
            c=slope,
            s=self._fdr_sizes(
                scoped[fdr_column],
                maximum_size=resolved_max_dot_size,
            ),
            cmap=cmap,
            norm=norm,
            edgecolors=edgecolors,
            linewidth=0.2,
        )
        ax.set_xticks(range(len(feature_order)))
        display_features = (
            [self._module_label(value) for value in feature_order]
            if feature_type == "module_score"
            else [value.upper() for value in feature_order]
        )
        self._set_vertical_matrix_xticklabels(ax, display_features)
        if feature_type == "gene_expression":
            for label in ax.get_xticklabels():
                label.set_fontstyle("italic")
        ax.set_yticks(range(len(context_order)))
        ax.set_yticklabels(self._strip_common_context(context_order))
        ax.set_xlim(-0.5, len(feature_order) - 0.5)
        ax.set_ylim(len(context_order) - 0.5, -0.5)
        self._matrix_grid(ax, len(feature_order), len(context_order))
        ax.tick_params(axis="both", which="major", pad=tick_label_pad)
        title = (
            "Age effects on NASP modules"
            if feature_type == "module_score"
            else "Age effects on nucleic acid sensor genes"
        )
        ax.set_title(title)
        if column_spacing != 1.0 or row_spacing != 1.0:
            fig.canvas.draw()
            fig.set_layout_engine(None)
            position = ax.get_position()
            scaled_height = position.height * row_spacing
            ax.set_position(
                (
                    position.x0,
                    position.y1 - scaled_height,
                    position.width * column_spacing,
                    scaled_height,
                )
            )
        resolved_colorbar_style = (
            colorbar_style or ColorbarStyle(height="25%", width="5%", pad=0.03)
        ).with_overrides(height=cbar_height, width=cbar_width)
        self._add_embedding_colorbar(
            fig,
            ax,
            resolved_colorbar_style,
            mappable=ScalarMappable(norm=norm, cmap=cmap),
            title=(
                "Score change per year"
                if feature_type == "module_score"
                else "Expression change per year"
            ),
        )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP age-effect dotplot -> %s",
        )

    def plot_age_effect_consistency(
        self,
        stability: pd.DataFrame,
        *,
        filename: str,
        analysis_scope: str = "within_tissue",
        feature_type: str | None = None,
        feature_labels: Sequence[str] | None = None,
        max_features: int = 30,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot median and range of age slopes across tested strata.

        The horizontal interval is the observed minimum-to-maximum slope, not
        a confidence interval. Set `feature_type` to keep module scores and
        gene-expression effects in separate figures. This plot is intended for
        the global aggregation stage after stability has been recomputed across
        tissues. Features are selected by absolute median slope, then ordered
        by signed median slope from the most positive effect at the top to the
        most negative effect at the bottom. Display label breaks ties
        deterministically. `figsize` overrides the adaptive figure dimensions.

        Example Usage:
          >>> plotter.plot_age_effect_consistency(
          ...     stability,
          ...     filename="age_consistency",
          ...     feature_type="module_score",
          ...     figsize=(3.2, 4.0),
          ... )
        """
        set_matplotlib_publication_parameters()
        required = [
            "feature_label",
            "analysis_scope",
            "median_slope",
            "min_slope",
            "max_slope",
            "direction_consistency_fraction",
            "n_significant_strata",
        ]
        if feature_type is not None:
            required.append("feature_type")
        self._require_columns(stability, required, table_name="stability")
        scoped = stability.loc[
            stability["analysis_scope"].astype(str) == analysis_scope
        ].copy()
        if feature_type is not None:
            scoped = scoped.loc[
                scoped["feature_type"].astype(str) == feature_type
            ]
        if feature_labels is not None:
            requested_labels = {str(label) for label in feature_labels}
            scoped = scoped.loc[
                scoped["feature_label"].astype(str).isin(requested_labels)
            ]
        if "meets_min_tested_strata" in scoped:
            scoped = scoped.loc[
                self._boolean_values(scoped["meets_min_tested_strata"])
            ]
        for column in (
            "median_slope",
            "min_slope",
            "max_slope",
            "direction_consistency_fraction",
            "n_significant_strata",
        ):
            scoped[column] = self._numeric_values(scoped[column])
        scoped = scoped.loc[scoped["median_slope"].notna()]
        scoped = scoped.assign(_abs=scoped["median_slope"].abs()).nlargest(
            max_features,
            "_abs",
        )
        scoped = scoped.assign(
            _display_feature=(
                scoped["feature_label"].astype(str).str.upper()
                if feature_type == "gene_expression"
                else scoped["feature_label"].map(self._module_label)
            ),
        )
        scoped = scoped.assign(
            _display_sort=scoped["_display_feature"].str.casefold()
        ).sort_values(
            ["median_slope", "_display_sort"],
            ascending=[False, True],
            kind="stable",
            na_position="last",
        )
        if scoped.empty:
            self._warn_empty(filename)
            return

        resolved_figsize = figsize or (
            1.25,
            max(2.2, 1.53 + scoped.shape[0] * 0.067),
        )
        fig, ax = plt.subplots(figsize=resolved_figsize)
        y = np.arange(scoped.shape[0])
        norm = Normalize(vmin=0.5, vmax=1.0)
        cmap = plt.get_cmap("viridis")
        for position, (_, row) in enumerate(scoped.iterrows()):
            ax.plot(
                [row["min_slope"], row["max_slope"]],
                [position, position],
                color="#999999",
                lw=0.7,
                zorder=1,
            )
        ax.scatter(
            scoped["median_slope"],
            y,
            c=scoped["direction_consistency_fraction"],
            s=10.0 + 6.0 * scoped["n_significant_strata"].fillna(0.0),
            cmap=cmap,
            norm=norm,
            edgecolor="#333333",
            linewidth=0.25,
            zorder=2,
        )
        ax.axvline(0.0, color="#555555", lw=0.5, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(scoped["_display_feature"].tolist())
        ax.invert_yaxis()
        if feature_type == "gene_expression":
            for label in ax.get_yticklabels():
                label.set_fontstyle("italic")
        quantity = (
            "Expression change per year"
            if feature_type == "gene_expression"
            else "Module-score change per year"
            if feature_type == "module_score"
            else "Age slope"
        )
        ax.set_xlabel(f"{quantity}\n(median and observed range)")
        title = "Cross-stratum age-effect"
        ax.set_title(title)
        self._minimal_axis(ax)
        resolved_colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.5, width=0.075, pad=0.03)
        ).with_overrides(height=cbar_height, width=cbar_width)
        self._add_embedding_colorbar(
            fig,
            ax,
            resolved_colorbar_style,
            mappable=ScalarMappable(norm=norm, cmap=cmap),
            title="Direction consistency",
        )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP age-effect consistency -> %s",
        )

    def plot_mechanistic_edge_barplot(
        self,
        edges: pd.DataFrame,
        *,
        filename: str,
        analysis: str | None = "within_context_centered",
        row_spacing: float = 0.8,
        bar_height: float = 0.56,
        tick_label_pad: float = 1.0,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot Spearman correlation for every curated mechanistic edge.

        Duplicate source-target rows are represented by their median finite
        Spearman correlation. Edges without an estimable correlation remain on
        the y axis and are labeled "not estimable" rather than being treated as
        zero. Finite edges are ordered from most negative at the bottom to most
        positive at the top on a shared -1 to 1 scale.

        Arrow-like edge labels show curated mechanistic direction; correlation
        sign and magnitude remain symmetric associations and do not establish
        causal direction. `row_spacing` and `bar_height` use y-axis units,
        `tick_label_pad` uses points, and `figsize` uses inches.

        Example Usage:
          >>> plotter.plot_mechanistic_edge_barplot(
          ...     mechanistic_edges,
          ...     filename="mechanistic_edge_correlations",
          ...     row_spacing=0.9,
          ...     bar_height=0.62,
          ...     figsize=(3.6, 4.5),
          ... )
        """
        set_matplotlib_publication_parameters()
        for parameter_name, value in (
            ("row_spacing", row_spacing),
            ("bar_height", bar_height),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{parameter_name} must be a positive finite value; "
                    f"received {value}"
                )
        if bar_height > row_spacing:
            raise ValueError(
                "bar_height must not exceed row_spacing; received "
                f"bar_height={bar_height}, row_spacing={row_spacing}"
            )
        if not np.isfinite(tick_label_pad) or tick_label_pad < 0.0:
            raise ValueError(
                "tick_label_pad must be a nonnegative finite value; "
                f"received {tick_label_pad}"
            )
        if figsize is not None and (
            len(figsize) != 2
            or any(
                not np.isfinite(dimension) or dimension <= 0.0
                for dimension in figsize
            )
        ):
            raise ValueError(
                "figsize must contain two positive finite dimensions in "
                f"inches; received {figsize}"
            )

        self._require_columns(
            edges,
            ["source_module", "target_module", "spearman_r"],
            table_name="edges",
        )
        scoped = self._filter_analysis(edges, analysis=analysis)
        if scoped.empty:
            self._warn_empty(filename)
            return

        summary = self._pair_summary(
            scoped,
            row_column="source_module",
            column_column="target_module",
            statistic="spearman_r",
            fdr_column="spearman_fdr",
        )
        finite_correlations = summary["_statistic"].dropna()
        if not finite_correlations.between(-1.0, 1.0).all():
            raise ValueError(
                "spearman_r values must be between -1 and 1; received range "
                f"[{finite_correlations.min()}, {finite_correlations.max()}]"
            )

        summary = summary.assign(
            _display_edge=[
                f"{self._module_label(source)} → {self._module_label(target)}"
                for source, target in summary[
                    ["source_module", "target_module"]
                ].itertuples(index=False, name=None)
            ]
        ).sort_values(
            "_statistic",
            kind="stable",
            na_position="first",
        )

        resolved_figsize = figsize or (
            3.2,
            max(2.2, summary.shape[0] * 0.134 * row_spacing),
        )
        fig, ax = plt.subplots(figsize=resolved_figsize, layout="constrained")
        positions = np.arange(summary.shape[0], dtype=float) * row_spacing
        correlations = summary["_statistic"].to_numpy(dtype=float)
        finite = np.isfinite(correlations)
        norm = Normalize(vmin=-1.0, vmax=1.0)
        cmap = plt.get_cmap("RdBu_r")

        if finite.any():
            bar_colors = np.asarray(cmap(norm(correlations[finite])))
            ax.barh(
                positions[finite],
                correlations[finite],
                height=bar_height,
                color=bar_colors,
                edgecolor=_darkened_rgba_colors(bar_colors, factor=0.75),
                linewidth=0.4,
            )
        for position in positions[~finite]:
            ax.text(
                0.02,
                position,
                "not estimable",
                ha="left",
                va="center",
                color="#777777",
            )

        ax.axvline(0.0, color="#555555", lw=0.5, ls="--", zorder=0)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(
            -row_spacing / 2.0,
            positions[-1] + row_spacing / 2.0,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(summary["_display_edge"].tolist())
        ax.tick_params(axis="y", pad=tick_label_pad)
        ax.set_xlabel("Spearman correlation")
        ax.set_title("Mechanistic-edge correlations")
        self._minimal_axis(ax)
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP mechanistic-edge barplot -> %s",
        )

    def plot_mechanistic_edge_network(
        self,
        edges: pd.DataFrame,
        *,
        filename: str,
        statistic: str = "spearman_r",
        fdr_column: str = "spearman_fdr",
        analysis: str | None = "within_context_centered",
        facet_column: str | None = None,
        max_edges: int = 24,
        max_layer_span: int | None = None,
        show_fdr: bool = True,
        node_size: tuple[float, float] = (0.265, 0.42),
        node_corner_radius: float = 0.012,
        label_gutter: float = 0.30,
        layer_spacing: float = 1.25,
        within_layer_spacing: float = 1.0,
        colorbar_style: ColorbarStyle | None = None,
        cbar_height: str | float | None = None,
        cbar_width: str | float | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot expected mechanistic directions annotated by correlation.

        Modules are stacked top to bottom by mechanistic layer, each layer
        drawn as a band labeled in the left gutter, so upstream ligand sources
        sit above sensing, signaling, output, and post-NASP layers. Equal-sized
        pastel rectangles encode layer by color and contain automatically
        wrapped 5-point module labels. Arrows are straight segments meeting
        each rectangle's boundary along their own direction of travel.

        `node_size` sets the exact rectangle width and height in layout units,
        where each layer band spans one unit vertically. The horizontal layout
        expands when needed so rectangles in the busiest layer never touch;
        the requested size is not silently clamped. `node_corner_radius`
        controls corner curvature in the same units; rendering compensates for
        unequal axis scales so the radius remains visually even.
        `label_gutter` is the fraction of panel width reserved for band labels,
        and should be widened if a layer name is long enough to reach the first
        rectangle.
        `layer_spacing` and `within_layer_spacing` scale panel height and width.
        `figsize` overrides the adaptive total figure dimensions in inches;
        reducing it can require taller nodes so wrapped labels still fit.

        `max_layer_span` optionally drops edges whose endpoints sit more than
        that many layers apart, so `max_layer_span=1` keeps only adjacent-layer
        steps. Layer-skipping edges are the only ones that can cross an
        intervening rectangle, so restricting the span yields a figure where
        every arrow runs through open space; the cost is that modules reached
        only by a skipping edge drop out of the figure entirely. Dropped edges
        are reported through a warning rather than silently discarded.

        Arrow direction comes from the curated edge label, while arrow color
        shows symmetric transcriptomic coupling. It must therefore not be
        interpreted as evidence of causal direction.

        Example Usage:
          >>> plotter.plot_mechanistic_edge_network(
          ...     mechanistic_edges,
          ...     filename="mechanistic_network",
          ...     max_layer_span=1,
          ...     node_size=(0.20, 0.46),
          ...     label_gutter=0.30,
          ...     layer_spacing=1.25,
          ...     within_layer_spacing=1.0,
          ...     figsize=(4.0, 3.5),
          ... )
        """
        set_matplotlib_publication_parameters()
        node_width, node_height = node_size
        if node_width <= 0 or node_height <= 0:
            raise ValueError(
                "node_size must contain positive width and height values; "
                f"received {node_size}"
            )
        maximum_corner_radius = min(node_size) / 2.0
        if not 0 <= node_corner_radius <= maximum_corner_radius:
            raise ValueError(
                "node_corner_radius must be between 0 and half the smaller "
                f"node dimension ({maximum_corner_radius}); received "
                f"{node_corner_radius}"
            )
        if not 0 <= label_gutter < 1:
            raise ValueError(
                f"label_gutter must be in [0, 1); received {label_gutter}"
            )
        if max_layer_span is not None and max_layer_span < 1:
            raise ValueError(
                "max_layer_span must be at least 1 when set; received "
                f"{max_layer_span}"
            )
        if layer_spacing <= 0:
            raise ValueError(
                f"layer_spacing must be positive; received {layer_spacing}"
            )
        if within_layer_spacing <= 0:
            raise ValueError(
                "within_layer_spacing must be positive; "
                f"received {within_layer_spacing}"
            )
        if figsize is not None and (
            len(figsize) != 2
            or any(
                not np.isfinite(dimension) or dimension <= 0.0
                for dimension in figsize
            )
        ):
            raise ValueError(
                "figsize must contain two positive finite dimensions in "
                f"inches; received {figsize}"
            )

        required = ["source_module", "target_module", statistic]
        self._require_columns(edges, required, table_name="edges")
        scoped = self._filter_analysis(edges, analysis=analysis).copy()
        if "skipped" in scoped:
            scoped = scoped.loc[~self._boolean_values(scoped["skipped"])]
        scoped[statistic] = self._numeric_values(scoped[statistic])
        scoped = scoped.loc[scoped[statistic].notna()]
        if max_layer_span is not None:
            scoped = self._within_layer_span(
                scoped,
                max_layer_span=max_layer_span,
                filename=filename,
            )
        if scoped.empty:
            self._warn_empty(filename)
            return

        facets = self._facet_blocks(scoped, facet_column=facet_column)
        facet_layouts = []
        for facet_label, block in facets:
            summary = self._edge_summary(
                block,
                statistic=statistic,
                fdr_column=fdr_column,
                max_edges=max_edges,
            )
            facet_nodes = sorted(
                set(summary["source_module"].astype(str))
                | set(summary["target_module"].astype(str)),
                key=lambda node: (
                    self._mechanistic_layer(node),
                    self._module_label(node).casefold(),
                ),
            )
            positions, layer_rows, horizontal_span = (
                self._layered_node_positions(
                    facet_nodes,
                    summary,
                    node_size=node_size,
                    label_gutter=label_gutter,
                )
            )
            facet_layouts.append(
                (
                    facet_label,
                    summary,
                    facet_nodes,
                    positions,
                    layer_rows,
                    horizontal_span,
                )
            )

        widest_layer = max(
            sum(self._mechanistic_layer(node) == layer for node in nodes)
            for _, _, nodes, _, layer_rows, _ in facet_layouts
            for layer in layer_rows
        )
        tallest_stack = max(
            len(layer_rows) for _, _, _, _, layer_rows, _ in facet_layouts
        )
        fig, axes = self._panel_figure(
            len(facets),
            panel_width=max(
                3.5,
                0.84 * widest_layer * within_layer_spacing,
            ),
            panel_height=max(
                3.0,
                0.48 * tallest_stack * layer_spacing,
            ),
            max_columns=2,
        )
        if figsize is not None:
            fig.set_size_inches(*figsize)

        norm = Normalize(vmin=-1.0, vmax=1.0)
        cmap = plt.get_cmap("RdBu_r")
        layer_colors = {
            0: "#f3b5b9",
            1: "#c6e4f5",
            2: "#c7e1bd",
            3: "#eaf7c3",
            4: "#dfd5e8",
        }
        node_labels: list[tuple[Text, FancyBboxPatch, str]] = []

        used_axes: list[Axes] = []
        for ax, (
            facet_label,
            _,
            _,
            _,
            layer_rows,
            horizontal_span,
        ) in zip(axes, facet_layouts, strict=False):
            ax.set_xlim(0.0, horizontal_span)
            ax.set_ylim(-0.5, max(layer_rows.values()) + 0.5)
            ax.axis("off")

            title = "Module correlation"
            if facet_column is not None:
                title += f"\n{facet_label}"
            ax.set_title(title)
            used_axes.append(ax)

        fig.canvas.draw()
        fig.set_layout_engine(None)
        for ax, (
            _,
            summary,
            facet_nodes,
            positions,
            layer_rows,
            horizontal_span,
        ) in zip(
            axes,
            facet_layouts,
            strict=False,
        ):
            self._draw_layer_bands(
                ax,
                layer_rows,
                gutter_width=horizontal_span * label_gutter,
            )
            origin = ax.transData.transform((0.0, 0.0))
            x_scale = abs(ax.transData.transform((1.0, 0.0))[0] - origin[0])
            y_scale = abs(ax.transData.transform((0.0, 1.0))[1] - origin[1])
            node_labels.extend(
                self._draw_layer_nodes(
                    ax,
                    facet_nodes,
                    positions=positions,
                    node_size=node_size,
                    node_corner_radius=node_corner_radius,
                    node_mutation_aspect=x_scale / y_scale,
                    layer_colors=layer_colors,
                )
            )
            for _, row in summary.iterrows():
                self._draw_coupling_arrow(
                    ax,
                    positions[str(row["source_module"])],
                    positions[str(row["target_module"])],
                    statistic_value=float(row["_statistic"]),
                    fdr=float(row["_fdr"]),
                    node_size=node_size,
                    show_fdr=show_fdr,
                    cmap=cmap,
                    norm=norm,
                )

        self._hide_unused_axes(axes[len(facets) :])

        resolved_colorbar_style = (
            colorbar_style or ColorbarStyle(height=0.4, width=0.09, pad=0.001)
        ).with_overrides(height=cbar_height, width=cbar_width)

        self._add_embedding_colorbar(
            fig,
            used_axes[-1],
            resolved_colorbar_style,
            mappable=ScalarMappable(norm=norm, cmap=cmap),
            title="Spearman correlation",
        )
        self._fit_node_labels(
            fig,
            node_labels,
            requested_node_size=node_size,
        )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP mechanistic-edge network -> %s",
        )

    @staticmethod
    def _require_columns(
        table: pd.DataFrame,
        columns: Sequence[str],
        *,
        table_name: str,
    ) -> None:
        """Require plotting inputs with actionable column errors."""
        if missing := [column for column in columns if column not in table]:
            raise KeyError(f"{table_name} missing required columns: {missing}")

    @staticmethod
    def _numeric_values(values: pd.Series) -> pd.Series:
        """Return finite numeric values with infinities replaced by NaN."""
        return pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )

    @staticmethod
    def _boolean_values(values: pd.Series) -> pd.Series:
        """Return permissive Boolean values from in-memory or CSV tables."""
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool)
        return values.astype(str).str.lower().isin({"true", "1", "yes"})

    @staticmethod
    def _filter_analysis(
        table: pd.DataFrame,
        *,
        analysis: str | None,
    ) -> pd.DataFrame:
        """Filter an optional analysis label when the table carries it."""
        if analysis is None or "analysis" not in table:
            return table.copy()
        return table.loc[table["analysis"].astype(str) == analysis].copy()

    @staticmethod
    def _warn_empty(filename: str) -> None:
        """Log that a requested NASP plot has no finite supported values."""
        logger.warning("[plot] No supported values for %s. Skipping.", filename)

    @staticmethod
    def _facet_blocks(
        table: pd.DataFrame,
        *,
        facet_column: str | None,
    ) -> list[tuple[str, pd.DataFrame]]:
        """Return stable plotting blocks for an optional facet column."""
        if facet_column is None:
            return [("", table)]
        if facet_column not in table:
            raise KeyError(f"facet column not found: {facet_column}")
        return [
            ("NA" if pd.isna(value) else str(value), block.copy())
            for value, block in table.groupby(
                facet_column,
                observed=True,
                dropna=False,
                sort=False,
            )
        ]

    @staticmethod
    def _panel_figure(
        panel_count: int,
        *,
        panel_width: float,
        panel_height: float,
        max_columns: int = 3,
    ) -> tuple[Figure, list[Axes]]:
        """Create a compact panel grid and return flattened axes."""
        columns = min(max_columns, max(1, panel_count))
        rows = max(1, math.ceil(panel_count / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(panel_width * columns, panel_height * rows),
            squeeze=False,
            layout="constrained",
        )
        return fig, axes.ravel().tolist()

    @staticmethod
    def _hide_unused_axes(axes: Sequence[Axes]) -> None:
        """Hide axes that do not correspond to a requested panel."""
        for ax in axes:
            ax.axis("off")

    @staticmethod
    def _module_order(
        table: pd.DataFrame,
        *,
        requested: Sequence[str] | None,
    ) -> list[str]:
        """Return requested present modules followed by stable defaults."""
        available = set(table["module_a"].astype(str)) | set(
            table["module_b"].astype(str)
        )
        if requested is None:
            return sorted(available)
        ordered = [module for module in requested if module in available]
        return [*ordered, *sorted(available.difference(ordered))]

    @classmethod
    def _coupling_matrices(
        cls,
        table: pd.DataFrame,
        *,
        modules: Sequence[str],
        statistic: str,
        fdr_column: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build symmetric statistic and FDR matrices."""
        summary = cls._pair_summary(
            table,
            row_column="module_a",
            column_column="module_b",
            statistic=statistic,
            fdr_column=fdr_column,
        )
        lookup = {module: index for index, module in enumerate(modules)}
        matrix = np.full((len(modules), len(modules)), np.nan, dtype=float)
        fdr = np.full_like(matrix, np.nan)
        np.fill_diagonal(matrix, 1.0)
        for _, row in summary.iterrows():
            module_a = str(row["module_a"])
            module_b = str(row["module_b"])
            if module_a not in lookup or module_b not in lookup:
                continue
            index_a = lookup[module_a]
            index_b = lookup[module_b]
            matrix[index_a, index_b] = matrix[index_b, index_a] = float(
                row["_statistic"]
            )
            fdr[index_a, index_b] = fdr[index_b, index_a] = float(row["_fdr"])
        return matrix, fdr

    @classmethod
    def _pair_summary(
        cls,
        table: pd.DataFrame,
        *,
        row_column: str,
        column_column: str,
        statistic: str,
        fdr_column: str,
    ) -> pd.DataFrame:
        """Summarize duplicate pair rows for plotting."""
        working = table.copy()
        working[statistic] = cls._numeric_values(working[statistic])
        if fdr_column in working:
            working[fdr_column] = cls._numeric_values(working[fdr_column])
        else:
            working[fdr_column] = np.nan
        return (
            working.groupby(
                [row_column, column_column],
                observed=True,
                dropna=False,
                sort=False,
            )
            .agg(
                _statistic=(statistic, "median"),
                _fdr=(fdr_column, "max"),
            )
            .reset_index()
        )

    @staticmethod
    def _style_matrix_axis(
        ax: Axes,
        x_labels: Sequence[str],
        y_labels: Sequence[str],
    ) -> None:
        """Style a labeled matrix with white cell boundaries."""
        NaspPlotter._set_vertical_matrix_xticklabels(
            ax,
            [NaspPlotter._module_label(value) for value in x_labels],
        )
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(
            [NaspPlotter._module_label(value) for value in y_labels]
        )
        NaspPlotter._matrix_grid(ax, len(x_labels), len(y_labels))

    @staticmethod
    def _set_vertical_matrix_xticklabels(
        ax: Axes,
        labels: Sequence[str],
    ) -> None:
        """Center vertical labels on their corresponding matrix columns."""
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            labels,
            rotation=90,
            ha="right",
            va="center",
            rotation_mode="anchor",
        )

    @staticmethod
    def _matrix_grid(ax: Axes, columns: int, rows: int) -> None:
        """Add subtle white boundaries to a matrix or dot map."""
        ax.set_xticks(np.arange(-0.5, columns, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _minimal_axis(ax: Axes) -> None:
        """Apply a clean axis style while retaining data ticks."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=1.5, width=0.35)

    @staticmethod
    def _adjust_point_labels(
        ax: Axes,
        texts: Sequence[Text],
        *,
        point_x: np.ndarray,
        point_y: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
        seed: int,
        force: tuple[float, float],
        static_force: tuple[float, float],
        explode_force: tuple[float, float],
        expand: tuple[float, float],
        max_move: tuple[int, int],
    ) -> None:
        """Repel point labels deterministically while retaining leader lines."""
        random_state = np.random.get_state()
        try:
            np.random.seed(seed)
            adjust_text(
                list(texts),
                x=point_x,
                y=point_y,
                target_x=target_x,
                target_y=target_y,
                ax=ax,
                prevent_crossings=True,
                ensure_inside_axes=True,
                expand_axes=False,
                iter_lim=200,
                min_arrow_len=3,
                force_text=force,
                force_static=static_force,
                force_explode=explode_force,
                expand=expand,
                max_move=max_move,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#777777",
                    "linewidth": 0.3,
                },
            )
        finally:
            np.random.set_state(random_state)

    @staticmethod
    def _support_sizes(
        values: pd.Series,
        *,
        minimum_size: float,
        maximum_size: float,
    ) -> np.ndarray:
        """Scale observed donor support linearly into a requested size range."""
        if minimum_size <= 0 or maximum_size < minimum_size:
            raise ValueError(
                "support_size_range must contain positive increasing values"
            )
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
        minimum = float(numeric.min())
        maximum = float(numeric.max())
        if maximum <= minimum:
            midpoint = minimum_size + (maximum_size - minimum_size) / 2.0
            return np.full(numeric.shape[0], midpoint)
        normalized = (numeric.to_numpy(dtype=float) - minimum) / (
            maximum - minimum
        )
        return minimum_size + (maximum_size - minimum_size) * normalized

    @staticmethod
    def _fdr_sizes(
        values: pd.Series,
        *,
        maximum_size: float = 37.0,
    ) -> np.ndarray:
        """Scale adjusted p-values into bounded evidence dot sizes."""
        fdr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        evidence = np.zeros_like(fdr)
        finite = np.isfinite(fdr)
        evidence[finite] = -np.log10(np.clip(fdr[finite], 1e-12, 1.0))
        return 7.0 + (maximum_size - 7.0) * np.clip(
            evidence / 5.0,
            0.0,
            1.0,
        )

    @staticmethod
    def _row_labels(
        table: pd.DataFrame,
        columns: Sequence[str],
    ) -> list[str]:
        """Join selected context columns into readable row labels."""
        return [
            " · ".join("NA" if pd.isna(value) else str(value) for value in row)
            for row in table.loc[:, list(columns)].itertuples(
                index=False,
                name=None,
            )
        ]

    @staticmethod
    def _display_label(value: str) -> str:
        """Turn a snake-case identifier into a compact display label."""
        return value.replace("_", " ").strip().capitalize()

    @staticmethod
    def _hypothesis_display_label(hypothesis: str) -> str:
        """Return the scientific display label for a hypothesis class."""
        labels = {
            "active_like": "Active-like",
            "responsive_like": "Responsive-like",
            "restricted_buffered": "Restricted/buffered",
            "feedback_dominant": "Feedback-dominant",
            "post_without_nasp": "Post without NASP",
        }
        return labels.get(hypothesis, hypothesis.replace("_", " ").capitalize())

    @staticmethod
    def _module_label(value: object) -> str:
        """Return compact display text for a module identifier."""
        return humanize_module_name(str(value))

    @staticmethod
    def _top_labels_by_fdr(
        table: pd.DataFrame,
        *,
        label_column: str,
        fdr_column: str,
        maximum: int,
    ) -> list[str]:
        """Return labels ordered by strongest finite FDR evidence."""
        ranked = table.assign(
            _plot_fdr=pd.to_numeric(table[fdr_column], errors="coerce").fillna(
                1.0
            )
        )
        return (
            ranked.groupby(label_column, observed=True)["_plot_fdr"]
            .min()
            .nsmallest(maximum)
            .index.astype(str)
            .tolist()
        )

    @staticmethod
    def _strip_common_context(values: Sequence[str]) -> list[str]:
        """Remove a shared tissue prefix from tissue::cell-type labels."""
        split = [str(value).split("::", maxsplit=1) for value in values]
        if split and all(len(parts) == 2 for parts in split):
            prefixes = {parts[0] for parts in split}
            if len(prefixes) == 1:
                return [parts[1] for parts in split]
        return [str(value) for value in values]

    @classmethod
    def _edge_summary(
        cls,
        table: pd.DataFrame,
        *,
        statistic: str,
        fdr_column: str,
        max_edges: int,
    ) -> pd.DataFrame:
        """Summarize and retain the strongest mechanistic edges."""
        summary = cls._pair_summary(
            table,
            row_column="source_module",
            column_column="target_module",
            statistic=statistic,
            fdr_column=fdr_column,
        )
        return summary.assign(_abs=summary["_statistic"].abs()).nlargest(
            max_edges,
            "_abs",
        )

    @classmethod
    def _layered_node_positions(
        cls,
        nodes: Sequence[str],
        edges: pd.DataFrame,
        *,
        node_size: tuple[float, float],
        label_gutter: float = 0.23,
    ) -> tuple[
        dict[str, tuple[float, float]],
        dict[int, float],
        float,
    ]:
        """Place nodes on evenly spaced top-down mechanistic layers.

        Returns node centers, the vertical position of each occupied layer, and
        the horizontal layout span. Only occupied layers consume vertical
        space, so a facet missing a layer leaves no empty band. The horizontal
        span grows as needed to preserve the requested rectangle width with a
        small gap between neighboring nodes.
        """
        grouped: dict[int, list[str]] = {}
        for node in nodes:
            grouped.setdefault(cls._mechanistic_layer(node), []).append(node)
        grouped = cls._crossing_reduced_layers(grouped, edges)

        occupied = sorted(grouped)
        layer_rows = {
            layer: float(len(occupied) - 1 - index)
            for index, layer in enumerate(occupied)
        }
        widest_layer = max(len(members) for members in grouped.values())
        required_node_field = widest_layer * node_size[0] / 0.92
        horizontal_span = max(
            1.0,
            required_node_field / (1.0 - label_gutter),
        )
        gutter_width = horizontal_span * label_gutter
        node_field = horizontal_span - gutter_width

        positions = {}
        for layer, members in grouped.items():
            span = node_field / len(members)
            for index, node in enumerate(members):
                positions[node] = (
                    gutter_width + span * (index + 0.5),
                    layer_rows[layer],
                )
        return positions, layer_rows, horizontal_span

    @classmethod
    def _crossing_reduced_layers(
        cls,
        grouped: dict[int, list[str]],
        edges: pd.DataFrame,
        *,
        sweeps: int = 4,
    ) -> dict[int, list[str]]:
        """Reorder nodes within layers to shorten and untangle arrows.

        Runs alternating down-and-up barycenter sweeps, the standard heuristic
        for layered graph drawing: each node moves toward the mean position of
        its neighbors in the adjacent layer. Ordering is deterministic because
        sweeps start from the caller's order and ties fall back to the incoming
        index.
        """
        neighbors: dict[str, list[str]] = {
            node: [] for members in grouped.values() for node in members
        }
        for source, target in zip(
            edges["source_module"].astype(str),
            edges["target_module"].astype(str),
            strict=True,
        ):
            if source in neighbors and target in neighbors:
                neighbors[source].append(target)
                neighbors[target].append(source)

        ordered = {layer: list(members) for layer, members in grouped.items()}
        layers = sorted(ordered)
        for sweep in range(sweeps):
            for layer in layers if sweep % 2 == 0 else layers[::-1]:
                members = ordered[layer]
                relative_position = {
                    node: position / max(1, len(other_members) - 1)
                    for other, other_members in ordered.items()
                    if other != layer
                    for position, node in enumerate(other_members)
                }
                ordered[layer] = sorted(
                    members,
                    key=lambda node: cls._barycenter_key(
                        node,
                        members=members,
                        neighbors=neighbors[node],
                        relative_position=relative_position,
                    ),
                )
        return ordered

    @staticmethod
    def _barycenter_key(
        node: str,
        *,
        members: Sequence[str],
        neighbors: Sequence[str],
        relative_position: dict[str, float],
    ) -> tuple[float, int]:
        """Return the mean neighbor position and a stable tiebreaker."""
        incoming_index = members.index(node)
        if relative := [
            relative_position[neighbor]
            for neighbor in neighbors
            if neighbor in relative_position
        ]:
            return (float(np.mean(relative)), incoming_index)
        else:
            return (
                incoming_index / max(1, len(members) - 1),
                incoming_index,
            )

    @classmethod
    def _draw_layer_bands(
        cls,
        ax: Axes,
        layer_rows: dict[int, float],
        *,
        gutter_width: float,
        band_half_height: float = 0.42,
    ) -> None:
        """Shade each occupied layer and label it in the left gutter.

        Labels are left-aligned at the panel's left edge so the layer names read
        as a column, rather than ragged-left against the rectangles.
        """
        for layer, row in layer_rows.items():
            ax.axhspan(
                row - band_half_height,
                row + band_half_height,
                color="#f4f4f2",
                zorder=0,
            )
            ax.text(
                0.02 * gutter_width,
                row,
                cls._layer_display_name(layer),
                ha="left",
                va="center",
                color="#444444",
                clip_on=False,
                zorder=3,
            )

    @classmethod
    def _draw_layer_nodes(
        cls,
        ax: Axes,
        nodes: Sequence[str],
        *,
        positions: dict[str, tuple[float, float]],
        node_size: tuple[float, float],
        node_corner_radius: float,
        node_mutation_aspect: float,
        layer_colors: dict[int, str],
    ) -> list[tuple[Text, FancyBboxPatch, str]]:
        """Draw pastel layer rectangles and return their fitted labels."""
        node_width, node_height = node_size
        drawn = []
        for node in nodes:
            x, y = positions[node]
            node_facecolor = mcolors.to_rgb(
                layer_colors[cls._mechanistic_layer(node)]
            )
            node_patch = FancyBboxPatch(
                (x - node_width / 2.0, y - node_height / 2.0),
                node_width,
                node_height,
                boxstyle=f"round,pad=0,rounding_size={node_corner_radius}",
                mutation_aspect=node_mutation_aspect,
                facecolor=node_facecolor,
                edgecolor=tuple(0.8 * channel for channel in node_facecolor),
                linewidth=0.35,
                zorder=2,
            )
            ax.add_patch(node_patch)
            module_label = cls._module_label(node)
            label = ax.text(
                x,
                y,
                module_label,
                ha="center",
                va="center",
                multialignment="center",
                fontsize=5,
                linespacing=1.0,
                color="#111111",
                clip_on=False,
                zorder=3,
            )
            drawn.append((label, node_patch, module_label))
        return drawn

    @classmethod
    def _draw_coupling_arrow(
        cls,
        ax: Axes,
        source_position: tuple[float, float],
        target_position: tuple[float, float],
        *,
        statistic_value: float,
        fdr: float,
        node_size: tuple[float, float],
        show_fdr: bool,
        cmap: mcolors.Colormap,
        norm: Normalize,
        boundary_gap: float = 0.012,
    ) -> None:
        """Draw one straight arrow between two node boundaries.

        Both endpoints are pulled back to where the center-to-center line
        crosses each rectangle's boundary, so the segment meets each box along
        its own direction of travel and no arrowhead lands inside a rectangle.

        Arrows draw above rectangle fills but below module labels. An edge that
        skips a layer can pass across an intervening rectangle, and drawing it
        on top keeps the line continuous rather than making it look like it
        originates from the box it disappears behind.
        """
        source = np.asarray(source_position, dtype=float)
        target = np.asarray(target_position, dtype=float)
        start = source + cls._boundary_offset(
            target - source,
            node_size=node_size,
            boundary_gap=boundary_gap,
        )
        end = target + cls._boundary_offset(
            source - target,
            node_size=node_size,
            boundary_gap=boundary_gap,
        )

        significant = np.isfinite(fdr) and fdr <= 0.05
        arrow_alpha = 0.72 if not show_fdr else 0.9 if significant else 0.28
        arrow_rgb = tuple(
            float(channel) for channel in cmap(norm(statistic_value))[:3]
        )
        arrow_color = (*arrow_rgb, arrow_alpha)
        arrow_outline = (
            *(0.8 * channel for channel in arrow_rgb),
            arrow_alpha,
        )
        ax.add_patch(
            FancyArrowPatch(
                tuple(start),
                tuple(end),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=7.0,
                linewidth=0.8,
                facecolor=arrow_color,
                edgecolor=arrow_color,
                path_effects=[
                    path_effects.Stroke(
                        linewidth=1.3,
                        foreground=arrow_outline,
                    ),
                    path_effects.Normal(),
                ],
                clip_on=False,
                zorder=2.5,
            )
        )

    @classmethod
    def _within_layer_span(
        cls,
        edges: pd.DataFrame,
        *,
        max_layer_span: int,
        filename: str,
    ) -> pd.DataFrame:
        """Drop edges whose endpoints span too many mechanistic layers.

        Warns with the dropped edge labels and the modules that leave the
        figure as a result, since losing a module is a bigger change to the
        diagram than losing an edge and should not be silent.
        """
        span = (
            edges["target_module"].astype(str).map(cls._mechanistic_layer)
            - edges["source_module"].astype(str).map(cls._mechanistic_layer)
        ).abs()
        retained = edges.loc[span <= max_layer_span]
        dropped = edges.loc[span > max_layer_span]
        if dropped.empty:
            return retained

        def modules(frame: pd.DataFrame) -> set[str]:
            """Return the module identifiers appearing in a frame."""
            return set(frame["source_module"].astype(str)) | set(
                frame["target_module"].astype(str)
            )

        orphaned = sorted(modules(edges) - modules(retained))
        pairs = ", ".join(
            f"{cls._module_label(str(row.source_module))} -> "
            f"{cls._module_label(str(row.target_module))}"
            for row in dropped.itertuples()
        )
        message = (
            f"{filename}: max_layer_span={max_layer_span} dropped "
            f"{len(dropped)} of {len(edges)} edges ({pairs})"
        )
        if orphaned:
            labels = ", ".join(cls._module_label(module) for module in orphaned)
            message += f"; modules no longer shown: {labels}"
        warnings.warn(message, stacklevel=3)
        return retained

    @staticmethod
    def _boundary_offset(
        direction: np.ndarray,
        *,
        node_size: tuple[float, float],
        boundary_gap: float,
    ) -> np.ndarray:
        """Return the step from a node center to its boundary crossing.

        Solves the ray-rectangle intersection in layout coordinates: the
        crossing sits at the smaller of the horizontal and vertical
        half-extents scaled by `direction`, which is what makes a steep arrow
        exit through the top or bottom edge and a shallow one exit through a
        side. `boundary_gap` extends the step so the arrowhead clears the
        rectangle's stroke.
        """
        half_width, half_height = node_size[0] / 2.0, node_size[1] / 2.0
        length = float(np.hypot(*direction))
        if length == 0.0:
            return np.zeros(2)
        unit = direction / length
        scales = [
            half_width / abs(unit[0]) if unit[0] else np.inf,
            half_height / abs(unit[1]) if unit[1] else np.inf,
        ]
        return unit * (min(scales) + boundary_gap)

    @staticmethod
    def _layer_display_name(layer: int) -> str:
        """Return the band label for a coarse mechanistic layer."""
        names = {
            0: "Upstream ligand source",
            1: "Sensing (competence)",
            2: "Proximal signaling",
            3: "Pathway output",
            4: "Feedback / post-NASP",
        }
        return names.get(layer, f"Layer {layer}")

    @staticmethod
    def _fit_node_labels(
        fig: Figure,
        labels: Sequence[tuple[Text, FancyBboxPatch, str]],
        *,
        requested_node_size: tuple[float, float],
    ) -> None:
        """Wrap 5-point labels against their rendered rectangle bounds."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        padding_pixels = 0.5 * fig.dpi / 72.0

        for label, node_patch, module_label in labels:
            node_bounds = node_patch.get_window_extent(renderer)
            available_width = node_bounds.width - 2.0 * padding_pixels
            available_height = node_bounds.height - 2.0 * padding_pixels
            fitted = False

            label.set_fontsize(5.0)
            for wrap_width in range(len(module_label), 0, -1):
                label.set_text(
                    textwrap.fill(
                        module_label,
                        width=wrap_width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )
                label_bounds = label.get_window_extent(renderer)
                if (
                    label_bounds.width <= available_width
                    and label_bounds.height <= available_height
                ):
                    fitted = True
                    break

            if not fitted:
                drawn_node_size = (
                    float(node_patch.get_width()),
                    float(node_patch.get_height()),
                )
                raise ValueError(
                    f"requested node_size {requested_node_size} produced drawn "
                    f"node size {drawn_node_size}, which is too small to fit "
                    f"module label {module_label!r} at 5 pt; increase the "
                    "requested width or height"
                )

    @staticmethod
    def _mechanistic_layer(module: str) -> int:
        """Assign a module to a coarse left-to-right pathway layer."""
        if module in {
            "MITOCHONDRIAL_NA_SENSING",
            "TE_DEREPRESSION",
        }:
            return 0
        if "SENSING" in module:
            return 1
        if module == "CGAMP_TRANSPORT" or module.startswith(
            "SIGNALING_CONTEXT"
        ):
            return 2
        if module in {
            "IFN_I_OUTPUT",
            "NFKB_CYTOKINE_OUTPUT",
            "ISR",
            "INFLAMMASOME",
        }:
            return 3
        return 4
