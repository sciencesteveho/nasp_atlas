"""Publication-style visualizations for NASP interpretation tables."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from nasp_atlas.single_cell.visualization.style import _VisualizationStyleMixin


logger = logging.getLogger(__name__)


class _NaspPlotMixin(_VisualizationStyleMixin):
    """Mechanistic NASP plotting methods for tidy analysis tables."""

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
    ) -> None:
        """Plot symmetric module-coupling matrices with FDR markers.

        Duplicate module pairs are summarized by their median correlation,
        which makes the same method usable for consensus plots after tissue
        result tables have been concatenated.

        Example Usage:
          >>> viz.plot_module_coupling_heatmap(
          ...     coupling,
          ...     filename="module_coupling",
          ... )
        """
        self._set_matplotlib_publication_parameters()
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
            title = "Module coupling"
            if not show_fdr:
                title += " · descriptive"
            if analysis is not None:
                title += f" · {self._display_label(analysis)}"
            if facet_column is not None:
                title += f"\n{facet_label}"
            ax.set_title(title)
            used_axes.append(ax)
        self._hide_unused_axes(axes[len(facets) :])
        if image is not None:
            colorbar = fig.colorbar(
                image,
                ax=used_axes,
                fraction=0.025,
                pad=0.025,
            )
            colorbar.set_label("Spearman correlation")
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
        competence_column: str = "relative_competence",
        output_column: str = "relative_output",
        color_column: str = "output_minus_competence_gap",
        donor_count_column: str = "n_donors",
        min_donors: int = 2,
        max_labels: int = 30,
    ) -> None:
        """Plot relative competence against output for supported contexts.

        Point color shows the output-minus-competence mismatch and point size
        shows donor support. Axes remain fixed from zero to one so separately
        faceted tissues share the same relative-state geometry.

        Example Usage:
          >>> viz.plot_competence_output_state_map(
          ...     contexts,
          ...     filename="competence_output",
          ...     label_columns=["cell_type"],
          ... )
        """
        self._set_matplotlib_publication_parameters()
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
        fig, axes = self._panel_figure(
            len(facets),
            panel_width=2.5,
            panel_height=2.3,
        )
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        cmap = plt.get_cmap("RdBu_r")
        used_axes: list[Axes] = []
        for ax, (facet_label, block) in zip(axes, facets, strict=False):
            sizes = self._support_sizes(block[donor_count_column])
            ax.axline(
                (0, 0),
                slope=1,
                color="#999999",
                lw=0.45,
                ls="--",
                zorder=0,
            )
            ax.axvline(0.8, color="#dddddd", lw=0.4, zorder=0)
            ax.axhline(0.8, color="#dddddd", lw=0.4, zorder=0)
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
            labels = self._row_labels(block, label_columns)
            for index, (_, row) in enumerate(block.iterrows()):
                if index >= max_labels:
                    break
                ax.annotate(
                    labels[index],
                    (
                        float(row[competence_column]),
                        float(row[output_column]),
                    ),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=4,
                )
            ax.text(
                0.96,
                0.96,
                "active-like",
                ha="right",
                va="top",
                color="#555555",
            )
            ax.text(
                0.04,
                0.96,
                "responsive-like",
                ha="left",
                va="top",
                color="#555555",
            )
            ax.text(
                0.96,
                0.04,
                "competent-like",
                ha="right",
                va="bottom",
                color="#555555",
            )
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.set_aspect("equal")
            ax.set_xlabel("Relative sensing competence")
            ax.set_ylabel("Relative pathway output")
            title = "NASP evidence state"
            if facet_column is not None:
                title += f" · {facet_label}"
            ax.set_title(title)
            self._minimal_axis(ax)
            used_axes.append(ax)
        self._hide_unused_axes(axes[len(facets) :])
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=used_axes,
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("Output - competence")
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
    ) -> None:
        """Plot the highest supported contexts for each NASP hypothesis.

        The method works for a single tissue or an atlas-wide concatenated
        table; callers control whether labels contain cell type alone or both
        tissue and cell type.

        Example Usage:
          >>> viz.plot_ranked_nasp_hypotheses(
          ...     priorities,
          ...     filename="hypothesis_priorities",
          ...     label_columns=["tissue", "cell_type"],
          ... )
        """
        self._set_matplotlib_publication_parameters()
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
            panel_width=3.0,
            panel_height=2.2,
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
            y = np.arange(block.shape[0])
            ax.barh(
                y,
                block["priority_score"],
                color=colors.get(hypothesis, "#4c78a8"),
                height=0.72,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Priority score")
            ax.set_title(self._display_label(hypothesis))
            self._minimal_axis(ax)
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
        max_genes: int = 24,
        exclude_output_members: bool = True,
        show_fdr: bool = True,
    ) -> None:
        """Plot sensor-gene coupling to canonical output modules.

        Color represents correlation and dot size represents FDR evidence.
        Output-module members are excluded by default to reduce circular
        gene-to-module relationships.

        Example Usage:
          >>> viz.plot_sensor_output_mismatch(
          ...     sensor_output,
          ...     filename="sensor_output_mismatch",
          ... )
        """
        self._set_matplotlib_publication_parameters()
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
        fig, axes = self._panel_figure(
            len(facets),
            panel_width=2.8,
            panel_height=3.0,
        )
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
            sizes = (
                self._fdr_sizes(summary["_fdr"])
                if show_fdr
                else 10.0 + 25.0 * summary["_statistic"].abs()
            )
            ax.scatter(
                x,
                y,
                c=summary["_statistic"],
                s=sizes,
                cmap=cmap,
                norm=norm,
                edgecolor="#333333",
                linewidth=0.2,
            )
            ax.set_xticks(range(len(output_order)))
            ax.set_xticklabels(
                [self._module_label(value) for value in output_order],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(range(len(gene_order)))
            ax.set_yticklabels(gene_order)
            for label in ax.get_yticklabels():
                label.set_fontstyle("italic")
            ax.set_xlim(-0.5, len(output_order) - 0.5)
            ax.set_ylim(len(gene_order) - 0.5, -0.5)
            self._matrix_grid(ax, len(output_order), len(gene_order))
            title = "Sensor-to-output coupling"
            if not show_fdr:
                title += " · descriptive"
            if facet_column is not None:
                title += f" · {facet_label}"
            ax.set_title(title)
            used_axes.append(ax)
        self._hide_unused_axes(axes[len(facets) :])
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=used_axes,
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("Spearman correlation")
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
        slope_column: str = "slope",
        fdr_column: str = "ols_pvalue_fdr",
        max_features: int = 20,
        max_contexts: int = 30,
    ) -> None:
        """Plot context-specific age slopes as a significance-sized dot map.

        This is suitable for one tissue or concatenated cross-tissue
        regressions because the y axis is taken from the explicit regression
        stratum labels.

        Example Usage:
          >>> viz.plot_age_effect_dotplot(
          ...     regressions,
          ...     filename="age_effects",
          ... )
        """
        self._set_matplotlib_publication_parameters()
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

        width = max(3.0, len(feature_order) * 0.32)
        height = max(2.2, len(context_order) * 0.22)
        fig, ax = plt.subplots(figsize=(width, height))
        ax.scatter(
            x,
            y,
            c=slope,
            s=self._fdr_sizes(scoped[fdr_column]),
            cmap=cmap,
            norm=norm,
            edgecolor="#333333",
            linewidth=0.2,
        )
        ax.set_xticks(range(len(feature_order)))
        ax.set_xticklabels(
            [self._module_label(value) for value in feature_order],
            rotation=90,
        )
        ax.set_yticks(range(len(context_order)))
        ax.set_yticklabels(self._strip_common_context(context_order))
        ax.set_xlim(-0.5, len(feature_order) - 0.5)
        ax.set_ylim(len(context_order) - 0.5, -0.5)
        self._matrix_grid(ax, len(feature_order), len(context_order))
        ax.set_title("Age-associated NASP effects by context")
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("Score change per year")
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
        max_features: int = 30,
    ) -> None:
        """Plot median and range of age slopes across tested strata.

        The horizontal interval is the observed minimum-to-maximum slope, not
        a confidence interval. This plot is intended for the global aggregation
        stage after stability has been recomputed across tissues.

        Example Usage:
          >>> viz.plot_age_effect_consistency(
          ...     stability,
          ...     filename="age_consistency",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        required = [
            "feature_label",
            "analysis_scope",
            "median_slope",
            "min_slope",
            "max_slope",
            "direction_consistency_fraction",
            "n_significant_strata",
        ]
        self._require_columns(stability, required, table_name="stability")
        scoped = stability.loc[
            stability["analysis_scope"].astype(str) == analysis_scope
        ].copy()
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
        scoped = scoped.sort_values("median_slope", kind="stable")
        if scoped.empty:
            self._warn_empty(filename)
            return

        fig, ax = plt.subplots(figsize=(3.2, max(2.2, scoped.shape[0] * 0.22)))
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
        ax.set_yticklabels(
            [self._module_label(value) for value in scoped["feature_label"]]
        )
        ax.set_xlabel("Age slope (median and observed range)")
        ax.set_title("Cross-stratum age-effect consistency")
        self._minimal_axis(ax)
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.04,
            pad=0.03,
        )
        colorbar.set_label("Direction consistency")
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] NASP age-effect consistency -> %s",
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
        show_fdr: bool = True,
    ) -> None:
        """Plot expected mechanistic directions annotated by correlation.

        Arrow direction comes from the curated edge label while color and
        width show symmetric transcriptomic coupling. It must therefore not be
        interpreted as evidence of causal direction.

        Example Usage:
          >>> viz.plot_mechanistic_edge_network(
          ...     mechanistic_edges,
          ...     filename="mechanistic_network",
          ... )
        """
        self._set_matplotlib_publication_parameters()
        required = ["source_module", "target_module", statistic]
        self._require_columns(edges, required, table_name="edges")
        scoped = self._filter_analysis(edges, analysis=analysis).copy()
        if "skipped" in scoped:
            scoped = scoped.loc[~self._boolean_values(scoped["skipped"])]
        scoped[statistic] = self._numeric_values(scoped[statistic])
        scoped = scoped.loc[scoped[statistic].notna()]
        if scoped.empty:
            self._warn_empty(filename)
            return

        facets = self._facet_blocks(scoped, facet_column=facet_column)
        fig, axes = self._panel_figure(
            len(facets),
            panel_width=4.2,
            panel_height=3.0,
            max_columns=2,
        )
        norm = Normalize(vmin=-1.0, vmax=1.0)
        cmap = plt.get_cmap("RdBu_r")
        used_axes: list[Axes] = []
        for ax, (facet_label, block) in zip(axes, facets, strict=False):
            summary = self._edge_summary(
                block,
                statistic=statistic,
                fdr_column=fdr_column,
                max_edges=max_edges,
            )
            nodes = sorted(
                set(summary["source_module"]) | set(summary["target_module"])
            )
            positions = self._layered_node_positions(nodes)
            for edge_index, (_, row) in enumerate(summary.iterrows()):
                source = str(row["source_module"])
                target = str(row["target_module"])
                value = float(row["_statistic"])
                fdr = float(row["_fdr"])
                significant = np.isfinite(fdr) and fdr <= 0.05
                arrow = FancyArrowPatch(
                    positions[source],
                    positions[target],
                    arrowstyle="-|>",
                    mutation_scale=5.0,
                    connectionstyle=(
                        f"arc3,rad={0.035 * ((edge_index % 3) - 1):.3f}"
                    ),
                    linewidth=0.45 + 1.8 * abs(value),
                    color=cmap(norm(value)),
                    alpha=(
                        0.72 if not show_fdr else 0.9 if significant else 0.28
                    ),
                    zorder=1,
                )
                ax.add_patch(arrow)
            layer_colors = {
                0: "#fddbc7",
                1: "#d1e5f0",
                2: "#c7e9c0",
                3: "#decbe4",
                4: "#fed9a6",
            }
            for node in nodes:
                x, y = positions[node]
                layer = self._mechanistic_layer(node)
                ax.scatter(
                    [x],
                    [y],
                    s=95,
                    color=layer_colors[layer],
                    edgecolor="#444444",
                    linewidth=0.35,
                    zorder=2,
                )
                ax.text(
                    x,
                    y,
                    self._module_label(node),
                    ha="center",
                    va="center",
                    fontsize=4,
                    zorder=3,
                )
            ax.set_xlim(-0.08, 1.08)
            ax.set_ylim(-0.08, 1.08)
            ax.axis("off")
            title = "Mechanistic hypotheses · correlation overlay"
            if not show_fdr:
                title += " · descriptive"
            if facet_column is not None:
                title += f"\n{facet_label}"
            ax.set_title(title)
            used_axes.append(ax)
        self._hide_unused_axes(axes[len(facets) :])
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=used_axes,
            fraction=0.025,
            pad=0.02,
        )
        colorbar.set_label("Spearman correlation")
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
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(
            [_NaspPlotMixin._module_label(value) for value in x_labels],
            rotation=90,
        )
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(
            [_NaspPlotMixin._module_label(value) for value in y_labels]
        )
        _NaspPlotMixin._matrix_grid(ax, len(x_labels), len(y_labels))

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
    def _support_sizes(values: pd.Series) -> np.ndarray:
        """Scale donor support into restrained scatter sizes."""
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
        maximum = float(numeric.max())
        if maximum <= 0.0:
            return np.full(numeric.shape[0], 16.0)
        return 12.0 + 36.0 * np.sqrt(numeric.to_numpy(dtype=float) / maximum)

    @staticmethod
    def _fdr_sizes(values: pd.Series) -> np.ndarray:
        """Scale adjusted p-values into bounded evidence dot sizes."""
        fdr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        evidence = np.zeros_like(fdr)
        finite = np.isfinite(fdr)
        evidence[finite] = -np.log10(np.clip(fdr[finite], 1e-12, 1.0))
        return 7.0 + 30.0 * np.clip(evidence / 5.0, 0.0, 1.0)

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
    def _module_label(value: object) -> str:
        """Return compact display text for a module identifier."""
        label = str(value)
        replacements = {
            "SIGNALING_CONTEXT_": "Context: ",
            "NFKB_CYTOKINE_OUTPUT": "NF-κB output",
            "IFN_I_OUTPUT": "IFN-I output",
            "NASP_DNA_SENSING": "DNA sensing",
            "NASP_RNA_SENSING": "RNA sensing",
            "NASP_RESTRICTION": "Restriction",
            "NASP_FEEDBACK": "Feedback",
            "MITOCHONDRIAL_NA_SENSING": "Mitochondrial NA",
        }
        for source, target in replacements.items():
            label = label.replace(source, target)
        return label.replace("_", " ")

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
    ) -> dict[str, tuple[float, float]]:
        """Lay mechanistic nodes out from initiating context to phenotype."""
        layers: dict[int, list[str]] = {index: [] for index in range(5)}
        for node in nodes:
            layers[cls._mechanistic_layer(node)].append(node)
        positions: dict[str, tuple[float, float]] = {}
        for layer, layer_nodes in layers.items():
            ordered = sorted(layer_nodes)
            if not ordered:
                continue
            y_values = (
                np.asarray([0.5])
                if len(ordered) == 1
                else np.linspace(0.1, 0.9, len(ordered))
            )
            for node, y in zip(ordered, y_values, strict=True):
                positions[node] = (layer / 4.0, float(y))
        return positions

    @staticmethod
    def _mechanistic_layer(module: str) -> int:
        """Assign a module to a coarse left-to-right pathway layer."""
        if module in {
            "MITOCHONDRIAL_NA_SENSING",
            "TE_DEREPRESSION",
            "CGAMP_TRANSPORT",
        }:
            return 0
        if "SENSING" in module:
            return 1
        if module.startswith("SIGNALING_CONTEXT"):
            return 2
        if module in {
            "IFN_I_OUTPUT",
            "NFKB_CYTOKINE_OUTPUT",
            "ISR",
            "INFLAMMASOME",
        }:
            return 3
        return 4
