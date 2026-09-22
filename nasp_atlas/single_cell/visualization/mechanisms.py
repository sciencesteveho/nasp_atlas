"""Compact publication figures for mechanism and reference diagnostics."""

from __future__ import annotations

import dataclasses
import textwrap
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nasp_compendium.display import humanize_module_name  # type: ignore[import]

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.reference_sets import reference_metadata
from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import set_matplotlib_publication_parameters


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Presentation:
    """Presentation overrides shared by one `plot_tables` call."""

    figsize: tuple[float, float] | None
    title: str | None
    x_label: str | None
    y_label: str | None
    title_fontsize: float | None
    label_fontsize: float | None
    tick_fontsize: float | None
    show_x_ticks: bool
    show_y_ticks: bool
    show_x_tick_labels: bool
    show_y_tick_labels: bool
    cmap_name: str | None
    cbar_label: str | None
    point_color: str


class MechanismPlotter(_PlotterBase):
    """Export descriptive coupling and component-expression summaries.

    Example Usage:
      >>> plotter = MechanismPlotter("path/to/output")
      >>> paths = plotter.plot_tables(tables, schema=ObsSchema())
    """

    def plot_tables(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        schema: ObsSchema,
        minimum_donors: int = 3,
        max_regulators: int = 12,
        max_contexts: int = 8,
        figsize: tuple[float, float] | None = None,
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
        cmap_name: str | None = None,
        cbar_label: str | None = None,
        point_color: str = "#4477aa",
    ) -> list[Path]:
        """Save PNG/PDF overviews and the full rows used for their display.

        `figsize` overrides inches; by default width is 3.5 inches and height
        fits wrapped rows. Positive row limits bound overview density.
        Missing estimates appear gray. Couplings have no significance marks;
        gene expression is scaled within each gene, not across genes.

        Presentation overrides apply to every figure rendered by one call, so
        pass one table family per call to tune one figure. `title`, `x_label`,
        `y_label` and `cbar_label` replace each figure's default text when not
        None; an empty string hides it. Font sizes are positive points; None
        keeps the shared publication style. The tick flags hide tick marks or
        tick labels. `cmap_name` replaces the default palettes ("RdBu_r" for
        correlations and z-scores, "Blues" for overlap) without changing their
        fixed limits; keep a diverging palette for signed values.
        `point_color` colors regulator scatter points.

        Example Usage:
          >>> paths = plotter.plot_tables(
          ...     tables, schema=ObsSchema(), max_regulators=8,
          ...     figsize=(3.5, 2.8), title="Coupling", cmap_name="PuOr_r",
          ... )
        """
        if min(max_regulators, max_contexts) < 1 or (
            figsize is not None
            and any(not np.isfinite(x) or x <= 0 for x in figsize)
        ):
            raise ValueError(
                "Figure dimensions and row limits must be positive"
            )
        for size in (title_fontsize, label_fontsize, tick_fontsize):
            if size is not None and (not np.isfinite(size) or size <= 0):
                raise ValueError("Font sizes must be positive and finite")

        presentation = _Presentation(
            figsize=figsize,
            title=title,
            x_label=x_label,
            y_label=y_label,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            show_x_ticks=show_x_ticks,
            show_y_ticks=show_y_ticks,
            show_x_tick_labels=show_x_tick_labels,
            show_y_tick_labels=show_y_tick_labels,
            cmap_name=cmap_name,
            cbar_label=cbar_label,
            point_color=point_color,
        )
        paths: list[Path] = []
        with plt.rc_context():
            set_matplotlib_publication_parameters()
            paths.extend(
                self._regulator_branches(
                    tables.get("regulator_output_coupling", pd.DataFrame()),
                    max_regulators=max_regulators,
                    presentation=presentation,
                )
            )
            paths.extend(
                self._regulator_points(
                    tables.get("regulator_output_points", pd.DataFrame()),
                    presentation=presentation,
                )
            )

            components = tables.get("mechanism_components", pd.DataFrame())
            if not components.empty:
                for name, block in (
                    (
                        "oas_rnasel",
                        components.loc[components.component.eq("OAS_RNaseL")],
                    ),
                    (
                        "interferon_components",
                        components.loc[components.component.ne("OAS_RNaseL")],
                    ),
                ):
                    paths.extend(
                        self._components(
                            block,
                            name,
                            schema,
                            minimum_donors,
                            max_contexts,
                            presentation,
                        )
                    )

            paths.extend(
                self._reference_comparisons(
                    tables.get("reference_comparisons", pd.DataFrame()),
                    presentation=presentation,
                )
            )

        return paths

    def _regulator_branches(
        self,
        regulators: pd.DataFrame,
        *,
        max_regulators: int,
        presentation: _Presentation,
    ) -> list[Path]:
        """Select supported regulators and render their branch comparison."""
        paths: list[Path] = []
        if not regulators.empty:
            supported = regulators.loc[regulators.status.eq("ok")]
            strongest = (
                supported.assign(strength=supported.spearman_r.abs())
                .groupby("gene")
                .strength.max()
                .nlargest(max_regulators)
                .index
            )
            shown = regulators.loc[regulators.gene.isin(strongest)].copy()
            if not shown.empty:
                shown["display_gene"] = (
                    shown.gene
                    + " (n="
                    + shown.groupby("gene")
                    .n_donors.transform("min")
                    .astype(str)
                    + ")"
                )
                paths.extend(
                    self._matrix(
                        shown,
                        "display_gene",
                        "output_module",
                        "spearman_r",
                        "Regulator-output coupling",
                        "regulator_branches",
                        presentation,
                    )
                )

        return paths

    def _regulator_points(
        self,
        points: pd.DataFrame,
        *,
        presentation: _Presentation,
    ) -> list[Path]:
        """Render selected pairs using centered donor-context values."""
        paths: list[Path] = []
        if not points.empty:
            for (gene, output), group in points.groupby(
                ["gene", "output_module"], sort=False
            ):
                figure, axis = plt.subplots(
                    figsize=presentation.figsize or (3.5, 2.2)
                )
                try:
                    axis.scatter(
                        group.gene_centered,
                        group.output_centered,
                        s=7,
                        color=presentation.point_color,
                        alpha=0.65,
                        linewidths=0,
                    )
                    self._decorate(
                        axis,
                        presentation,
                        title=(
                            f"Exploratory r={group.spearman_r.iloc[0]:.2f}; "
                            f"{group.independent_donor.nunique()} donors "
                            f"/ {len(group)} donor-contexts"
                        ),
                        x_label=f"{gene}: context-centered expression",
                        y_label=textwrap.fill(
                            f"{humanize_module_name(str(output))}: "
                            "centered score",
                            35,
                        ),
                    )
                    axis.spines[["top", "right"]].set_visible(False)
                    paths.extend(
                        self._export(
                            figure, group, f"regulator_{gene}_{output}"
                        )
                    )
                finally:
                    plt.close(figure)

        return paths

    def _reference_comparisons(
        self,
        comparison: pd.DataFrame,
        *,
        presentation: _Presentation,
    ) -> list[Path]:
        """Render reference agreement and overlap on matching module axes."""
        paths: list[Path] = []
        if not comparison.empty:
            selected = [
                "NASP_DNA_SENSING",
                "NASP_RNA_SENSING",
                "IFN_I_OUTPUT",
                "NFKB_CYTOKINE_OUTPUT",
                "ISR",
                "INFLAMMASOME",
                "SASP",
            ]
            shown = comparison.loc[
                comparison.curated_module.isin(selected)
            ].copy()
            names = reference_metadata().set_index("module_id")["display_name"]
            shown["reference_name"] = shown.reference_module.map(names)
            for value, title in (
                ("spearman_r", "Reference agreement"),
                (
                    "gene_jaccard",
                    "Reference overlap (not independent validation)",
                ),
            ):
                paths.extend(
                    self._matrix(
                        shown,
                        "reference_name",
                        "curated_module",
                        value,
                        title,
                        f"reference_{value}",
                        presentation,
                    )
                )

        return paths

    def _matrix(
        self,
        frame: pd.DataFrame,
        row: str,
        column: str,
        value: str,
        title: str,
        filename: str,
        presentation: _Presentation,
    ) -> list[Path]:
        """Plot signed correlations or unsigned overlap without filling gaps."""
        if frame.empty or not frame[value].notna().any():
            return []

        matrix = frame.pivot(index=row, columns=column, values=value)
        labels = [textwrap.fill(str(x), 32) for x in matrix.index]
        lines = sum(label.count("\n") + 1 for label in labels)
        dimensions = presentation.figsize or (
            3.5,
            max(1.5, 0.9 + 0.14 * lines),
        )

        figure, axis = plt.subplots(figsize=dimensions)
        try:
            default_cmap = "RdBu_r" if value == "spearman_r" else "Blues"
            cmap = plt.get_cmap(
                presentation.cmap_name or default_cmap
            ).with_extremes(bad="#dddddd")
            artist = axis.imshow(
                np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
                cmap=cmap,
                vmin=-1 if value == "spearman_r" else 0,
                vmax=1,
                aspect="equal",
            )

            axis.set_yticks(
                range(len(matrix)),
                labels,
            )
            axis.set_xticks(
                range(len(matrix.columns)),
                [
                    textwrap.fill(humanize_module_name(str(x)), 18)
                    for x in matrix.columns
                ],
                rotation=90,
            )
            axis.tick_params(length=0, pad=2)
            self._decorate(axis, presentation, title=title, title_pad=4)
            axis.spines[:].set_visible(False)
            default_label = "Spearman r" if value == "spearman_r" else "Jaccard"
            figure.colorbar(
                artist,
                ax=axis,
                fraction=0.035,
                pad=0.025,
                shrink=0.35,
                label=(
                    default_label
                    if presentation.cbar_label is None
                    else presentation.cbar_label
                ),
            )
            return self._export(figure, frame, filename)
        finally:
            plt.close(figure)

    def _components(
        self,
        frame: pd.DataFrame,
        name: str,
        schema: ObsSchema,
        minimum_donors: int,
        max_contexts: int,
        presentation: _Presentation,
    ) -> list[Path]:
        """Show donor-median expression and detection for ordered components."""
        display = _component_display(
            frame,
            schema=schema,
            minimum_donors=minimum_donors,
            max_contexts=max_contexts,
        )
        if display is None:
            return []

        shown, genes, selected = display
        matrix = shown.pivot(
            index="context", columns="gene", values="expression_z"
        ).reindex(index=selected.index, columns=genes)

        detection = shown.pivot(
            index="context", columns="gene", values="median_fraction_detected"
        ).reindex_like(matrix)

        labels = [
            textwrap.fill(f"{label} (max n={int(count)})", 35)
            for label, count in selected.items()
        ]
        lines = sum(label.count("\n") + 1 for label in labels)
        dimensions = presentation.figsize or (
            3.5,
            max(1.3, 0.75 + 0.15 * lines),
        )

        figure, axis = plt.subplots(figsize=dimensions)
        try:
            yy, xx = np.indices(matrix.shape)
            values = matrix.to_numpy(dtype=float)
            valid = np.isfinite(values)
            axis.scatter(
                xx[~valid],
                yy[~valid],
                marker="x",
                s=8,
                color="#aaaaaa",
                linewidths=0.4,
            )
            artist = axis.scatter(
                xx[valid],
                yy[valid],
                c=values[valid],
                s=3 + 28 * detection.to_numpy(dtype=float)[valid],
                cmap=presentation.cmap_name or "RdBu_r",
                vmin=-2,
                vmax=2,
                edgecolors="#777777",
                linewidths=0.2,
            )

            axis.set_xticks(
                range(len(genes)), genes, rotation=90, fontstyle="italic"
            )
            axis.set_yticks(
                range(len(selected)),
                labels,
            )
            axis.set_ylim(len(selected) - 0.5, -0.5)
            axis.set_xlim(-0.5, len(genes) - 0.5)
            axis.tick_params(length=0, pad=2)
            self._decorate(
                axis,
                presentation,
                title=(
                    "OAS / RNase L components"
                    if name == "oas_rnasel"
                    else "IFN ligands / receptor context / response"
                ),
                x_label="Dot area: donor-median detection; x: unavailable",
                title_pad=4,
                x_label_pad=3,
            )
            axis.spines[:].set_visible(False)
            figure.colorbar(
                artist,
                ax=axis,
                fraction=0.035,
                pad=0.025,
                shrink=0.35,
                label=(
                    "Gene-wise z"
                    if presentation.cbar_label is None
                    else presentation.cbar_label
                ),
            )
            return self._export(figure, shown, name)
        finally:
            plt.close(figure)

    def _decorate(
        self,
        axis: Axes,
        presentation: _Presentation,
        *,
        title: str,
        x_label: str = "",
        y_label: str = "",
        title_pad: float | None = None,
        x_label_pad: float | None = None,
    ) -> None:
        """Apply default or overridden axis text, font sizes and tick flags."""
        axis.set_title(
            title if presentation.title is None else presentation.title,
            loc="left",
            pad=title_pad,
            fontsize=presentation.title_fontsize,
        )
        axis.set_xlabel(
            x_label if presentation.x_label is None else presentation.x_label,
            labelpad=x_label_pad,
            fontsize=presentation.label_fontsize,
        )
        axis.set_ylabel(
            y_label if presentation.y_label is None else presentation.y_label,
            fontsize=presentation.label_fontsize,
        )
        if presentation.tick_fontsize is not None:
            axis.tick_params(labelsize=presentation.tick_fontsize)
        self._set_tick_visibility(
            axis,
            show_x_ticks=presentation.show_x_ticks,
            show_y_ticks=presentation.show_y_ticks,
            show_x_tick_labels=presentation.show_x_tick_labels,
            show_y_tick_labels=presentation.show_y_tick_labels,
        )

    def _export(
        self, figure: Figure, frame: pd.DataFrame, filename: str
    ) -> list[Path]:
        """Save compact vector/raster figures and their displayed data."""
        figure.tight_layout(pad=0.5)
        paths = []
        for extension in ("png", "pdf"):
            path = self.output_dir / f"{filename}.{extension}"
            figure.savefig(path, dpi=self.dpi)
            paths.append(path)

        path = self.output_dir / f"{filename}_displayed.csv"
        frame.to_csv(path, index=False)
        return [*paths, path]


def _component_display(
    frame: pd.DataFrame,
    *,
    schema: ObsSchema,
    minimum_donors: int,
    max_contexts: int,
) -> tuple[pd.DataFrame, list[str], pd.Series] | None:
    """Select supported contexts and scale expression within each gene."""
    if frame.empty:
        return None

    genes = frame.gene.drop_duplicates().tolist()
    contexts = [
        key
        for key in (
            schema.tissue_key,
            schema.cell_type_key,
            schema.assay_key,
        )
        if key in frame
    ]
    shown = frame.dropna(subset=contexts).copy()
    if shown.empty:
        return None

    shown["context"] = shown[contexts].astype(str).agg(" / ".join, axis=1)
    support = (
        shown.groupby("context")
        .n_donors.max()
        .sort_values(ascending=False, kind="stable")
    )
    selected = support.loc[support.ge(minimum_donors)].head(max_contexts)
    if selected.empty:
        return None

    shown = shown.loc[shown.context.isin(selected.index)].copy()
    shown["expression"] = shown.median_mean_expression.where(
        shown.n_donors.ge(minimum_donors)
    )
    means = shown.groupby("gene").expression.transform("mean")
    deviations = shown.groupby("gene").expression.transform("std")
    shown["expression_z"] = (shown.expression - means) / deviations.where(
        deviations.gt(0), 1
    )

    return shown, genes, selected
