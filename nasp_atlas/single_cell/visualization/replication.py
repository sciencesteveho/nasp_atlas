"""Donor support, registered decisions and paired expression contrasts."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from nasp_compendium.display import humanize_module_name

from nasp_atlas.visualization import set_matplotlib_publication_parameters


__all__ = [
    "plot_cohort_support",
    "plot_donor_contrasts",
    "plot_gene_support",
    "plot_replication_matrix",
    "plot_sensitivity_effects",
    "plot_subtype_extension",
]


def plot_gene_support(
    summary: pd.DataFrame,
    paired_genes: pd.DataFrame,
    *,
    module_ids: Sequence[str],
    target: str,
    reference: str,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 6,
) -> tuple[Figure, list[Axes]]:
    """Show donor expression, detection and differences for every module gene.

    Summary values come from gene_support_summary; gray difference points are
    individual people and black diamonds are their mean. Plus/minus/context
    labels show curated membership, not expected gene-level effects. There
    are no gene tests. Dimensions are inches and label_fontsize is points.

    Example Usage:
      >>> figure, axes = plot_gene_support(
      ...     summary, pairs, module_ids=["IFN_I_OUTPUT"],
      ...     target="spleen", reference="blood", figsize=(9, 5),
      ... )
    """
    sizes = [int(summary.module_id.eq(module).sum()) for module in module_ids]
    if (
        not module_ids
        or min(sizes) == 0
        or summary.duplicated(["module_id", "gene"]).any()
    ):
        raise ValueError(
            "Gene support requires all modules and one row per module gene"
        )
    dimensions = figsize or (10, max(5, sum(sizes) * 0.18 + len(sizes)))
    _validate_dimensions(dimensions, label_fontsize)
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, grid = plt.subplots(
            len(module_ids),
            3,
            figsize=dimensions,
            squeeze=False,
            layout="constrained",
            gridspec_kw={"height_ratios": sizes},
        )
        for row, module in enumerate(module_ids):
            genes = summary.loc[summary.module_id.eq(module)].sort_values(
                ["arm", "gene"]
            )
            labels = []
            for position, gene in enumerate(genes.itertuples()):
                arm = {
                    "positive": "+",
                    "inverse": "-",
                    "context_dependent": "context",
                }[str(gene.arm)]
                labels.append(f"{gene.gene} [{arm}] | n={gene.n_pairs}")
                for column, values in enumerate(
                    (
                        (gene.mean_reference, gene.mean_target),
                        (gene.detected_reference, gene.detected_target),
                    )
                ):
                    grid[row, column].plot(
                        values,
                        [position, position],
                        color="#bbbbbb",
                        linewidth=0.5,
                    )
                    grid[row, column].scatter(
                        values,
                        [position, position],
                        c=["#0072B2", "#D55E00"],
                        s=10,
                    )
                pairs = paired_genes.loc[
                    paired_genes.module_id.eq(module)
                    & paired_genes.gene.eq(gene.gene)
                    & paired_genes.arm.eq(gene.arm)
                ]
                grid[row, 2].scatter(
                    pairs.mean_expression_difference,
                    np.full(len(pairs), position),
                    s=5,
                    color="#9da7ad",
                    alpha=0.6,
                )
                if pd.notna(gene.mean_difference):
                    grid[row, 2].scatter(
                        [gene.mean_difference],
                        [position],
                        s=13,
                        marker="D",
                        color="black",
                    )
                else:
                    grid[row, 2].text(
                        0.02,
                        position,
                        "unavailable",
                        transform=grid[row, 2].get_yaxis_transform(),
                        fontsize=label_fontsize,
                    )
            for column, axis in enumerate(grid[row]):
                axis.set_yticks(
                    range(len(genes)),
                    labels if column == 0 else [],
                    fontsize=label_fontsize,
                )
                axis.set_ylim(len(genes) - 0.5, -0.5)
                axis.tick_params(axis="x", labelsize=label_fontsize)
                axis.set_xlabel(
                    (
                        "Mean donor log1p(counts/10k)",
                        "Mean donor fraction detected",
                        "Paired log-expression difference",
                    )[column]
                )
            grid[row, 0].set_title(humanize_module_name(module), loc="left")
            grid[row, 1].set_xlim(-0.04, 1.04)
            grid[row, 2].axvline(0, color="#555555", linewidth=0.5, zorder=0)
        figure.suptitle(
            f"Gene support: {target} (orange) vs {reference} (blue)\n"
            "Differences: people (gray), mean (black); no gene tests",
            fontsize=9,
        )
    return figure, list(grid.flat)


def plot_sensitivity_effects(
    effects: pd.DataFrame,
    *,
    module_ids: Sequence[str],
    title: str,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 6,
) -> tuple[Figure, list[Axes]]:
    """Compare saved within-cohort sensitivities on separate module/method axes.

    Filled dots and bars are variant effects and saved pointwise intervals,
    except the explicitly labeled donor-deletion range. Open diamonds are
    primary effects on the same donor intersection. When donor sets differ,
    blue squares show the variant on that intersection. Unavailable results
    stay visible. Dimensions are inches and label_fontsize is points.

    Example Usage:
      >>> figure, axes = plot_sensitivity_effects(
      ...     effects, module_ids=["IFN_I_OUTPUT"], title="Wells checks",
      ...     figsize=(10, 4),
      ... )
    """
    dimensions = figsize or (11, 2.8 * len(module_ids) + 0.6)
    _validate_dimensions(dimensions, label_fontsize)
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, grid = plt.subplots(
            len(module_ids),
            2,
            figsize=dimensions,
            squeeze=False,
            layout="constrained",
        )
        for row, module in enumerate(module_ids):
            for column, scorer in enumerate(("scanpy", "aucell")):
                axis = grid[row, column]
                selected = effects.loc[
                    effects.module_id.eq(module) & effects.scorer.eq(scorer)
                ]
                labels = []
                for position, (_, effect) in enumerate(selected.iterrows()):
                    labels.append(
                        f"{effect.label} | n={int(effect.n_paired_donors)}"
                    )
                    if np.isfinite(effect.estimate):
                        axis.scatter(
                            [effect.estimate],
                            [position],
                            s=13,
                            color="black",
                            zorder=3,
                        )
                    if np.isfinite(effect.ci_lower) and np.isfinite(
                        effect.ci_upper
                    ):
                        axis.hlines(
                            position,
                            effect.ci_lower,
                            effect.ci_upper,
                            color="#4d6473",
                            linewidth=1,
                        )
                    if np.isfinite(effect.baseline_matched_estimate):
                        axis.scatter(
                            [effect.baseline_matched_estimate],
                            [position],
                            marker="D",
                            s=17,
                            facecolors="none",
                            edgecolors="#D55E00",
                            linewidths=0.6,
                            zorder=4,
                        )
                    if (
                        effect.n_matched_donors != effect.n_paired_donors
                        and np.isfinite(effect.variant_matched_estimate)
                    ):
                        axis.scatter(
                            [effect.variant_matched_estimate],
                            [position],
                            marker="s",
                            s=14,
                            color="#0072B2",
                            zorder=5,
                        )
                    if (
                        not effect.eligible
                        and effect.interval_kind != "effect_range_not_CI"
                    ):
                        axis.text(
                            0.02,
                            position,
                            str(effect.eligibility_reason).replace("_", " "),
                            transform=axis.get_yaxis_transform(),
                            fontsize=label_fontsize,
                            va="center",
                        )
                axis.set_yticks(
                    range(len(labels)), labels, fontsize=label_fontsize
                )
                axis.set_ylim(len(labels) - 0.5, -0.5)
                axis.axvline(0, color="#aaaaaa", linewidth=0.5, zorder=0)
                axis.tick_params(axis="x", labelsize=label_fontsize)
                axis.set_title(
                    f"{humanize_module_name(module)} | {scorer}", loc="left"
                )
                axis.set_xlabel("Mean paired score difference")
        figure.suptitle(
            title + "\nBars: pointwise 95% CI, except donor-deletion min-max; "
            "open diamonds: matched primary effect\n"
            "Filled dots: all variant pairs; blue squares: matched variant "
            "when donor sets differ",
            fontsize=9,
        )
    return figure, list(grid.flat)


def plot_cohort_support(
    support: pd.DataFrame,
    *,
    target: str,
    reference: str,
    title: str,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 7,
) -> tuple[Figure, Axes]:
    """Show every audited person, including excluded or empty contexts.

    Rows require donor_id, n_reference_cells, n_target_cells, eligible and
    annotation (chemistry/site/modality/overlap). Dimensions are inches;
    label_fontsize is points. Counts use a log1p color scale and exact text.

    Example Usage:
      >>> figure, axis = plot_cohort_support(
      ...     support, target="spleen", reference="blood",
      ...     title="Wells support", figsize=(7, 5), label_fontsize=7,
      ... )
    """
    dimensions = figsize or (7, max(2.5, 0.23 * len(support) + 1))
    _validate_dimensions(dimensions, label_fontsize)
    rows = support.sort_values("donor_id")
    counts = rows[["n_reference_cells", "n_target_cells"]].to_numpy(float)
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=dimensions, layout="constrained")
        axis.imshow(np.log1p(counts), cmap="Blues", aspect="auto")
        maximum = np.log1p(counts).max()
        for (row, column), value in np.ndenumerate(counts):
            axis.text(
                column,
                row,
                str(int(value)),
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color="white" if np.log1p(value) > maximum * 0.6 else "black",
            )
        labels = [
            f"{row.donor_id} | {row.annotation}"
            + (" | excluded" if not row.eligible else "")
            for row in rows.itertuples(index=False)
        ]
        axis.set_yticks(range(len(rows)), labels, fontsize=label_fontsize)
        axis.set_xticks([0, 1], [reference, target], fontsize=label_fontsize)
        axis.tick_params(length=0)
        axis.set_title(title, loc="left")
        axis.set_xlabel(
            "Cells before paired-donor exclusion; color = log1p(count)"
        )
    return figure, axis


def plot_replication_matrix(
    results: pd.DataFrame,
    discovery: pd.DataFrame,
    *,
    title: str,
    figsize: tuple[float, float] = (9, 2.6),
    label_fontsize: float = 7,
) -> tuple[Figure, Axes]:
    """Display all primary questions, support and separate method agreement.

    Inputs are classified results and saved TS estimates. No inference is
    recalculated. Dimensions are inches; label_fontsize is points. Colors
    encode decisions redundantly with text; unavailable is not a zero effect.

    Example Usage:
      >>> figure, axis = plot_replication_matrix(
      ...     results, discovery, title="Frozen questions", figsize=(9, 3),
      ... )
    """
    _validate_dimensions(figsize, label_fontsize)
    rows = results.loc[
        results.analysis_role.eq("primary")
        & results.scorer.eq(results.primary_scorer)
    ]
    reference = (
        discovery.loc[discovery.scorer.eq("scanpy")]
        .set_index("module_id")
        .n_paired_donors.to_dict()
    )
    cells = []
    for _, row in rows.iterrows():
        n_reference = (
            str(int(reference[row.module_id]))
            if row.module_id in reference
            else "unavailable"
        )
        n_external = (
            str(int(row.n_paired_donors))
            if pd.notna(row.n_paired_donors)
            else "unavailable"
        )
        cells.append(
            [
                humanize_module_name(str(row.module_id)),
                "+" if row.expected_direction == 1 else "-",
                n_reference,
                n_external,
                str(row.status).replace("_", "\n"),
                str(row.method_concordance).replace("_", "\n"),
            ]
        )
    colors = {
        "supported": "#d6e8df",
        "opposite_direction": "#efd5ba",
        "inconclusive": "#e3e6ef",
        "unavailable": "#ededed",
    }
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=figsize, layout="constrained")
        axis.axis("off")
        table = axis.table(
            cellText=cells,
            colLabels=[
                "Module",
                "Expected\nsign",
                "TS people",
                "External\npeople",
                "Scanpy decision",
                "AUCell vs Scanpy",
            ],
            colWidths=[0.26, 0.10, 0.10, 0.12, 0.19, 0.23],
            cellLoc="center",
            bbox=Bbox.from_bounds(0, 0, 1, 0.85),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(label_fontsize)
        for index, status in enumerate(rows.status, start=1):
            table[index, 4].set_facecolor(colors[status])
        for cell in table.get_celld().values():
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.4)
        axis.set_title(
            title + "\nScanpy: full-family Holm; AUCell: sensitivity",
            loc="left",
        )
    return figure, axis


def plot_donor_contrasts(
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    *,
    module_ids: Sequence[str],
    title: str,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 6,
    marker_size: float = 12,
) -> tuple[Figure, list[Axes]]:
    """Show complete paired people and mean/pointwise CI on separate scales.

    Each panel owns its scorer/module scale. Gray points are people; the final
    black diamond is the mean with its pointwise CI. Unavailable intervals are
    explicitly labeled. Dimensions are inches, marker_size is squared points,
    and label_fontsize is points. Inputs retain unmatched pairs as diagnostics.

    Example Usage:
      >>> figure, axes = plot_donor_contrasts(
      ...     estimates, differences, module_ids=["IFN_I_OUTPUT"],
      ...     title="Spleen minus blood", figsize=(8, 2.8), marker_size=15,
      ... )
    """
    dimensions = figsize or (8, 2.2 * len(module_ids) + 0.5)
    _validate_dimensions(dimensions, label_fontsize)
    if not module_ids or not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("Provide modules and a positive finite marker_size")
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, grid = plt.subplots(
            len(module_ids),
            2,
            figsize=dimensions,
            squeeze=False,
            layout="constrained",
        )
        for row, module in enumerate(module_ids):
            for column, scorer in enumerate(("scanpy", "aucell")):
                axis = grid[row, column]
                pairs = differences.loc[
                    differences.feature_label.eq(module)
                    & differences.scorer.eq(scorer)
                    & differences.status.eq("complete")
                ].sort_values("donor_id")
                summary = estimates.loc[
                    estimates.module_id.eq(module) & estimates.scorer.eq(scorer)
                ]
                axis.axhline(0, color="#aaaaaa", linewidth=0.6, zorder=0)
                axis.scatter(
                    range(len(pairs)),
                    pairs.difference,
                    s=marker_size,
                    color="#6e8391",
                )
                if not summary.empty:
                    estimate = summary.iloc[0]
                    if np.isfinite(estimate.estimate):
                        axis.scatter(
                            len(pairs),
                            estimate.estimate,
                            marker="D",
                            s=marker_size,
                            color="black",
                        )
                    if np.isfinite(estimate.ci_lower) and np.isfinite(
                        estimate.ci_upper
                    ):
                        axis.vlines(
                            len(pairs),
                            estimate.ci_lower,
                            estimate.ci_upper,
                            color="black",
                            linewidth=1,
                        )
                    else:
                        axis.text(
                            0.02,
                            0.96,
                            "CI unavailable",
                            transform=axis.transAxes,
                            va="top",
                            fontsize=label_fontsize,
                        )
                labels = [*pairs.donor_id.astype(str), "mean"]
                axis.set_xticks(
                    range(len(labels)),
                    labels,
                    rotation=90,
                    fontsize=label_fontsize,
                )
                axis.set_title(
                    f"{humanize_module_name(module)} | {scorer} "
                    f"| n={len(pairs)}",
                    fontsize=label_fontsize + 1,
                )
                axis.set_ylabel("Target - reference score")
                axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(
            title + "\nSeparate score scales; pointwise 95% intervals"
        )
    return figure, list(grid.ravel())


def plot_subtype_extension(
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    frequencies: pd.DataFrame,
    *,
    module_ids: Sequence[str],
    subtypes: Sequence[str],
    title: str,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 6,
    marker_size: float = 12,
) -> tuple[Figure, list[Axes]]:
    """Show each subtype's paired Scanpy contrasts and its donor composition.

    Rows are modules, columns are subtypes; gray points are people and the
    black diamond is the mean with its pointwise CI on that panel's own scale.
    The bottom row gives each subtype's share of the person's vascular cells,
    because a broad mean can reflect composition. AUCell stays in the saved
    tables. Dimensions are inches, marker_size is squared points and
    label_fontsize is points. Missing estimates are labeled, never zero.

    Every row of a column shares one donor axis covering everyone with cells
    of that subtype or a test of it, so a person's bar sits under their own
    points and a person missing from one module leaves a visible gap. A bar is
    hatched when that person contributed to no test in the column, which keeps
    a composition bar from reading as support for the contrast above it.

    Example Usage:
      >>> figure, axes = plot_subtype_extension(
      ...     estimates, differences, frequencies,
      ...     module_ids=["IFN_I_OUTPUT"], subtypes=["ArtEC", "CapEC"],
      ...     title="Vascular subtypes minus CD4 T cells", marker_size=15,
      ... )
    """
    dimensions = figsize or (3.0 * len(subtypes) + 1, 2.0 * len(module_ids) + 2)
    _validate_dimensions(dimensions, label_fontsize)
    if (
        not module_ids
        or not subtypes
        or not np.isfinite(marker_size)
        or marker_size <= 0
    ):
        raise ValueError("Provide modules, subtypes and a positive marker_size")
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, grid = plt.subplots(
            len(module_ids) + 1,
            len(subtypes),
            figsize=dimensions,
            squeeze=False,
            layout="constrained",
        )
        for column, subtype in enumerate(subtypes):
            tested = differences.loc[
                differences.scorer.eq("scanpy")
                & differences.target_level.eq(subtype)
                & differences.status.eq("complete")
            ]
            composition = frequencies.loc[frequencies.subtype.eq(subtype)]
            donors = sorted(
                set(composition.donor_id.astype(str))
                | set(tested.donor_id.astype(str))
            )
            positions = {donor: index for index, donor in enumerate(donors)}
            for row, module in enumerate(module_ids):
                axis = grid[row, column]
                pairs = tested.loc[tested.feature_label.eq(module)].sort_values(
                    "donor_id"
                )
                summary = estimates.loc[
                    estimates.module_id.eq(module)
                    & estimates.scorer.eq("scanpy")
                    & estimates.target_context.eq(subtype)
                ]
                axis.axhline(0, color="#aaaaaa", linewidth=0.6, zorder=0)
                axis.scatter(
                    [positions[str(donor)] for donor in pairs.donor_id],
                    pairs.difference,
                    s=marker_size,
                    color="#6e8391",
                )
                estimate = summary.iloc[0] if not summary.empty else None
                if estimate is not None and np.isfinite(estimate.estimate):
                    axis.scatter(
                        len(donors),
                        estimate.estimate,
                        marker="D",
                        s=marker_size,
                        color="black",
                    )
                if (
                    estimate is not None
                    and np.isfinite(estimate.ci_lower)
                    and np.isfinite(estimate.ci_upper)
                ):
                    axis.vlines(
                        len(donors),
                        estimate.ci_lower,
                        estimate.ci_upper,
                        color="black",
                        linewidth=1,
                    )
                else:
                    axis.text(
                        0.02,
                        0.96,
                        "interval unavailable"
                        if estimate is not None
                        and np.isfinite(estimate.estimate)
                        else "estimate unavailable",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=label_fontsize,
                    )
                axis.set_xticks(
                    range(len(donors) + 1),
                    [*donors, "mean"],
                    rotation=90,
                    fontsize=label_fontsize,
                )
                axis.set_title(
                    f"{humanize_module_name(module)} | {subtype} "
                    f"| n={len(pairs)}",
                    fontsize=label_fontsize + 1,
                )
                axis.set_ylabel("Subtype - CD4 T score")
                axis.spines[["top", "right"]].set_visible(False)
            axis = grid[len(module_ids), column]
            shares = (
                composition.set_index(composition.donor_id.astype(str))
                .fraction_of_vascular_cells.reindex(donors)
                .fillna(0.0)
            )
            contributed = set(tested.donor_id.astype(str))
            axis.bar(
                range(len(donors)),
                shares,
                color=[
                    "#6e8391" if donor in contributed else "#dde3e7"
                    for donor in donors
                ],
                hatch=[
                    "" if donor in contributed else "///" for donor in donors
                ],
                edgecolor="#6e8391",
                linewidth=0.4,
            )
            axis.set_xticks(
                range(len(donors)),
                donors,
                rotation=90,
                fontsize=label_fontsize,
            )
            axis.set_ylim(0, 1)
            axis.set_title(
                f"{subtype}: share of vascular cells",
                fontsize=label_fontsize + 1,
            )
            axis.set_ylabel("Fraction")
            axis.spines[["top", "right"]].set_visible(False)
            for donor_axis in grid[:, column]:
                donor_axis.set_xlim(-0.6, len(donors) + 0.6)
        figure.suptitle(
            title
            + "\nSeparate score scales; pointwise 95% intervals"
            + "\nEvery row of a column shares one donor axis; hatched bars "
            "are people contributing to no test in that column"
        )
    return figure, list(grid.ravel())


def _validate_dimensions(
    dimensions: tuple[float, float], fontsize: float
) -> None:
    """Reject unusable presentation sizes before allocating a figure."""
    if any(
        not np.isfinite(value) or value <= 0
        for value in (*dimensions, fontsize)
    ):
        raise ValueError(
            "Figure dimensions and font size must be positive and finite"
        )
