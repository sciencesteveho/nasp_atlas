"""Individual-sensor effects, expression, detection and donor support."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nasp_atlas.visualization import set_matplotlib_publication_parameters


__all__ = ["plot_sensor_effects", "plot_sensor_measurements"]


def plot_sensor_effects(
    estimates: pd.DataFrame,
    *,
    title: str,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 7,
) -> tuple[Figure, list[Axes]]:
    """Show every sensor, paired expression intervals and descriptive detection.

    Rows must be unique by dataset/gene. Intervals are pointwise; markers with
    no interval retain descriptive estimates. Missing estimates get an explicit
    unavailable annotation, never a point at zero. n labels count paired people.

    Example Usage:
      >>> figure, axes = plot_sensor_effects(
      ...     estimates, title="Spleen minus blood", label_fontsize=7,
      ... )
    """
    genes, datasets = _plot_order(estimates)
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, grid = plt.subplots(
            1,
            2,
            figsize=figsize or (10, 1.8 + len(genes) * 0.23),
            layout="constrained",
            sharey=True,
        )
        axes = list(grid)
        colors = ["#0072B2", "#D55E00"]
        offsets = (
            np.linspace(-0.16, 0.16, len(datasets))
            if len(datasets) > 1
            else [0.0]
        )
        for dataset, offset, color in zip(
            datasets, offsets, colors, strict=False
        ):
            selected = (
                estimates.loc[estimates.dataset.eq(dataset)]
                .set_index("gene")
                .reindex(genes)
            )
            positions = np.arange(len(genes)) + offset
            values = selected[
                ["estimate", "ci_lower", "ci_upper", "detection_difference"]
            ].to_numpy(dtype=float)
            for position, (effect, lower, upper, detection) in zip(
                positions, values, strict=True
            ):
                if np.isfinite(effect):
                    interval_available = np.isfinite(lower) and np.isfinite(
                        upper
                    )
                    if interval_available:
                        axes[0].plot(
                            [lower, upper],
                            [position, position],
                            color=color,
                            lw=0.8,
                        )
                    axes[0].scatter(
                        effect,
                        position,
                        s=13,
                        marker="o" if interval_available else "D",
                        facecolors=color if interval_available else "none",
                        edgecolors=color,
                    )
                else:
                    axes[0].text(
                        0.98,
                        position,
                        f"{dataset}: unavailable",
                        transform=axes[0].get_yaxis_transform(),
                        ha="right",
                        fontsize=5.5,
                        color=color,
                    )
                if np.isfinite(detection):
                    axes[1].scatter(
                        100 * detection,
                        position,
                        color=color,
                        s=13,
                    )
            axes[0].scatter([], [], color=color, s=15, label=dataset)
        labels = _support_labels(estimates, genes, datasets)
        axes[0].set_yticks(range(len(genes)), labels, fontsize=label_fontsize)
        axes[0].invert_yaxis()
        for axis in axes:
            axis.axvline(0, color="#888888", lw=0.6, zorder=0)
            axis.grid(axis="x", color="#eeeeee", lw=0.5)
        axes[0].set_xlabel(
            "Mean paired expression difference\n"
            "log1p(counts per 10,000); pointwise 95% CI\n"
            "Open diamond: descriptive estimate, interval unavailable"
        )
        axes[1].set_xlabel(
            "Mean paired detection difference\npercentage points; descriptive"
        )
        axes[0].legend(
            loc="upper left",
            bbox_to_anchor=(0, 1.065),
            frameon=False,
            ncol=len(datasets),
        )
        figure.suptitle(
            f"{title}\nExploratory · all curated sensors · "
            "paired people (n in dataset order)",
            fontsize=10,
        )
    return figure, axes


def plot_sensor_measurements(
    estimates: pd.DataFrame,
    *,
    title: str,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 7,
) -> tuple[Figure, list[Axes]]:
    """Show expression and detection for the same complete donor pairs.

    Mean expression and detection are averaged equally across people, with
    separate panels per dataset. Unavailable pairs stay visible as missing.

    Example Usage:
      >>> figure, axes = plot_sensor_measurements(
      ...     estimates, title="Classical monocytes", figsize=(12, 11),
      ... )
    """
    genes, datasets = _plot_order(estimates)
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, grid = plt.subplots(
            1,
            2 * len(datasets),
            figsize=figsize or (5 * len(datasets), 1.8 + 0.23 * len(genes)),
            layout="constrained",
            sharey=True,
            squeeze=False,
        )
        axes = list(grid[0])
        for number, dataset in enumerate(datasets):
            selected = (
                estimates.loc[estimates.dataset.eq(dataset)]
                .set_index("gene")
                .reindex(genes)
            )
            for metric, prefix, multiplier in (
                (0, "mean", 1),
                (1, "detected", 100),
            ):
                axis = axes[2 * number + metric]
                for position, row in enumerate(selected.itertuples()):
                    values = [
                        getattr(row, f"{prefix}_reference") * multiplier,
                        getattr(row, f"{prefix}_target") * multiplier,
                    ]
                    if np.isfinite(values).all():
                        axis.plot(
                            values,
                            [position, position],
                            color="#bbbbbb",
                            lw=0.6,
                        )
                        axis.scatter(
                            values,
                            [position, position],
                            c=["#0072B2", "#D55E00"],
                            s=12,
                        )
                    else:
                        axis.text(
                            0.5,
                            position,
                            "unavailable",
                            transform=axis.get_yaxis_transform(),
                            ha="center",
                            fontsize=5.5,
                        )
                axis.set_title(dataset, fontsize=9)
                axis.set_xlabel(
                    "Mean log1p expression"
                    if metric == 0
                    else "Detected cells (%)"
                )
                axis.set_xlim(left=0, right=100 if metric else None)
        finite_means = estimates[["mean_target", "mean_reference"]].to_numpy(
            dtype=float
        )
        maximum = (
            np.nanmax(finite_means) if np.isfinite(finite_means).any() else 1
        )
        for axis in axes[::2]:
            axis.set_xlim(0, max(1, maximum * 1.05))
        axes[0].set_yticks(
            range(len(genes)),
            _support_labels(estimates, genes, datasets),
            fontsize=label_fontsize,
        )
        axes[0].invert_yaxis()
        axes[0].scatter([], [], color="#0072B2", s=15, label="Reference")
        axes[0].scatter([], [], color="#D55E00", s=15, label="Target")
        axes[0].legend(
            loc="upper left", bbox_to_anchor=(0, 1.065), frameon=False, ncol=2
        )
        figure.suptitle(
            f"{title}\nEqual-person means · complete pairs only · "
            "log1p(counts per 10,000)",
            fontsize=10,
        )
    return figure, axes


def _plot_order(estimates: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Validate one or two datasets and retain their complete sensor union."""
    if estimates.duplicated(["dataset", "gene"]).any():
        raise ValueError("Sensor figure requires unique dataset/gene rows")
    datasets = estimates.dataset.drop_duplicates().astype(str).tolist()
    if not 1 <= len(datasets) <= 2:
        raise ValueError("Sensor figure accepts one or two datasets")
    return sorted(estimates.gene.unique()), datasets


def _support_labels(
    estimates: pd.DataFrame, genes: list[str], datasets: list[str]
) -> list[str]:
    """Make the independent-unit denominator visible for each sensor."""
    support = estimates.pivot(
        index="gene", columns="dataset", values="n_paired_donors"
    ).reindex(index=genes, columns=datasets)
    labels = []
    for gene, values in zip(genes, support.to_numpy(dtype=float), strict=True):
        counts = "/".join(
            str(int(value)) if np.isfinite(value) else "0" for value in values
        )
        labels.append(f"{gene}  (n={counts})")
    return labels
