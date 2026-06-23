"""Composition visualization via sankey for CELLxGENE metadata."""

import colorsys
from collections.abc import Callable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from nasp_atlas.cellxgene.categorize import _categorize_disease
from nasp_atlas.cellxgene.categorize import _categorize_tissue
from nasp_atlas.cellxgene.filter import _humanize_label
from nasp_atlas.cellxgene.visualization.composition import _count_dataset_labels
from nasp_atlas.cellxgene.visualization.composition import _format_plot_title
from nasp_atlas.cellxgene.visualization.composition import (
    _order_labels_by_category,
)
from nasp_atlas.cellxgene.visualization.composition import _select_plot_obs


def _label_colors(
    base_rgb: tuple[float, float, float],
    n_siblings: int,
    sibling_index: int,
) -> tuple[float, float, float]:
    """Return a lightness-shifted variant of base_rgb for a sibling node.

    Args:
      base_rgb: Parent category color as (r, g, b) floats in [0, 1].
      n_siblings: Total number of raw labels in this category.
      sibling_index: 0-based rank of this label within the category (0 =
        largest).

    Returns:
      (r, g, b) with lightness shifted.
    """
    if n_siblings == 1:
        return base_rgb

    hue, lightness, saturation = colorsys.rgb_to_hls(*base_rgb)
    offsets = [-0.15 + 0.30 * i / (n_siblings - 1) for i in range(n_siblings)]
    new_lightness = max(0.15, min(0.85, lightness + offsets[sibling_index]))
    return colorsys.hls_to_rgb(hue, new_lightness, saturation)


def _build_sankey_layout(
    counts: pd.DataFrame,
    node_gap: float,
    cmap: str,
) -> dict:
    """Compute geometry for two-column Sankey.

    Args:
      counts: DF with columns 'label', 'n_cells', 'fraction', 'category',
        ordered as they should appear top-to-bottom (largest category first,
        largest label first within each category).
      node_gap: Vertical gap between right-column nodes.
      cmap: Colormap palette to use.

    Returns:
      Dict with keys:
        'left_segments': list of dicts {category, y_top, height, color}
        'right_nodes': list of dicts {label, n_cells, fraction, y_top, height,
          color}
        'ribbons': list of dicts {y_src_top, ribbon_height, y_dst_top,
          dst_height, color}
        'category_colors': dict mapping category name to base (r,g,b)
    """
    total_cells = counts["n_cells"].sum()
    categories = counts["category"].unique().tolist()
    n_categories = len(categories)

    palette_colors = sns.color_palette(cmap, n_colors=n_categories)
    category_colors: dict[str, tuple[float, float, float]] = {
        category: tuple(palette_colors[index])  # type: ignore
        for index, category in enumerate(categories)
    }

    n_raw = len(counts)
    total_gap = node_gap * (n_raw - 1)
    scale = (1.0 - total_gap) / total_cells

    left_segments: list[dict] = []
    right_nodes: list[dict] = []
    ribbons: list[dict] = []

    left_height = total_cells * scale
    right_height = left_height + total_gap

    left_cursor = 0.5 + left_height / 2.0
    right_cursor = 0.5 + right_height / 2.0

    category_right_cursors: dict[str, float] = {}

    for category in categories:
        group = counts[counts["category"] == category]
        category_cells = group["n_cells"].sum()
        category_height = category_cells * scale
        category_y_top = left_cursor
        left_cursor -= category_height

        left_segments.append(
            {
                "category": category,
                "y_top": category_y_top,
                "height": category_height,
                "color": category_colors[category],
            }
        )
        category_right_cursors[category] = category_y_top

    category_sibling_counters: dict[str, int] = dict.fromkeys(categories, 0)
    category_sibling_totals: dict[str, int] = {
        category: int((counts["category"] == category).sum())
        for category in categories
    }

    for row in counts.itertuples():
        node_height = row.n_cells * scale
        node_y_top = right_cursor
        right_cursor -= node_height + node_gap

        sibling_index = category_sibling_counters[row.category]  # type: ignore
        category_sibling_counters[row.category] += 1  # type: ignore

        node_color = _label_colors(
            base_rgb=category_colors[row.category],  # type: ignore
            n_siblings=category_sibling_totals[row.category],  # type: ignore
            sibling_index=sibling_index,
        )

        right_nodes.append(
            {
                "label": row.label,
                "n_cells": row.n_cells,
                "fraction": row.fraction,
                "y_top": node_y_top,
                "height": node_height,
                "color": node_color,
            }
        )

        src_y_top = category_right_cursors[row.category]  # type: ignore
        category_right_cursors[row.category] -= node_height  # type: ignore

        ribbons.append(
            {
                "y_src_top": src_y_top,
                "ribbon_height": node_height,
                "y_dst_top": node_y_top,
                "dst_height": node_height,
                "color": node_color,
            }
        )

    return {
        "left_segments": left_segments,
        "right_nodes": right_nodes,
        "ribbons": ribbons,
        "category_colors": category_colors,
    }


def _draw_bezier_ribbon(
    ax: Axes,
    x_src: float,
    x_dst: float,
    y_src_top: float,
    ribbon_height: float,
    y_dst_top: float,
    dst_height: float,
    color: tuple[float, float, float],
    alpha: float = 0.45,
) -> None:
    """Draw one filled ribbon between two nodes.

    Args:
      ax: Target axes.
      x_src: Right x-edge of the source node (axes fraction).
      x_dst: Left x-edge of the destination node (axes fraction).
      y_src_top: Top y of the ribbon's attachment on the source node.
      ribbon_height: Vertical height of the ribbon at the source.
      y_dst_top: Top y of the destination node.
      dst_height: Full height of the destination node.
      color: RGB tuple for the ribbon fill.
      alpha: Ribbon transparency.
    """
    x_ctrl = x_src + 0.5 * (x_dst - x_src)

    y_src_bot = y_src_top - ribbon_height
    y_dst_bot = y_dst_top - dst_height

    verts = [
        (x_src, y_src_top),
        (x_ctrl, y_src_top),
        (x_ctrl, y_dst_top),
        (x_dst, y_dst_top),
        (x_dst, y_dst_bot),
        (x_ctrl, y_dst_bot),
        (x_ctrl, y_src_bot),
        (x_src, y_src_bot),
        (x_src, y_src_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=(*color, alpha),
        edgecolor="none",
        zorder=1,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)


def _plot_dataset_sankey(
    obs: pd.DataFrame,
    *,
    dataset_id: str | None,
    label_column: str,
    categorizer: Callable[[object], str],
    outpath: str | Path,
    node_height_in: float = 0.22,
    fig_width_in: float = 4.5,
    cmap: str = "tab20",
    node_width: float = 0.035,
    node_gap: float = 0.008,
    display_names: Mapping[str, str] | None = None,
) -> Path:
    """Render a two-column Sankey diagram of dataset composition.

    Args:
      obs: Metadata df with dataset_id and label_column.
      dataset_id: Optional dataset to visualize. If None, plots all current obs.
      label_column: Column whose unique values become right-column nodes.
      categorizer: Function mapping a raw label string to a broad category.
      outpath: Output dir for saved figure.
      node_height_in: Height allowed per node.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
      node_width: Width of nodes.
      node_gap: Vertical gap between right-column nodes.
      display_names: Optional human-readable labels for category values.

    Returns:
      Path to the written figure file.
    """
    counts = _count_dataset_labels(obs, dataset_id, label_column)
    plot_obs = _select_plot_obs(obs, dataset_id)
    total_cells = int(counts["n_cells"].sum())
    if "dataset_id" in plot_obs:
        n_datasets = int(plot_obs["dataset_id"].nunique())
    else:
        n_datasets = 1

    ordered, _ = _order_labels_by_category(
        counts=counts,
        categorizer=categorizer,
        ascending=False,
    )

    layout = _build_sankey_layout(counts=ordered, node_gap=node_gap, cmap=cmap)

    n_raw = len(ordered)
    fig_height = max(3.0, node_height_in * n_raw + 0.75)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_left_left = 0.18
    x_left_right = x_left_left + node_width
    x_right_left = 0.52
    x_right_right = x_right_left + node_width

    for segment in layout["left_segments"]:
        ax.add_patch(
            plt.Rectangle(  # type: ignore
                (x_left_left, segment["y_top"] - segment["height"]),
                node_width,
                segment["height"],
                facecolor=segment["color"],
                edgecolor="none",
                zorder=2,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

    for node in layout["right_nodes"]:
        ax.add_patch(
            plt.Rectangle(  # type: ignore
                (x_right_left, node["y_top"] - node["height"]),
                node_width,
                node["height"],
                facecolor=node["color"],
                edgecolor="none",
                zorder=2,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

    for ribbon in layout["ribbons"]:
        _draw_bezier_ribbon(
            ax=ax,
            x_src=x_left_right,
            x_dst=x_right_left,
            y_src_top=ribbon["y_src_top"],
            ribbon_height=ribbon["ribbon_height"],
            y_dst_top=ribbon["y_dst_top"],
            dst_height=ribbon["dst_height"],
            color=ribbon["color"],
        )

    fontsize = plt.rcParams["xtick.labelsize"]
    label_pad = 0.008

    for segment in layout["left_segments"]:
        y_mid = segment["y_top"] - segment["height"] / 2.0
        ax.text(
            x_left_left - label_pad,
            y_mid,
            _humanize_label(segment["category"], display_names),
            ha="right",
            va="center",
            fontsize=fontsize,
            transform=ax.transAxes,
        )

    for node in layout["right_nodes"]:
        y_mid = node["y_top"] - node["height"] / 2.0
        label_text = (
            f"{node['label']}  {node['n_cells']:,} ({node['fraction']:.1%})"
        )
        ax.text(
            x_right_right + label_pad,
            y_mid,
            label_text,
            ha="left",
            va="center",
            fontsize=fontsize,
            transform=ax.transAxes,
        )

    ax.set_title(_format_plot_title(dataset_id, total_cells, n_datasets))

    output_path = Path(outpath)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _default_categorizer(label_column: str) -> Callable[[object], str] | None:
    """Return a default categorizer for raw CELLxGENE metadata columns."""
    categorizers = {
        "tissue": _categorize_tissue,
        "disease": _categorize_disease,
    }
    return categorizers.get(label_column)


def _metadata_sankey(
    obs: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    label_column: str,
    outpath: str | Path,
    node_height_in: float = 0.22,
    fig_width_in: float = 4.5,
    cmap: str = "tab20",
    categorizer: Callable[[object], str] | None = None,
    display_names: Mapping[str, str] | None = None,
) -> Path:
    """Plot a Sankey diagram of composition along one metadata axis.

    Args:
      obs: Metadata df with dataset_id and label_column.
      dataset_id: Optional dataset to visualize. If None, plots all current obs.
      label_column: Column whose raw labels become right-column nodes.
      outpath: Output path for saved figure.
      node_height_in: Height allowed per raw-label node.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
      categorizer: Optional function mapping raw labels to broad categories.
      display_names: Optional human-readable labels for category values.

    Returns:
      Path to the written figure file.
    """
    active_categorizer = categorizer or _default_categorizer(label_column)
    if active_categorizer is None:
        raise ValueError(
            "metadata_sankey requires a categorizer for label_column "
            f"'{label_column}'."
        )

    return _plot_dataset_sankey(
        obs=obs,
        dataset_id=dataset_id,
        label_column=label_column,
        categorizer=active_categorizer,
        outpath=outpath,
        node_height_in=node_height_in,
        fig_width_in=fig_width_in,
        cmap=cmap,
        display_names=display_names,
    )
