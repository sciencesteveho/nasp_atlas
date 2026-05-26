"""Composition visualization via barplot for CELLxGENE metadata."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from nasp_atlas.cellxgene.categorize import categorize_disease
from nasp_atlas.cellxgene.categorize import categorize_tissue
from nasp_atlas.cellxgene.filter import humanize_label
from nasp_atlas.cellxgene.filter import order_categories
from nasp_atlas.visualization import _darken_color


def build_makeup_table(
    obs: pd.DataFrame,
    category_column: str,
    dataset_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a long-format dataset x category cell-count table.

    Args:
      obs: Cell-level metadata; must contain dataset_id and category_column.
      category_column: Column whose values become the category axis.
      dataset_meta: Optional dataset-level table merged in for display fields
        such as collection_name.

    Returns:
      DF with dataset_id, category, n_cells, fraction, and any merged dataset
        metadata.
    """
    _validate_columns(obs, ("dataset_id", category_column))
    if obs.empty:
        raise ValueError("Cannot build makeup table from empty obs.")

    plot_obs = obs.loc[:, ["dataset_id", category_column]].copy()
    plot_obs[category_column] = _clean_label_series(plot_obs[category_column])

    counts = (
        plot_obs.groupby(["dataset_id", category_column], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    dataset_totals = counts.groupby("dataset_id", observed=True)[
        "n_cells"
    ].transform("sum")

    counts["fraction"] = counts["n_cells"] / dataset_totals
    counts = counts.rename(columns={category_column: "category"})

    if dataset_meta is not None and "dataset_id" in dataset_meta.columns:
        merge_cols = [
            col
            for col in ("dataset_id", "collection_name")
            if col in dataset_meta.columns
        ]
        counts = counts.merge(
            dataset_meta[merge_cols], on="dataset_id", how="left"
        )

    return counts


def _validate_columns(obs: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when required obs columns are missing."""
    if missing := [column for column in columns if column not in obs.columns]:
        raise ValueError(f"Missing required obs columns: {', '.join(missing)}.")


def _clean_label_series(values: pd.Series) -> pd.Series:
    """Return labels with missing and blank values converted to unknown."""
    cleaned = values.fillna("unknown").astype(str).str.strip()
    return cleaned.replace("", "unknown")


def _font_size_points(size: object) -> float:
    """Return a matplotlib font size in points."""
    return float(FontProperties(size=size).get_size_in_points())  # type: ignore


def _select_plot_obs(
    obs: pd.DataFrame,
    dataset_id: str | None,
) -> pd.DataFrame:
    """Return all current obs or the subset matching one dataset."""
    if dataset_id is None:
        plot_obs = obs
    else:
        _validate_columns(obs, ("dataset_id",))
        plot_obs = obs.loc[obs["dataset_id"] == dataset_id]

    if plot_obs.empty:
        if dataset_id is None:
            raise ValueError("No cells found in current obs.")
        raise ValueError(f"No cells found for dataset_id '{dataset_id}'.")

    return plot_obs


def _format_plot_title(
    dataset_id: str | None,
    total_cells: int,
    n_datasets: int,
) -> str:
    """Format title text for whole-obs and single-dataset plots."""
    if dataset_id is not None:
        return f"{dataset_id}\n{total_cells:,} cells"

    dataset_label = "dataset" if n_datasets == 1 else "datasets"
    return f"{n_datasets:,} {dataset_label} | {total_cells:,} cells"


def _shorten_identifier(value: object, short_id_chars: int) -> str:
    """Shorten an identifier for plotting."""
    text = str(value)
    return text if short_id_chars <= 0 else text[:short_id_chars]


def _chunk_dataset_ids(
    dataset_ids: Sequence[object],
    chunk_size: int,
) -> list[list[object]]:
    """Split dataset IDs into ordered chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    return [
        list(dataset_ids[start : start + chunk_size])
        for start in range(0, len(dataset_ids), chunk_size)
    ]


def _format_chunk_output_path(
    outpath: str | Path,
    chunk_index: int,
    n_chunks: int,
) -> Path:
    """Return output path for one plot chunk."""
    path = Path(outpath)

    if n_chunks == 1:
        return path

    return path.with_name(f"{path.stem}_{chunk_index + 1:03d}{path.suffix}")


def _prepare_stacked_bar_data(
    makeup: pd.DataFrame,
    category_order: Sequence[str],
    sort_by_total_cells: bool,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Pivot makeup, filter/order categories, order datasets by total cells.

    Returns:
      Tuple of (dataset x category fraction pivot in plot order, total cells
      per dataset in plot order, ordered category names present in data).
    """
    if makeup.empty:
        raise ValueError("Cannot plot an empty makeup table.")

    pivot_fraction = makeup.pivot_table(
        index="dataset_id",
        columns="category",
        values="fraction",
        fill_value=0.0,
    )
    pivot_counts = makeup.pivot_table(
        index="dataset_id",
        columns="category",
        values="n_cells",
        fill_value=0,
    )

    categories = [
        category
        for category in category_order
        if category in pivot_fraction.columns
    ]
    if not categories:
        raise ValueError("No categories were available to plot.")

    pivot_fraction = pivot_fraction[categories]
    pivot_counts = pivot_counts[categories]

    total_cells = pivot_counts.sum(axis=1)
    if sort_by_total_cells:
        total_cells = total_cells.sort_values(ascending=True)

    pivot_fraction = pivot_fraction.loc[total_cells.index]
    return pivot_fraction, total_cells, categories


def _plot_stacked_bar_chunk(
    chunk_fraction: pd.DataFrame,
    chunk_total_cells: pd.Series,
    categories: list[str],
    color_map: dict[str, tuple[float, float, float]],
    outpath: Path,
    short_id_chars: int,
    bar_height_in: float,
    fig_width_in: float,
    display_names: Mapping[str, str] | None,
    num_legend_cols: int = 3,
) -> None:
    """Render one stacked-bar chunk figure to disk."""
    n_datasets = len(chunk_fraction)
    fig_height = max(2.0, bar_height_in * n_datasets + 0.9)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height))

    y_positions = np.arange(n_datasets)
    cursor = np.zeros(n_datasets)
    for category in categories:
        widths = chunk_fraction[category].to_numpy()
        ax.barh(
            y_positions,
            widths,
            height=0.675,
            left=cursor,
            color=color_map[category],
            label=humanize_label(category, display_names),
            edgecolor="white",
            linewidth=0.3,
        )
        cursor += widths

    short_ids = [
        _shorten_identifier(dataset_id, short_id_chars)
        for dataset_id in chunk_fraction.index
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(short_ids)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fraction of cells")
    ax.margins(y=0.001)

    annotation_fontsize = _font_size_points(plt.rcParams["xtick.labelsize"])
    for position, total in zip(
        y_positions, chunk_total_cells.to_numpy(), strict=False
    ):
        ax.text(
            1.01,
            position,  # type: ignore
            f"{int(total):,}",
            va="center",
            ha="left",
            fontsize=annotation_fontsize,
        )

    n_legend_cols = min(len(categories), num_legend_cols) if categories else 1
    ax.legend(
        bbox_to_anchor=(0.0, -0.225 / max(fig_height, 1.0) - 0.05),
        loc="upper left",
        ncol=n_legend_cols,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )
    sns.despine(ax=ax)

    fig.savefig(outpath, bbox_inches="tight", dpi=450)
    plt.close(fig)


def _plot_stacked_bar(
    makeup: pd.DataFrame,
    outpath: str | Path,
    category_order: Sequence[str],
    datasets_per_plot: int = 35,
    short_id_chars: int = 0,
    bar_height_in: float = 0.15,
    fig_width_in: float = 2.85,
    cmap: str = "tab20c",
    sort_by_total_cells: bool = True,
    display_names: Mapping[str, str] | None = None,
    num_legend_cols: int = 3,
) -> None:
    """Plot dataset makeup as chunked horizontal stacked bar figures.

    Args:
      makeup: long-format dataset x category cell-count table from
        `build_makeup_table`.
      outpath: Output dir. If multiple plots are written, chunk
        numbers are appended before the suffix.
      category_order: Categories in left-to-right plotting order.
      datasets_per_plot: Max num of datasets to show per figure.
      short_id_chars: Number of leading dataset_id characters to label. Values
        <= 0 use the full identifier.
      bar_height_in: Height of each dataset bar.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
      sort_by_total_cells: Order dataset by total cell count (largest sit at the
        top of each figure).
      display_names: Optional human-readable labels for category values.
    """
    pivot_fraction, total_cells, categories = _prepare_stacked_bar_data(
        makeup=makeup,
        category_order=category_order,
        sort_by_total_cells=sort_by_total_cells,
    )
    dataset_chunks = _chunk_dataset_ids(
        dataset_ids=list(total_cells.index),
        chunk_size=datasets_per_plot,
    )
    color_map = dict(
        zip(
            categories,
            sns.color_palette(cmap, n_colors=len(categories)),
            strict=False,
        )
    )

    for chunk_index, chunk_dataset_ids in enumerate(dataset_chunks):
        chunk_output_path = _format_chunk_output_path(
            outpath=outpath,
            chunk_index=chunk_index,
            n_chunks=len(dataset_chunks),
        )
        _plot_stacked_bar_chunk(
            chunk_fraction=pivot_fraction.loc[chunk_dataset_ids],
            chunk_total_cells=total_cells.loc[chunk_dataset_ids],  # type: ignore
            categories=categories,
            color_map=color_map,
            outpath=chunk_output_path,
            short_id_chars=short_id_chars,
            bar_height_in=bar_height_in,
            fig_width_in=fig_width_in,
            display_names=display_names,
            num_legend_cols=num_legend_cols,
        )


def _count_dataset_labels(
    obs: pd.DataFrame,
    dataset_id: str | None,
    label_column: str,
) -> pd.DataFrame:
    """Return per-label cell counts for current obs or one dataset.

    Args:
      obs: Metadata df with dataset_id and label_column.
      dataset_id: Optional dataset to filter to. If None, counts all current
        obs rows.
      label_column: Column to count unique values of.

    Returns:
      DF with columns 'label', 'n_cells', 'fraction', sorted ascending
        by n_cells.
    """
    _validate_columns(obs, (label_column,))
    plot_obs = _select_plot_obs(obs, dataset_id)

    labels = _clean_label_series(plot_obs[label_column])
    counts = (
        labels.value_counts().rename_axis("label").reset_index(name="n_cells")
    )
    counts = counts[counts["n_cells"] > 0].copy()
    if counts.empty:
        raise ValueError(f"No non-empty labels found in '{label_column}'.")

    counts["fraction"] = counts["n_cells"] / counts["n_cells"].sum()
    return counts.sort_values("n_cells", ascending=True).reset_index(drop=True)


def _annotate_bars(
    ax: Axes,
    y_positions: np.ndarray,
    n_cells: np.ndarray,
    fractions: np.ndarray,
    x_max: float,
) -> None:
    """Annotate horizontal bars with count and percentage labels. If the
    rendered label fits inside its bar, it is placed inside the bar,
    right-aligned in white. Otherwise, it is placed to the right of the bar.

    Args:
      ax: Axes containing the bars to annotate.
      y_positions: Y positions of the bars.
      n_cells: Cell counts for each bar.
      fractions: Cell count fractions for each bar.
      x_max: Maximum x value across all bars (used to calculate padding).
    """
    annotation_fontsize = _font_size_points(plt.rcParams["font.size"])

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()  # type: ignore

    inside_padding_px = annotation_fontsize * 0.25 * ax.figure.dpi / 72
    outside_padding_data = x_max * 0.01

    for position, count, fraction in zip(
        y_positions, n_cells, fractions, strict=False
    ):
        label_text = f"{int(count):,} ({fraction:.1%})"

        probe = ax.text(
            0,
            0,
            label_text,
            fontsize=annotation_fontsize,
            va="center_baseline",
            ha="left",
        )
        text_width_px = probe.get_window_extent(renderer=renderer).width
        probe.remove()

        bar_left_px = ax.transData.transform((0, position))[0]
        bar_right_px = ax.transData.transform((count, position))[0]
        bar_width_px = bar_right_px - bar_left_px

        if text_width_px + 1.5 * inside_padding_px <= bar_width_px:
            inside_x_data = ax.transData.inverted().transform(
                (bar_right_px - inside_padding_px, 0)
            )[0]
            ax.text(
                inside_x_data,
                position,
                label_text,
                va="center_baseline",
                ha="right",
                fontsize=annotation_fontsize,
                color="white",
                zorder=10,
            )
        else:
            ax.text(
                count + outside_padding_data,
                position,
                label_text,
                va="center_baseline",
                ha="left",
                fontsize=annotation_fontsize,
                color="black",
                zorder=10,
            )


def _composition_barplot(
    obs: pd.DataFrame,
    dataset_id: str | None,
    label_column: str,
    outpath: str | Path,
    bar_height_in: float = 0.18,
    fig_width_in: float = 3.5,
    cmap: str = "tab20",
    bar_spacing: float = 0.5,
    bar_height: float = 0.4125,
) -> None:
    """Plot a horizontal bar depicting composition along a categorical axis.

    Args:
      obs: Metadata df with dataset_id and label_column.
      dataset_id: Optional dataset to visualize. If None, plots all current obs.
      label_column: Column with categorical labels to plot composition of.
      outpath: Output dir for saved figure.
      bar_height_in: Height of each dataset bar.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
      bar_spacing: Vertical spacing between adjacent bars.
      bar_height: Height of each horizontal bar in y-axis units.
    """
    counts = _count_dataset_labels(obs, dataset_id, label_column)
    plot_obs = _select_plot_obs(obs, dataset_id)
    total_cells = int(counts["n_cells"].sum())
    if "dataset_id" in plot_obs:
        n_datasets = int(plot_obs["dataset_id"].nunique())
    else:
        n_datasets = 1
    n_labels = len(counts)

    n_cells = counts["n_cells"].to_numpy()
    fractions = counts["fraction"].to_numpy()
    labels = counts["label"].tolist()

    y_positions = np.arange(n_labels) * bar_spacing
    plot_height_units = max(1.0, (n_labels - 1) * bar_spacing + bar_height)
    fig_height = max(1.2, bar_height_in * plot_height_units + 0.75)

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height))

    colors = sns.color_palette(cmap, n_colors=n_labels)
    edge_colors = [_darken_color(color, factor=0.80) for color in colors]

    ax.barh(
        y_positions,
        n_cells,
        height=bar_height,
        color=colors,
        linewidth=0.3,
        edgecolor=edge_colors,
    )

    _annotate_bars(
        ax=ax,
        y_positions=y_positions,
        n_cells=n_cells,
        fractions=fractions,
        x_max=float(n_cells.max()),
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cells")
    ax.set_title(_format_plot_title(dataset_id, total_cells, n_datasets))

    y_padding = bar_spacing / 2
    ax.set_ylim(
        y_positions.min() - y_padding,
        y_positions.max() + y_padding,
    )

    ax.margins(y=0)
    sns.despine(ax=ax)

    fig.savefig(Path(outpath), bbox_inches="tight", dpi=450)
    plt.close(fig)


def _order_labels_by_category(
    counts: pd.DataFrame,
    categorizer: Callable[[object], str],
    ascending: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Assign categories to labels and order them for plotting.

    Categories are ordered by total cell count. Within each category, labels are
    sorted by count.
    """
    counts = counts.assign(category=counts["label"].map(categorizer))

    category_totals = (
        counts.groupby("category", observed=True)["n_cells"]
        .sum()
        .sort_values(ascending=ascending)
    )
    ordered_categories = [str(category) for category in category_totals.index]

    ordered = counts.assign(
        category=pd.Categorical(
            counts["category"],
            categories=ordered_categories,
            ordered=True,
        )
    ).sort_values(
        ["category", "n_cells"],
        ascending=[True, ascending],
        ignore_index=True,
    )

    return ordered, ordered_categories


def _compute_group_boundaries(
    ordered: pd.DataFrame,
    ordered_categories: list[str],
) -> list[tuple[int, int, str]]:
    """Compute (start, end, category) index spans for each category group."""
    boundaries: list[tuple[int, int, str]] = []
    current_start = 0
    for category in ordered_categories:
        group_size = int((ordered["category"] == category).sum())
        if group_size > 0:
            boundaries.append(
                (current_start, current_start + group_size, category)
            )
        current_start += group_size
    return boundaries


def _draw_alternating_bands(
    ax: Axes,
    y_positions: np.ndarray,
    group_boundaries: list[tuple[int, int, str]],
    y_padding: float,
) -> None:
    """Shade every other category group with a light background band."""
    for band_index, (start, end, _) in enumerate(group_boundaries):
        if band_index % 2 == 1:
            band_start = y_positions[start] - y_padding
            band_end = y_positions[end - 1] + y_padding
            ax.axhspan(band_start, band_end, color="0.965", zorder=0)


def _draw_category_sidebar(
    ax: Axes,
    y_positions: np.ndarray,
    group_boundaries: list[tuple[int, int, str]],
    category_colors: dict[str, tuple[float, float, float]],
    y_padding: float,
    display_names: Mapping[str, str] | None = None,
) -> None:
    """Draw colored category labels to the right of the axes.

    Measures the widest humanized label via the renderer to size a
    consistent band width, then places a colored rectangle and centered
    white text per category in a blended (axes-x, data-y) transform.

    Args:
      ax: Target axes.
      y_positions: Y positions of all bars.
      group_boundaries: Output of `_compute_group_boundaries`.
      category_colors: Mapping from category to RGB tuple.
      y_padding: Half the bar spacing, used to extend bands beyond bars.
      display_names: Optional human-readable labels for category values.
    """
    annotation_fontsize = _font_size_points(plt.rcParams["font.size"])
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()  # type: ignore

    max_label_width_px = 0.0
    for _, _, category in group_boundaries:
        probe = ax.text(
            0,
            0,
            humanize_label(category, display_names),
            fontsize=annotation_fontsize,
        )
        max_label_width_px = max(
            max_label_width_px,
            probe.get_window_extent(renderer=renderer).width,
        )
        probe.remove()

    ax_width_px = ax.get_window_extent(renderer=renderer).width
    padding_px = annotation_fontsize * 1.5 * ax.figure.dpi / 72
    band_width = (max_label_width_px + padding_px) / ax_width_px

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    band_x = 1.02

    for start, end, category in group_boundaries:
        band_start = y_positions[start] - y_padding
        band_end = y_positions[end - 1] + y_padding
        band_height = band_end - band_start
        mid_y = (y_positions[start] + y_positions[end - 1]) / 2.0

        ax.add_patch(
            Rectangle(
                (band_x, band_start),
                band_width,
                band_height,
                facecolor=category_colors[category],
                edgecolor="none",
                transform=trans,
                clip_on=False,
                zorder=3,
            )
        )
        ax.text(
            band_x + band_width / 2,
            mid_y,
            humanize_label(category, display_names),
            va="center",
            ha="center",
            fontsize=annotation_fontsize,
            color="white",
            transform=trans,
            clip_on=False,
            zorder=4,
        )


def _grouped_composition_barplot(
    obs: pd.DataFrame,
    dataset_id: str | None,
    label_column: str,
    *,
    categorizer: Callable[[object], str],
    outpath: str | Path,
    bar_height_in: float = 0.18,
    fig_width_in: float = 2.75,
    cmap: str = "tab20",
    bar_spacing: float = 0.5,
    bar_height: float = 0.4125,
    display_names: Mapping[str, str] | None = None,
) -> None:
    """Plot a grouped horizontal bar depicting composition by metadata.

    Args:
      obs: Metadata df with dataset_id and label_column.
      dataset_id: Optional dataset to visualize. If None, plots all current obs.
      label_column: Column with categorical labels to plot composition of.
      categorizer: Function mapping a raw label string to a broad category.
      outpath: Output dir for saved figure.
      bar_height_in: Height of each dataset bar.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
      bar_spacing: Vertical spacing between adjacent bars.
      bar_height: Height of each horizontal bar in y-axis units.
      display_names: Optional human-readable labels for category values.
    """
    counts = _count_dataset_labels(obs, dataset_id, label_column)
    plot_obs = _select_plot_obs(obs, dataset_id)
    total_cells = int(counts["n_cells"].sum())
    if "dataset_id" in plot_obs:
        n_datasets = int(plot_obs["dataset_id"].nunique())
    else:
        n_datasets = 1

    ordered, ordered_categories = _order_labels_by_category(counts, categorizer)
    group_boundaries = _compute_group_boundaries(ordered, ordered_categories)

    category_colors = dict(
        zip(
            ordered_categories,
            sns.color_palette(cmap, n_colors=len(ordered_categories)),
            strict=False,
        )
    )

    n_labels = len(ordered)
    n_cells = ordered["n_cells"].to_numpy()
    fractions = ordered["fraction"].to_numpy()
    labels = ordered["label"].tolist()

    y_positions = np.arange(n_labels) * bar_spacing
    y_padding = bar_spacing / 2
    plot_height_units = max(1.0, (n_labels - 1) * bar_spacing + bar_height)
    fig_height = max(1.2, bar_height_in * plot_height_units + 0.75)

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height))

    _draw_alternating_bands(ax, y_positions, group_boundaries, y_padding)

    bar_colors = [category_colors[category] for category in ordered["category"]]
    edge_colors = [_darken_color(color, factor=0.80) for color in bar_colors]
    ax.barh(
        y_positions,
        n_cells,
        height=bar_height,
        color=bar_colors,
        linewidth=0.3,
        edgecolor=edge_colors,
    )

    _annotate_bars(
        ax=ax,
        y_positions=y_positions,
        n_cells=n_cells,
        fractions=fractions,
        x_max=float(n_cells.max()),
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cells")
    ax.set_title(_format_plot_title(dataset_id, total_cells, n_datasets))
    ax.set_ylim(
        y_positions.min() - y_padding,
        y_positions.max() + y_padding,
    )
    ax.margins(y=0)

    _draw_category_sidebar(
        ax=ax,
        y_positions=y_positions,
        group_boundaries=group_boundaries,
        category_colors=category_colors,
        y_padding=y_padding,
        display_names=display_names,
    )

    sns.despine(ax=ax)
    fig.savefig(Path(outpath), bbox_inches="tight", dpi=450)
    plt.close(fig)


def _default_categorizer(label_column: str) -> Callable[[object], str] | None:
    """Return a default categorizer for raw CELLxGENE metadata columns."""
    categorizers = {
        "tissue": categorize_tissue,
        "disease": categorize_disease,
    }
    return categorizers.get(label_column)


def plot_category_makeup(
    obs: pd.DataFrame,
    datasets: pd.DataFrame,
    category_column: str,
    outpath: str | Path,
    front: tuple[str, ...] = (),
    back: tuple[str, ...] = (),
    datasets_per_plot: int = 35,
    display_names: Mapping[str, str] | None = None,
) -> None:
    """Plot dataset makeup along one categorical axis."""
    _validate_columns(obs, (category_column,))
    category_order = order_categories(
        _clean_label_series(obs[category_column]).unique(),
        front=front,
        back=back,
    )
    makeup = build_makeup_table(
        obs,
        category_column=category_column,
        dataset_meta=datasets,
    )
    _plot_stacked_bar(
        makeup=makeup,
        outpath=outpath,
        category_order=category_order,
        datasets_per_plot=datasets_per_plot,
        display_names=display_names,
    )


def metadata_barplot(
    obs: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    label_column: str,
    outpath: str | Path,
    bar_height_in: float = 0.18,
    fig_width_in: float = 2.75,
    cmap: str = "tab20",
    grouped: bool = False,
    categorizer: Callable[[object], str] | None = None,
    display_names: Mapping[str, str] | None = None,
) -> None:
    """Horizontal bar plot of composition along one categorical metadata axis.

    Args:
      obs: Metadata df with dataset_id and label_column.
      dataset_id: Optional dataset to visualize. If None, plots all current obs.
      label_column: Column with categorical labels to plot composition of.
      outpath: Output dir for saved figure.
      bar_height_in: Height of each dataset bar.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
      grouped: If True, group raw labels under their broad category derived with
        alternating background bands and colored category labels.
      categorizer: Optional function mapping raw labels to broad categories.
      display_names: Optional human-readable labels for category values.
    """
    if grouped:
        active_categorizer = categorizer or _default_categorizer(label_column)
        if active_categorizer is None:
            raise ValueError(
                "grouped=True requires a categorizer for label_column "
                f"'{label_column}'."
            )
        _grouped_composition_barplot(
            obs=obs,
            dataset_id=dataset_id,
            label_column=label_column,
            categorizer=active_categorizer,
            outpath=outpath,
            bar_height_in=bar_height_in,
            fig_width_in=fig_width_in,
            cmap=cmap,
            display_names=display_names,
        )
        return

    _composition_barplot(
        obs=obs,
        dataset_id=dataset_id,
        label_column=label_column,
        outpath=outpath,
        bar_height_in=bar_height_in,
        fig_width_in=fig_width_in,
        cmap=cmap,
    )
