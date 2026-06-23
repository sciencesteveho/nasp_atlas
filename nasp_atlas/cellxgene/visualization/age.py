"""Age-range visualization for CELLxGENE metadata."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nasp_atlas.cellxgene.categorize import _categorize_development_stage
from nasp_atlas.cellxgene.categorize import stage_age_value
from nasp_atlas.cellxgene.visualization.composition import _annotate_bars
from nasp_atlas.cellxgene.visualization.composition import _clean_label_series
from nasp_atlas.cellxgene.visualization.composition import _format_plot_title
from nasp_atlas.cellxgene.visualization.composition import _plot_stacked_bar
from nasp_atlas.cellxgene.visualization.composition import _select_plot_obs
from nasp_atlas.cellxgene.visualization.composition import _validate_columns
from nasp_atlas.visualization import _darken_color


def _age_labels_and_values(
    stages: pd.Series,
    collapsed: bool,
) -> tuple[pd.Series, pd.Series]:
    """Return plot labels and numeric sort values for stage labels."""
    labels = stages.map(_categorize_development_stage) if collapsed else stages
    age_values = stages.map(stage_age_value)
    return labels, age_values


def _ordered_age_categories(age_table: pd.DataFrame) -> list[str]:
    """Return age categories ordered by minimum numeric age value."""
    ordered = (
        age_table.groupby("category", observed=True)["sort_age"]
        .min()
        .reset_index()
        .sort_values(["sort_age", "category"], ascending=True)
    )
    return ordered["category"].astype(str).tolist()


def _build_age_range_table(
    obs: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    stage_column: str = "development_stage",
    collapsed: bool = True,
) -> pd.DataFrame:
    """Build an age-range count table from current obs or one dataset.

    Args:
      obs: Metadata df with dataset_id and stage_column.
      dataset_id: Optional dataset to summarize. If None, summarizes all current
        obs rows.
      stage_column: Column containing CELLxGENE development-stage labels.
      collapsed: If True, collapse raw development-stage labels to approximate
        age labels using the existing development-stage categorizer.

    Returns:
      DF with columns label, n_cells, fraction, and sort_age.
    """
    _validate_columns(obs, (stage_column,))
    plot_obs = _select_plot_obs(obs, dataset_id)
    stages = _clean_label_series(plot_obs[stage_column])
    labels, age_values = _age_labels_and_values(stages, collapsed)

    table = pd.DataFrame(
        {
            "label": labels,
            "age_value": age_values,
        }
    )

    counts = (
        table.groupby("label", observed=True)
        .agg(
            n_cells=("label", "size"),
            sort_age=("age_value", "min"),
        )
        .reset_index()
    )
    counts["fraction"] = counts["n_cells"] / counts["n_cells"].sum()

    return counts.sort_values(
        ["sort_age", "label"],
        ascending=True,
        ignore_index=True,
    )


def _build_age_makeup_table(
    obs: pd.DataFrame,
    *,
    stage_column: str = "development_stage",
    collapsed: bool = True,
    dataset_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a long-format dataset x age-range cell-count table.

    Args:
      obs: Cell-level metadata; must contain dataset_id and stage_column.
      stage_column: Column containing CELLxGENE development-stage labels.
      collapsed: If True, collapse raw development-stage labels to approximate
        age labels using the existing development-stage categorizer.
      dataset_meta: Optional dataset-level table merged in for display fields
        such as collection_name.

    Returns:
      DF with dataset_id, category, n_cells, fraction, sort_age, and any merged
      dataset metadata.
    """
    _validate_columns(obs, ("dataset_id", stage_column))
    if obs.empty:
        raise ValueError("Cannot build age makeup table from empty obs.")

    plot_obs = obs.loc[:, ["dataset_id", stage_column]].copy()
    stages = _clean_label_series(plot_obs[stage_column])
    labels, age_values = _age_labels_and_values(stages, collapsed)

    age_table = pd.DataFrame(
        {
            "dataset_id": plot_obs["dataset_id"],
            "category": labels,
            "sort_age": age_values,
        }
    )

    counts = (
        age_table.groupby(["dataset_id", "category"], observed=True)
        .agg(
            n_cells=("category", "size"),
            sort_age=("sort_age", "min"),
        )
        .reset_index()
    )
    dataset_totals = counts.groupby("dataset_id", observed=True)[
        "n_cells"
    ].transform("sum")
    counts["fraction"] = counts["n_cells"] / dataset_totals

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


def _plot_age_ranges(
    obs: pd.DataFrame,
    *,
    outpath: str | Path,
    dataset_id: str | None = None,
    stage_column: str = "development_stage",
    collapsed: bool = True,
    bar_height_in: float = 0.18,
    fig_width_in: float = 3.5,
    cmap: str = "tab20",
    bar_spacing: float = 0.5,
    bar_height: float = 0.4125,
) -> Path:
    """Plot age-range composition for current obs or one dataset.

    Args:
      obs: Metadata df with dataset_id and stage_column.
      outpath: Output path for saved figure.
      dataset_id: Optional dataset to visualize. If None, plots all current obs.
      stage_column: Column containing CELLxGENE development-stage labels.
      collapsed: If True, collapse raw development-stage labels to approximate
        age labels using the existing development-stage categorizer.
      bar_height_in: Height of each age-range bar.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
      bar_spacing: Vertical spacing between adjacent bars.
      bar_height: Height of each horizontal bar in y-axis units.

    Returns:
      Path to the written figure file.
    """
    counts = _build_age_range_table(
        obs,
        dataset_id=dataset_id,
        stage_column=stage_column,
        collapsed=collapsed,
    )
    plot_obs = _select_plot_obs(obs, dataset_id)
    total_cells = int(counts["n_cells"].sum())
    if "dataset_id" in plot_obs:
        n_datasets = int(plot_obs["dataset_id"].nunique())
    else:
        n_datasets = 1

    n_cells = counts["n_cells"].to_numpy()
    fractions = counts["fraction"].to_numpy()
    labels = counts["label"].tolist()

    y_positions = np.arange(len(counts)) * bar_spacing
    plot_height_units = max(1.0, (len(counts) - 1) * bar_spacing + bar_height)
    fig_height = max(1.2, bar_height_in * plot_height_units + 0.75)

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height))

    colors = sns.color_palette(cmap, n_colors=len(counts))
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
    title = _format_plot_title(
        dataset_id=dataset_id, total_cells=total_cells, n_datasets=n_datasets
    )
    ax.set_title(f"Age ranges\n{title}")

    y_padding = bar_spacing / 2
    ax.set_ylim(
        y_positions.min() - y_padding,
        y_positions.max() + y_padding,
    )

    ax.margins(y=0)
    sns.despine(ax=ax)

    output_path = Path(outpath)
    fig.savefig(output_path, bbox_inches="tight", dpi=450)
    plt.close(fig)

    return output_path


def _plot_age_makeup(
    obs: pd.DataFrame,
    datasets: pd.DataFrame,
    *,
    outpath: str | Path,
    stage_column: str = "development_stage",
    collapsed: bool = True,
    datasets_per_plot: int = 35,
    bar_height_in: float = 0.15,
    fig_width_in: float = 2.85,
    cmap: str = "tab20c",
) -> None:
    """Plot dataset makeup across development-stage age ranges.

    Args:
      obs: Metadata df with dataset_id and stage_column.
      datasets: Dataset-level metadata table.
      outpath: Output path. If multiple plots are written, chunk numbers are
        appended before the suffix.
      stage_column: Column containing CELLxGENE development-stage labels.
      collapsed: If True, collapse raw development-stage labels to approximate
        age labels using the existing development-stage categorizer.
      datasets_per_plot: Max num of datasets to show per figure.
      bar_height_in: Height of each dataset bar.
      fig_width_in: Figure width.
      cmap: Colormap palette to use.
    """
    makeup = _build_age_makeup_table(
        obs,
        stage_column=stage_column,
        collapsed=collapsed,
        dataset_meta=datasets,
    )
    category_order = _ordered_age_categories(makeup)
    _plot_stacked_bar(
        makeup=makeup,
        outpath=outpath,
        category_order=category_order,
        datasets_per_plot=datasets_per_plot,
        bar_height_in=bar_height_in,
        fig_width_in=fig_width_in,
        cmap=cmap,
        num_legend_cols=6,
    )
