"""Transcriptomic-clock analysis workflow for Tabula Sapiens tissues."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import (  # type: ignore[import]
    inset_axes,
)
from scipy import stats  # type: ignore[import]

from nasp_atlas.cellxgene.metadata import add_development_stage_age_obs
from nasp_atlas.single_cell.clocks.model import SPECIES_MAX_LIFESPAN
from nasp_atlas.single_cell.clocks.model import ClockModel
from nasp_atlas.single_cell.clocks.model import load_clock
from nasp_atlas.single_cell.clocks.model import model_feature_coverage
from nasp_atlas.single_cell.clocks.model import predict_metacells
from nasp_atlas.single_cell.clocks.preprocess import build_human_entrez_map
from nasp_atlas.single_cell.clocks.preprocess import build_mouse_ortholog_map
from nasp_atlas.single_cell.clocks.preprocess import preprocess_metacells
from nasp_atlas.single_cell.io import read_h5ad
from nasp_atlas.single_cell.metacells import aggregate_metacells
from nasp_atlas.single_cell.utils import normalize_h5ad_string_storage
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)

REPRESENTATION_TOKENS = {
    "scaleddiff": "scaled_diff",
    "yugenediff": "yugene_diff",
}
STRATUM_COLUMN = "stratum"
LEVEL_COLUMN = "level"
FEATURE_COVERAGE_SUFFIX = "_feature_coverage"


@dataclass(frozen=True, kw_only=True)
class ClockRegressionStyle:
    """Rendering options shared across clock regression plots."""

    cbar_height: str | float = "21%"
    cbar_width: str | float = "4%"
    cbar_pad: float = 0.075
    title: str | None = None
    x_tick_pad: float = 1
    y_tick_pad: float = 1
    x_tick_length: float = 2.5
    y_tick_length: float = 2.5
    scatter_cmap: str = "Blues"
    scatter_cmap_min: float = 0.25
    scatter_count_bins: int = 30
    scatter_alpha: float = 0.6


@dataclass(frozen=True, kw_only=True)
class ClockConfig:
    """Parameters to apply the per-tissue transcriptomic clock.

    Attributes:
      counts_layer: Layer holding counts to aggregate. None uses `.X`.
      ensembl_column: var column with Ensembl gene IDs for mapping.
      donor_key: obs column identifying donors.
      cell_type_key: obs column identifying cell types.
      tissue_key: obs column identifying tissue (constant per file).
      assay_key: obs colum identifying sequencing platform.
      assay_allowlist: Choose to limit analysis to specific platforms as they
        vary in coverage.
      development_stage_key: obs column with CxG development stage.
      age_key: obs column created to hold numeric age in years.
      levels: Aggregation levels to run ("tissue" and/or "cell_type").
      coverage_threshold: Minimum cumulative counts per metacell.
      count_threshold: Gene-filter minimum count.
      percent_threshold: Gene-filter minimum percent of metacells.
      min_metacells_per_stratum: Strata with fewer metacells are skipped.
      species: Query species selecting the max-lifespan adjustment.
      chronological_clock_key: clock_key used for age acceleration.
      broadcast_level: Level whose predictions are broadcast to `.obs`.
      random_seed: Seed used throughout.
      shuffle_metacells: Whether to randomize cells before metacell pooling.
    """

    counts_layer: str | None = "decontXcounts"
    ensembl_column: str = "ensembl_id"
    donor_key: str = "donor_id"
    cell_type_key: str = "cell_type"
    tissue_key: str = "tissue_in_publication"
    assay_key: str | None = "assay"
    assay_allowlist: tuple[str, ...] | None = None
    development_stage_key: str = "development_stage"
    age_key: str = "age_years"
    levels: Sequence[str] = ("tissue", "cell_type")
    coverage_threshold: float = 1e6
    count_threshold: float = 10.0
    percent_threshold: float = 20.0
    min_metacells_per_stratum: int = 3
    species: str = "human"
    chronological_clock_key: str = "chronoage"
    broadcast_level: str = "cell_type"
    random_seed: int = 42
    shuffle_metacells: bool = True

    def level_grouping(self, level: str) -> tuple[list[str], list[str] | None]:
        """Return (group_by, split_by) for an aggregation level.

        Assay, when configured, is added to both the metacell grouping and
        the relative-scaling stratum so platforms are never pooled into the
        same per-gene reference.

        Args:
          level: "tissue" or "cell_type".
        """
        group_by = [self.donor_key]
        stratum: list[str] = []
        if self.assay_key is not None:
            group_by.append(self.assay_key)
            stratum.append(self.assay_key)
        if level == "tissue":
            return group_by, (stratum or None)
        if level == "cell_type":
            return (
                [*group_by, self.cell_type_key],
                [*stratum, self.cell_type_key],
            )
        raise ValueError(
            f"Unknown level {level!r}; expected tissue or cell_type"
        )


def tissue_clock_analysis(
    *,
    h5ad_path: str | Path,
    output_dir: str | Path,
    gene_table: pd.DataFrame,
    ortholog_table: pd.DataFrame,
    model_paths: Sequence[str | Path],
    config: ClockConfig | None = None,
    save_tables: bool = True,
    annotate_adata: bool = True,
    save_adata: bool = False,
    plot_regressions: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the transcriptomic-clock workflow on one tissue h5ad.

    Args:
      h5ad_path: Path to a single tissue h5ad (cellxgene-standardized).
      output_dir: Directory for tidy tables and the annotated h5ad.
      gene_table: tAge human gene table (Ensembl -> human Entrez).
      ortholog_table: tAge ortholog table (human Entrez -> mouse Entrez).
      model_paths: Paths to the clock .pkl models to apply.
      config: Run configuration. Defaults to `ClockConfig()`.
      save_tables: Write one tidy CSV per level to `output_dir`.
      annotate_adata: Broadcast the configured level's predictions to `.obs`.
      save_adata: Write the annotated AnnData to `output_dir`.
      plot_regressions: Write predicted-vs-chronological-age regression plots
        for every clock prediction column at each non-empty level.

    Returns:
      Mapping of level name to its tidy metacell DataFrame.
    """
    config = config or ClockConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adata, _ = read_h5ad(h5ad_path)

    if config.assay_allowlist is not None and config.assay_key is not None:
        keep = adata.obs[config.assay_key].isin(config.assay_allowlist)
        logger.info(
            "[clock] assay filter %s: keeping %d/%d cells",
            config.assay_allowlist,
            int(keep.sum()),
            adata.n_obs,
        )
        adata = adata[keep.to_numpy()].copy()

    add_development_stage_age_obs(
        adata,
        stage_column=config.development_stage_key,
        age_column=config.age_key,
    )

    human_map = build_human_entrez_map(gene_table)
    mouse_map = build_mouse_ortholog_map(ortholog_table)
    clocks = _load_clocks(model_paths)

    results: dict[str, pd.DataFrame] = {}
    for level in config.levels:
        cell_assignment_key = f"metacell_id_{level}"
        tidy = _run_level(
            adata,
            level=level,
            clocks=clocks,
            human_map=human_map,
            mouse_map=mouse_map,
            config=config,
            cell_assignment_key=cell_assignment_key,
        )
        _add_age_acceleration(tidy, clocks=clocks, config=config)
        results[level] = tidy

        if not tidy.empty:
            adata.uns[f"nasp_clocks_{level}"] = tidy.reset_index()
        if save_tables and not tidy.empty:
            tidy.to_csv(output_path / f"clock_{level}_metacells.csv")
        if plot_regressions and not tidy.empty:
            plot_clock_regressions(
                tidy,
                output_dir=output_path,
                level=level,
                age_key=config.age_key,
            )
        logger.info("[clock] level=%s | metacells=%d", level, tidy.shape[0])

    broadcast = results.get(config.broadcast_level)
    if annotate_adata and broadcast is not None and not broadcast.empty:
        _broadcast_to_obs(
            adata,
            broadcast,
            cell_assignment_key=f"metacell_id_{config.broadcast_level}",
        )

    if save_adata:
        normalize_h5ad_string_storage(adata)
        adata.write_h5ad(output_path / f"{Path(h5ad_path).stem}_clocks.h5ad")

    return results


def plot_clock_regressions(
    tidy: pd.DataFrame,
    *,
    output_dir: str | Path,
    level: str,
    age_key: str,
    style: ClockRegressionStyle | None = None,
    cbar_height: str | float | None = None,
    cbar_width: str | float | None = None,
    cbar_pad: float | None = None,
    title: str | None = None,
    x_tick_pad: float | None = None,
    y_tick_pad: float | None = None,
    x_tick_length: float | None = None,
    y_tick_length: float | None = None,
    scatter_cmap: str | None = None,
    scatter_alpha: float | None = None,
) -> None:
    """Plot predicted clock age against chronological age for each clock.

    Args:
      tidy: Tidy metacell frame with chronological and predicted ages.
      output_dir: Directory where regression images are written.
      level: Aggregation level token used in output filenames.
      age_key: Chronological-age column in `tidy`.
      style: Base rendering controls. Defaults to `ClockRegressionStyle()`.
      cbar_height: Optional override for inset colorbar height.
      cbar_width: Optional override for inset colorbar width.
      cbar_pad: Optional override for plotting-axis to colorbar padding.
      title: Optional override for each regression plot title.
      x_tick_pad: Optional override for x-axis tick-label padding.
      y_tick_pad: Optional override for y-axis tick-label padding.
      x_tick_length: Optional override for x-axis tick mark length.
      y_tick_length: Optional override for y-axis tick mark length.
      scatter_cmap: Optional override for scatter-count colormap.
      scatter_alpha: Optional override for scatter marker alpha.

    Example Usage:
      >>> plot_clock_regressions(
      ...     tidy,
      ...     output_dir="clock_figures",
      ...     level="cell_type",
      ...     age_key="age_years",
      ...     title="Tabula Sapiens",
      ...     x_tick_pad=0.25,
      ...     x_tick_length=1.0,
      ...     scatter_cmap="Greens",
      ...     scatter_alpha=0.35,
      ... )
    """
    if age_key not in tidy.columns:
        logger.warning("[clock.%s] age column missing; skip regressions", level)
        return

    set_matplotlib_publication_parameters()
    style = replace(
        style or ClockRegressionStyle(),
        **{
            key: value
            for key, value in {
                "cbar_height": cbar_height,
                "cbar_width": cbar_width,
                "cbar_pad": cbar_pad,
                "title": title,
                "x_tick_pad": x_tick_pad,
                "y_tick_pad": y_tick_pad,
                "x_tick_length": x_tick_length,
                "y_tick_length": y_tick_length,
                "scatter_cmap": scatter_cmap,
                "scatter_alpha": scatter_alpha,
            }.items()
            if value is not None
        },
    )
    prediction_columns = [
        column
        for column in tidy.columns
        if column.endswith("_tage") and not column.endswith("_tage_std")
    ]
    for prediction_column in prediction_columns:
        _plot_clock_regression(
            tidy,
            age_key=age_key,
            prediction_column=prediction_column,
            output_path=(
                Path(output_dir)
                / f"clock_{level}_{_safe_filename_token(prediction_column)}"
                "_regression.png"
            ),
            style=style,
        )


def metacell_counts_frame(
    metacell_adata: ad.AnnData,
    *,
    ensembl_column: str,
) -> pd.DataFrame:
    """Return metacell counts as a metacells x Ensembl-id DataFrame."""
    var = cast(pd.DataFrame, metacell_adata.var)
    if ensembl_column in var.columns:
        gene_ids = var[ensembl_column].astype(str)
    else:
        gene_ids = metacell_adata.var_names.astype(str)
    gene_ids = gene_ids.str.split(".").str[0]
    return pd.DataFrame(
        np.asarray(metacell_adata.X),
        index=metacell_adata.obs_names,
        columns=gene_ids.to_numpy(),
    )


def stratum_indices(
    metacell_obs: pd.DataFrame,
    split_by: Sequence[str] | None,
) -> list[tuple[str, np.ndarray]]:
    """Return (stratum_label, positional indices) for each stratum.

    Args:
      metacell_obs: Metacell-level obs frame.
      split_by: One or more columns whose unique combinations define strata,
        or None for a single stratum spanning all metacells.

    Returns:
      A list of (composite label, positional indices) pairs.
    """
    if not split_by:
        return [("all", np.arange(metacell_obs.shape[0]))]
    positions = np.arange(metacell_obs.shape[0])
    composite = (
        metacell_obs[list(split_by)]
        .astype(str)
        .agg(" | ".join, axis=1)
        .to_numpy()
    )
    grouped = pd.Series(positions).groupby(composite, sort=True)
    return [(str(label), group.to_numpy()) for label, group in grouped]


def discover_tissue_h5ads(directory: str | Path) -> list[Path]:
    """Return sorted tissue h5ad paths in a directory."""
    return sorted(Path(directory).glob("*.h5ad"))


def _parse_model_spec(
    model_path: str | Path,
    *,
    model_type_prefixes: Sequence[str] = ("br", "en"),
) -> tuple[str, str, str]:
    """Parse a clock filename into (clock_key, representation, repr_token).

    Args:
      model_path: Path to a model whose stem encodes the clock and the
        representation, e.g. "BR_Chronoage_Multispecies_Multitissue_scaleddiff".
      model_type_prefixes: Filename tokens that identify the model type rather
        than the clock key.

    Returns:
      The short clock key, the preprocessing representation key, and the
      representation token found in the filename.
    """
    stem = Path(model_path).stem.lower()
    repr_token = next(
        (token for token in REPRESENTATION_TOKENS if token in stem),
        None,
    )
    if repr_token is None:
        raise ValueError(
            f"Cannot infer representation from model filename: {model_path}"
        )

    parts = stem.split("_")
    has_type_prefix = bool(parts) and parts[0] in model_type_prefixes
    clock_key = parts[1] if has_type_prefix else parts[0]
    return clock_key, REPRESENTATION_TOKENS[repr_token], repr_token


def _clock_metadata(clock_name: str | Path) -> tuple[str, str, str]:
    """Return (clock_key, representation, column_prefix)."""
    clock_key, representation, repr_token = _parse_model_spec(clock_name)
    return clock_key, representation, f"{clock_key}_{repr_token}"


def _load_clocks(model_paths: Sequence[str | Path]) -> list[ClockModel]:
    """Load clock models, validating that filenames encode metadata.

    Args:
      model_paths: Paths to serialized clock models.

    Returns:
      Loaded clocks, in input order.
    """
    clocks: list[ClockModel] = []
    for model_path in model_paths:
        _clock_metadata(model_path)
        clocks.append(load_clock(model_path))
    return clocks


def _predict_stratum(
    counts_frame: pd.DataFrame,
    base_obs: pd.DataFrame,
    *,
    clocks: Sequence[ClockModel],
    human_map: pd.Series,
    mouse_map: pd.Series,
    config: ClockConfig,
) -> pd.DataFrame:
    """Preprocess one stratum and predict every clock into a tidy frame."""
    representations = preprocess_metacells(
        counts_frame,
        human_map,
        mouse_map,
        count_threshold=config.count_threshold,
        percent_threshold=config.percent_threshold,
    )
    stratum_frame = base_obs.copy()
    for clock in clocks:
        _, representation, column_prefix = _clock_metadata(clock.name)
        features = representations[representation]
        prediction = predict_metacells(
            clock,
            features,
            species=config.species,
            return_std=True,
        )
        stratum_frame[f"{column_prefix}_tage"] = prediction["tage"]
        stratum_frame[f"{column_prefix}_tage_std"] = prediction["tage_std"]
        stratum_frame[f"{column_prefix}{FEATURE_COVERAGE_SUFFIX}"] = (
            model_feature_coverage(features, clock)
        )
    return stratum_frame


def _run_level(
    adata: ad.AnnData,
    *,
    level: str,
    clocks: Sequence[ClockModel],
    human_map: pd.Series,
    mouse_map: pd.Series,
    config: ClockConfig,
    cell_assignment_key: str,
) -> pd.DataFrame:
    """Build metacells and predict all clocks for one aggregation level."""
    group_by, split_by = config.level_grouping(level)
    carry_obs = [config.age_key]
    if config.tissue_key in adata.obs.columns:
        carry_obs.append(config.tissue_key)

    metacell_adata = aggregate_metacells(
        adata,
        group_by=group_by,
        counts_layer=config.counts_layer,
        coverage_threshold=config.coverage_threshold,
        carry_obs=carry_obs,
        shuffle=config.shuffle_metacells,
        random_seed=config.random_seed,
        cell_assignment_key=cell_assignment_key,
    )
    counts_frame = metacell_counts_frame(
        metacell_adata,
        ensembl_column=config.ensembl_column,
    )

    stratum_frames: list[pd.DataFrame] = []
    metacell_obs = cast(pd.DataFrame, metacell_adata.obs)
    strata = stratum_indices(metacell_obs, split_by)
    for stratum_label, positions in strata:
        if positions.shape[0] < config.min_metacells_per_stratum:
            logger.info(
                "[clock.%s] skip stratum %r (%d < %d metacells)",
                level,
                stratum_label,
                positions.shape[0],
                config.min_metacells_per_stratum,
            )
            continue

        base_obs = metacell_obs.iloc[positions].copy()
        base_obs[STRATUM_COLUMN] = stratum_label
        try:
            stratum_frames.append(
                _predict_stratum(
                    counts_frame.iloc[positions],
                    base_obs,
                    clocks=clocks,
                    human_map=human_map,
                    mouse_map=mouse_map,
                    config=config,
                )
            )
        except (ValueError, KeyError) as error:
            logger.warning(
                "[clock.%s] stratum %r failed: %s",
                level,
                stratum_label,
                error,
            )

    if not stratum_frames:
        return pd.DataFrame()

    tidy = pd.concat(stratum_frames)
    tidy[LEVEL_COLUMN] = level
    return tidy


def _add_age_acceleration(
    tidy: pd.DataFrame,
    *,
    clocks: Sequence[ClockModel],
    config: ClockConfig,
) -> None:
    """Add chronological age acceleration columns to a tidy metacell frame.

    Acceleration is a within-stratum deviation: the expected target is each
    metacell's chronological age over the species max lifespan, centered on the
    stratum median, and acceleration is the chronological clock prediction minus
    that expected value, with a years-scaled convenience column.

    Args:
      tidy: Tidy metacell frame with predictions and `age_key`.
      clocks: Loaded clocks; chronological ones are selected by clock_key
        parsed from the model name.
      config: Run configuration providing species and the age column.
    """
    if tidy.empty:
        return

    max_lifespan = SPECIES_MAX_LIFESPAN[config.species]
    expected_relative = tidy[config.age_key].astype(float) / max_lifespan
    stratum_keys = [LEVEL_COLUMN, STRATUM_COLUMN]
    reference = expected_relative.groupby(
        [tidy[key] for key in stratum_keys]
    ).transform("median")
    expected_centered = expected_relative - reference

    for clock in clocks:
        clock_key, _, column_prefix = _clock_metadata(clock.name)
        if clock_key != config.chronological_clock_key:
            continue
        tage_column = f"{column_prefix}_tage"
        if tage_column not in tidy.columns:
            continue
        predicted_relative = tidy[tage_column].astype(float) / max_lifespan
        acceleration = predicted_relative - expected_centered
        tidy[f"{column_prefix}_age_accel"] = acceleration
        tidy[f"{column_prefix}_age_accel_years"] = acceleration * max_lifespan


def _plot_clock_regression(
    tidy: pd.DataFrame,
    *,
    age_key: str,
    prediction_column: str,
    output_path: Path,
    style: ClockRegressionStyle,
) -> None:
    """Plot one predicted-vs-chronological-age regression figure."""
    x, y = _clock_regression_points(
        tidy,
        age_key=age_key,
        prediction_column=prediction_column,
    )
    if x.size < 2:
        logger.info(
            "[clock] skip regression %s (%d valid points)",
            prediction_column,
            x.size,
        )
        return

    style = replace(
        style,
        title=f"{style.title or prediction_column} (n={x.size})",
    )
    fig, ax = plt.subplots(figsize=(1.1, 1.1))
    mappable, ticks = _draw_clock_count_scatter(ax, x, y, style=style)
    _add_clock_count_colorbar(
        fig,
        ax,
        mappable=mappable,
        ticks=ticks,
        style=style,
    )
    _style_clock_regression_axes(ax, x, y, style=style)

    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    logger.info("[clock] regression plot -> %s", output_path)


def _clock_regression_points(
    tidy: pd.DataFrame,
    *,
    age_key: str,
    prediction_column: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return valid chronological and predicted age values for plotting."""
    plot_df = tidy.loc[:, [age_key, prediction_column]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    plot_df = plot_df.dropna()
    return (
        plot_df[age_key].to_numpy(dtype=float),
        plot_df[prediction_column].to_numpy(dtype=float),
    )


def _draw_clock_count_scatter(
    ax: Axes,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    style: ClockRegressionStyle,
) -> tuple[mcm.ScalarMappable, npt.NDArray[np.int_]]:
    """Draw metacells colored by integer local count."""
    counts = _point_bin_counts(x, y, bins=style.scatter_count_bins)
    order = np.argsort(counts)

    base_cmap = plt.get_cmap(style.scatter_cmap)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{style.scatter_cmap}_truncated",
        base_cmap(np.linspace(style.scatter_cmap_min, 1.0, base_cmap.N)),
    )
    min_count = int(np.min(counts))
    max_count = int(np.max(counts))
    norm = mcolors.Normalize(vmin=float(min_count), vmax=float(max_count))
    facecolors = cmap(norm(counts[order]))
    facecolors[:, -1] = style.scatter_alpha

    ax.scatter(
        x[order],
        y[order],
        s=4,
        facecolors=facecolors,
        linewidths=0,
        rasterized=True,
    )
    ticks = np.unique(np.array([min_count, max_count], dtype=int))
    mappable = mcm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.array([min_count, max_count], dtype=int))
    return mappable, ticks


def _add_clock_count_colorbar(
    fig: Figure,
    ax: Axes,
    *,
    mappable: mcm.ScalarMappable,
    ticks: npt.NDArray[np.int_],
    style: ClockRegressionStyle,
) -> None:
    """Add a compact metacell-count colorbar."""
    cax = inset_axes(
        ax,
        width=style.cbar_width,
        height=style.cbar_height,
        loc="center left",
        bbox_to_anchor=(1.02 + style.cbar_pad, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cbar.ax.tick_params(length=1.5, pad=0.5)


def _style_clock_regression_axes(
    ax: Axes,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    style: ClockRegressionStyle,
) -> None:
    """Apply regression annotation, labels, ticks, title, and spines."""
    if np.unique(x).size >= 2:
        _add_regression_fit_annotation(x, y, ax)

    ax.set_xlabel("Chronological age")
    ax.set_ylabel("Predicted relative age")
    x_tick_params = {"pad": style.x_tick_pad}
    y_tick_params = {"pad": style.y_tick_pad}

    if style.x_tick_length is not None:
        x_tick_params["length"] = style.x_tick_length
    if style.y_tick_length is not None:
        y_tick_params["length"] = style.y_tick_length

    ax.tick_params(axis="x", **x_tick_params)
    ax.tick_params(axis="y", **y_tick_params)

    if style.title is not None:
        ax.set_title(style.title, pad=3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)


def _add_regression_fit_annotation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    ax: Axes,
) -> None:
    """Draw a fitted regression line and Pearson correlation label."""
    slope, intercept = np.polyfit(x, y, deg=1)
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_line = np.array(x_limits, dtype=float)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="lightskyblue",
        linewidth=0.5,
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    if np.unique(y).size >= 2:
        pearson = cast(tuple[float, float], stats.pearsonr(x, y))
        r_value = pearson[0]
        ax.text(
            0.03,
            0.97,
            f"r={r_value:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )


def _point_bin_counts(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    bins: int,
) -> npt.NDArray[np.int_]:
    """Return the local x/y-bin metacell count for each point."""
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_bins = np.searchsorted(x_edges, x, side="right") - 1
    y_bins = np.searchsorted(y_edges, y, side="right") - 1
    x_bins = np.clip(x_bins, 0, hist.shape[0] - 1)
    y_bins = np.clip(y_bins, 0, hist.shape[1] - 1)
    return hist[x_bins, y_bins].astype(np.int_)


def _safe_filename_token(value: str) -> str:
    """Return a filesystem-safe token for generated plot filenames."""
    token = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )
    return token.strip("_") or "clock"


def _broadcast_to_obs(
    adata: ad.AnnData,
    tidy: pd.DataFrame,
    *,
    cell_assignment_key: str,
    column_prefix: str = "clock_",
    n_cells_column: str = "n_cells",
) -> None:
    """Broadcast metacell-level scalar columns to cells via the assignment."""
    assignment = adata.obs[cell_assignment_key].astype(str)
    scalar_suffixes = ("_tage", "_tage_std", "_age_accel", "_age_accel_years")
    scalar_columns = [
        column for column in tidy.columns if column.endswith(scalar_suffixes)
    ]
    scalar_columns.append(n_cells_column)
    for column in scalar_columns:
        series = tidy[column]
        adata.obs[f"{column_prefix}{column}"] = (
            assignment.map(series).astype(float).to_numpy()
        )
