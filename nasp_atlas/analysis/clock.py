"""Transcriptomic-clock analysis workflow for Tabula Sapiens tissues."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd

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
from nasp_atlas.single_cell.metacells import N_CELLS_COLUMN
from nasp_atlas.single_cell.metacells import aggregate_metacells
from nasp_atlas.single_cell.utils import normalize_h5ad_string_storage


logger = logging.getLogger(__name__)

REPRESENTATION_TOKENS = {
    "scaleddiff": "scaled_diff",
    "yugenediff": "yugene_diff",
}
STRATUM_COLUMN = "stratum"
LEVEL_COLUMN = "level"
FEATURE_COVERAGE_SUFFIX = "_feature_coverage"


@dataclass
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


def run_tissue_clock_analysis(
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


def _metacell_counts_frame(
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


def _stratum_indices(
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
    counts_frame = _metacell_counts_frame(
        metacell_adata,
        ensembl_column=config.ensembl_column,
    )

    stratum_frames: list[pd.DataFrame] = []
    metacell_obs = cast(pd.DataFrame, metacell_adata.obs)
    strata = _stratum_indices(metacell_obs, split_by)
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


def _broadcast_to_obs(
    adata: ad.AnnData,
    tidy: pd.DataFrame,
    *,
    cell_assignment_key: str,
    column_prefix: str = "clock_",
) -> None:
    """Broadcast metacell-level scalar columns to cells via the assignment."""
    assignment = adata.obs[cell_assignment_key].astype(str)
    scalar_suffixes = ("_tage", "_tage_std", "_age_accel", "_age_accel_years")
    scalar_columns = [
        column for column in tidy.columns if column.endswith(scalar_suffixes)
    ]
    scalar_columns.append(N_CELLS_COLUMN)
    for column in scalar_columns:
        series = tidy[column]
        adata.obs[f"{column_prefix}{column}"] = (
            assignment.map(series).astype(float).to_numpy()
        )


def discover_tissue_h5ads(directory: str | Path) -> list[Path]:
    """Return sorted tissue h5ad paths in a directory."""
    return sorted(Path(directory).glob("*.h5ad"))
