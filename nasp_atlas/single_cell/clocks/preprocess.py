"""Transcriptomic-clock preprocessing on metacell count matrices."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def preprocess_metacells(
    counts: pd.DataFrame,
    human_entrez_map: pd.Series,
    mouse_ortholog_map: pd.Series,
    *,
    count_threshold: float = 10.0,
    percent_threshold: float = 20.0,
    reference_mask: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full tAge relative pipeline for one stratum of metacells.

    Pipeline:
      filter → ortholog mapping → RLE → log10 → per-sample Z-score → YuGene →
      per-gene median subtraction

    Args:
      counts: Metacells x source-identifier counts for a single stratum.
      human_entrez_map: Source identifier -> human Entrez.
      mouse_ortholog_map: Human Entrez -> mouse Entrez.
      count_threshold: Gene-filter minimum count.
      percent_threshold: Gene-filter minimum percent of metacells.
      reference_mask: Reference metacells for median subtraction. None uses all.

    Returns:
      Mapping with relative representations in mouse-Entrez space:
        "scaled": z-scored log-RLE values.
        "yugene": YuGene of the scaled values.
        "scaled_diff": median-subtracted scaled values.
        "yugene_diff": median-subtracted YuGene values.
    """
    filtered = _filter_low_count_genes(
        counts,
        count_threshold=count_threshold,
        percent_threshold=percent_threshold,
    )
    mapped = map_counts_to_model_features(
        filtered, human_entrez_map, mouse_ortholog_map
    )
    normalized = _rle_normalize(mapped)
    log_normalized = _log10_transform(normalized)

    scaled = _zscore_per_metacell(log_normalized)
    yugene_scaled = _yugene(scaled)

    scaled_diff = _subtract_reference_median(
        scaled, reference_mask=reference_mask
    )
    yugene_diff = _subtract_reference_median(
        yugene_scaled, reference_mask=reference_mask
    )

    return {
        "scaled": scaled,
        "yugene": yugene_scaled,
        "scaled_diff": scaled_diff,
        "yugene_diff": yugene_diff,
    }


def build_human_entrez_map(
    gene_table: pd.DataFrame,
    *,
    id_column: str = "Ensembl",
    entrez_column: str = "Entrez",
) -> pd.Series:
    """Map human gene identifiers to human Entrez IDs.

    Rows without an Entrez ID are dropped and the source identifier is
    deduplicated keeping the first occurrence.

    Args:
      gene_table: tAge `Gene_table_human.csv` as a DataFrame.
      id_column: Source identifier column ("Ensembl" or "Gene.Symbol").
      entrez_column: Human Entrez column.

    Returns:
      Series indexed by source identifier with human Entrez (string) values.
    """
    table = gene_table.dropna(subset=[entrez_column]).copy()
    table = table.drop_duplicates(subset=[id_column], keep="first")
    entrez = table[entrez_column].astype("int64").astype(str)
    return pd.Series(entrez.to_numpy(), index=table[id_column].astype(str))


def build_mouse_ortholog_map(
    ortholog_table: pd.DataFrame,
    *,
    source_column: str = "Entrez.Human",
    mouse_column: str = "Entrez.Mouse",
) -> pd.Series:
    """Map human Entrez IDs to mouse Entrez orthologs.

    Rows without a mouse Entrez are dropped and the source Entrez is
    deduplicated keeping the first occurrence.

    Args:
      ortholog_table: tAge `Table_of_orthologs.csv` as a DataFrame.
      source_column: Source-species Entrez column ("Entrez.Human").
      mouse_column: Mouse Entrez column.

    Returns:
      Series indexed by human Entrez (string) with mouse Entrez (string) values.
    """
    table = ortholog_table.dropna(subset=[mouse_column]).copy()
    table = table.drop_duplicates(subset=[source_column], keep="first")
    source = table[source_column].astype("int64").astype(str)
    mouse = table[mouse_column].astype("int64").astype(str)
    return pd.Series(mouse.to_numpy(), index=source.to_numpy())


def map_counts_to_model_features(
    counts: pd.DataFrame,
    human_entrez_map: pd.Series,
    mouse_ortholog_map: pd.Series,
) -> pd.DataFrame:
    """Translate counts from human gene IDs into the mouse Entrez feature space.

    Mapping happens at the count stage in two steps:
      1. Collapse to human Entrez by summing many-to-one duplicates, then
         relabel to mouse
      2. Entrez dropping duplicate orthologs (first wins, no summing).

    Args:
      counts: Metacells x source-identifier counts (columns are gene IDs).
      human_entrez_map: Source identifier -> human Entrez.
      mouse_ortholog_map: Human Entrez -> mouse Entrez.

    Returns:
      Metacells x mouse-Entrez counts.
    """
    shared = counts.columns.intersection(human_entrez_map.index)
    human_counts = counts[shared].copy()
    human_counts.columns = human_entrez_map.loc[shared].to_numpy()
    human_counts = human_counts.T.groupby(level=0).sum().T

    mappable = human_counts.columns.intersection(mouse_ortholog_map.index)
    mouse_counts = human_counts[mappable].copy()
    mouse_ids = mouse_ortholog_map.loc[mappable].to_numpy()
    keep = ~pd.Index(mouse_ids).duplicated(keep="first")
    mouse_counts = mouse_counts.loc[:, keep]
    mouse_counts.columns = mouse_ids[keep]

    logger.info(
        "[clock.preprocess] mapped %d source genes -> %d mouse-Entrez features",
        counts.shape[1],
        mouse_counts.shape[1],
    )
    return mouse_counts


def _filter_low_count_genes(
    counts: pd.DataFrame,
    *,
    count_threshold: float = 10.0,
    percent_threshold: float = 20.0,
) -> pd.DataFrame:
    """Drop genes expressed below threshold in too few metacells.

    A gene is retained when at least `percent_threshold` percent of metacells
    have counts at or above `count_threshold`.

    Args:
      counts: Metacells x genes counts.
      count_threshold: Minimum per-metacell count to count as expressed.
      percent_threshold: Minimum percent of metacells that must be expressed.

    Returns:
      Counts restricted to the retained genes.
    """
    n_metacells = counts.shape[0]
    expressed_metacells = (counts >= count_threshold).sum(axis=0)
    required = n_metacells * (percent_threshold / 100.0)
    retained = expressed_metacells >= required
    return counts.loc[:, retained.to_numpy()]


def _rle_normalize(
    counts: pd.DataFrame, rle_scale: float = 1e7
) -> pd.DataFrame:
    """Apply edgeR RLE library normalization per metacell.

    Computes median-of-ratios size factors against the per-gene geometric mean
    (genes with any zero are excluded from the reference), scales factors to
    unit geometric mean, then divides each metacell by its effective library
    size and rescales by 1e7.

    Args:
      counts: Metacells x genes counts.
      rle_scale: Scaling factor used by RLE.

    Returns:
      RLE-normalized metacells x genes matrix.
    """
    values = counts.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_values = np.log(values)
    geometric_mean = np.exp(log_values.mean(axis=0))
    reference_genes = geometric_mean > 0
    if not reference_genes.any():
        raise ValueError(
            "RLE reference is empty: no gene is expressed across all metacells "
            "in this stratum, so median-of-ratios size factors are undefined."
        )

    ratios = values[:, reference_genes] / geometric_mean[reference_genes]
    raw_factors = np.median(ratios, axis=1)
    scaled_factors = raw_factors / np.exp(np.mean(np.log(raw_factors)))

    library_size = values.sum(axis=1)
    effective_library = library_size * scaled_factors
    normalized = values / effective_library[:, np.newaxis] * rle_scale

    return pd.DataFrame(normalized, index=counts.index, columns=counts.columns)


def _log10_transform(values: pd.DataFrame) -> pd.DataFrame:
    """Apply log10(x + 1) to an expression matrix."""
    return pd.DataFrame(
        np.log10(values.to_numpy(dtype=np.float64) + 1.0),
        index=values.index,
        columns=values.columns,
    )


def _zscore_per_metacell(values: pd.DataFrame) -> pd.DataFrame:
    """Z-score each metacell across genes (sample-wise, ddof=1), standardizes
    each sample using the sample standard deviation.

    Args:
      values: Metacells x genes matrix.

    Returns:
      Per-metacell standardized matrix.
    """
    matrix = values.to_numpy(dtype=np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, ddof=1, keepdims=True)
    scaled = (matrix - mean) / std
    return pd.DataFrame(scaled, index=values.index, columns=values.columns)


def _yugene_row(row: np.ndarray) -> np.ndarray:
    """Apply the YuGene transform to a single metacell vector."""
    shifted = row - np.min(row)
    total = shifted.sum()
    if total == 0:
        return np.ones_like(shifted)

    order = np.argsort(shifted, kind="stable")[::-1]
    sorted_values = shifted[order]
    cumulative_proportion = np.cumsum(sorted_values) / total

    run_start = np.empty(sorted_values.shape[0], dtype=bool)
    run_start[0] = True
    run_start[1:] = sorted_values[1:] != sorted_values[:-1]
    run_index = np.cumsum(run_start) - 1
    carried = cumulative_proportion[run_start][run_index]

    result = np.empty_like(shifted)
    result[order] = 1.0 - carried
    return result


def _yugene(values: pd.DataFrame) -> pd.DataFrame:
    """Apply YuGene cumulative-proportion normalization per metacell.

    Each metacell is shifted to non-negative range, ranked by descending value,
    and mapped to one minus its cumulative proportion, with ties carrying the
    leading run value forward.

    Args:
      values: Metacells x genes matrix (typically z-scored).

    Returns:
      YuGene-normalized metacells x genes matrix.
    """
    matrix = values.to_numpy(dtype=np.float64)
    transformed = np.vstack(
        [_yugene_row(matrix[index]) for index in range(matrix.shape[0])]
    )
    return pd.DataFrame(transformed, index=values.index, columns=values.columns)


def _subtract_reference_median(
    values: pd.DataFrame,
    *,
    reference_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Subtract a per-gene reference median to form relative deviations.

    Uses the median across reference metacells (all metacells when no mask is
    given). The result expresses each gene as a deviation from the stratum
    reference.

    Args:
      values: Metacells x genes matrix.
      reference_mask: Boolean mask selecting reference metacells. None uses all.

    Returns:
      Median-subtracted metacells x genes matrix.
    """
    matrix = values.to_numpy(dtype=np.float64)
    reference = matrix if reference_mask is None else matrix[reference_mask]
    gene_median = np.nanmedian(reference, axis=0)
    centered = matrix - gene_median[np.newaxis, :]
    return pd.DataFrame(centered, index=values.index, columns=values.columns)
