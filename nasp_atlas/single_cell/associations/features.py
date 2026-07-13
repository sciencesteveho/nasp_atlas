"""Feature resolution and cell-level association frames."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]
from nasp_compendium import GeneModules  # type: ignore[import]
from nasp_compendium.types import GeneModule  # type: ignore[import]

from nasp_atlas.single_cell.associations.core import FeatureSpec
from nasp_atlas.single_cell.associations.core import ObsSchema
from nasp_atlas.single_cell.associations.core import metadata_columns
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.utils import expression_matrix


__all__ = [
    "build_cell_feature_frame",
    "resolve_feature_specs",
]


def resolve_feature_specs(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    *,
    score_keys: Sequence[str] | None = None,
    module_ids: Sequence[str] | None = None,
    gene_symbols: Sequence[str] | None = None,
    sensor_group: str | None = None,
    scored_module_ids: Sequence[str] | None = None,
    scorer: ScorerName = "scanpy",
    gene_symbol_column: str = "feature_name",
    use_raw: bool = False,
) -> tuple[list[FeatureSpec], list[dict[str, str]]]:
    """Resolve requested features to concrete score columns and var names.

    Module-score features are validated against the score-table columns (never
    recomputed); gene-expression features are validated against `adata.var`.
    Anything that cannot be resolved is returned in a skipped list with a reason
    rather than silently dropped.

    Args:
    adata: AnnData providing `var` for gene resolution.
    scores: Score table whose columns hold module score values.
    score_keys: Explicit score-table column names to use as features.
    module_ids: Module ids to resolve to score-column names via
      `module_score_name`.
    gene_symbols: Explicit gene symbols to resolve to var names.
    sensor_group: Optional sensor group resolved through
      `GeneModules.sensors`.
    scored_module_ids: Module ids whose score columns should be included when
      present, used to associate every scored module without listing each.
    scorer: Scorer whose naming convention maps module ids to columns.
    gene_symbol_column: Source var column holding gene symbols.
    use_raw: Resolve gene-expression features against `adata.raw.var`.

    Returns:
    A tuple of resolved feature specs and a list of skip records, each a
    mapping with `feature_type`, `requested` and `skip_reason`.
    """
    specs: list[FeatureSpec] = []
    skipped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for column in score_keys or []:
        _add_score_spec(
            scores,
            specs=specs,
            skipped=skipped,
            seen=seen,
            column=column,
            label=column,
        )

    requested_module_ids = list(module_ids or []) + list(
        scored_module_ids or []
    )
    for module_id in requested_module_ids:
        module: GeneModule = GeneModules.modules(module_id)
        _add_score_spec(
            scores,
            specs=specs,
            skipped=skipped,
            seen=seen,
            column=module_score_name(module, scorer=scorer),
            label=module.module_id,
        )

    if symbols := _requested_gene_symbols(
        adata,
        gene_symbols=gene_symbols,
        sensor_group=sensor_group,
        gene_symbol_column=gene_symbol_column,
        use_raw=use_raw,
    ):
        _add_gene_specs(
            adata,
            symbols,
            specs=specs,
            skipped=skipped,
            seen=seen,
            gene_symbol_column=gene_symbol_column,
            use_raw=use_raw,
        )

    return specs, skipped


def _add_score_spec(
    scores: pd.DataFrame,
    *,
    specs: list[FeatureSpec],
    skipped: list[dict[str, str]],
    seen: set[tuple[str, str]],
    column: str,
    label: str,
) -> None:
    """Append one unique score feature or its explicit skip record."""
    if column not in scores.columns:
        skipped.append(
            {
                "feature_type": "module_score",
                "requested": label,
                "skip_reason": f"score column absent: {column}",
            }
        )
        return

    key = ("module_score", column)
    if key in seen:
        return

    seen.add(key)
    specs.append(
        FeatureSpec(
            feature_type="module_score",
            feature_id=column,
            feature_label=label,
            source="scores",
        )
    )


def _requested_gene_symbols(
    adata: ad.AnnData,
    *,
    gene_symbols: Sequence[str] | None,
    sensor_group: str | None,
    gene_symbol_column: str,
    use_raw: bool,
) -> list[str]:
    """Combine explicit genes and compendium sensors in one identifier space."""
    symbols = list(gene_symbols or [])
    if sensor_group is None:
        return symbols

    resolution_adata = adata
    if use_raw:
        raw = adata.raw
        if raw is None:
            raise ValueError("use_raw=True requires adata.raw to be set")
        resolution_adata = ad.AnnData(shape=raw.shape, var=raw.var)

    symbols.extend(
        GeneModules.sensors(
            sensor_group,
            adata=resolution_adata,  # type: ignore[arg-type]
            gene_symbol_column=gene_symbol_column,
            output="symbols",
        )
    )
    return symbols


def _add_gene_specs(
    adata: ad.AnnData,
    symbols: Sequence[str],
    *,
    specs: list[FeatureSpec],
    skipped: list[dict[str, str]],
    seen: set[tuple[str, str]],
    gene_symbol_column: str,
    use_raw: bool,
) -> None:
    """Resolve unique gene features and retain every unresolved request."""
    raw = adata.raw
    if use_raw and raw is None:
        raise ValueError("use_raw=True requires adata.raw to be set")

    source_var = raw.var if use_raw and raw is not None else adata.var
    source_names = (
        raw.var_names if use_raw and raw is not None else adata.var_names
    )
    var_names = pd.Index(source_names).astype(str)
    symbol_to_var = {name: name for name in var_names}
    if gene_symbol_column in source_var.columns:
        for var_name, symbol in zip(
            var_names,
            source_var[gene_symbol_column].astype(str),
            strict=True,
        ):
            if symbol and symbol != "nan":
                symbol_to_var.setdefault(symbol, var_name)

    for symbol in symbols:
        var_name = symbol_to_var.get(symbol)
        if var_name is None:
            skipped.append(
                {
                    "feature_type": "gene_expression",
                    "requested": symbol,
                    "skip_reason": "gene absent from adata.var",
                }
            )
            continue

        key = ("gene_expression", var_name)
        if key in seen:
            continue

        seen.add(key)
        specs.append(
            FeatureSpec(
                feature_type="gene_expression",
                feature_id=var_name,
                feature_label=symbol,
                source="expression",
            )
        )


def _dense_expression(
    adata: ad.AnnData,
    var_names: Sequence[str],
    *,
    expression_layer: str | None,
    use_raw: bool = False,
) -> pd.DataFrame:
    """Return a dense cells-by-genes expression frame for present var names.

    Args:
    adata: AnnData whose matrix supplies expression.
    var_names: Var names to extract; names absent from `adata` are omitted.
    expression_layer: Layer to read, or None for `adata.X`/raw when requested.
    use_raw: Read from `adata.raw.X` instead of `adata.X` or a layer.

    Returns:
    Frame indexed by `adata.obs_names` with one column per present var name.
    Empty (index-only) when no requested var name is present.
    """
    obs_index = pd.Index(adata.obs_names).astype(str)
    present, matrix = expression_matrix(
        adata,
        var_names,
        expression_layer=expression_layer,
        use_raw=use_raw,
    )
    if not present:
        return pd.DataFrame(index=obs_index)
    assert matrix is not None
    if sp.issparse(matrix):
        matrix = matrix.toarray()  # type: ignore[union-attr]
    return pd.DataFrame(
        np.asarray(matrix, dtype=float),
        index=obs_index,
        columns=present,
    )


def build_cell_feature_frame(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    feature_specs: Sequence[FeatureSpec],
    *,
    schema: ObsSchema,
    expression_layer: str | None = "log1p",
    use_raw: bool = False,
) -> pd.DataFrame:
    """Build a tidy long cell-level feature frame with metadata.

    Module-score features are taken from `scores` and gene-expression features
    from `adata` (matrix), joined by obs name. The result is long: one row per
    (cell, feature). No AnnData copy is made beyond the requested gene columns.

    Args:
    adata: AnnData providing obs metadata and gene expression.
    scores: Score table indexed by obs name with module score columns.
    feature_specs: Features to include, from `resolve_feature_specs`.
    schema: Column-name schema for metadata columns.
    expression_layer: Layer to read gene expression from, or None for
      `adata.X`/raw when requested.
    use_raw: Read gene expression from `adata.raw.X`.

    Returns:
    Long frame with columns `obs_name`, `feature_type`, `feature_id`,
    `feature_label`, `feature_value` and the available metadata columns.
    """
    # AnnData exposes a DataFrame obs for the in-memory objects accepted here.
    obs: pd.DataFrame = adata.obs  # type: ignore[assignment]
    obs_index = pd.Index(adata.obs_names).astype(str)
    metadata_keys = [key for key in metadata_columns(schema) if key in obs]
    metadata = obs.loc[:, metadata_keys].copy()
    metadata.index = obs_index

    score_specs = [s for s in feature_specs if s.source == "scores"]
    gene_specs = [s for s in feature_specs if s.source == "expression"]

    aligned_scores = scores.reindex(obs_index)
    expression = _dense_expression(
        adata,
        [s.feature_id for s in gene_specs],
        expression_layer=expression_layer,
        use_raw=use_raw,
    )

    frames: list[pd.DataFrame] = []
    for spec in score_specs:
        score_values = aligned_scores.get(spec.feature_id)
        if score_values is None:
            raise KeyError(f"score column not found: {spec.feature_id}")
        values = pd.to_numeric(score_values, errors="coerce")
        frames.append(_long_feature_block(spec, values, metadata, obs_index))
    frames.extend(
        _long_feature_block(
            spec, expression[spec.feature_id], metadata, obs_index
        )
        for spec in gene_specs
        if spec.feature_id in expression.columns
    )
    if not frames:
        return pd.DataFrame(
            columns=[
                "obs_name",
                "feature_type",
                "feature_id",
                "feature_label",
                "feature_value",
                *metadata_keys,
            ]
        )
    return pd.concat(frames, axis="index", ignore_index=True)


def _long_feature_block(
    spec: FeatureSpec,
    values: pd.Series,
    metadata: pd.DataFrame,
    obs_index: pd.Index,
) -> pd.DataFrame:
    """Assemble one feature's long-format block with metadata.

    Args:
    spec: Feature specification providing type, id and label.
    values: Per-cell feature values indexed by obs name.
    metadata: Per-cell metadata indexed by obs name.
    obs_index: Canonical obs-name index for alignment.

    Returns:
    Long frame for the single feature with metadata columns attached.
    """
    block = metadata.copy()
    block.insert(0, "feature_value", pd.to_numeric(values, errors="coerce"))
    block.insert(0, "feature_label", spec.feature_label)
    block.insert(0, "feature_id", spec.feature_id)
    block.insert(0, "feature_type", spec.feature_type)
    block.insert(0, "obs_name", np.asarray(obs_index))
    return block.reset_index(drop=True)
