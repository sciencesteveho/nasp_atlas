"""Donor and assay resolved expression diagnostics for curated modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]
from anndata.typing import XDataType  # type: ignore[import]
from nasp_compendium.types import GeneModule  # type: ignore[import]

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.utils import expression_matrix


__all__ = ["GeneDiagnosticResults", "module_gene_diagnostics"]


@dataclass(frozen=True)
class GeneDiagnosticResults:
    """Expression/detection per donor and descriptive gene-score coupling."""

    donor_expression: pd.DataFrame
    context_summary: pd.DataFrame


def module_gene_diagnostics(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    modules: Sequence[GeneModule],
    *,
    schema: ObsSchema,
    scorer: ScorerName = "scanpy",
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    use_raw: bool = False,
    detection_threshold: float = 0.0,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
) -> GeneDiagnosticResults:
    """Inspect every declared gene, retaining absent and undetected states.

    Modules must use symbols and include unmeasured genes. One gene vector is
    materialized at a time. Repeated assays are kept separate; study qualifies
    donor identity. Correlations are descriptive across donors within one
    tissue, cell type and assay, with no independence claim across contexts.
    Gene-score correlations include the gene itself and are labelled circular.
    Mean expression per signed arm member is a driver-selection heuristic,
    not attribution of Scanpy or AUCell scores. Inputs are not modified.
    """
    if minimum_cells < 1 or minimum_donors < 2:
        raise ValueError("Require minimum_cells >= 1 and minimum_donors >= 2")
    if not np.isfinite(detection_threshold):
        raise ValueError("detection_threshold must be finite")
    if any(module.gene_id_output != "symbols" for module in modules):
        raise ValueError("Gene diagnostics require modules in symbol space")
    if not scores.index.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Cell identifiers must be unique")
    if not adata.obs_names.isin(scores.index).all():
        raise ValueError(
            "Scores are missing cells required by gene diagnostics"
        )

    names, matrix = expression_matrix(
        adata, expression_layer=expression_layer, use_raw=use_raw
    )
    if matrix is None:
        raise ValueError("Gene diagnostics require an expression matrix")

    var = adata.raw.var if use_raw and adata.raw is not None else adata.var
    obs = adata.obs
    if not isinstance(var, pd.DataFrame) or not isinstance(obs, pd.DataFrame):
        raise TypeError("Gene diagnostics require in-memory obs and var tables")

    positions = _gene_positions(var, names, gene_symbol_column)
    metadata, keys, contexts = _diagnostic_metadata(obs, schema)
    cache: dict[str, pd.DataFrame] = {}
    donor_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for module in modules:
        module_tables = _module_gene_donors(
            matrix,
            positions,
            metadata,
            scores,
            module,
            keys=keys,
            cache=cache,
            scorer=scorer,
            minimum_cells=minimum_cells,
            detection_threshold=detection_threshold,
        )
        donor_tables.extend(module_tables)
        for donor in module_tables:
            summaries.extend(
                _gene_context_record(
                    group,
                    contexts,
                    context,
                    minimum_donors,
                    minimum_cells,
                )
                for context, group in donor.groupby(
                    contexts, observed=True, dropna=False
                )
            )

    donor_columns = [
        *keys,
        "module_id",
        "gene",
        "arm",
        "status",
        "mean_expression",
        "fraction_detected",
        "n_finite_cells",
    ]
    summary_columns = [
        *contexts,
        "module_id",
        "gene",
        "arm",
        "status",
        "n_donors",
        "gene_score_spearman",
        "driver_magnitude",
    ]
    return GeneDiagnosticResults(
        donor_expression=pd.concat(donor_tables, ignore_index=True)
        if donor_tables
        else pd.DataFrame(columns=donor_columns),
        context_summary=pd.DataFrame(summaries)
        if summaries
        else pd.DataFrame(columns=summary_columns),
    )


def _gene_positions(
    var: pd.DataFrame, names: list[str], gene_symbol_column: str
) -> dict[str, list[int]]:
    """Index symbols without discarding ambiguous feature identifiers."""
    symbols = (
        var.loc[names, gene_symbol_column]
        .fillna(pd.Series(names, index=names))
        .astype(str)
    )
    positions: dict[str, list[int]] = {}
    for position, symbol in enumerate(symbols):
        positions.setdefault(symbol.strip(), []).append(position)

    return positions


def _diagnostic_metadata(
    obs: pd.DataFrame, schema: ObsSchema
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Resolve donor/context keys and study-qualified donor identities."""
    required = [schema.donor_key, schema.tissue_key, schema.cell_type_key]
    if missing := [key for key in required if key not in obs]:
        raise KeyError(f"Missing diagnostic metadata: {missing}")

    keys = [
        key
        for key in [schema.study_key, *required, schema.assay_key]
        if key in obs
    ]
    metadata = obs[keys].copy()
    if metadata[keys].isna().any(axis=None):
        raise ValueError(
            "Gene diagnostic grouping identifiers must not be missing"
        )

    contexts = [
        key
        for key in [schema.tissue_key, schema.cell_type_key, schema.assay_key]
        if key in metadata
    ]
    donor_keys = [
        key for key in (schema.study_key, schema.donor_key) if key in metadata
    ]
    metadata["independent_donor"] = pd.factorize(
        pd.MultiIndex.from_frame(metadata[donor_keys]), sort=True
    )[0]

    return metadata, keys, contexts


def _module_gene_donors(
    matrix: XDataType,
    positions: dict[str, list[int]],
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    module: GeneModule,
    *,
    keys: list[str],
    cache: dict[str, pd.DataFrame],
    scorer: ScorerName,
    minimum_cells: int,
    detection_threshold: float,
) -> list[pd.DataFrame]:
    """Join cached gene aggregates to one module's scores and signed arms."""
    score_key = module_score_name(module, scorer=scorer)
    if score_key not in scores:
        raise KeyError(f"Missing module score: {score_key}")

    score_metadata = metadata.assign(
        score=pd.to_numeric(
            scores.loc[metadata.index, score_key], errors="raise"
        ).replace([np.inf, -np.inf], np.nan)
    )
    score_means = (
        score_metadata.groupby(keys, observed=True, dropna=False)
        .agg(module_score=("score", "mean"), score_n_cells=("score", "count"))
        .reset_index()
    )

    donor_tables: list[pd.DataFrame] = []
    for arm, genes, direction in (
        ("positive", module.positive_genes, 1),
        ("inverse", module.inverse_genes, -1),
        ("context_dependent", module.context_dependent_genes, 0),
    ):
        for gene in genes:
            if gene not in cache:
                cache[gene] = _donor_gene_expression(
                    matrix,
                    positions.get(gene, []),
                    metadata,
                    keys,
                    detection_threshold,
                )

            donor = cache[gene].merge(
                score_means, on=keys, validate="one_to_one"
            )
            donor["module_id"] = module.module_id
            donor["gene"] = gene
            donor["arm"] = arm
            donor["scoring_direction"] = direction
            donor["signed_expression_per_arm_gene"] = (
                direction * donor.mean_expression / len(genes)
                if direction
                else np.nan
            )
            donor["supported"] = donor.n_finite_cells.ge(minimum_cells)
            donor_tables.append(donor)

    return donor_tables


def _donor_gene_expression(
    matrix: XDataType,
    matches: list[int],
    metadata: pd.DataFrame,
    keys: list[str],
    detection_threshold: float,
) -> pd.DataFrame:
    """Read and aggregate one gene, preserving absent and ambiguous states."""
    values = np.full(len(metadata), np.nan)
    state = "ambiguous_identifier" if matches else "absent"
    if len(matches) == 1:
        column = matrix[:, matches[0] : matches[0] + 1]
        if sp.issparse(column):
            values = column.toarray().ravel()  # type: ignore[union-attr]
        else:
            values = np.asarray(column).ravel()
        state = "present"

    return _aggregate_gene(metadata, keys, values, state, detection_threshold)


def _aggregate_gene(
    metadata: pd.DataFrame,
    keys: list[str],
    values: np.ndarray,
    state: str,
    threshold: float,
) -> pd.DataFrame:
    """Aggregate one gene without conflating missing expression and zero."""
    finite = np.isfinite(values)
    frame = metadata.assign(
        expression=np.where(finite, values, np.nan),
        detected=np.where(finite, values > threshold, np.nan),
    )
    result = (
        frame.groupby(keys, observed=True, dropna=False)
        .agg(
            independent_donor=("independent_donor", "first"),
            n_cells=("expression", "size"),
            n_finite_cells=("expression", "count"),
            mean_expression=("expression", "mean"),
            fraction_detected=("detected", "mean"),
        )
        .reset_index()
    )

    result["n_nonfinite_cells"] = result.n_cells - result.n_finite_cells
    result["status"] = state
    if state == "present":
        result.loc[result.n_finite_cells.eq(0), "status"] = "nonfinite"
        result.loc[result.fraction_detected.eq(0), "status"] = "undetected"

    return result


def _gene_context_record(
    group: pd.DataFrame,
    contexts: list[str],
    context: object,
    minimum_donors: int,
    minimum_cells: int,
) -> dict[str, object]:
    """Describe one module gene across donors within an assay and context."""
    key = context if isinstance(context, tuple) else (context,)
    supported = group.loc[
        group.supported & group.score_n_cells.ge(minimum_cells)
    ]
    paired = supported[["mean_expression", "module_score"]].dropna()
    n_donors = len(paired)

    status = "ok"
    correlation = np.nan
    if n_donors < minimum_donors:
        status = "insufficient_donors"
    elif paired.nunique().min() < 2:
        status = "constant_expression_or_score"
    else:
        correlation = paired.mean_expression.corr(
            paired.module_score, method="spearman"
        )

    return {
        **dict(zip(contexts, key, strict=True)),
        "module_id": group.module_id.iloc[0],
        "gene": group.gene.iloc[0],
        "arm": group.arm.iloc[0],
        "n_donors": n_donors,
        "n_total_donors": group.independent_donor.nunique(),
        "median_mean_expression": supported.mean_expression.median(),
        "median_fraction_detected": supported.fraction_detected.median(),
        "driver_magnitude": (
            supported.signed_expression_per_arm_gene.abs().median()
        ),
        "gene_score_spearman": correlation,
        "gene_in_score": group.arm.iloc[0] != "context_dependent",
        "gene_states": ";".join(sorted(group.status.unique())),
        "status": status,
    }
