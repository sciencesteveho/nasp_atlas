"""Rescore derived signatures after dominant or overlapping gene removal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
from nasp_compendium.types import GeneModule  # type: ignore[import]

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.context_summary import summarize_module_contexts
from nasp_atlas.single_cell.gene_diagnostics import GeneDiagnosticResults
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.module_scoring import inverse_module_score_name
from nasp_atlas.single_cell.module_scoring import module_score_name
from nasp_atlas.single_cell.module_scoring import positive_module_score_name
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_module
from nasp_atlas.single_cell.utils import expression_matrix


__all__ = ["GeneSensitivityResults", "gene_removal_sensitivity"]


@dataclass(frozen=True)
class GeneSensitivityResults:
    """Derived signatures, context changes and overlap sensitivity."""

    variants: pd.DataFrame
    donor_scores: pd.DataFrame
    context_changes: pd.DataFrame
    overlap_coupling: pd.DataFrame
    cell_scores: pd.DataFrame = field(default_factory=pd.DataFrame)


def gene_removal_sensitivity(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    modules: Sequence[GeneModule],
    diagnostics: GeneDiagnosticResults,
    *,
    module_pairs: Sequence[tuple[str, str]] = (),
    schema: ObsSchema,
    scorer: ScorerName = "scanpy",
    gene_symbol_column: str = "feature_name",
    expression_layer: str | None = None,
    use_raw: bool = False,
    dominant_genes: int = 1,
    dominant_per_arm: bool = False,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
    random_state: int = 42,
    aucell_chunk_size: int = 1000,
    aucell_num_workers: int = 1,
) -> GeneSensitivityResults:
    """Remove selected genes and compare actual scorer outputs with baseline.

    Dominant candidates have the largest maximum context-level median absolute
    expression per arm member, among donor-supported diagnostic contexts.
    Each candidate is removed separately. With dominant_per_arm=True, select
    candidates separately within each scored arm; the default retains the
    module-wide selection. Returned cell_scores retain scored components so
    callers can diagnose degenerate arms in changed measurements.
    Shared signed genes are removed
    from both members of each requested pair. Context-dependent genes are never
    scored. Losing a previously present score arm is explicitly unscorable.

    Scoring uses all input cells and the full expression background. Scanpy
    reselects controls with the same seed; signed arms are restandardized over
    the same cell universe. AUCell batches every variant in one ranking pass.
    Inputs and curated module definitions are not modified. The caller must
    supply baseline scores generated with matching definitions, source and seed.
    Correlations and rank changes are descriptive, with no p-values.
    """
    if dominant_genes < 1 or minimum_cells < 1 or minimum_donors < 2:
        raise ValueError(
            "Invalid dominant-gene or donor/cell support threshold"
        )
    if any(module.gene_id_output != "symbols" for module in modules):
        raise ValueError("Sensitivity modules must use symbols")
    if (
        not scores.index.is_unique
        or not adata.obs_names.is_unique
        or len(scores) != adata.n_obs
        or not adata.obs_names.isin(scores.index).all()
    ):
        raise ValueError(
            "Sensitivity scores must identify exactly the input cells"
        )

    scores = scores.reindex(adata.obs_names)
    names, matrix = expression_matrix(
        adata, expression_layer=expression_layer, use_raw=use_raw
    )

    var = adata.raw.var if use_raw and adata.raw is not None else adata.var
    obs = adata.obs
    if not isinstance(var, pd.DataFrame) or not isinstance(obs, pd.DataFrame):
        raise TypeError(
            "Gene sensitivity requires in-memory obs and var tables"
        )

    symbols = (
        var.loc[names, gene_symbol_column]
        .fillna(pd.Series(names, index=names))
        .astype(str)
        .str.strip()
    )
    unique = ~symbols.duplicated(keep=False)
    symbol_to_name = dict(
        zip(symbols[unique], np.asarray(names)[unique], strict=True)
    )

    variants, definitions = _removal_variants(
        modules,
        diagnostics.context_summary,
        module_pairs,
        set(symbol_to_name),
        set(symbols[~unique]),
        dominant_genes,
        minimum_donors,
        dominant_per_arm,
    )

    # Share expression storage; only variant score columns belong to this view.
    working = ad.AnnData(
        X=matrix, obs=pd.DataFrame(index=adata.obs_names), var=var
    )
    variant_scores = _score_variants(
        working,
        definitions,
        symbol_to_name,
        scorer=scorer,
        gene_symbol_column=gene_symbol_column,
        random_state=random_state,
        aucell_chunk_size=aucell_chunk_size,
        aucell_num_workers=aucell_num_workers,
    )

    return _summarize_variants(
        variants,
        variant_scores,
        scores,
        modules,
        obs,
        schema=schema,
        scorer=scorer,
        minimum_cells=minimum_cells,
        minimum_donors=minimum_donors,
    )


def _score_variants(
    working: ad.AnnData,
    definitions: Sequence[GeneModule],
    symbol_to_name: dict[str, str],
    *,
    scorer: ScorerName,
    gene_symbol_column: str,
    random_state: int,
    aucell_chunk_size: int,
    aucell_num_workers: int,
) -> pd.DataFrame:
    """Score derived signatures on the shared full expression background."""
    variant_scores = pd.DataFrame(index=working.obs_names)
    if definitions and scorer == "aucell":
        _, variant_scores, _ = score_aucell_modules(
            working,
            [module.module_id for module in definitions],
            gene_modules=definitions,
            gene_symbol_column=gene_symbol_column,
            expression_layer=None,
            random_state=random_state,
            chunk_size=aucell_chunk_size,
            num_workers=aucell_num_workers,
        )
    elif scorer == "scanpy":
        if not isinstance(working.obs, pd.DataFrame):
            raise TypeError("Gene sensitivity requires in-memory scores")
        for definition in definitions:
            resolved = replace(
                definition,
                gene_id_output="var_names",
                positive_genes=tuple(
                    symbol_to_name[g] for g in definition.positive_genes
                ),
                inverse_genes=tuple(
                    symbol_to_name[g] for g in definition.inverse_genes
                ),
            )
            column = score_scanpy_module(
                working,
                resolved,
                expression_layer=None,
                random_state=random_state,
            )
            columns = [column]
            if definition.positive_genes:
                columns.append(
                    positive_module_score_name(definition, scorer=scorer)
                )
            inverse = inverse_module_score_name(definition, scorer=scorer)
            if inverse is not None:
                columns.append(inverse)
            columns = list(dict.fromkeys(columns))
            variant_scores[columns] = working.obs[columns].to_numpy()
    elif scorer != "aucell":
        raise ValueError(f"Unsupported scorer: {scorer}")

    return variant_scores


def _summarize_variants(
    variants: pd.DataFrame,
    variant_scores: pd.DataFrame,
    scores: pd.DataFrame,
    modules: Sequence[GeneModule],
    obs: pd.DataFrame,
    *,
    schema: ObsSchema,
    scorer: ScorerName,
    minimum_cells: int,
    minimum_donors: int,
) -> GeneSensitivityResults:
    """Compare variant donor scores, context ranks and overlap coupling."""
    donor_keys = [
        key for key in (schema.study_key, schema.donor_key) if key in obs
    ]
    contexts = [schema.tissue_key, schema.cell_type_key]
    keys = [*donor_keys, *contexts]
    metadata = obs[keys].copy()
    if metadata.isna().any(axis=None):
        raise ValueError("Sensitivity grouping identifiers must not be missing")
    metadata["independent_donor"] = pd.factorize(
        pd.MultiIndex.from_frame(metadata[donor_keys]), sort=True
    )[0]

    baseline = {
        module.module_id: _donor_scores(
            metadata,
            scores[module_score_name(module, scorer=scorer)],
            keys,
            minimum_cells,
        )
        for module in modules
    }

    changed: dict[str, pd.DataFrame] = {}
    context_tables: list[pd.DataFrame] = []
    for record in variants.loc[variants.status.eq("ok")].to_dict("records"):
        variant_id = str(record["variant_id"])
        original_id = str(record["module_id"])
        suffix = "score" if scorer == "scanpy" else "auc"
        donor = _donor_scores(
            metadata,
            variant_scores[f"{variant_id}_{suffix}"],
            keys,
            minimum_cells,
        )
        changed[variant_id] = donor
        comparison = _rank_changes(
            baseline[original_id], donor, contexts, minimum_donors
        )
        comparison["variant_id"] = variant_id
        comparison["module_id"] = original_id
        context_tables.append(comparison)

    donor_tables = [
        frame.assign(variant_id=variant_id)
        for variant_id, frame in changed.items()
    ]

    return GeneSensitivityResults(
        variants=variants,
        donor_scores=pd.concat(donor_tables, ignore_index=True)
        if donor_tables
        else pd.DataFrame(
            columns=[*keys, "variant_id", "feature_value", "n_cells"]
        ),
        context_changes=pd.concat(context_tables, ignore_index=True)
        if context_tables
        else pd.DataFrame(
            columns=[*contexts, "variant_id", "module_id", "rank_change"]
        ),
        overlap_coupling=_overlap_coupling(
            variants, baseline, changed, contexts, minimum_donors
        ),
        cell_scores=variant_scores.reindex(obs.index),
    )


def _removal_requests(
    modules: Sequence[GeneModule],
    diagnostics: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    dominant_genes: int,
    minimum_donors: int,
    dominant_per_arm: bool,
) -> list[tuple[str, str, str, str, set[str]]]:
    """Select supported drivers and paired shared-gene removal requests."""
    lookup = {module.module_id: module for module in modules}
    requests: list[tuple[str, str, str, str, set[str]]] = []
    for module in modules:
        selected = diagnostics.loc[
            diagnostics.module_id.eq(module.module_id)
            & diagnostics.n_donors.ge(minimum_donors)
            & diagnostics.arm.ne("context_dependent")
        ]
        arms = (
            [
                arm
                for arm, genes in (
                    ("positive", module.positive_genes),
                    ("inverse", module.inverse_genes),
                )
                if genes
            ]
            if dominant_per_arm
            else ["all_scored"]
        )
        for arm in arms:
            candidates = (
                selected.loc[selected.arm.eq(arm)]
                if dominant_per_arm
                else selected
            )
            drivers = (
                candidates.groupby("gene")
                .driver_magnitude.max()
                .dropna()
                .sort_values(ascending=False, kind="stable")
                .head(dominant_genes)
            )
            requests.extend(
                (module.module_id, "dominant_gene", "", arm, {gene})
                for gene in drivers.index
            )
            if drivers.empty:
                requests.append(
                    (module.module_id, "dominant_gene", "", arm, set())
                )

    for left, right in dict.fromkeys(tuple(sorted(pair)) for pair in pairs):
        if left not in lookup or right not in lookup or left == right:
            raise ValueError(
                f"Invalid sensitivity module pair: {left}, {right}"
            )
        first, second = lookup[left], lookup[right]
        shared = set(first.positive_genes + first.inverse_genes).intersection(
            second.positive_genes + second.inverse_genes
        )
        requests.extend(
            [
                (left, "shared_genes", right, "shared_scored", shared),
                (right, "shared_genes", left, "shared_scored", shared),
            ]
        )

    return requests


def _removal_variants(
    modules: Sequence[GeneModule],
    diagnostics: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    available: set[str],
    ambiguous: set[str],
    dominant_genes: int,
    minimum_donors: int,
    dominant_per_arm: bool,
) -> tuple[pd.DataFrame, list[GeneModule]]:
    """Define deletions explicitly, retaining failed and empty-arm variants."""
    lookup = {module.module_id: module for module in modules}
    requests = _removal_requests(
        modules,
        diagnostics,
        pairs,
        dominant_genes,
        minimum_donors,
        dominant_per_arm,
    )

    records: list[dict[str, object]] = []
    definitions: list[GeneModule] = []
    for index, (module_id, mode, partner, arm, removed) in enumerate(requests):
        module = lookup[module_id]
        positive = tuple(
            g for g in module.positive_genes if g in available - removed
        )
        inverse = tuple(
            g for g in module.inverse_genes if g in available - removed
        )

        status = "ok"
        if not removed:
            status = "no_shared_genes" if partner else "no_supported_driver"
        elif ambiguous.intersection(
            module.positive_genes + module.inverse_genes
        ):
            status = "ambiguous_identifier"
        elif (module.positive_genes and not positive) or (
            module.inverse_genes and not inverse
        ):
            status = "unscorable_arm"

        variant_id = f"sensitivity_{index}"
        records.append(
            {
                "variant_id": variant_id,
                "module_id": module_id,
                "mode": mode,
                "partner_module": partner,
                "selection_arm": arm,
                "removed_genes": ";".join(sorted(removed)),
                "ambiguous_genes": ";".join(
                    sorted(
                        ambiguous.intersection(
                            module.positive_genes + module.inverse_genes
                        )
                    )
                ),
                "positive_genes": ";".join(positive),
                "inverse_genes": ";".join(inverse),
                "status": status,
            }
        )
        if status == "ok":
            definitions.append(
                replace(
                    module,
                    module_id=variant_id,
                    positive_genes=positive,
                    inverse_genes=inverse,
                    context_dependent_genes=(),
                )
            )

    return pd.DataFrame(
        records,
        columns=[
            "variant_id",
            "module_id",
            "mode",
            "partner_module",
            "selection_arm",
            "removed_genes",
            "ambiguous_genes",
            "positive_genes",
            "inverse_genes",
            "status",
        ],
    ), definitions


def _donor_scores(
    metadata: pd.DataFrame,
    values: pd.Series,
    keys: list[str],
    minimum_cells: int,
) -> pd.DataFrame:
    """Average scores by donor and context, preserving unsupported rows."""
    frame = metadata.assign(value=values.replace([np.inf, -np.inf], np.nan))
    result = (
        frame.groupby(keys, observed=True, dropna=False)
        .agg(
            feature_value=("value", "mean"),
            n_cells=("value", "count"),
            independent_donor=("independent_donor", "first"),
        )
        .reset_index()
    )
    result.loc[result.n_cells.lt(minimum_cells), "feature_value"] = np.nan
    return result


def _rank_changes(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    contexts: list[str],
    minimum_donors: int,
) -> pd.DataFrame:
    """Compare donor-median ranks across a fixed observed context universe."""
    tables = [
        summarize_module_contexts(
            frame.assign(feature_label="module"),
            context_columns=contexts,
            unit_columns=["independent_donor", *contexts],
            donor_column="independent_donor",
            feature_type_column=None,
            min_units=minimum_donors,
            min_donors=minimum_donors,
        )
        for frame in (baseline, variant)
    ]

    result = tables[0].merge(
        tables[1],
        on=["feature_label", *contexts],
        suffixes=("_baseline", "_removed"),
        validate="one_to_one",
    )
    result["rank_change"] = (
        result.context_rank_removed - result.context_rank_baseline
    )
    result["median_change"] = result.median_removed - result.median_baseline
    result["status"] = np.where(
        result.eligible_baseline & result.eligible_removed,
        "ok",
        "insufficient_donors",
    )

    return result.drop(columns="feature_label")


def _overlap_coupling(
    variants: pd.DataFrame,
    baseline: dict[str, pd.DataFrame],
    changed: dict[str, pd.DataFrame],
    contexts: list[str],
    minimum_donors: int,
) -> pd.DataFrame:
    """Compare correlations on the same complete donor-context rows."""
    records: list[dict[str, object]] = []
    shared = variants.loc[variants["mode"].eq("shared_genes")]
    for row in shared.to_dict("records"):
        left, right = str(row["module_id"]), str(row["partner_module"])
        if left >= right:
            continue
        partner = shared.loc[
            shared.module_id.eq(right) & shared.partner_module.eq(left)
        ].iloc[0]
        for mode in ("across_contexts", "within_context_centered"):
            record = {
                "module_a": left,
                "module_b": right,
                "analysis": mode,
                "baseline_spearman": np.nan,
                "removed_spearman": np.nan,
                "n_donors": 0,
                "n_units": 0,
                "status": row["status"],
            }
            if row["status"] != "ok" or partner.status != "ok":
                record["status"] = (
                    row["status"] if row["status"] != "ok" else partner.status
                )
                records.append(record)
                continue

            keys = ["independent_donor", *contexts]
            frames = [
                baseline[left],
                baseline[right],
                changed[str(row["variant_id"])],
                changed[str(partner.variant_id)],
            ]
            wide = pd.concat(
                [
                    frame.set_index(keys).feature_value.rename(str(index))
                    for index, frame in enumerate(frames)
                ],
                axis=1,
            ).dropna()

            record.update(
                n_donors=wide.index.get_level_values(0).nunique(),
                n_units=len(wide),
            )
            if mode == "within_context_centered":
                wide = wide - wide.groupby(level=contexts).transform("mean")
            if record["n_donors"] < minimum_donors:
                record["status"] = "insufficient_donors"
            elif wide.nunique().min() < 2:
                record["status"] = "constant_score"
            else:
                record.update(
                    status="ok",
                    baseline_spearman=wide["0"].corr(
                        wide["1"], method="spearman"
                    ),
                    removed_spearman=wide["2"].corr(
                        wide["3"], method="spearman"
                    ),
                )
            records.append(record)

    return pd.DataFrame(
        records,
        columns=[
            "module_a",
            "module_b",
            "analysis",
            "baseline_spearman",
            "removed_spearman",
            "n_donors",
            "n_units",
            "status",
        ],
    )
