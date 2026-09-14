"""Donor-supported, descriptive mechanism and reference-score comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from nasp_compendium import GeneModules  # type: ignore[import]

from nasp_atlas.single_cell.associations import ObsSchema
from nasp_atlas.single_cell.gene_diagnostics import GeneDiagnosticResults
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.reference_sets import reference_gene_sets


def donor_score_units(
    obs: pd.DataFrame,
    scores: pd.DataFrame,
    columns: Sequence[str],
    *,
    schema: ObsSchema,
    minimum_cells: int,
) -> pd.DataFrame:
    """Aggregate score means by donor/context/assay, masking low support."""
    required = [schema.donor_key, schema.tissue_key, schema.cell_type_key]
    if missing := [key for key in required if key not in obs]:
        raise KeyError(f"Missing mechanism metadata: {missing}")
    if minimum_cells < 1:
        raise ValueError("minimum_cells must be positive")
    if not obs.index.isin(scores.index).all():
        raise ValueError("Mechanism scores are missing required cells")

    keys = [
        key
        for key in (
            schema.study_key,
            schema.donor_key,
            schema.tissue_key,
            schema.cell_type_key,
            schema.assay_key,
        )
        if key in obs
    ]
    if obs[keys].isna().any(axis=None):
        raise ValueError("Mechanism grouping identifiers must not be missing")

    frame = obs[keys].join(
        scores[list(columns)].replace([np.inf, -np.inf], np.nan),
        validate="one_to_one",
    )
    grouped = frame.groupby(keys, observed=True, dropna=False)
    means = (
        grouped[list(columns)]
        .mean()
        .where(grouped[list(columns)].count().ge(minimum_cells))
    )

    result = means.reset_index()
    donors = [
        key for key in (schema.study_key, schema.donor_key) if key in result
    ]
    result["independent_donor"] = pd.factorize(
        pd.MultiIndex.from_frame(result[donors]), sort=True
    )[0]

    return result


def centered_coupling(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    contexts: Sequence[str],
    minimum_donors: int,
) -> dict[str, object]:
    """Estimate centered Spearman coupling and donor-deletion sensitivity.

    Each deletion removes the donor from all contexts, reapplies support,
    and recenters. Ranges are descriptive, not confidence intervals. No
    p-values treat repeated donor contexts as independent observations.
    """
    if minimum_donors < 3:
        raise ValueError("minimum_donors must be at least 3")

    paired = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[left, right]
    )
    paired = paired.loc[
        paired.groupby(list(contexts), observed=True)
        .independent_donor.transform("nunique")
        .ge(minimum_donors)
    ]

    estimate = _centered_correlation(paired, left, right, contexts)

    deletions = []
    for donor in paired.independent_donor.unique():
        retained = paired.loc[paired.independent_donor.ne(donor)]
        retained = retained.loc[
            retained.groupby(list(contexts), observed=True)
            .independent_donor.transform("nunique")
            .ge(minimum_donors)
        ]
        deletions.append(_centered_correlation(retained, left, right, contexts))

    finite = np.asarray(deletions, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "spearman_r": estimate,
        "n_donors": paired.independent_donor.nunique(),
        "n_units": len(paired),
        "n_contexts": paired.groupby(list(contexts), observed=True).ngroups,
        "deletion_low": finite.min() if finite.size else np.nan,
        "deletion_high": finite.max() if finite.size else np.nan,
        "valid_deletions": finite.size,
        "status": "ok" if np.isfinite(estimate) else "insufficient_or_constant",
    }


def mechanism_tables(
    diagnostics: GeneDiagnosticResults,
    score_units: pd.DataFrame,
    *,
    schema: ObsSchema,
    scorer: ScorerName,
    minimum_cells: int = 10,
    minimum_donors: int = 3,
) -> dict[str, pd.DataFrame]:
    """Resolve regulator branches, IFN components, and the OAS/RNase L pilot."""
    contexts = [
        key
        for key in (schema.tissue_key, schema.cell_type_key, schema.assay_key)
        if key in score_units
    ]

    keys = [
        key
        for key in (schema.study_key, schema.donor_key, *contexts)
        if key in score_units
    ]

    donor = diagnostics.donor_expression
    regulators = donor.loc[
        donor.module_id.isin(["NASP_FEEDBACK", "NASP_RESTRICTION"])
    ].copy()
    regulators["gene_expression"] = regulators.mean_expression.where(
        regulators.n_finite_cells.ge(minimum_cells)
    )

    coupling = _regulator_output_coupling(
        regulators,
        score_units,
        keys=keys,
        contexts=contexts,
        scorer=scorer,
        minimum_donors=minimum_donors,
    )
    points = _regulator_output_points(
        regulators,
        score_units,
        coupling,
        keys=keys,
        contexts=contexts,
        scorer=scorer,
        minimum_donors=minimum_donors,
    )
    components = _mechanism_components(diagnostics.context_summary, contexts)

    return {
        "regulator_output_coupling": coupling,
        "regulator_output_points": points,
        "mechanism_components": components,
    }


def reference_comparisons(
    score_units: pd.DataFrame,
    module_ids: Sequence[str],
    *,
    schema: ObsSchema,
    scorer: ScorerName,
    minimum_donors: int = 3,
) -> pd.DataFrame:
    """Compare reference and curated scores, retaining signed gene overlap."""
    contexts = [
        key
        for key in (schema.tissue_key, schema.cell_type_key, schema.assay_key)
        if key in score_units
    ]
    suffix = "score" if scorer == "scanpy" else "auc"
    records = []
    for reference in reference_gene_sets():
        left = f"{reference.module_id}_{suffix}"
        if left not in score_units:
            continue
        reference_genes = set(reference.positive_genes)
        for identifier in module_ids:
            right = f"{identifier}_{suffix}"
            if right not in score_units:
                continue
            module = GeneModules.modules(identifier)
            positive = set(module.positive_genes)
            inverse = set(module.inverse_genes)
            result = centered_coupling(
                score_units,
                left=left,
                right=right,
                contexts=contexts,
                minimum_donors=minimum_donors,
            )

            records.append(
                {
                    "reference_module": reference.module_id,
                    "curated_module": identifier,
                    "n_shared_positive": len(reference_genes & positive),
                    "n_shared_inverse": len(reference_genes & inverse),
                    "gene_jaccard": len(reference_genes & (positive | inverse))
                    / len(reference_genes | positive | inverse),
                    **result,
                }
            )

    return pd.DataFrame(records)


def _regulator_output_coupling(
    regulators: pd.DataFrame,
    score_units: pd.DataFrame,
    *,
    keys: list[str],
    contexts: list[str],
    scorer: ScorerName,
    minimum_donors: int,
) -> pd.DataFrame:
    """Compare supported regulators against separate, non-circular outputs."""
    suffix = "score" if scorer == "scanpy" else "auc"
    records: list[dict[str, object]] = []
    for output in (
        "IFN_I_OUTPUT",
        "NFKB_CYTOKINE_OUTPUT",
        "ISR",
        "INFLAMMASOME",
        "SASP",
    ):
        column = f"{output}_{suffix}"
        if column not in score_units:
            continue
        members = set(GeneModules.genes(output))
        joined = regulators.merge(
            score_units[[*keys, column]], on=keys, validate="many_to_one"
        )
        for (module, gene), group in joined.groupby(
            ["module_id", "gene"], observed=True
        ):
            circular = gene in members
            result = centered_coupling(
                group,
                left="gene_expression",
                right=column,
                contexts=contexts,
                minimum_donors=minimum_donors,
            )
            if circular:
                result.update(
                    spearman_r=np.nan,
                    deletion_low=np.nan,
                    deletion_high=np.nan,
                    status="excluded_gene_in_output",
                )

            records.append(
                {
                    "regulator_module": module,
                    "gene": gene,
                    "output_module": output,
                    "gene_in_output": circular,
                    **result,
                }
            )

    return pd.DataFrame(records)


def _regulator_output_points(
    regulators: pd.DataFrame,
    score_units: pd.DataFrame,
    coupling: pd.DataFrame,
    *,
    keys: list[str],
    contexts: list[str],
    scorer: ScorerName,
    minimum_donors: int,
) -> pd.DataFrame:
    """Select and center donor-context points for the strongest two pairs."""
    suffix = "score" if scorer == "scanpy" else "auc"
    if coupling.empty:
        return pd.DataFrame()

    points = []

    strongest = (
        coupling.loc[coupling.status.eq("ok")]
        .assign(strength=lambda frame: frame.spearman_r.abs())
        .sort_values("strength", ascending=False, kind="stable")
        .drop_duplicates(["gene", "output_module"])
        .head(2)
    )

    for _, pair in strongest.iterrows():
        column = f"{pair.output_module}_{suffix}"
        paired = regulators.loc[
            regulators.gene.eq(pair.gene)
            & regulators.module_id.eq(pair.regulator_module)
        ].merge(score_units[[*keys, column]], on=keys, validate="one_to_one")
        paired = paired.dropna(subset=["gene_expression", column])
        paired = paired.loc[
            paired.groupby(contexts, observed=True)
            .independent_donor.transform("nunique")
            .ge(minimum_donors)
        ].copy()

        centered = paired[["gene_expression", column]] - paired.groupby(
            contexts, observed=True
        )[["gene_expression", column]].transform("mean")
        paired["gene_centered"] = centered.gene_expression
        paired["output_centered"] = centered[column]
        paired["output_module"] = pair.output_module
        paired["spearman_r"] = pair.spearman_r
        points.append(paired)

    return pd.concat(points, ignore_index=True) if points else pd.DataFrame()


def _mechanism_components(
    context_summary: pd.DataFrame, contexts: list[str]
) -> pd.DataFrame:
    """Retain ordered IFN and OAS/RNase L components and missing gene states."""
    expression = context_summary.copy()
    expression = expression.drop_duplicates([*contexts, "gene"])
    components = {
        "IFN_ligands": ("IFNA1", "IFNB1"),
        "IFNG_ligand": ("IFNG",),
        "IFN_I_receptor_context": ("IFNAR1", "IFNAR2"),
        "IFN_response": ("ISG15", "IFIT1", "MX1", "RSAD2"),
        "OAS_RNaseL": ("OAS1", "OAS2", "OAS3", "RNASEL"),
    }

    rows = []
    for component, genes in components.items():
        for order, gene in enumerate(genes):
            selected = expression.loc[expression.gene.eq(gene)].copy()
            if selected.empty:
                selected = pd.DataFrame(
                    [
                        {
                            "gene": gene,
                            "status": "unavailable_in_scored_modules",
                            "n_donors": 0,
                        }
                    ]
                )
            selected["component"] = component
            selected["component_order"] = order
            rows.append(selected)

    return pd.concat(rows, ignore_index=True)


def _centered_correlation(
    frame: pd.DataFrame,
    left: str,
    right: str,
    contexts: Sequence[str],
) -> float:
    """Correlate residuals after removing context means on complete rows."""
    if len(frame) < 3:
        return np.nan

    centered = frame[[left, right]] - frame.groupby(
        list(contexts), observed=True
    )[[left, right]].transform("mean")
    if centered.nunique().min() < 2:
        return np.nan
    return float(centered[left].corr(centered[right], method="spearman"))
