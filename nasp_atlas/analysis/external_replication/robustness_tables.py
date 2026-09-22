"""Auditable display summaries for gene and paired-effect diagnostics."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pandas as pd


__all__ = [
    "RescoredVariantLabel",
    "gene_support_summary",
    "sensitivity_display_table",
    "variant_label",
]


@dataclasses.dataclass(frozen=True)
class RescoredVariantLabel:
    """Names for one rescored selection variant in prose and in a figure.

    Attributes:
      statement: Sentence subject used in the robustness prose.
      axis: Shorter label used on the sensitivity figure's axis.
    """

    statement: str
    axis: str


def variant_label(
    variant_id: object,
    *,
    names: Mapping[str, RescoredVariantLabel] | None = None,
) -> RescoredVariantLabel:
    """Name a rescored selection variant, rejecting unregistered ids.

    Rescored variants are registered sensitivities, so an id absent here means
    a stage published a variant this report cannot describe. That must fail
    rather than reach a reader as an unexplained identifier.
    """
    registered = (
        {
            "pooled_chemistry": RescoredVariantLabel(
                "Pooled-chemistry rescoring", "All chemistries; rescored"
            ),
            "as_released": RescoredVariantLabel(
                "As-released QC rescoring", "As-released cells; rescored"
            ),
            "nuclei": RescoredVariantLabel(
                "Nuclei rescoring (separate modality)", "Nuclei; rescored"
            ),
        }
        if names is None
        else names
    )
    key = str(variant_id)
    if key not in registered:
        raise ValueError(
            f"Unregistered rescored variant: {key}. Known variants are "
            f"{', '.join(sorted(registered))}."
        )
    return registered[key]


def gene_support_summary(paired_genes: pd.DataFrame) -> pd.DataFrame:
    """Summarize people equally, retaining genes with unavailable expression.

    Each person contributes once to a gene's mean even when the cohort pools
    assays; the contributing assays are listed for context, not as strata.
    """
    keys = ["module_id", "gene", "arm"]
    if paired_genes.duplicated([*keys, "cohort_id", "donor_id"]).any():
        raise ValueError("Gene support requires one paired record per person")
    summary = (
        paired_genes.groupby(keys, observed=True, dropna=False)
        .agg(
            mean_target=("mean_expression_target", "mean"),
            mean_reference=("mean_expression_reference", "mean"),
            detected_target=("fraction_detected_target", "mean"),
            detected_reference=("fraction_detected_reference", "mean"),
            mean_difference=("mean_expression_difference", "mean"),
            n_pairs=("mean_expression_difference", "count"),
            assay=("assay", lambda values: ";".join(sorted(values.unique()))),
            target_status=(
                "status_target",
                lambda values: ";".join(
                    sorted(values.fillna("missing_context").unique())
                ),
            ),
            reference_status=(
                "status_reference",
                lambda values: ";".join(
                    sorted(values.fillna("missing_context").unique())
                ),
            ),
        )
        .reset_index()
    )
    return summary


def sensitivity_display_table(
    fixed: pd.DataFrame,
    balanced: pd.DataFrame,
    gene_removals: pd.DataFrame,
    variants: pd.DataFrame,
) -> pd.DataFrame:
    """Collect all required checks without changing primary classifications.

    The donor-deletion bar is the range of point estimates, not a confidence
    interval. Other bars retain their saved pointwise donor intervals; an
    unavailable variant remains a row. Descriptive technical strata are saved
    separately, not mixed with inferential interval displays.
    """
    selected = fixed.loc[
        ~fixed.variant_kind.isin(["donor_deletion", "descriptive_stratum"])
    ].copy()
    selected["label"] = selected.variant_id.replace(
        {
            "baseline": "Primary",
            "minimum_cells:20": "At least 20 cells/context",
            "minimum_cells:50": "At least 50 cells/context",
            "exclude:primary_cohort_overlap": "Exclude shared participant",
            "exclude:any_wells_participant": "Exclude all Wells participants",
        }
    )
    selected["interval_kind"] = "pointwise_95_CI"
    deletions = []
    for (module, scorer), rows in fixed.loc[
        fixed.variant_kind.eq("donor_deletion")
    ].groupby(["module_id", "scorer"]):
        baseline = fixed.loc[
            fixed.module_id.eq(str(module))
            & fixed.scorer.eq(str(scorer))
            & fixed.variant_kind.eq("baseline")
        ].iloc[0]
        n_unavailable = int((~rows.eligible).sum())
        deletions.append(
            {
                "module_id": module,
                "scorer": scorer,
                "variant_id": "donor_deletion_range",
                "label": f"Donor deletion range ({n_unavailable} unavailable)",
                "estimate": baseline.estimate,
                "ci_lower": rows.estimate.min(),
                "ci_upper": rows.estimate.max(),
                "n_paired_donors": rows.n_paired_donors.min(),
                "n_unavailable_deletions": n_unavailable,
                "eligible": rows.eligible.all(),
                "eligibility_reason": "effect_range_not_CI",
                "interval_kind": "effect_range_not_CI",
            }
        )
    balanced = balanced.assign(
        label="Balanced cells; rescored", interval_kind="pointwise_95_CI"
    )
    removal = gene_removals.copy()
    removal["label"] = (
        "Remove "
        + removal.removed_genes.fillna("unavailable")
        + " ("
        + removal.selection_arm
        + ")"
    )
    removal["interval_kind"] = "pointwise_95_CI"
    variants = variants.assign(
        label=[
            variant_label(variant_id).axis for variant_id in variants.variant_id
        ],
        interval_kind="pointwise_95_CI",
    )
    display = pd.concat(
        [selected, pd.DataFrame(deletions), balanced, removal, variants],
        ignore_index=True,
    )
    display = display.loc[display.module_id.isin(fixed.module_id.unique())]
    return display.reset_index(drop=True)
