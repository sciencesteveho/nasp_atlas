"""Readable reports from saved estimates, without scoring or inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nasp_atlas.analysis.external_replication.robustness_tables import (
    variant_label,
)


__all__ = [
    "read_report_limitations",
    "render_extension_sensitivity",
    "render_replication_report",
    "render_robustness_section",
]


def read_report_limitations(
    registered: object, reviewed_path: Path | None = None
) -> list[str]:
    """Return the limitation statements a report should print.

    The frozen registration carries one global list. A cohort whose report
    needs different reviewed text supplies it as a YAML list, which replaces
    the registered list for that report only.
    """
    limitations = (
        registered
        if reviewed_path is None
        else yaml.safe_load(reviewed_path.read_text())
    )
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(
            isinstance(text, str) and text.strip() for text in limitations
        )
    ):
        raise ValueError(
            "Report limitations must be a nonempty list of statements"
            + ("" if reviewed_path is None else f": {reviewed_path}")
        )
    return [" ".join(text.split()) for text in limitations]


def render_replication_report(
    results: pd.DataFrame,
    discovery: pd.DataFrame,
    *,
    title: str,
    registration_identity: str,
    report_scope: str,
    limitations: Sequence[str],
    figures: Mapping[str, str] | None = None,
) -> str:
    """Render every classified question and its separate method estimates.

    `report_scope` states completed and unfinished analyses. A preliminary
    primary report must not imply robustness or extensions are complete.
    Figure paths are relative to the eventual report. Performs no file I/O,
    scoring, fitting, multiplicity adjustment or eligibility selection.
    """
    primary_method = results.scorer.eq(results.primary_scorer)
    primary = results.loc[primary_method & results.analysis_role.eq("primary")]
    counts = primary.status.value_counts()
    states = ("supported", "opposite_direction", "inconclusive", "unavailable")
    summary = "; ".join(
        f"{int(counts.get(state, 0))} {state.replace('_', ' ')}"
        for state in states
    )
    sections = [
        f"# {title}",
        f"**Primary registered questions: {summary}.**",
        report_scope,
        "Effects are equal-person means of paired target-minus-reference "
        "differences. Cells contribute to each person's context mean; they "
        "are not independent replicates. Confidence intervals are pointwise "
        "95% intervals. Holm correction includes every registered Scanpy "
        "question within its declared family, including unavailable members.",
        "## Primary results",
        _result_table(primary),
    ]
    dependent = primary.status.eq("supported") & primary.method_concordance.eq(
        "opposite_direction"
    )
    if dependent.any():
        sections.append(
            f"{int(dependent.sum())} supported primary result(s) reverse "
            "direction under AUCell and must be described as method-dependent."
        )
    sensitivity = results.loc[
        ~primary_method & results.analysis_role.eq("primary")
    ]
    sections.extend(
        [
            "## AUCell sensitivity",
            _result_table(sensitivity),
            "AUCell uses the same participants as a method sensitivity. It is "
            "not an independent biological replication. Its p-values are "
            "unadjusted and do not determine the primary decision.",
            "## Harmonized TS discovery reference",
            _discovery_table(discovery),
            "TS and external scores depend on the fitted scoring populations. "
            "Compare direction and uncertainty on separate cohort/scorer "
            "scales. The corrected reference retains the frozen panel and "
            "reconciles identifiers; historical evidence is preserved.",
        ]
    )
    extension = results.loc[
        primary_method & results.analysis_role.eq("extension")
    ]
    if not extension.empty:
        sections.extend(
            [
                "## Registered extension",
                _result_table(extension),
                "New association questions use a separate Holm family. "
                "Different p-values do not establish population interactions.",
            ]
        )
        extension_sensitivity = render_extension_sensitivity(results)
        if extension_sensitivity:
            sections.append(extension_sensitivity.rstrip())
    if figures:
        sections.extend(
            [
                "## Figures",
                *[f"![{label}]({path})" for label, path in figures.items()],
            ]
        )
    sections.extend(
        [
            "## Interpretation and limitations",
            "Signature-expression associations are hypothesis-generating. They "
            "do not establish activation, mechanism or causality. An "
            "inconclusive test does not establish equivalence or absence.",
            "\n".join(f"- {limitation}" for limitation in limitations),
            f"Frozen registration: `{registration_identity}`.",
        ]
    )
    return "\n\n".join(sections) + "\n"


def render_extension_sensitivity(results: pd.DataFrame) -> str:
    """Render AUCell extension effects separately from the Scanpy family."""
    selected = results.loc[
        results.analysis_role.eq("extension")
        & results.scorer.ne(results.primary_scorer)
    ]
    if selected.empty:
        return ""
    return (
        "## Registered extension: AUCell sensitivity\n\n"
        + _result_table(selected)
        + "\n\nAUCell is a measurement sensitivity using the same people. "
        "Its raw "
        "p-values do not enter the Scanpy Holm family or define extension "
        "association status.\n"
    )


def _result_table(results: pd.DataFrame) -> str:
    """Keep estimate, support, decision and unavailability readable together."""
    rows = [
        "| Module / comparison | Expected sign | Paired people "
        "| Effect [95% CI] "
        "| p (Holm for Scanpy; raw for AUCell) | Decision "
        "| Method agreement / reason |",
        "| --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in results.itertuples(index=False):
        pvalue = (
            row.pvalue_adjusted
            if row.scorer == row.primary_scorer
            else row.pvalue
        )
        reason = (
            row.eligibility_reason
            if not row.eligible
            else row.method_concordance
        )
        label = (
            f"{row.module_id}: {row.target_context} - {row.reference_context}"
        )
        if row.analysis_role == "extension":
            expected = "two-sided"
        else:
            expected = "+" if row.expected_direction == 1 else "-"
        interval = (
            f"{_number(row.estimate)} "
            f"[{_number(row.ci_lower)}, {_number(row.ci_upper)}]"
        )
        rows.append(
            f"| {label} | {expected} | {_number(row.n_paired_donors)} | "
            f"{interval} | {_number(pvalue)} | {row.status} | {reason} |"
        )
    return "\n".join(rows)


def _discovery_table(discovery: pd.DataFrame) -> str:
    """Show discovery methods without implying cross-method calibration."""
    rows = [
        "| Module | Scorer | Paired people | Effect [95% CI] | Measurement |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in discovery.itertuples(index=False):
        rows.append(
            f"| {row.module_id} | {row.scorer} | "
            f"{_number(row.n_paired_donors)} | {_number(row.estimate)} "
            f"[{_number(row.ci_lower)}, {_number(row.ci_upper)}] | "
            f"{row.eligibility_reason} |"
        )
    return "\n".join(rows)


def _number(value: object) -> str:
    """Show missing inference explicitly and readable numerical precision."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return "unavailable"
    return f"{value:.4g}" if np.isfinite(value) else "unavailable"


def _donor_support_clause(compared: pd.DataFrame) -> str:
    """State a variant's donor support when its people differ from the primary.

    `n_matched_donors` counts the people complete in both analyses, so the
    donor sets are identical only when it equals both analyses' paired counts.
    Returns an empty string in that case.
    """
    matched = compared.n_matched_donors
    differs = matched.ne(compared.n_paired_donors) | matched.ne(
        compared.baseline_paired_donors
    )
    if not differs.any():
        return ""
    counts = []
    for column in (
        "n_paired_donors",
        "baseline_paired_donors",
        "n_matched_donors",
    ):
        low, high = int(compared[column].min()), int(compared[column].max())
        counts.append(str(low) if low == high else f"{low}-{high}")
    paired, primary, shared = counts
    return (
        f"; {paired} paired donors versus {primary} in the primary, "
        f"{shared} in common"
    )


def render_robustness_section(
    fixed: pd.DataFrame,
    balanced: pd.DataFrame,
    removals: pd.DataFrame,
    variants: pd.DataFrame,
    *,
    registered_notes: Sequence[str] = (),
) -> str:
    """Describe sign stability and unavailable variants without retesting.

    `variants` holds rescored selection variants (chemistry, released QC,
    modality); each is summarized on its own cells and donors.
    `registered_notes` are complete list items accounting for registered
    checks that have no rescored variant of their own.
    """
    baseline = fixed.loc[
        fixed.variant_kind.eq("baseline"),
        ["module_id", "scorer", "estimate", "n_paired_donors"],
    ].rename(
        columns={
            "estimate": "baseline_estimate",
            "n_paired_donors": "baseline_paired_donors",
        }
    )
    lines = []
    rescored_labels = ["Balanced external rescoring"]
    variant_check_labels = []
    checks = [
        (
            "Whole-person deletion",
            fixed.loc[fixed.variant_kind.eq("donor_deletion")],
        ),
        ("Balanced external rescoring", balanced),
        ("Gene-removal rescoring", removals),
    ]
    for variant_id, frame in variants.groupby("variant_id", sort=False):
        label = variant_label(variant_id).statement
        rescored_labels.append(label)
        variant_check_labels.append(label)
        checks.append((label, frame))
    for label, frame in checks:
        compared = frame.merge(
            baseline,
            on=["module_id", "scorer"],
            how="inner",
            validate="many_to_one",
        )
        reversals = (compared.estimate * compared.baseline_estimate < 0).sum()
        unavailable = (~compared.eligible).sum()
        donors = (
            _donor_support_clause(compared)
            if label in variant_check_labels
            else ""
        )
        lines.append(
            f"- {label}: {int(reversals)} point-estimate sign reversals; "
            f"{int(unavailable)} unavailable module/method variants{donors}. "
            "These are sensitivity checks, not additional replication tests."
        )
        if label in rescored_labels:
            imprecise = compared.loc[
                compared.eligible
                & compared.ci_lower.le(0)
                & compared.ci_upper.ge(0)
            ]
            if not imprecise.empty:
                names = ", ".join(
                    f"{row.module_id} ({row.scorer})"
                    for row in imprecise.itertuples()
                )
                lines.append(
                    f"- {label}: pointwise intervals include zero for {names}. "
                    "A retained point-estimate sign does not establish "
                    "precision or independence from the scoring population."
                )
    unavailable_removals = removals.loc[
        ~removals.eligible, ["module_id", "removed_genes", "eligibility_reason"]
    ].drop_duplicates()
    for row in unavailable_removals.itertuples():
        lines.append(
            f"- {row.module_id}, removal of {row.removed_genes}: "
            f"{row.eligibility_reason}. This cannot establish robustness "
            "to that arm's definition."
        )
    lines.extend(registered_notes)
    text = "\n".join(lines) + "\n"
    text += (
        "\nGene panels include every registered module member and context-only "
        "genes, with signed arms labeled. Gray gene differences are people; "
        "black diamonds are means. Detection and expression panels use "
        "equal-person means. There are no gene-level tests. A gene need not "
        "share its module's contrast sign.\n"
    )
    text += (
        "\nFigure 5 shows saved pointwise intervals, except the explicitly "
        "labeled donor-deletion range. Open diamonds are primary effects on "
        "the same donor intersection. Filled dots use all available variant "
        "pairs; blue squares show the variant on the matched intersection "
        "when these donor sets differ. Read the saved tables for donor "
        "identities and unavailable variants. Descriptive technical strata "
        "carry no inferential intervals. "
        "[Technical strata](technical_strata.csv); "
        "[balanced TS estimates](balanced_TS_contrasts.csv). TS and external "
        "score magnitudes are not calibrated across cohorts.\n"
    )
    return text
