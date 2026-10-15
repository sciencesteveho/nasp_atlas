"""Markdown summary of cross-cohort sensor target evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from nasp_atlas.analysis.condition_cohorts.target_evidence import TargetEvidence


def render_evidence_report(
    evidence: TargetEvidence,
    *,
    run_dirs: Sequence[Path],
    settings: Mapping[str, object],
    figure_name: str = "target_evidence_matrix.png",
    max_cell_types: int = 6,
) -> str:
    """Render the tiers and drug-discovery shortlist as Markdown.

    The report restates the inputs and thresholds, counts each tier, and
    lists every drug-discovery "priority" sensor with the evidence behind
    it. Long cell-type lists are truncated to `max_cell_types` with a count;
    the CSV tables hold everything. Nothing here adds inference.
    """
    summary = evidence.summary
    shortlist = summary.loc[summary["drug_discovery_tier"].eq("priority")]

    lines = [
        "# Sensor target evidence across condition cohorts",
        "",
        "Hypothesis-generating synthesis of published cohort results; no "
        "estimate is refitted. Tier rules are in `summarize_target_evidence`.",
        "",
        "## Inputs",
        "",
        *(f"- Run: `{run}`" for run in run_dirs),
        *(f"- {name}: `{value}`" for name, value in settings.items()),
        "",
        "## Tiers",
        "",
        _markdown_table(_tier_counts(summary)),
        "",
        f"![Evidence matrix]({figure_name})",
        "",
        "## Drug-discovery priority sensors",
        "",
        "Ordered by the number of cohorts modulated, then the largest "
        "sensor-level shift. Directions disagreeing between cohorts are "
        "listed, not averaged.",
        "",
        _markdown_table(_shortlist_table(shortlist, max_cell_types)),
        "",
    ]
    return "\n".join(lines)


def _tier_counts(summary: pd.DataFrame) -> pd.DataFrame:
    """Count sensors per tier value for each of the three tiers."""
    return (
        pd.DataFrame(
            {
                tier: summary[tier].value_counts()
                for tier in (
                    "drug_discovery_tier",
                    "sirna_tier",
                    "crispri_tier",
                )
            }
        )
        .fillna(0)
        .astype(int)
        .rename_axis("tier")
        .reset_index()
    )


def _shortlist_table(
    shortlist: pd.DataFrame,
    max_cell_types: int,
) -> pd.DataFrame:
    """Display columns for the drug-discovery priority sensors."""
    return pd.DataFrame(
        {
            "sensor": shortlist["gene"],
            "siRNA": shortlist["sirna_tier"],
            "CRISPRi": shortlist["crispri_tier"],
            "direction by cohort": shortlist[
                "level_direction_by_cohort"
            ].fillna(""),
            "max |level shift|": shortlist["max_abs_level_effect"].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            ),
            "IFN-independent in": shortlist["ifn_independent_cell_types"].map(
                lambda value: _truncate(value, max_cell_types)
            ),
            "within-coupled to": shortlist["within_coupled_responses"].fillna(
                ""
            ),
        }
    )


def _truncate(value: object, limit: int) -> str:
    """Shorten a semicolon list to `limit` items plus a count."""
    if not isinstance(value, str) or not value:
        return ""

    items = value.split(";")
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} (+{len(items) - limit})"


def _markdown_table(table: pd.DataFrame) -> str:
    """Format a small table as GitHub Markdown without extra dependencies."""
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    rule = "| " + " | ".join("---" for _ in table.columns) + " |"

    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])
