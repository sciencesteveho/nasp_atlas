"""Tests for NASP hypothesis-priority reports."""

from __future__ import annotations

import pandas as pd
import pytest

from nasp_atlas.single_cell.hypothesis_priorities import (
    expected_module_coupling_report,
)
from nasp_atlas.single_cell.hypothesis_priorities import rank_nasp_hypotheses


def test_expected_edges_preserve_analyses_and_overlap_annotations() -> None:
    """An expected direction annotates both rows of a symmetric pair."""
    coupling = pd.DataFrame(
        {
            "module_a": ["IFN_OUTPUT", "IFN_OUTPUT", "POST", "POST"],
            "module_b": ["SENSOR", "SENSOR", "SENSOR", "SENSOR"],
            "analysis": ["raw", "within_context", "raw", "within_context"],
            "spearman_r": [0.8, 0.3, 0.4, 0.1],
            "spearman_fdr": [0.01, 0.2, 0.1, 0.8],
            "signed_gene_jaccard": [0.25, 0.25, 0.0, 0.0],
            "concordant_shared_genes": ["ISG15", "ISG15", "", ""],
        }
    )

    report = expected_module_coupling_report(
        coupling,
        edge_specs=[("SENSOR", "IFN_OUTPUT", "sensing_to_ifn")],
    )

    assert report["source_module"].tolist() == ["SENSOR", "SENSOR"]
    assert report["target_module"].tolist() == ["IFN_OUTPUT", "IFN_OUTPUT"]
    assert report["mechanistic_edge"].tolist() == [
        "sensing_to_ifn",
        "sensing_to_ifn",
    ]
    assert report["analysis"].tolist() == ["raw", "within_context"]
    assert report["spearman_r"].tolist() == [0.8, 0.3]
    assert report["signed_gene_jaccard"].tolist() == [0.25, 0.25]
    assert report["concordant_shared_genes"].tolist() == ["ISG15", "ISG15"]
    assert report["module_a"].tolist() == ["IFN_OUTPUT", "IFN_OUTPUT"]


def test_expected_edges_honor_spec_order_and_custom_pair_columns() -> None:
    """Caller order, labels, and nonstandard pair names are supported."""
    coupling = pd.DataFrame(
        {
            "left": ["A", "B", "A"],
            "right": ["B", "C", "C"],
            "raw_r": [0.1, 0.2, 0.3],
            "within_context_r": [0.01, 0.02, 0.03],
            "overlap": ["G1", "G2", "G3"],
        }
    )

    report = expected_module_coupling_report(
        coupling,
        edge_specs=[("C", "A", "second"), ("A", "B", "first")],
        module_a_column="left",
        module_b_column="right",
    )

    assert report["mechanistic_edge"].tolist() == ["second", "first"]
    assert report["raw_r"].tolist() == [0.3, 0.1]
    assert report["within_context_r"].tolist() == [0.03, 0.01]
    assert report["overlap"].tolist() == ["G3", "G1"]


def test_expected_edges_reject_duplicate_specifications() -> None:
    """Duplicate annotations cannot silently duplicate coupling evidence."""
    coupling = pd.DataFrame({"module_a": ["A"], "module_b": ["B"]})

    with pytest.raises(ValueError, match="duplicate edge specification"):
        expected_module_coupling_report(
            coupling,
            edge_specs=[("A", "B", "edge"), ("A", "B", "edge")],
        )


def _profiles() -> pd.DataFrame:
    """Build eligible and under-supported relative context profiles."""
    return pd.DataFrame(
        {
            "tissue": ["lung", "liver", "skin", "brain"],
            "cell_type": ["macrophage", "hepatocyte", "fibroblast", "glia"],
            "n_units": [6, 5, 2, 5],
            "n_donors": [4, 3, 2, 1],
            "c": [0.6, 0.9, 0.8, 0.7],
            "o": [0.8, 0.3, 0.9, 0.8],
            "r": [0.9, 0.95, 0.9, 0.9],
            "f": [0.9, 0.7, 0.9, 0.9],
            "p": [0.95, 0.4, 0.95, 0.9],
        }
    )


def test_rank_hypotheses_filters_support_and_emits_all_patterns() -> None:
    """Only replicated contexts enter five biologically distinct rankings."""
    result = rank_nasp_hypotheses(
        _profiles(),
        context_columns=["tissue", "cell_type"],
        competence_column="c",
        output_column="o",
        restriction_column="r",
        feedback_column="f",
        post_column="p",
        min_units=3,
        min_donors=2,
    )

    assert set(result["tissue"]) == {"lung", "liver"}
    assert set(result["hypothesis"]) == {
        "active_like",
        "responsive_like",
        "restricted_buffered",
        "feedback_dominant",
        "post_without_nasp",
    }
    responsive = result[result["hypothesis"] == "responsive_like"].iloc[0]
    assert responsive["tissue"] == "lung"
    assert responsive["contrast_value"] == pytest.approx(0.2)
    assert responsive["priority_score"] == pytest.approx(0.16)
    assert responsive["rank_within_hypothesis"] == 1


def test_restriction_priority_is_competence_aware() -> None:
    """Competence breaks equal restriction-output contrasts biologically."""
    profiles = pd.DataFrame(
        {
            "context": ["competent", "low_competence"],
            "n_units": [4, 4],
            "n_donors": [3, 3],
            "relative_competence": [0.9, 0.1],
            "relative_output": [0.2, 0.2],
            "relative_restriction": [0.9, 0.9],
            "relative_feedback": [0.2, 0.2],
            "relative_post": [0.2, 0.2],
        }
    )

    result = rank_nasp_hypotheses(
        profiles,
        context_columns=["context"],
    )
    restricted = result[result["hypothesis"] == "restricted_buffered"]

    top = restricted.sort_values("rank_within_hypothesis").iloc[0]
    bottom = restricted.sort_values("rank_within_hypothesis").iloc[1]
    assert top["context"] == "competent"
    assert top["priority_score"] == pytest.approx(0.63)
    assert bottom["priority_score"] == pytest.approx(0.35)


def test_rank_hypotheses_supports_disabled_count_filters() -> None:
    """Unit profiles can be ranked when count filters are explicitly off."""
    profiles = pd.DataFrame(
        {
            "unit": ["D1", "D2"],
            "relative_competence": [0.2, 0.8],
            "relative_output": [0.9, 0.7],
            "relative_restriction": [0.1, 0.2],
            "relative_feedback": [0.1, 0.2],
            "relative_post": [0.1, 0.2],
        }
    )

    result = rank_nasp_hypotheses(
        profiles,
        context_columns=["unit"],
        unit_count_column=None,
        donor_count_column=None,
        min_units=None,
        min_donors=None,
    )

    responsive = result[result["hypothesis"] == "responsive_like"]
    assert responsive.iloc[0]["unit"] == "D1"
    assert responsive.iloc[0]["rank_within_hypothesis"] == 1


def test_rank_hypotheses_rejects_nonrelative_axes() -> None:
    """Out-of-range axes fail before producing misleading priorities."""
    profiles = _profiles()
    profiles.loc[0, "o"] = 1.2

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        rank_nasp_hypotheses(
            profiles,
            context_columns=["tissue", "cell_type"],
            competence_column="c",
            output_column="o",
            restriction_column="r",
            feedback_column="f",
            post_column="p",
        )
