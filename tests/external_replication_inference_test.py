"""Registered questions survive unavailable measurements and method reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nasp_atlas.analysis.external_replication.registration import (
    freeze_registration,
)
from nasp_atlas.analysis.external_replication.registration import (
    read_registration,
)
from nasp_atlas.analysis.external_replication.replication import (
    adjust_registered_families,
)
from nasp_atlas.analysis.external_replication.replication import (
    classify_registered_contrasts,
)
from nasp_atlas.analysis.external_replication.reporting import (
    render_replication_report,
)


def test_frozen_family_preserves_negative_missing_and_method_dependent_results(
    tmp_path,
) -> None:
    """Frozen questions yield all decisions; changing the freeze is rejected."""
    hypotheses = pd.DataFrame(
        {
            "hypothesis_id": ["a", "b", "c", "d"],
            "module_id": ["A", "B", "C", "D"],
            "cohort_id": "fixture",
            "family_id": "primary",
            "analysis_role": "primary",
            "comparison_axis": "tissue",
            "target_context": "spleen",
            "reference_context": "blood",
            "expected_direction": 1,
            "primary_scorer": "scanpy",
            "eligibility_status": "eligible",
            "eligibility_reason": "eligible",
        }
    )
    output = tmp_path / "registration"
    frozen = freeze_registration(
        hypotheses,
        {"independent_unit": "person", "seed": 42},
        output_dir=output,
    )
    recovered = read_registration(output)
    assert frozen.identity == recovered.identity
    estimates = pd.DataFrame(
        {
            "module_id": ["A", "B", "D"] * 2,
            "scorer": ["scanpy"] * 3 + ["aucell"] * 3,
            "comparison_axis": "tissue",
            "target_level": "spleen",
            "reference_level": "blood",
            "estimate": [1, -1, 0.01, -1, -1, 0.01],
            "pvalue": [0.001, 0.002, 0.8] * 2,
            "eligible": True,
            "eligibility_reason": "ok",
            "n_paired_donors": 6,
        }
    )
    result = classify_registered_contrasts(recovered.hypotheses, estimates)
    primary = result.loc[result.scorer.eq("scanpy")].set_index("hypothesis_id")
    assert primary.status.tolist() == [
        "supported",
        "opposite_direction",
        "unavailable",
        "inconclusive",
    ]
    assert primary.loc["a", "pvalue_adjusted"] == pytest.approx(0.004)
    assert primary.loc["b", "pvalue_adjusted"] == pytest.approx(0.006)
    assert np.isnan(primary.loc["c", "estimate"])
    assert np.isnan(primary.loc["c", "pvalue_adjusted"])
    assert primary.n_registered_family.eq(4).all()
    assert primary.n_tested_family.eq(3).all()
    assert primary.loc["a", "method_concordance"] == "opposite_direction"
    assert (
        result.loc[result.scorer.eq("aucell"), "pvalue_adjusted"].isna().all()
    )

    result["ci_lower"] = result.estimate - 0.5
    result["ci_upper"] = result.estimate + 0.5
    saved = tmp_path / "contrasts.csv"
    result.to_csv(saved, index=False)
    discovery = estimates.assign(
        ci_lower=estimates.estimate - 0.5, ci_upper=estimates.estimate + 0.5
    )
    report = render_replication_report(
        pd.read_csv(saved),
        discovery,
        title="Synthetic replication",
        registration_identity=recovered.identity,
        report_scope="Synthetic primary report; robustness is not computed.",
        limitations=["Synthetic fixture, not biological evidence."],
    )
    (tmp_path / "report.md").write_text(report)
    expected_summary = (
        "1 supported; 1 opposite direction; 1 inconclusive; 1 unavailable"
    )
    assert expected_summary in report
    assert "method-dependent" in report
    assert "0.004" in report
    assert "missing_estimate" in report

    unsupported = recovered.hypotheses.copy()
    unsupported.loc[unsupported.module_id.eq("A"), "eligibility_status"] = (
        "unavailable"
    )
    unsupported.loc[unsupported.module_id.eq("A"), "eligibility_reason"] = (
        "discovery_not_supported"
    )
    gated = classify_registered_contrasts(unsupported, estimates)
    row = gated.loc[
        gated.hypothesis_id.eq("a") & gated.scorer.eq("scanpy")
    ].iloc[0]
    assert row.status == "unavailable"
    assert row.estimate == 1
    assert np.isnan(row.pvalue)

    extension = recovered.hypotheses.assign(
        analysis_role="extension",
        family_id="extension",
        expected_direction=0,
        eligibility_status="pending",
    )
    extended = classify_registered_contrasts(extension, estimates)
    assert (
        extended.loc[
            extended.hypothesis_id.eq("b") & extended.scorer.eq("scanpy"),
            "status",
        ].item()
        == "association_detected"
    )

    (output / "hypotheses.csv").write_text(
        hypotheses.iloc[:2].to_csv(index=False)
    )
    with pytest.raises(ValueError, match="Changed registration"):
        read_registration(output)


def test_holm_retains_unavailable_members_as_internal_nonrejections() -> None:
    """Two tested p-values still belong to the four planned tests."""
    table = pd.DataFrame(
        {
            "hypothesis_id": ["a", "b", "c", "d"],
            "family_id": "primary",
            "scorer": "scanpy",
            "primary_scorer": "scanpy",
            "pvalue": [0.01, 0.04, np.nan, np.nan],
            "eligible": [True, True, False, False],
        }
    )
    result = adjust_registered_families(table)
    np.testing.assert_allclose(result.pvalue_adjusted[:2], [0.04, 0.12])
    assert result.pvalue_adjusted[2:].isna().all()
    assert result.pvalue[2:].isna().all()
