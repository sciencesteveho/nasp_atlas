"""Robustness tests for single-cell NASP module scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from nasp_compendium.types import GeneIdOutput
from nasp_compendium.types import GeneModule
from pyscenic.aucell import GeneSignature

import nasp_atlas.single_cell.module_scoring as module_scoring
from nasp_atlas.single_cell.module_scoring import combine_module_scores
from nasp_atlas.single_cell.module_scoring import score_aucell_modules
from nasp_atlas.single_cell.module_scoring import score_scanpy_modules


def _gene_module(
    *,
    positive_genes: tuple[str, ...] = ("CGAS",),
    inverse_genes: tuple[str, ...] = (),
    gene_id_output: GeneIdOutput = "symbols",
) -> GeneModule:
    """Return a minimal signed gene module for scoring tests."""
    return GeneModule(
        module_id="NASP_TEST",
        positive_genes=positive_genes,
        inverse_genes=inverse_genes,
        context_dependent_genes=(),
        gene_id_output=gene_id_output,
    )


def _adata(
    var_names: Sequence[str],
    symbols: Sequence[object],
    *,
    n_obs: int = 2,
) -> ad.AnnData:
    """Return a small AnnData with aligned var names and gene symbols."""
    values = np.arange(n_obs * len(var_names), dtype=float).reshape(
        n_obs,
        len(var_names),
    )
    obs_names = [f"cell_{index}" for index in range(n_obs)]
    return ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(
            {"feature_name": pd.array(symbols, dtype="string")},
            index=pd.Index(var_names),
        ),
    )


def _patch_aucell(
    monkeypatch: pytest.MonkeyPatch,
    module: GeneModule,
    scorer: Callable[..., pd.DataFrame],
) -> None:
    """Replace module resolution and AUCell with deterministic test fakes."""
    monkeypatch.setattr(
        module_scoring.GeneModules,
        "modules",
        lambda _module_id, **_kwargs: module,
    )
    monkeypatch.setattr(module_scoring, "aucell", scorer)


def _constant_aucell(
    expression: pd.DataFrame,
    signatures: Sequence[GeneSignature],
    **_kwargs,
) -> pd.DataFrame:
    """Return one constant-valued AUC column per signature."""
    return pd.DataFrame(
        {
            signature.name: np.full(len(expression), 0.5)
            for signature in signatures
        },
        index=expression.index,
    )


def test_near_constant_module_arm_is_centered_before_combining() -> None:
    """Numerical noise in a constant arm is not amplified by z-scoring."""
    module = _gene_module(inverse_genes=("LMNB1",))
    scores = pd.DataFrame(
        {
            "NASP_TEST_pos": [1.0, 1.0 + 1e-12],
            "NASP_TEST_inv": [0.0, 2.0],
        }
    )

    combined = combine_module_scores(module, scores, scorer="scanpy")

    np.testing.assert_allclose(combined.to_numpy(), [1.0, -1.0])


def test_aucell_rejects_empty_cell_input() -> None:
    """AUCell reports an actionable error when no cells are present."""
    adata = _adata(["CGAS"], ["CGAS"], n_obs=0)

    with pytest.raises(ValueError, match="at least one cell"):
        score_aucell_modules(
            adata,
            ["NASP_TEST"],
            expression_layer=None,
        )


def test_aucell_rejects_nonpositive_chunk_size() -> None:
    """AUCell rejects chunk sizes that cannot advance block iteration."""
    adata = _adata(["CGAS"], ["CGAS"])

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        score_aucell_modules(
            adata,
            ["NASP_TEST"],
            expression_layer=None,
            chunk_size=0,
        )


def test_aucell_rejects_empty_signature_selection() -> None:
    """AUCell reports when no positive or inverse signatures were selected."""
    adata = _adata(["CGAS"], ["CGAS"])

    with pytest.raises(ValueError, match="at least one positive or inverse"):
        score_aucell_modules(adata, [], expression_layer=None)


def test_aucell_propagates_seed_and_uses_one_worker_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every AUCell block uses the requested seed and conservative workers."""
    adata = _adata(["CGAS", "OTHER"], ["CGAS", "OTHER"], n_obs=3)
    calls: list[tuple[tuple[str, ...], int, int]] = []

    def fake_aucell(
        expression: pd.DataFrame,
        signatures: Sequence[GeneSignature],
        *,
        seed: int,
        num_workers: int,
    ) -> pd.DataFrame:
        calls.append((tuple(expression.index), seed, num_workers))
        return _constant_aucell(expression, signatures)

    _patch_aucell(monkeypatch, _gene_module(), fake_aucell)

    score_aucell_modules(
        adata,
        ["NASP_TEST"],
        expression_layer=None,
        chunk_size=2,
        random_state=17,
    )

    assert Counter(
        cell for cell_indices, _, _ in calls for cell in cell_indices
    ) == Counter(adata.obs_names)
    assert all(len(cell_indices) <= 2 for cell_indices, _, _ in calls)
    assert all(seed == 17 for _, seed, _ in calls)
    assert all(num_workers == 1 for _, _, num_workers in calls)


def test_aucell_uses_var_names_when_symbols_are_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Null symbols fall back to matching var names in the ranking matrix."""
    adata = _adata(["CGAS", "OTHER"], [pd.NA, "OTHER"])
    ranked_columns: list[tuple[str, ...]] = []
    signature_genes: list[tuple[str, ...]] = []

    def fake_aucell(
        expression: pd.DataFrame,
        signatures: Sequence[GeneSignature],
        **_kwargs,
    ) -> pd.DataFrame:
        ranked_columns.append(tuple(expression.columns))
        signature_genes.extend(signature.genes for signature in signatures)
        return _constant_aucell(expression, signatures)

    _patch_aucell(monkeypatch, _gene_module(), fake_aucell)

    score_aucell_modules(
        adata,
        ["NASP_TEST"],
        expression_layer=None,
    )

    assert ranked_columns == [("CGAS", "OTHER")]
    assert signature_genes == [("CGAS",)]


def test_aucell_result_exposes_signed_arm_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AUCell results retain positive and inverse arms beside composites."""
    adata = _adata(["CGAS", "LMNB1"], ["CGAS", "LMNB1"])
    module = _gene_module(inverse_genes=("LMNB1",))

    _patch_aucell(monkeypatch, module, _constant_aucell)

    scored, auc_scores, _ = score_aucell_modules(
        adata,
        ["NASP_TEST"],
        expression_layer=None,
    )

    expected_columns = {
        "NASP_TEST_pos_auc",
        "NASP_TEST_inv_auc",
        "NASP_TEST_auc",
    }
    assert expected_columns.issubset(scored.obs.columns)
    assert expected_columns.issubset(auc_scores.columns)


def test_aucell_result_reuses_expression_without_mutating_input_obs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AUCell avoids copying expression while isolating score annotations."""
    adata = _adata(["CGAS"], ["CGAS"])

    _patch_aucell(monkeypatch, _gene_module(), _constant_aucell)

    scored, _, _ = score_aucell_modules(
        adata,
        ["NASP_TEST"],
        expression_layer=None,
    )

    assert scored.X is adata.X
    assert "NASP_TEST_pos_auc" not in adata.obs
    assert "NASP_TEST_pos_auc" in scored.obs


def test_scanpy_module_resolution_uses_raw_gene_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw scoring resolves module genes against raw rather than current var."""
    adata = _adata(["OTHER"], ["OTHER"])
    raw_source = _adata(["CGAS"], ["CGAS"])
    adata.raw = raw_source
    resolved_var_names: list[tuple[str, ...]] = []
    score_gene_lists: list[tuple[str, ...]] = []

    def fake_modules(_module_id: str, *, adata: ad.AnnData, **_kwargs):
        genes = tuple(adata.var_names.astype(str))
        resolved_var_names.append(genes)
        return _gene_module(
            positive_genes=genes,
            gene_id_output="var_names",
        )

    def fake_score_genes(adata_arg: ad.AnnData, **kwargs) -> None:
        score_gene_lists.append(tuple(kwargs["gene_list"]))
        adata_arg.obs[kwargs["score_name"]] = 0.5

    monkeypatch.setattr(module_scoring.GeneModules, "modules", fake_modules)
    monkeypatch.setattr(module_scoring.sc.tl, "score_genes", fake_score_genes)

    modules = score_scanpy_modules(
        adata,
        ["NASP_TEST"],
        expression_layer=None,
        use_raw=True,
    )

    assert resolved_var_names == [("CGAS",)]
    assert score_gene_lists == [("CGAS",)]
    assert modules[0].positive_genes == ("CGAS",)
