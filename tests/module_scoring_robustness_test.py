"""Robustness tests for single-cell NASP module scoring."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from nasp_compendium import GeneModules
from nasp_compendium.types import GeneIdOutput
from nasp_compendium.types import GeneModule
from pyscenic.aucell import GeneSignature
from pyscenic.aucell import aucell

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


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_aucell_preserves_signed_scores_across_chunks_and_missing_symbols(
    caplog,
    dtype,
) -> None:
    """Chunked scores match pySCENIC with ties and report progress."""
    rng = np.random.default_rng(8)
    genes = ["CGAS", "LMNB1", *[f"background_{i}" for i in range(98)]]
    values = rng.integers(0, 4, size=(12, 100)).astype(dtype)
    values[:6, 0] = 10
    values[6:, 1] = 10
    values[:, 2:8] = 10
    if np.issubdtype(dtype, np.floating):
        values[1, 2:8] = [np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0]
    else:
        limits = np.iinfo(dtype)
        values[1, 2:6] = [limits.min, limits.max, limits.max - 1, 0]

    adata = ad.AnnData(
        values,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(12)]),
        var=pd.DataFrame({"feature_name": genes}, index=genes),
    )
    module = _gene_module(inverse_genes=("LMNB1",))
    reference = aucell(
        pd.DataFrame(values, index=adata.obs_names, columns=genes),
        [
            GeneSignature("NASP_TEST_pos_auc", {"CGAS": 1.0}),
            GeneSignature("NASP_TEST_inv_auc", {"LMNB1": 1.0}),
        ],
        seed=17,
        num_workers=1,
    )
    reference["NASP_TEST_auc"] = combine_module_scores(
        module, reference, scorer="aucell"
    )
    _, single_chunk, _ = score_aucell_modules(
        adata,
        [module.module_id],
        gene_modules=[module],
        expression_layer=None,
        chunk_size=12,
        random_state=17,
    )
    adata.var.loc["CGAS", "feature_name"] = None
    original_obs = adata.obs.copy()
    original_values = values.copy()
    caplog.set_level(
        logging.INFO, logger="nasp_atlas.single_cell.module_scoring"
    )
    caplog.clear()

    _, chunked, _ = score_aucell_modules(
        adata,
        [module.module_id],
        gene_modules=[module],
        expression_layer=None,
        chunk_size=3,
        random_state=17,
    )

    pd.testing.assert_frame_equal(
        reference.sort_index(), single_chunk.sort_index(), check_exact=True
    )
    pd.testing.assert_frame_equal(reference.sort_index(), chunked.sort_index())
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    np.testing.assert_array_equal(adata.X, original_values)
    scores = chunked.reindex(adata.obs_names)["NASP_TEST_auc"]
    assert scores.iloc[:6].mean() > scores.iloc[6:].mean()
    assert "AUCell chunk 1/4" in caplog.text
    assert "AUCell chunk 4/4 complete: 12/12 cells (100.0%)" in caplog.text
    assert "estimated remaining 0.0min" in caplog.text
    assert "AUCell finished" in caplog.text


def test_scanpy_raw_scoring_uses_the_raw_gene_universe() -> None:
    """Raw scoring matches direct scoring after current genes are filtered."""
    module = GeneModules.modules("NASP_DNA_SENSING")
    genes = list(module.positive_genes + module.inverse_genes)
    genes += [f"background_{index}" for index in range(200)]
    rng = np.random.default_rng(11)
    raw = ad.AnnData(
        rng.uniform(0, 4, size=(12, len(genes))),
        var=pd.DataFrame({"feature_name": genes}, index=genes),
    )
    filtered = raw[:, [-1, -2]].copy()
    filtered.raw = raw.copy()

    score_scanpy_modules(raw, [module.module_id], expression_layer=None)
    score_scanpy_modules(
        filtered, [module.module_id], expression_layer=None, use_raw=True
    )

    np.testing.assert_allclose(
        filtered.obs["NASP_DNA_SENSING_score"],
        raw.obs["NASP_DNA_SENSING_score"],
    )
