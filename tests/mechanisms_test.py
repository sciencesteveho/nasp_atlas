"""Worked examples for donor-supported mechanism and reference scoring."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from nasp_compendium import GeneModules

from nasp_atlas.single_cell import ObsSchema
from nasp_atlas.single_cell import score_aucell_modules
from nasp_atlas.single_cell import score_scanpy_modules
from nasp_atlas.single_cell.mechanisms import centered_coupling
from nasp_atlas.single_cell.mechanisms import donor_score_units
from nasp_atlas.single_cell.module_scoring import ScorerName
from nasp_atlas.single_cell.reference_sets import reference_gene_sets


def test_coupling_controls_context_and_counts_donors() -> None:
    """Opposing within-context biology survives a misleading pooled trend."""
    rows = [
        {
            "donor_id": str(donor),
            "tissue_in_publication": tissue,
            "cell_type": "endothelial",
            "left": offset + donor,
            "right": offset + 3 - donor,
        }
        for tissue, offset in [("Lung", 0), ("Muscle", 100)]
        for donor in range(4)
        for _ in range(3)
    ]
    cells = pd.DataFrame(rows)
    units = donor_score_units(
        cells.drop(columns=["left", "right"]),
        cells[["left", "right"]],
        ["left", "right"],
        schema=ObsSchema(),
        minimum_cells=2,
    )
    result = centered_coupling(
        units,
        left="left",
        right="right",
        contexts=["tissue_in_publication", "cell_type"],
        minimum_donors=3,
    )
    assert cells.left.corr(cells.right, method="spearman") > 0
    assert result["spearman_r"] == pytest.approx(-1)
    assert result["n_donors"] == 4
    assert result["deletion_low"] == pytest.approx(-1)
    assert result["deletion_high"] == pytest.approx(-1)
    unsupported = centered_coupling(
        units,
        left="left",
        right="right",
        contexts=["tissue_in_publication", "cell_type"],
        minimum_donors=5,
    )
    assert pd.isna(unsupported["spearman_r"])


@pytest.mark.parametrize("scorer", ["scanpy", "aucell"])
def test_reference_scores_preserve_curated_scores(scorer: ScorerName) -> None:
    """Reference signatures preserve NASP scores on Ensembl-indexed data."""
    curated = GeneModules.modules("IFN_I_OUTPUT")
    references = reference_gene_sets()
    definitions = [curated, *references]
    genes = sorted(
        {
            gene
            for module in definitions
            for gene in module.positive_genes + module.inverse_genes
        }
    ) + [f"background_{index}" for index in range(300)]
    rng = np.random.default_rng(37)
    adata = ad.AnnData(
        sp.csr_matrix(rng.uniform(0, 5, size=(24, len(genes)))),
        var=pd.DataFrame(
            {"feature_name": genes},
            index=[f"ENSG{index}" for index in range(len(genes))],
        ),
    )
    baseline = adata.copy()
    if scorer == "scanpy":
        score_scanpy_modules(
            baseline, [curated.module_id], expression_layer=None
        )
        score_scanpy_modules(
            adata,
            [module.module_id for module in definitions],
            gene_modules=definitions,
            expression_layer=None,
        )
        expected = baseline.obs[f"{curated.module_id}_score"]
        actual = adata.obs[f"{curated.module_id}_score"]
        reference_values = adata.obs[
            [f"{module.module_id}_score" for module in references]
        ]
    else:
        _, baseline_scores, _ = score_aucell_modules(
            baseline,
            [curated.module_id],
            expression_layer=None,
        )
        _, all_scores, _ = score_aucell_modules(
            adata,
            [module.module_id for module in definitions],
            gene_modules=definitions,
            expression_layer=None,
        )
        expected = baseline_scores[f"{curated.module_id}_auc"]
        actual = all_scores[f"{curated.module_id}_auc"]
        reference_values = all_scores[
            [f"{module.module_id}_auc" for module in references]
        ]
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(reference_values.to_numpy()).all()
    assert reference_values.nunique().gt(1).all()
