"""Processing utilities for single-cell datasets."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
import scipy.sparse as sp  # type: ignore
from anndata.typing import XDataType  # type: ignore

from nasp_atlas.single_cell.config import EmbeddingConfig


logger = logging.getLogger(__name__)


def _optional_module(name: str) -> ModuleType | None:
    """Import an optional module if it is available."""
    try:
        return import_module(name)
    except ImportError:
        return None


dc = _optional_module("decoupler")
harmonypy = _optional_module("harmonypy")


@runtime_checkable
class _TorchArrayLike(Protocol):
    """Minimal torch-like array protocol used by Harmony outputs."""

    def detach(self) -> _TorchArrayLike:
        """Return a detached array."""
        ...

    def cpu(self) -> _TorchArrayLike:
        """Return a CPU-backed array."""
        ...

    def numpy(self) -> np.ndarray:
        """Return a NumPy array."""
        ...


class SCProcessor:
    """Processor that transforms AnnData objects.

    Handles embedding generation (normalization, HVG, PCA, Harmony, UMAP) and
    Leiden clustering.

    Supports two normalization pipelines via the `pipeline` parameter:
      "standard": normalize_total -> log1p -> HVGs -> optional scale -> PCA
      "pearson_residuals": Pearson-residual HVG selection + normalization -> PCA
        (uses sc.experimental.pp.recipe_pearson_residuals)

    Example usage (standalone):
      >>> from nasp_atlas.single_cell import EmbeddingConfig
      >>> from nasp_atlas.single_cell import SCProcessor
      >>> config = EmbeddingConfig(
      ...     name="standard_harmony",
      ...     harmony_key="condition",
      ... )
      >>> proc = SCProcessor(output_dir="results/standard_harmony")
      >>> proc.generate_embeddings(adata, config=config)
      >>> proc.cluster(adata, resolution=0.5)

    Attributes:
      output_dir: Directory where outputs are written
      random_seed: Seed used throughout
    """

    def __init__(
        self,
        output_dir: str | Path,
        random_seed: int = 42,
    ) -> None:
        """Initialize the processor.

        Args:
          output_dir: Directory where outputs are written
          random_seed: Seed used throughout
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed

        np.random.seed(random_seed)

    def __repr__(self) -> str:
        """Return a concise object representation."""
        return (
            f"SCProcessor("
            f"output_dir={self.output_dir}, "
            f"seed={self.random_seed})"
        )

    def generate_embeddings(
        self,
        adata: ad.AnnData,
        *,
        config: EmbeddingConfig,
        save_h5ad: bool = True,
        out_h5ad: str = "embedded.h5ad",
    ) -> ad.AnnData:
        """Run the full embedding pipeline defined by `config`.

        Currently runs a standard Scanpy pipeline or one that utilizes Pearson
        residuals. See docstrings for `_normalize_standard()` and
        `_normalize_pearson_residuals()` for more details.

        Args:
          adata: AnnData to process
          config: EmbeddingConfig defining algorithm parameters
          save_h5ad: Write the embedded object to disk
          out_h5ad: Filename for the embedded h5ad
        """
        logger.info(
            "[generate_embeddings] config=%r (pipeline=%r)",
            config.name,
            config.pipeline,
        )

        adata.raw = adata.copy()

        if config.pipeline == "standard":
            self._normalize_standard(
                adata,
                target_sum=config.target_sum,
                exclude_highly_expressed=config.exclude_highly_expressed,
                max_fraction=config.max_fraction,
                n_top_genes=config.n_top_genes,
                hvg_kwargs=config.hvg_kwargs,
                scale=config.scale,
                regress_out=config.regress_out,
            )
        elif config.pipeline == "pearson_residuals":
            self._normalize_pearson_residuals(
                adata,
                n_top_genes=config.n_top_genes,
                pearson_residuals_kwargs=config.pearson_residuals_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown pipeline {config.pipeline!r}. "
                f"Expected 'standard' or 'pearson_residuals'"
            )

        logger.info("[generate_embeddings] PCA: %s components", config.n_pcs)
        sc.tl.pca(
            adata,
            n_comps=config.n_pcs,
            mask_var="highly_variable",
            svd_solver="arpack",
            random_state=self.random_seed,
        )

        rep_key = self._run_harmony_if_requested(adata, config=config)

        logger.info(
            "[generate_embeddings] kNN graph: k=%s, n_pcs=%s",
            config.n_neighbors,
            config.n_pcs,
        )
        sc.pp.neighbors(
            adata,
            n_neighbors=config.n_neighbors,
            n_pcs=config.n_pcs,
            use_rep=rep_key,
            key_added="neighborhood",
            random_state=self.random_seed,
        )

        logger.info("[generate_embeddings] computing UMAP")
        sc.tl.umap(
            adata,
            min_dist=config.umap_min_dist,
            neighbors_key="neighborhood",
            random_state=self.random_seed,
        )

        if config.force_directed is not None:
            logger.info("[generate_embeddings] computing force-directed layout")
            sc.tl.draw_graph(
                adata,
                layout=config.force_directed,
                init_pos="X_umap",
                neighbors_key="neighborhood",
                random_state=self.random_seed,
            )

        adata.uns["embedding_config"] = {
            **config.to_dict(),
            "random_seed": self.random_seed,
        }

        if save_h5ad:
            out = self.output_dir / out_h5ad
            adata.write_h5ad(out)
            logger.info("[generate_embeddings] saved -> %s", out)

        return adata

    def cluster(
        self,
        adata: ad.AnnData,
        *,
        resolution: float = 0.5,
        neighbors_key: str = "neighborhood",
    ) -> ad.AnnData:
        """Run Leiden clustering.

        Labels are stored in `adata.obs[leiden_{resolution}]`.

        Args:
          adata: AnnData
          resolution: Leiden resolution
          neighbors_key: Neighbour graph to use for Leiden
        """
        key = f"leiden_{resolution}"
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=self.random_seed,
            key_added=key,
            neighbors_key=neighbors_key,
            flavor="igraph",
            n_iterations=2,
        )
        n_clusters = adata.obs[key].nunique()
        logger.info(
            "[cluster] Leiden resolution=%s -> %s clusters -> obs[%s]",
            resolution,
            n_clusters,
            key,
        )
        return adata

    def _run_harmony_if_requested(
        self,
        adata: ad.AnnData,
        *,
        config: EmbeddingConfig,
    ) -> str:
        """Run Harmony if requested and return the representation key."""
        if config.harmony_key is None:
            logger.info(
                "[generate_embeddings] no harmony_key; "
                "skipping batch correction"
            )
            return "X_pca"

        if harmonypy is None:
            raise ImportError(
                "harmonypy is required when config.harmony_key is set. "
                "Install harmonypy to run Harmony integration."
            )

        logger.info(
            "[generate_embeddings] Harmony integration on %s",
            config.harmony_key,
        )
        ho = harmonypy.run_harmony(
            adata.obsm["X_pca"],
            adata.obs,
            config.harmony_key,
            random_state=self.random_seed,
        )
        adata.obsm["X_pca_harmony"] = self._coerce_harmony_output(
            ho.Z_corr, adata.n_obs
        )
        return "X_pca_harmony"

    def _normalize_standard(
        self,
        adata: ad.AnnData,
        *,
        target_sum: float,
        exclude_highly_expressed: bool,
        max_fraction: float,
        n_top_genes: int | None,
        hvg_kwargs: dict[str, Any] | None,
        scale: bool,
        regress_out: list[str] | None,
    ) -> None:
        """Standard pipeline: normalize_total -> log1p -> HVG -> optional scale.

        Args:
          adata: AnnData
          target_sum: Normalization target sum per cell
          exclude_highly_expressed: Exclude highly expressed genes during
            normalization
          max_fraction: Max fraction for highly expressed gene exclusion
          n_top_genes: Number of HVGs to select
          hvg_kwargs: Additional kwargs forwarded to sc.pp.highly_variable_genes
          scale: Whether to z-score scale (max_value=10) prior to PCA
          regress_out: obs column name(s) to regress out
        """
        logger.info("[standard] normalize_total")
        sc.pp.normalize_total(
            adata,
            target_sum=target_sum,
            exclude_highly_expressed=exclude_highly_expressed,
            max_fraction=max_fraction,
        )

        logger.info("[standard] log1p")
        sc.pp.log1p(adata)
        adata.layers["log1p"] = self._copy_expression_matrix(adata)

        logger.info("[standard] selecting highly variable genes")
        hvg_kw: dict[str, Any] = dict(hvg_kwargs or {})
        if n_top_genes is not None:
            hvg_kw["n_top_genes"] = n_top_genes
        sc.pp.highly_variable_genes(adata, **hvg_kw)

        n_hvg = int(adata.var["highly_variable"].sum())
        logger.info("[standard] %s highly variable genes selected", n_hvg)

        if regress_out is not None:
            logger.info("[standard] regressing out %s", regress_out)
            sc.pp.regress_out(adata, keys=regress_out)

        if scale:
            logger.info("[standard] scale (max_value=10)")
            sc.pp.scale(adata, max_value=10)

    def _normalize_pearson_residuals(
        self,
        adata: ad.AnnData,
        *,
        n_top_genes: int | None,
        pearson_residuals_kwargs: dict[str, Any] | None,
        log1p_target_sum: float = 1e4,
    ) -> None:
        """Pearson-residuals pipeline.

        Also computes a standard log1p layer from raw counts so downstream
        consumers have a consistent layer available regardless of which
        embedding pipeline was used.

        After this call:
          adata.X: clipped Pearson residuals
          adata.layers["log1p"]: standard log-normalized values
          adata.layers["pearson_residuals"]: copy of the clipped residuals
          adata.var["highly_variable"]: set by the recipe

        Args:
          adata: AnnData
          n_top_genes: Number of HVGs to select
          pearson_residuals_kwargs: Additional kwargs forwarded to
            sc.experimental.pp.recipe_pearson_residuals
          log1p_target_sum: Target sum for the log1p layer normalization
        """
        logger.info("[pearson_residuals] computing log1p layer from raw counts")
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=log1p_target_sum)
        sc.pp.log1p(tmp)
        adata.layers["log1p"] = self._copy_expression_matrix(tmp)
        del tmp

        pr_kw: dict[str, Any] = dict(pearson_residuals_kwargs or {})
        if n_top_genes is not None:
            pr_kw["n_top_genes"] = n_top_genes

        logger.info(
            "[pearson_residuals] recipe_pearson_residuals (n_top_genes=%s)",
            pr_kw.get("n_top_genes", "scanpy default"),
        )
        sc.experimental.pp.recipe_pearson_residuals(adata, **pr_kw)
        adata.layers["pearson_residuals"] = self._copy_expression_matrix(adata)

        n_hvg = int(adata.var["highly_variable"].sum())
        logger.info(
            "[pearson_residuals] %s highly variable genes selected; "
            "adata.X contains clipped Pearson residuals; "
            "log1p and pearson_residuals layers stored",
            n_hvg,
        )

    @staticmethod
    def _copy_expression_matrix(adata: ad.AnnData) -> XDataType:
        """Copy an in-memory AnnData expression matrix."""
        x = adata.X
        if x is None:
            raise ValueError("adata.X is None; cannot copy expression matrix.")

        if isinstance(x, np.ndarray):
            return x.copy()
        if isinstance(
            x,
            (
                sp.csr_matrix,
                sp.csc_matrix,
                sp.csr_array,
                sp.csc_array,
            ),
        ):
            return x.copy()

        raise TypeError(
            "SCProcessor expects adata.X to be an in-memory NumPy or SciPy "
            f"sparse matrix, got {type(x).__name__}."
        )

    @staticmethod
    def _coerce_harmony_output(
        Z_corr: np.ndarray | _TorchArrayLike,
        n_obs: int,
    ) -> np.ndarray:
        """Force Harmony output to shape (n_obs, n_pcs).

        Args:
          Z_corr: ho.Z_corr from harmonypy.run_harmony()
          n_obs: Expected number of cells
        """
        z = (
            Z_corr.detach().cpu().numpy()
            if isinstance(Z_corr, _TorchArrayLike)
            else np.asarray(Z_corr)
        )

        if z.ndim != 2:
            raise ValueError(f"Harmony output must be 2D, got shape {z.shape}")

        if z.shape[0] == n_obs:
            return z
        if z.shape[1] == n_obs:
            return z.T
        raise ValueError(
            f"Harmony output shape {z.shape} has neither dimension "
            f"equal to n_obs={n_obs}"
        )

    @staticmethod
    def subset_to_raw_counts(
        adata: ad.AnnData,
        *,
        obs_key: str,
        values: str | list[str],
    ) -> ad.AnnData:
        """Subset cells by annotation label and revert to raw counts.

        Args:
          adata: AnnData with .raw set
          obs_key: obs column to filter on
          values: Label(s) to keep

        Returns:
          A new AnnData with raw counts in .X
        """
        if adata.raw is None:
            raise ValueError(
                "adata.raw is None. subset_to_raw_counts requires .raw to be "
                "set (generate_embeddings does this automatically)"
            )

        if isinstance(values, str):
            values = [values]

        mask = adata.obs[obs_key].isin(values)
        n_selected = int(mask.sum())

        if n_selected == 0:
            raise ValueError(
                f"No cells matched {values!r} in obs[{obs_key!r}]. "
                f"Unique values: {adata.obs[obs_key].unique().tolist()}"
            )

        adata_raw = adata.raw.to_adata()
        adata_raw.obs = adata.obs.copy()
        adata_sub = adata_raw[mask].copy()

        logger.info(
            "[subset_to_raw_counts] %s / %s cells selected "
            "(%s=%r); reverted to raw counts",
            n_selected,
            adata.n_obs,
            obs_key,
            values,
        )
        return adata_sub

    @staticmethod
    def auto_annotate_from_scores(
        adata: ad.AnnData,
        *,
        leiden_key: str,
        score_key: str = "score_ulm",
        prefix: str = "ctscore",
        ambiguity_margin: float = 0.1,
    ) -> tuple[pd.DataFrame, dict[Any, str], ad.AnnData]:
        """Assign automated cluster labels from precomputed per-cell scores.

        For each cluster, the cell type with the highest mean enrichment score
        is selected. If the margin between the top two scores falls below
        `ambiguity_margin`, the label is suffixed with "-like".

        Args:
          adata: anndata with decoupleR scores in obsm
          leiden_key: obs column for Leiden clustering
          score_key: obsm key for the decoupleR score matrix
          prefix: Prefix for per-cell score columns added to adata.obs
          ambiguity_margin: Minimum margin between the top two scores
            for a confident assignment

        Returns:
          cluster_means: Per-cluster mean scores (clusters x cell types)
          cluster_labels: {cluster_id: assigned_label}
          score: Per-cell score anndata from decoupleR
        """
        if dc is None:
            raise ImportError(
                "decoupler is required for auto_annotate_from_scores. "
                "Install decoupler to use this method."
            )

        score = dc.pp.get_obsm(adata, key=score_key)
        score.obs[leiden_key] = adata.obs[leiden_key].values
        score_df = score.to_df()

        for col in score_df.columns:
            obs_col = f"{prefix}_{col}"
            if obs_col not in adata.obs.columns:
                adata.obs[obs_col] = score_df[col].values

        cluster_means = score_df.groupby(
            score.obs[leiden_key], observed=True
        ).mean()

        top_labels = cluster_means.idxmax(axis=1)
        top_scores = cluster_means.max(axis=1)
        second_scores = cluster_means.apply(
            lambda x: x.nlargest(2).iloc[-1], axis=1
        )
        margins = top_scores - second_scores

        cluster_labels = {}
        for cluster in cluster_means.index:
            label = str(top_labels.loc[cluster])
            cluster_labels[cluster] = (
                f"{label}-like"
                if margins.loc[cluster] < ambiguity_margin
                else label
            )

        adata.obs[f"{leiden_key}_auto"] = (
            adata.obs[leiden_key].map(cluster_labels).astype("category")
        )

        return cluster_means, cluster_labels, score
