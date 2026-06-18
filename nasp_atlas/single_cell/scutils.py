"""Single-cell utilities / toolkit."""

from __future__ import annotations

import contextlib
import gzip
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad  # type: ignore
import scanpy as sc  # type: ignore

from nasp_atlas.single_cell.config import EmbeddingConfig
from nasp_atlas.single_cell.scprocessor import SCProcessor
from nasp_atlas.single_cell.visualization import SCVisualizer


logger = logging.getLogger(__name__)


class SCUtils:
    """Single-cell analysis utilities and toolkit.

    When an `EmbeddingConfig` is provided, all outputs are written to a
    subdirectory named after the config (`output_dir / config.name`), and the
    config JSON is persisted alongside the results for full traceability.

    Example usage:
      >>> from nasp_atlas.single_cell import EmbeddingConfig
      >>> from nasp_atlas.single_cell import SCUtils
      >>> config = EmbeddingConfig(
      ...     name="standard_harmony_fa",
      ...     harmony_key="condition",
      ...     force_directed="fa",
      ... )
      >>> adata = SCUtils.load_h5ad("data/dataset.h5ad")
      >>> sc_utils = SCUtils(output_dir="results", config=config)
      >>> sc_utils.processor.generate_embeddings(adata, config=config)
      >>> sc_utils.viz.plot_embedding(adata, color="condition", filename="umap")

    Attributes:
      output_dir: Directory where all outputs are written
      config: EmbeddingConfig driving the run (if provided)
      random_seed: Seed used throughout
      processor: SCProcessor instance
      viz: SCVisualizer instance
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        config: EmbeddingConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize the toolkit.

        Args:
          output_dir: Root directory for outputs. If `config` is provided,
            outputs are written to `output_dir / config.name`
          config: EmbeddingConfig for the run (optional)
          random_seed: Seed used throughout
        """
        root = Path(output_dir)
        self.config = config
        self.random_seed = random_seed

        self.output_dir = root / config.name if config is not None else root
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processor = SCProcessor(
            output_dir=self.output_dir,
            random_seed=self.random_seed,
        )
        self.viz = SCVisualizer(output_dir=self.output_dir)

        if config is not None:
            self._dump_config(config)

        logger.info(
            "SCUtils | output_dir=%s | seed=%s",
            self.output_dir,
            random_seed,
        )

    def __repr__(self) -> str:
        """Return a concise object representation."""
        config_name = self.config.name if self.config else None
        return (
            f"SCUtils("
            f"output_dir={self.output_dir}, "
            f"config={config_name!r}, "
            f"seed={self.random_seed})"
        )

    def _dump_config(self, config: EmbeddingConfig) -> None:
        """Persist config at the start of a run."""
        config_path = self.output_dir / "embedding_config.json"
        config_path.write_text(config.to_json())
        logger.info("SCUtils | config=%r saved -> %s", config.name, config_path)

    @staticmethod
    def load_h5ad(path: str | Path) -> ad.AnnData:
        """Load an h5ad.

        Args:
          path: Path to the .h5ad or .h5ad.gz file

        Returns:
          Loaded AnnData object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        adata = SCUtils._handle_h5ad(path)
        logger.info("[load] h5ad loaded: %s", adata)
        return adata

    @staticmethod
    def filter_obs_doublets(
        adata: ad.AnnData,
        *,
        doublet_key: str = "doublet",
    ) -> ad.AnnData:
        """Drop flagged doublets if a doublet `obs` column is present.

        Assumes `True` in the doublet column marks cells to remove.

        Args:
          adata: AnnData to filter
          doublet_key: obs column containing doublet flags
        """
        if doublet_key not in adata.obs.columns:
            return adata

        n_before = adata.n_obs
        mask = adata.obs[doublet_key].astype(str) != "True"
        adata = adata[mask].copy()

        if n_removed := n_before - adata.n_obs:
            logger.info("[filter] removed %s flagged doublets", n_removed)
        return adata

    @staticmethod
    def map_categorical_column(
        adata: ad.AnnData,
        *,
        source_col: str,
        mapping: dict[Any, Any],
        destination_col: str,
        axis: str = "obs",
    ) -> ad.AnnData:
        """Applies a dictionary lookup to remap a categorical column.

        Values not present in `mapping` become NaN in the new column.

        Args:
          adata: AnnData to modify
          source_col: Existing column to remap
          mapping: {old_value: new_label}
          destination_col: Name of the new categorical column
          axis: "obs" or "var"
        """
        if axis not in ("obs", "var"):
            raise ValueError(f"axis must be 'obs' or 'var', got '{axis}'")

        df = getattr(adata, axis)

        if source_col not in df.columns:
            logger.warning(
                "[map_column] %r not found in %s; %r not created.",
                source_col,
                axis,
                destination_col,
            )
            return adata

        source_values = df[source_col]
        key_dtype = type(next(iter(mapping)))
        with contextlib.suppress(ValueError, TypeError):
            source_values = source_values.astype(key_dtype)
        df[destination_col] = source_values.map(mapping).astype("category")
        counts = df[destination_col].value_counts().sort_index()
        logger.info(
            "[map_column] %s[%r] counts:\n%s",
            axis,
            destination_col,
            counts.to_string(),
        )
        return adata

    @staticmethod
    def _handle_h5ad(path: Path) -> ad.AnnData:
        """Read an .h5ad or .h5ad.gz."""
        if path.suffix != ".gz":
            return sc.read_h5ad(path)

        logger.info("[load] decompressing %s", path.name)

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                with gzip.open(path, "rb") as src:
                    shutil.copyfileobj(src, tmp)
            finally:
                tmp.flush()

        try:
            return sc.read_h5ad(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
