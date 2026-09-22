"""Content identities for external replication measurements and results."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from nasp_compendium.types import GeneModule
from pyscenic.aucell import aucell4r

from nasp_atlas.analysis.external_replication import cohort
from nasp_atlas.analysis.external_replication import feature_mapping
from nasp_atlas.analysis.external_replication import scoring
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.single_cell import io
from nasp_atlas.single_cell import module_scoring
from nasp_atlas.single_cell import utils as expression_utils


__all__ = ["file_identity", "request_identity", "scoring_request"]


def file_identity(path: Path) -> dict[str, str | int]:
    """Hash source bytes; fail if a writer changes the file during reading."""
    before = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Input changed during hashing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": after.st_size,
        "sha256": digest,
    }


def request_identity(request: Mapping[str, object]) -> str:
    """Identify a JSON-compatible scientific request without hidden defaults."""
    payload = json.dumps(request, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def scoring_request(
    spec: CohortSpec,
    selected_obs: pd.DataFrame,
    features: pd.DataFrame,
    shared_gene_ids: Sequence[str],
    modules: Sequence[GeneModule],
    coverage: pd.DataFrame,
    *,
    panel_path: Path,
    target_sum: float,
    random_state: int,
    aucell_chunk_size: int,
    aucell_num_workers: int,
) -> dict[str, object]:
    """Fingerprint scores, excluding later inference/report choices.

    Exact cell order, source matrix, canonical identifiers, curated membership,
    shared background, preparation/scorer implementations and actual versions
    determine compatibility. Covariates, hypothesis directions and inference
    thresholds do not change a fixed population's scores.
    """
    if not selected_obs.index.is_unique or selected_obs.empty:
        raise ValueError("A scoring request requires unique selected cells")
    dependencies = (
        cohort,
        feature_mapping,
        scoring,
        io,
        module_scoring,
        expression_utils,
    )
    code = {
        dependency.__name__: file_identity(Path(inspect.getfile(dependency)))[
            "sha256"
        ]
        for dependency in dependencies
    }
    versions = {
        package: importlib.metadata.version(package)
        for package in (
            "anndata",
            "h5py",
            "numpy",
            "pandas",
            "scanpy",
            "scipy",
            "pyscenic",
            "ctxcore",
            "numba",
            "nasp_compendium",
        )
    }
    scanpy_defaults = str(inspect.signature(sc.tl.score_genes))
    request = {
        "source": file_identity(spec.input_path),
        "counts_source": spec.counts_source,
        "selected_cells": selected_obs.index.astype(str).tolist(),
        "panel": file_identity(panel_path),
        "modules": [asdict(module) for module in modules],
        "feature_crosswalk_sha256": hashlib.sha256(
            features.to_csv(index=True).encode()
        ).hexdigest(),
        "coverage_sha256": hashlib.sha256(
            coverage.to_csv(index=False).encode()
        ).hexdigest(),
        "shared_gene_ids_sha256": request_identity(
            {
                "genes": list(shared_gene_ids),
            }
        ),
        "shared_gene_count": len(shared_gene_ids),
        "preparation": "full_source_library_size_then_log1p_then_shared_subset",
        "target_sum": target_sum,
        "scorers": ["scanpy", "aucell"],
        "random_state": random_state,
        "aucell_chunk_size": aucell_chunk_size,
        "aucell_num_workers": aucell_num_workers,
        "scanpy_score_genes_signature": scanpy_defaults,
        "aucell4r_signature": str(inspect.signature(aucell4r)),
        "python": sys.version,
        "compendium_install": json.loads(
            importlib.metadata.distribution("nasp_compendium").read_text(
                "direct_url.json"
            )
            or "null"
        ),
        "versions": versions,
        "implementation_sha256": code,
        "expression_dtype": str(np.dtype(np.float64)),
    }
    # Immutable definitions must round-trip with JSON arrays replacing tuples.
    return json.loads(json.dumps(request, allow_nan=False))
