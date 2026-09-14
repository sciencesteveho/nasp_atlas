"""Versioned, repository-shipped Reactome and Hallmark reference signatures."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import pandas as pd
from nasp_compendium.types import GeneModule  # type: ignore[import]


def reference_gene_sets() -> list[GeneModule]:
    """Load unsigned human reference signatures without network access."""
    text = files("nasp_atlas.data").joinpath("reference_sets.gmt").read_text()
    modules = []
    for line in text.splitlines():
        identifier, _url, *genes = line.split("\t")
        if not genes or len(genes) != len(set(genes)):
            raise ValueError(
                f"Invalid bundled reference signature: {identifier}"
            )
        modules.append(GeneModule(identifier, tuple(genes), (), (), "symbols"))
    return modules


def reference_metadata() -> pd.DataFrame:
    """Return source identifiers, releases, names, and download provenance."""
    text = files("nasp_atlas.data").joinpath("reference_sets.json").read_text()
    return pd.DataFrame(json.loads(text))


def reference_bundle_hash() -> str:
    """Identify both shipped gene membership and source metadata for resume."""
    digest = hashlib.sha256()
    for filename in ("reference_sets.gmt", "reference_sets.json"):
        digest.update(files("nasp_atlas.data").joinpath(filename).read_bytes())
    return digest.hexdigest()
