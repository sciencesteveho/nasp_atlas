"""Curated sensor identities and explicit source-feature resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nasp_compendium import GeneModules


__all__ = ["resolve_sensor_features", "sensor_catalog"]


def sensor_catalog(
    *,
    panel_path: str | Path | None = None,
    sensor_group: str = "nucleic_acid_sensors",
) -> pd.DataFrame:
    """Return one canonical row per annotated sensor, without dataset filtering.

    Membership and aliases come only from the compendium public catalog.
    Module memberships annotate a sensor; they never duplicate measurements.
    """
    catalog = GeneModules(panel_path=panel_path)
    genes = catalog.get_sensors(sensor_group)
    if not genes:
        raise ValueError(
            f"No {sensor_group} annotations in {catalog.panel_path}"
        )
    records = []
    for gene in sorted(genes):
        rows = catalog.panel.loc[catalog.panel.gene_symbol.eq(gene)]
        record = {"gene": gene}
        for column, output in (
            ("sensor", "sensor_class"),
            ("aliases", "aliases"),
            ("module_id", "module_memberships"),
        ):
            values = rows[column].dropna() if column in rows else []
            record[output] = ";".join(
                sorted(
                    {
                        token.strip()
                        for value in values
                        for token in str(value).split(";")
                        if token.strip()
                    }
                )
            )
        records.append(record)
    return pd.DataFrame(
        records,
        columns=["gene", "sensor_class", "aliases", "module_memberships"],
    )


def resolve_sensor_features(
    var: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    gene_symbol_column: str = "feature_name",
) -> pd.DataFrame:
    """Resolve canonical names/aliases; retain absent and ambiguous rows.

    A symbol mapping to multiple source features, or multiple requested genes
    mapping to the same feature, is unavailable. Inputs are not modified.
    """
    if not var.index.is_unique or catalog.gene.duplicated().any():
        raise ValueError("Source feature IDs and catalog genes must be unique")
    symbols = (
        var[gene_symbol_column].astype("string")
        if gene_symbol_column in var
        else pd.Series(var.index.astype(str), index=var.index, dtype="string")
    )
    lookup: dict[str, set[str]] = {}
    for identifier, symbol in symbols.items():
        for label in (
            str(identifier),
            str(symbol).strip() if pd.notna(symbol) else "",
        ):
            if label:
                lookup.setdefault(label, set()).add(str(identifier))

    records = []
    for row in catalog.itertuples(index=False):
        labels = [str(row.gene), *str(row.aliases).split(";")]
        matches = set().union(*(lookup.get(label, set()) for label in labels))
        identifier = next(iter(matches)) if len(matches) == 1 else None
        records.append(
            {
                "gene": row.gene,
                "source_feature_id": identifier,
                "source_symbol": symbols.loc[identifier]
                if identifier is not None
                else None,
                "mapping_status": "present"
                if identifier is not None
                else ("ambiguous_identifier" if matches else "absent"),
            }
        )
    resolved = catalog.merge(
        pd.DataFrame(records), on="gene", validate="one_to_one"
    )
    shared = (
        resolved.source_feature_id.notna()
        & resolved.source_feature_id.duplicated(keep=False)
    )
    resolved.loc[shared, "mapping_status"] = "ambiguous_identifier"
    resolved.loc[shared, "source_feature_id"] = None
    return resolved
