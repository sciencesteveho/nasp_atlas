"""Reference-arm sensor profiles compared with a healthy atlas (AB8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import yaml  # type: ignore[import]
from scipy import stats

from nasp_atlas.analysis.condition_cohorts.units import SENSOR_MEASURES
from nasp_atlas.analysis.condition_cohorts.units import ConditionUnits
from nasp_atlas.single_cell.count_normalization import log_normalize_counts
from nasp_atlas.single_cell.io import read_h5ad_metadata
from nasp_atlas.single_cell.io import read_h5ad_rows
from nasp_atlas.single_cell.sensor_effects import sensor_donor_expression
from nasp_atlas.single_cell.sensor_features import sensor_catalog


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class AtlasCalibrationSpec:
    """Healthy-atlas cells matched to a cohort's reference arm.

    Attributes:
      atlas_id: Short atlas name used in outputs.
      input_filename: Atlas h5ad file name, found in a host data directory.
      input_sha256: Atlas checksum.
      tissue: Atlas obs `tissue` value to keep, such as "blood".
      assays: Atlas obs `assay` values to keep.
      gene_symbol_column: Atlas var column holding gene symbols.
      cell_types: Each cohort cell type with the atlas `cell_type` labels
        pooled to match it, declared before any sensor value is compared.
    """

    atlas_id: str
    input_filename: str
    input_sha256: str
    tissue: str
    assays: tuple[str, ...]
    gene_symbol_column: str
    cell_types: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class AtlasCalibration:
    """Reference-arm and atlas sensor profiles side by side.

    Attributes:
      profiles: One row per matched cell type, sensor and measure: donor
        medians in the cohort reference arm and in the atlas, with donors.
      agreement: One row per cell type and measure: Spearman correlation
        across sensors of the two donor-median profiles, with sensor and
        donor counts. Descriptive.
    """

    profiles: pd.DataFrame
    agreement: pd.DataFrame


def read_atlas_calibration_spec(path: str | Path) -> AtlasCalibrationSpec:
    """Read a strict YAML calibration request; unknown fields fail."""
    with Path(path).open() as handle:
        record = dict(yaml.safe_load(handle))

    try:
        cell_types = record.pop("cell_types")
        spec = AtlasCalibrationSpec(
            atlas_id=str(record.pop("atlas_id")),
            input_filename=str(record.pop("input_filename")),
            input_sha256=str(record.pop("input_sha256")),
            tissue=str(record.pop("tissue")),
            assays=tuple(str(value) for value in record.pop("assays")),
            gene_symbol_column=str(record.pop("gene_symbol_column")),
            cell_types=tuple(
                (str(cohort), tuple(str(label) for label in labels))
                for cohort, labels in dict(cell_types).items()
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid atlas calibration spec {path}: {error}"
        ) from error

    if record:
        raise ValueError(f"Unknown calibration fields: {sorted(record)}")

    labels = [label for _, values in spec.cell_types for label in values]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("Each atlas cell type may match one cohort type only")
    return spec


def atlas_sensor_profiles(
    spec: AtlasCalibrationSpec,
    *,
    input_path: Path,
    minimum_cells: int = 10,
    target_sum: float = 10_000.0,
) -> pd.DataFrame:
    """Aggregate atlas sensor expression per donor and matched cell type.

    Raw counts of the selected tissue, assays and cell types are normalized
    to log1p CP10k over all genes, as cohort intake does, so both sides use
    the same scale. Returns one row per donor, matched cohort cell type and
    sensor with `mean_expression`, `fraction_detected`, cells and mapping
    status; donor-cell-type groups below `minimum_cells` are dropped.
    """
    obs, var = read_h5ad_metadata(input_path)
    atlas_to_cohort = {
        label: cohort for cohort, labels in spec.cell_types for label in labels
    }

    selected = obs.loc[
        obs["tissue"].astype(str).eq(spec.tissue)
        & obs["assay"].astype(str).isin(spec.assays)
        & obs["cell_type"].astype(str).isin(atlas_to_cohort)
    ]
    logger.info(
        "Atlas cells selected: %d from %d donors",
        len(selected),
        selected["donor_id"].nunique(),
    )

    source = read_h5ad_rows(
        input_path, selected.index.tolist(), read_x=False, read_raw=True
    )
    if source.raw is None:
        raise ValueError(f"No raw counts in {input_path}")

    normalized = log_normalize_counts(
        source.raw.X,
        target_sum=target_sum,
        source=f"{input_path}:raw/X",
        dtype=np.float32,
    )

    raw_var = source.raw.var
    expression = ad.AnnData(
        X=normalized.expression.tocsc(),
        obs=pd.DataFrame(
            {
                "donor_id": selected["donor_id"].astype(str).to_numpy(),
                "cell_type": selected["cell_type"]
                .astype(str)
                .map(atlas_to_cohort)
                .to_numpy(),
            },
            index=selected.index.astype(str),
        ),
        var=var.reindex(raw_var.index)[[spec.gene_symbol_column]],
    )

    genes, _ = sensor_donor_expression(
        expression,
        sensor_catalog(),
        group_keys=("donor_id", "cell_type"),
        gene_symbol_column=spec.gene_symbol_column,
    )
    return genes.loc[genes["n_cells"].ge(minimum_cells)]


def calibrate_reference_arm(
    units: ConditionUnits,
    atlas: pd.DataFrame,
    *,
    reference: str,
    minimum_donors: int = 3,
) -> AtlasCalibration:
    """Compare reference-arm and atlas donor-median sensor profiles.

    Cohort values are supported reference-arm units averaged per donor;
    atlas values are `atlas_sensor_profiles` rows. For each matched cell
    type and measure, the donor medians of each sensor are compared, and
    their Spearman correlation across sensors summarizes agreement. Atlas
    and cohort differ in donors, chemistry and processing, so agreement is
    descriptive: high correlation says the reference arm ranks sensors as
    the healthy atlas does, not that absolute levels match.
    """
    cohort = _reference_donor_values(units, reference)
    atlas_long = _atlas_long(atlas)

    profiles = _donor_medians(cohort, "cohort").merge(
        _donor_medians(atlas_long, "atlas"),
        on=["feature_type", "gene", "cell_type"],
        validate="one_to_one",
    )
    profiles = profiles.loc[
        profiles["cohort_donors"].ge(minimum_donors)
        & profiles["atlas_donors"].ge(minimum_donors)
    ]

    return AtlasCalibration(
        profiles=profiles.reset_index(drop=True),
        agreement=_agreement(profiles),
    )


def _reference_donor_values(
    units: ConditionUnits,
    reference: str,
) -> pd.DataFrame:
    """Average reference-arm sensor units per donor, cell type and measure."""
    metadata = units.metadata.loc[
        units.metadata["supported"] & units.metadata["condition"].eq(reference),
        ["donor_id", "sample_id", "cell_type"],
    ]
    return (
        units.features.loc[
            units.features["feature_type"].isin(SENSOR_MEASURES)
            & units.features["status"].isin(["present", "undetected"])
        ]
        .merge(metadata, on=["donor_id", "sample_id", "cell_type"])
        .groupby(["feature_type", "feature_label", "cell_type", "donor_id"])[
            "feature_value"
        ]
        .mean()
        .reset_index()
        .rename(columns={"feature_label": "gene"})
    )


def _atlas_long(atlas: pd.DataFrame) -> pd.DataFrame:
    """Reshape measured atlas profiles to the cohort's feature types."""
    atlas_long = atlas.loc[
        atlas["status"].isin(["present", "undetected"])
    ].melt(
        id_vars=["donor_id", "cell_type", "gene"],
        value_vars=["mean_expression", "fraction_detected"],
        var_name="feature_type",
        value_name="feature_value",
    )
    atlas_long["feature_type"] = atlas_long["feature_type"].map(
        dict(
            zip(
                ("mean_expression", "fraction_detected"),
                SENSOR_MEASURES,
                strict=True,
            )
        )
    )
    return atlas_long


def _donor_medians(values: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return the donor median and donor count per sensor and cell type."""
    return (
        values.groupby(["feature_type", "gene", "cell_type"])
        .agg(
            median=("feature_value", "median"),
            donors=("donor_id", "nunique"),
        )
        .add_prefix(f"{prefix}_")
        .reset_index()
    )


def _agreement(profiles: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of the two profiles per cell type and measure."""
    rows = []
    for (measure, cell_type), group in profiles.groupby(
        ["feature_type", "cell_type"]
    ):
        result = stats.spearmanr(group["cohort_median"], group["atlas_median"])
        rows.append(
            {
                "feature_type": measure,
                "cell_type": cell_type,
                "spearman_rho": float(result.statistic),  # type: ignore[attr-defined]
                "n_sensors": len(group),
                "cohort_donors": int(group["cohort_donors"].min()),
                "atlas_donors": int(group["atlas_donors"].min()),
            }
        )

    return pd.DataFrame(rows)
