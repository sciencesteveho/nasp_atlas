"""Per-sensor evidence across condition cohorts, with assay-specific tiers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nasp_atlas.analysis.condition_cohorts.units import SENSOR_MEASURES
from nasp_atlas.analysis.stage_artifacts import read_published_table


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetEvidence:
    """Evidence axes per sensor and the tiers derived from them.

    Attributes:
      long: One row per sensor, cohort, contrast, cell type and evidence
        axis, with the estimate, interval, adjusted p-value, support and
        the flag the axis contributes.
      summary: One row per sensor of the analysed sensor universe: flagged
        cell types per axis, breadth and direction of modulation,
        essentiality and protein annotations, and one tier per assay type.
      sensors: One row per sensor and cohort: whether the cohort measured it.
    """

    long: pd.DataFrame
    summary: pd.DataFrame
    sensors: pd.DataFrame


def summarize_target_evidence(
    run_dirs: Sequence[Path],
    *,
    annotation_panel: Path,
    alpha: float = 0.05,
    sirna_detection: float = 0.10,
    crispri_detection: float = 0.25,
    donor_fraction: float = 0.75,
    minimum_donors: int = 3,
) -> TargetEvidence:
    """Combine published cohort results into per-sensor evidence and tiers.

    Only stage tables whose bytes match their completion manifest are read;
    nothing is refitted. The sensor universe is the union of the sensors
    each run analysed. For each primary contrast of each run:

    - **Assayability (E1):** in the disease arm, the share of supported
      donors whose detected fraction in a cell type reaches the siRNA or
      CRISPRi threshold. A cell type is assayable when that share is at
      least `donor_fraction` over at least `minimum_donors` donors.
    - **Disease modulation (E2):** level or detection effect with BH-FDR
      below `alpha`.
    - **IFN independence (E3):** a modulated effect whose IFN-adjusted
      estimate keeps its sign with an interval excluding zero (AB3), or,
      when the run has IFN strata, that is also modulated in the same
      direction in IFN-low disease donors (AB2).
    - **Within-sample coupling (E4):** a disease-arm within-sample slope to
      an output module with BH-FDR below `alpha`.
    - **Essentiality (E7):** DepMap common-essential and s_het from
      `annotation_panel`; absent annotations stay missing.

    Tiers are rules over these flags, not a composite score:

    - drug discovery: "priority" when modulated and either IFN-independent
      or within-coupled, and not common-essential; "candidate" when
      modulated; otherwise "context";
    - siRNA and CRISPRi: "priority" when assayable at the assay's threshold
      in a cell type where the sensor is modulated; "assayable" when
      assayable anywhere; otherwise "not_assayable". CRISPRi also flags
      common-essential genes, which drop out of pooled screens.

    A sensor no run measured is "not_measured" in every tier. Unavailable
    and non-estimable rows remain in `long` with their status; they never
    count as negative evidence.
    """
    if not 0 < alpha < 1 or not 0 < donor_fraction <= 1:
        raise ValueError("alpha and donor_fraction must lie in (0, 1]")

    runs = [
        _run_evidence(
            run_dir,
            alpha=alpha,
            sirna_detection=sirna_detection,
            crispri_detection=crispri_detection,
            donor_fraction=donor_fraction,
            minimum_donors=minimum_donors,
        )
        for run_dir in run_dirs
    ]

    long = pd.concat([evidence for evidence, _ in runs], ignore_index=True)
    sensors = pd.concat([measured for _, measured in runs], ignore_index=True)
    annotations = _panel_annotations(
        annotation_panel, sorted(sensors["gene"].unique())
    )

    return TargetEvidence(
        long=long,
        summary=_assign_tiers(_axis_summary(long, annotations, sensors)),
        sensors=sensors,
    )


def evidence_matrix(evidence: TargetEvidence) -> pd.DataFrame:
    """Lay out each sensor's evidence per cohort as display categories.

    Returns one row per sensor and evidence column with a `category`:
    modulation direction ("up", "down", "mixed", "none"), flags ("yes",
    "no"), assayability ("in_modulated", "elsewhere", "no"), essentiality
    and tiers, or "not_measured". Sensors keep the summary's order.
    """
    long, summary = evidence.long, evidence.summary
    cohorts = list(dict.fromkeys(evidence.sensors["cohort"]))
    measured = evidence.sensors.set_index(["gene", "cohort"])["measured"]

    rows = []
    for gene in summary["gene"]:
        gene_rows = long.loc[long["gene"].eq(gene)]
        for cohort in cohorts:
            if not bool(measured.get((gene, cohort), False)):
                rows.extend(
                    {
                        "gene": gene,
                        "cohort": cohort,
                        "column": column,
                        "category": "not_measured",
                    }
                    for column in _cohort_columns()
                )
                continue
            rows.extend(
                {"gene": gene, "cohort": cohort, **cell}
                for cell in _cohort_cells(
                    gene_rows.loc[gene_rows["cohort"].eq(cohort)]
                )
            )

    rows.extend(_shared_cells(summary))
    return pd.DataFrame(rows)


def _shared_cells(summary: pd.DataFrame) -> list[dict[str, str]]:
    """Cross-cohort matrix cells: essentiality and the three tiers."""
    shared = summary.set_index("gene")
    rows = []
    for gene in summary["gene"]:
        essential = shared.at[gene, "depmap_essentiality"]
        rows.append(
            {
                "gene": gene,
                "cohort": "all",
                "column": "common_essential",
                "category": (
                    "missing"
                    if pd.isna(essential)
                    else ("yes" if str(essential) == "yes" else "no")
                ),
            }
        )
        rows.extend(
            {
                "gene": gene,
                "cohort": "all",
                "column": tier,
                "category": str(shared.at[gene, tier]),
            }
            for tier in ("drug_discovery_tier", "sirna_tier", "crispri_tier")
        )

    return rows


def _cohort_columns() -> tuple[str, ...]:
    """Evidence columns shown for each cohort, in display order."""
    return (
        "modulation",
        "ifn_independent",
        "within_coupled",
        "sirna_assayable",
        "crispri_assayable",
    )


def _cohort_cells(rows: pd.DataFrame) -> list[dict[str, str]]:
    """Classify one sensor's evidence within one cohort."""
    flagged = rows.loc[rows["flag"].fillna(False).astype(bool)]
    modulated = flagged.loc[flagged["axis"].eq("disease_modulation")]
    level = modulated.loc[modulated["measure"].eq("gene_expression")]

    signs = np.sign(
        (level if not level.empty else modulated)["estimate"].to_numpy()
    )
    direction = (
        "none"
        if signs.size == 0
        else "up"
        if (signs > 0).all()
        else "down"
        if (signs < 0).all()
        else "mixed"
    )

    changed = set(modulated["cell_type"].dropna())
    cells = {
        "modulation": direction,
        "ifn_independent": "yes"
        if modulated["ifn_independent"].fillna(False).astype(bool).any()
        else "no",
        "within_coupled": "yes"
        if flagged["axis"].eq("within_sample_coupling").any()
        else "no",
    }

    for assay in ("sirna", "crispri"):
        assayable = set(
            flagged.loc[
                flagged["axis"].eq(f"assayable_{assay}"), "cell_type"
            ].dropna()
        )
        cells[f"{assay}_assayable"] = (
            "in_modulated"
            if assayable & changed
            else ("elsewhere" if assayable else "no")
        )

    return [
        {"column": column, "category": category}
        for column, category in cells.items()
    ]


def _run_evidence(
    run_dir: Path,
    *,
    alpha: float,
    sirna_detection: float,
    crispri_detection: float,
    donor_fraction: float,
    minimum_donors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect every evidence axis and the measured sensors for one run."""
    cohort = f"{run_dir.parent.name}/{run_dir.name}"
    primary = _primary_sensor_effects(run_dir)

    adjusted = read_published_table(
        run_dir / "ifn_adjusted_effects",
        "ifn_adjusted_contrasts.csv",
        stage="ifn_adjusted_effects",
    )
    strata = (
        read_published_table(
            run_dir / "ifn_strata",
            "ifn_strata_contrasts.csv",
            stage="ifn_strata",
        )
        if (run_dir / "ifn_strata").is_dir()
        else None
    )
    coupling = read_published_table(
        run_dir / "condition_coupling",
        "coupling_by_arm.csv",
        stage="condition_coupling",
    )
    metadata, features, mapping = (
        read_published_table(run_dir / "units", name, stage="units")
        for name in (
            "unit_metadata.csv",
            "unit_features.csv.gz",
            "sensor_mapping.csv",
        )
    )

    frames = [
        _modulation(primary, adjusted, strata, alpha=alpha),
        _assayability(
            metadata,
            features,
            contrasts=primary[["estimand", "target"]].drop_duplicates(),
            thresholds={"sirna": sirna_detection, "crispri": crispri_detection},
            donor_fraction=donor_fraction,
            minimum_donors=minimum_donors,
        ),
        _within_coupling(
            coupling,
            arms=primary["target"].dropna().unique().tolist(),
            alpha=alpha,
        ),
    ]

    evidence = pd.concat(frames, ignore_index=True).assign(cohort=cohort)
    return evidence, _measured_sensors(mapping, cohort)


def _primary_sensor_effects(run_dir: Path) -> pd.DataFrame:
    """Read a run's estimable primary condition contrasts of sensors.

    Target evidence combines disease or exposure contrasts only. A run
    without one, such as a population read for a within-donor change, is
    refused rather than contributing nothing silently.
    """
    effects = read_published_table(
        run_dir / "condition_effects",
        "condition_contrasts.csv",
        stage="condition_effects",
    )

    primary = effects.loc[
        effects["role"].eq("primary")
        & effects["feature_type"].isin(SENSOR_MEASURES)
        & effects["cell_type"].notna()
    ]

    if primary.empty:
        raise ValueError(
            f"{run_dir} has no estimable primary condition contrast of "
            "sensors; target evidence combines disease or exposure "
            "contrasts, so leave this run out"
        )
    return primary


def _measured_sensors(mapping: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Which compendium sensors a cohort measured, with mapping status."""
    return pd.DataFrame(
        {
            "gene": mapping["gene"],
            "cohort": cohort,
            "measured": mapping["mapping_status"].eq("present"),
            "mapping_status": mapping["mapping_status"],
        }
    )


def _modulation(
    primary: pd.DataFrame,
    adjusted: pd.DataFrame,
    strata: pd.DataFrame | None,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Flag modulated sensors and whether the shift survives IFN checks."""
    keys = ["estimand", "feature_type", "feature_label", "cell_type"]
    columns = [*keys, "estimate", "ci_low", "ci_high", "pvalue_fdr", "status"]
    merged = primary[[*columns, "target", "reference", "n_level_donors"]].merge(
        adjusted.loc[adjusted["feature_type"].isin(SENSOR_MEASURES), columns],
        on=keys,
        how="left",
        suffixes=("", "_adjusted"),
        validate="one_to_one",
    )

    modulated = merged["status"].eq("ok") & merged["pvalue_fdr"].lt(alpha)
    adjusted_holds = (
        merged["status_adjusted"].eq("ok")
        & (np.sign(merged["estimate"]) == np.sign(merged["estimate_adjusted"]))
        & ((merged["ci_low_adjusted"] > 0) | (merged["ci_high_adjusted"] < 0))
    )

    if strata is None:
        merged = merged.assign(
            ifn_low_estimate=np.nan, ifn_low_pvalue_fdr=np.nan
        )
    else:
        merged = merged.merge(
            _ifn_low_rows(strata),
            on=[
                "target",
                "reference",
                "feature_type",
                "feature_label",
                "cell_type",
            ],
            how="left",
            validate="one_to_one",
        )

    # Missing IFN-low results compare as False, never as agreement.
    ifn_low_holds = merged["ifn_low_pvalue_fdr"].lt(alpha) & (
        np.sign(merged["ifn_low_estimate"]) == np.sign(merged["estimate"])
    )

    return merged.assign(
        axis="disease_modulation",
        flag=modulated,
        ifn_adjusted_holds=modulated & adjusted_holds,
        ifn_low_holds=modulated & ifn_low_holds,
        ifn_independent=modulated & (adjusted_holds | ifn_low_holds),
    ).rename(
        columns={
            "feature_label": "gene",
            "target": "disease_arm",
            "feature_type": "measure",
            "estimand": "contrast",
        }
    )


def _ifn_low_rows(strata: pd.DataFrame) -> pd.DataFrame:
    """Return IFN-low-versus-reference sensor rows keyed by the parent arm."""
    low = strata.loc[
        strata["target"].astype(str).str.endswith("_ifn_low")
        & strata["feature_type"].isin(SENSOR_MEASURES)
    ]

    return low.assign(
        target=low["target"].str.removesuffix("_ifn_low"),
    )[
        [
            "target",
            "reference",
            "feature_type",
            "feature_label",
            "cell_type",
            "estimate",
            "pvalue_fdr",
        ]
    ].rename(
        columns={
            "estimate": "ifn_low_estimate",
            "pvalue_fdr": "ifn_low_pvalue_fdr",
        }
    )


def _assayability(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    *,
    contrasts: pd.DataFrame,
    thresholds: dict[str, float],
    donor_fraction: float,
    minimum_donors: int,
) -> pd.DataFrame:
    """Score per-cell-type detection breadth across donors of each arm."""
    detection = features.loc[
        features["feature_type"].eq("gene_detection")
        & features["status"].isin(["present", "undetected"])
    ].merge(
        metadata.loc[
            metadata["supported"],
            ["donor_id", "sample_id", "cell_type", "condition"],
        ],
        on=["donor_id", "sample_id", "cell_type"],
        validate="many_to_one",
    )

    donors = (
        detection.groupby(
            ["feature_label", "cell_type", "condition", "donor_id"],
            observed=True,
        )["feature_value"]
        .mean()
        .reset_index()
    )

    rows = []
    for contrast in contrasts.itertuples(index=False):
        arm = donors.loc[donors["condition"].eq(contrast.target)]
        for assay, threshold in thresholds.items():
            summary = (
                arm.assign(passes=arm["feature_value"].ge(threshold))
                .groupby(["feature_label", "cell_type"], observed=True)
                .agg(
                    n_donors=("donor_id", "nunique"),
                    share_passing=("passes", "mean"),
                    median_detection=("feature_value", "median"),
                )
                .reset_index()
            )
            rows.append(
                summary.assign(
                    contrast=contrast.estimand,
                    disease_arm=contrast.target,
                    axis=f"assayable_{assay}",
                    threshold=threshold,
                    flag=summary["n_donors"].ge(minimum_donors)
                    & summary["share_passing"].ge(donor_fraction),
                )
            )

    return pd.concat(rows, ignore_index=True).rename(
        columns={"feature_label": "gene"}
    )


def _within_coupling(
    coupling: pd.DataFrame,
    *,
    arms: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    """Flag sensors with disease-arm within-sample coupling to an output."""
    within = coupling.loc[
        coupling["estimand"].eq("within_sample_slope")
        & coupling["condition"].isin(arms)
    ]

    return within.assign(
        axis="within_sample_coupling",
        flag=within["status"].eq("ok") & within["pvalue_fdr"].lt(alpha),
        disease_arm=within["condition"],
    ).rename(columns={"predictor": "gene"})[
        [
            "gene",
            "response",
            "disease_arm",
            "axis",
            "estimate",
            "ci_low",
            "ci_high",
            "pvalue_fdr",
            "status",
            "flag",
        ]
    ]


def _panel_annotations(panel: Path, genes: Sequence[str]) -> pd.DataFrame:
    """Read one row of essentiality and protein annotations per sensor.

    Columns absent from an older panel are kept as missing so the summary
    shows the annotation as unavailable rather than favourable.
    """
    columns = [
        "depmap_essentiality",
        "s_het_mean",
        "s_het_lower_95",
        "s_het_upper_95",
        "monomer_kDa",
        "tractability",
        "precedent_construct",
        "disorder_split",
    ]

    table = pd.read_csv(panel, sep="\t")
    annotations = pd.DataFrame({"gene": list(genes)})
    for column in columns:
        values = (
            table.loc[table[column].notna()]
            .drop_duplicates("gene_symbol")
            .set_index("gene_symbol")[column]
            if column in table
            else pd.Series(dtype=object)
        )
        annotations[column] = annotations["gene"].map(values)

    annotations["annotation_panel"] = str(panel)
    return annotations


def _axis_summary(
    long: pd.DataFrame,
    annotations: pd.DataFrame,
    sensors: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each sensor's flagged cell types, breadth and direction."""
    flags = long.loc[long["flag"].fillna(False).astype(bool)]
    modulated = flags.loc[flags["axis"].eq("disease_modulation")]

    def joined(frame: pd.DataFrame, column: str) -> pd.Series:
        return frame.groupby("gene")[column].agg(
            lambda values: ";".join(sorted(set(values.dropna().astype(str))))
        )

    summary = annotations.set_index("gene")
    summary["modulated_cell_types"] = joined(modulated, "cell_type")
    summary["ifn_independent_cell_types"] = joined(
        modulated.loc[modulated["ifn_independent"].fillna(False).astype(bool)],
        "cell_type",
    )
    summary["within_coupled_responses"] = joined(
        flags.loc[flags["axis"].eq("within_sample_coupling")], "response"
    )

    for assay in ("sirna", "crispri"):
        summary[f"{assay}_assayable_cell_types"] = joined(
            flags.loc[flags["axis"].eq(f"assayable_{assay}")], "cell_type"
        )

    # Cell-type names are cohort-specific (lupus and PsychAD both have
    # "Prolif"), so assay tiers match assayable and modulated cell types
    # within a cohort.
    contexts = flags.assign(context=flags["cohort"] + "::" + flags["cell_type"])
    summary["_modulated_contexts"] = joined(
        contexts.loc[contexts["axis"].eq("disease_modulation")], "context"
    )
    for assay in ("sirna", "crispri"):
        summary[f"_{assay}_assayable_contexts"] = joined(
            contexts.loc[contexts["axis"].eq(f"assayable_{assay}")], "context"
        )

    summary["cohorts_modulated"] = joined(modulated, "cohort")
    summary["n_cohorts_modulated"] = modulated.groupby("gene")[
        "cohort"
    ].nunique()
    summary["n_modulated_contexts"] = modulated.groupby("gene").size()

    level = modulated.loc[modulated["measure"].eq("gene_expression")]
    summary["max_abs_level_effect"] = (
        level.assign(magnitude=level["estimate"].abs())
        .groupby("gene")["magnitude"]
        .max()
    )

    summary["level_direction_by_cohort"] = _level_directions(level)
    summary["measured"] = sensors.groupby("gene")["measured"].any()

    return summary.reset_index()


def _assign_tiers(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the assay tier rules and order sensors within tiers."""
    summary = summary.copy()

    def has(column: str) -> pd.Series:
        return summary[column].fillna("").astype(str).ne("")

    essential = summary["depmap_essentiality"].astype(str).eq("yes")
    is_modulated = has("modulated_cell_types")
    summary["drug_discovery_tier"] = np.select(
        [
            is_modulated
            & (
                has("ifn_independent_cell_types")
                | has("within_coupled_responses")
            )
            & ~essential,
            is_modulated,
        ],
        ["priority", "candidate"],
        default="context",
    )

    for assay in ("sirna", "crispri"):
        overlap = [
            bool(set(assayable.split(";")) & set(changed.split(";")) - {""})
            for assayable, changed in zip(
                summary[f"_{assay}_assayable_contexts"].fillna(""),
                summary["_modulated_contexts"].fillna(""),
                strict=True,
            )
        ]
        summary[f"{assay}_tier"] = np.select(
            [np.array(overlap), has(f"{assay}_assayable_cell_types")],
            ["priority", "assayable"],
            default="not_assayable",
        )

    summary["crispri_essentiality_caution"] = essential

    measured = summary["measured"].fillna(False).astype(bool)
    for column in ("drug_discovery_tier", "sirna_tier", "crispri_tier"):
        summary.loc[~measured, column] = "not_measured"

    order = summary["drug_discovery_tier"].map(
        {"priority": 0, "candidate": 1, "context": 2, "not_measured": 3}
    )

    # Breadth and size order genes within a tier; the rules do not use them.
    return (
        summary.assign(_order=order)
        .sort_values(
            ["_order", "n_cohorts_modulated", "max_abs_level_effect", "gene"],
            ascending=[True, False, False, True],
            na_position="last",
        )
        .drop(
            columns=[
                "_order",
                "_modulated_contexts",
                "_sirna_assayable_contexts",
                "_crispri_assayable_contexts",
            ]
        )
        .reset_index(drop=True)
    )


def _level_directions(level: pd.DataFrame) -> pd.Series:
    """Describe each cohort's direction of modulated sensor-level effects.

    A cohort is "up" or "down" when every modulated cell type agrees, and
    "mixed" otherwise, so modulation in two cohorts is not mistaken for
    agreement.
    """
    if level.empty:
        return pd.Series(dtype=object)

    signs = level.assign(up=level["estimate"].gt(0)).groupby(
        ["gene", "cohort"]
    )["up"]

    direction = np.select(
        [signs.all(), ~signs.any()], ["up", "down"], default="mixed"
    )

    labels = pd.Series(direction, index=signs.all().index).reset_index()
    labels.columns = ["gene", "cohort", "direction"]
    return (
        labels.assign(text=labels["cohort"] + ":" + labels["direction"])
        .groupby("gene")["text"]
        .agg(";".join)
    )
