"""Query CELLxGENE for dataset metadata."""

import re

import cellxgene_census  # type: ignore
from pandas import DataFrame, Series  # type: ignore


CENSUS_VERSION: str = "stable"

OBS_COLS: list[str] = [
    "dataset_id",
    "donor_id",
    "assay",
    "cell_type",
    "development_stage",
    "disease",
    "sex",
    "tissue",
    "suspension_type",
]

DATASET_COLS: list[str] = [
    "dataset_id",
    "collection_name",
    "collection_id",
    "dataset_total_cell_count",
    "dataset_h5ad_path",
]

DISEASE_KEYWORDS: list[str] = [
    "normal",
    "age",
    "aging",
    "alzheimer",
    "dementia",
    "parkinson",
    "lewy",
    "frontotemporal",
    "tauopathy",
    "amyotrophic lateral sclerosis",
    "multiple sclerosis",
    "macular degeneration",
    "glaucoma",
    "cataract",
    "atherosclerosis",
    "cardiomyopathy",
    "heart failure",
    "myocardial",
    "coronary",
    "atrial fibrillation",
    "diabetes",
    "prediabetes",
    "chronic kidney",
    "kidney failure",
    "covid",
    "influenza",
    "pneumonia",
    "pulmonary fibrosis",
    "interstitial lung",
    "chronic obstructive pulmonary disease",
    "emphysema",
    "sarcoidosis",
    "inflammatory",
    "inflammation",
    "crohn",
    "colitis",
    "celiac",
    "gastritis",
    "periodontitis",
    "gingivitis",
    "arthritis",
    "lupus",
    "sjogren",
    "dermatomyositis",
    "scleroderma",
    "fibrosis",
    "injury",
    "keloid",
    "clonal hematopoiesis",
    "myelodysplastic",
    "myeloproliferative",
    "leukemia",
    "lymphoma",
    "myeloma",
    "cancer",
    "carcinoma",
    "melanoma",
    "neoplasm",
    "tumor",
    "glioblastoma",
    "blastoma",
]


def _collapse_unique_series(values: Series, delimiter: str = ", ") -> str:
    """Collapse unique series values into a delimiter-separated string."""
    vals = sorted(values.dropna().astype(str).unique())
    return delimiter.join(vals)


def _filter_datasets_with_relevant_disease(obs: DataFrame) -> DataFrame:
    """Keep cells from datasets with normal or aging/inflammation-relevant
    disease annotations.
    """
    disease_pattern = "|".join(
        re.escape(keyword) for keyword in DISEASE_KEYWORDS
    )

    relevant_dataset_ids = obs.loc[
        obs["disease"]
        .astype(str)
        .str.lower()
        .str.contains(disease_pattern, na=False),
        "dataset_id",
    ].unique()

    return obs[obs["dataset_id"].isin(relevant_dataset_ids)].copy()


def _collapse_sex_series(values: Series) -> str:
    """Collapse sex annotations merging male and female when both are
    present.
    """
    vals = set(values.dropna().astype(str).str.strip().str.lower())
    collapsed: list[str] = []

    if {"male", "female"}.issubset(vals):
        collapsed.append("male & female")
        vals.discard("male")
        vals.discard("female")

    collapsed.extend(sorted(vals))

    return ", ".join(collapsed)


def _stage_age_value(stage: str) -> float:
    """Return an approximate numeric age for sorting development stages."""
    value = stage.strip().lower()

    if value == "unknown":
        return float("inf")

    match = re.search(r"carnegie stage\s+(\d+)", value)
    if match:
        return -1.0 + int(match.group(1)) / 100.0

    match = re.search(r"(\d+)(?:st|nd|rd|th)? week post-fertilization", value)
    if match:
        return -0.75 + int(match.group(1)) / 100.0

    match = re.search(r"(\w+) lmp month", value)
    if match:
        month_map = {
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
        }
        return -0.5 + month_map.get(match.group(1), 0) / 100.0

    if (
        "embryonic" in value
        or "organogenesis" in value
        or "blastula" in value
        or "prenatal" in value
    ):
        return -0.25

    if "newborn" in value:
        return 0.0

    match = re.search(r"(\d+)-month-old", value)
    if match:
        return int(match.group(1)) / 12.0

    match = re.search(r"(\d+)-year-old", value)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+)\s*year-old and over", value)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+)-(\d+)\s*year-old", value)
    if match:
        return float(match.group(1))

    decade_map = {
        "third decade": 20.0,
        "fourth decade": 30.0,
        "fifth decade": 40.0,
        "sixth decade": 50.0,
        "seventh decade": 60.0,
        "eighth decade": 70.0,
        "ninth decade": 80.0,
        "tenth decade": 90.0,
    }

    for key, age in decade_map.items():
        if key in value:
            return age

    stage_map = {
        "nursing stage": 0.0,
        "infant stage": 0.0,
        "child stage": 1.0,
        "juvenile stage": 5.0,
        "pediatric stage": 1.0,
        "postnatal stage": 0.0,
        "young adult stage": 20.0,
        "adult stage": 18.0,
        "prime adult stage": 30.0,
        "middle aged stage": 45.0,
        "late adult stage": 60.0,
    }

    for key, age in stage_map.items():
        if key in value:
            return age

    return float("inf") - 1.0


def _stage_range_label(age: float, stage: str) -> str:
    """Format an approximate age for the development-stage range prefix."""
    value = stage.strip().lower()

    if age < 0:
        return "prenatal"

    if "newborn" in value or age == 0:
        return "neonatal"

    if age < 1:
        return f"{round(age * 12):g}mo"

    if "year-old and over" in value or "and over" in value:
        return f"{int(age)}yo+"

    return f"{int(age)}yo"


def _summarize_development_stage(values: Series, delimiter: str = ", ") -> str:
    """Summarize development stages as range plus sorted stage counts."""
    counts = values.dropna().astype(str).str.strip().value_counts()
    counts = counts[counts.index != ""]

    if counts.empty:
        return ""

    ordered = sorted(
        counts.index, key=lambda stage: (_stage_age_value(stage), stage)
    )

    known = [
        (stage, _stage_age_value(stage))
        for stage in ordered
        if _stage_age_value(stage) != float("inf")
        and _stage_age_value(stage) != float("inf") - 1.0
    ]

    if known:
        min_stage, min_age = min(known, key=lambda item: item[1])
        max_stage, max_age = max(known, key=lambda item: item[1])
        range_prefix = f"{_stage_range_label(min_age, min_stage)}"
        range_prefix += f"-{_stage_range_label(max_age, max_stage)}"
    else:
        range_prefix = "unknown"

    terms = [f"{stage} ({counts[stage]})" for stage in ordered]

    return delimiter.join([range_prefix, *terms])


def _read_cxg_census_metadata(
    organism: str = "homo_sapiens",
) -> tuple[DataFrame, DataFrame]:
    """Read CELLxGENE CENSUS metadata at the dataset and cell levels. As the
    metadata exists at separate levels, we read the metadata out at the cell
    level before collapsing back into dataset level downstream.

    Returns:
      [DataFrame, DataFrame]: The first DF consists of dataset-level CxG
      metadata. The second DF consists of cell-level CxG metadata to be further
      collapsed.
    """
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

    try:
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()

        obs = (
            census["census_data"][organism]
            .obs.read(
                value_filter="is_primary_data == True",
                column_names=OBS_COLS,
            )
            .concat()
            .to_pandas()
        )

    finally:
        census.close()

    return datasets, obs


def _summarize_dataset(datasets: DataFrame, obs: DataFrame) -> DataFrame:
    """Summarize primary-cell metadata by dataset.

    Aggregates per-cell CELLxGENE Census obs metadata to one row per dataset_id,
    including cell count, donor count, cell-type count, and collapsed
    categorical annotations. Dataset-level metadata from census_info["datasets"]
    is then merged onto the summary.

    Args:
      datasets: Dataset-level Census metadata from census_info["datasets"].
      obs: Primary-cell metadata from census_data[organism].obs.

    Returns:
      Dataset-level summary table with one row per dataset.
    """
    dataset_summary = (
        obs.groupby("dataset_id", observed=True)
        .agg(
            n_cells=("dataset_id", "size"),
            n_donors=("donor_id", "nunique"),
            n_cell_types=("cell_type", "nunique"),
            assays=("assay", _collapse_unique_series),
            tissues=("tissue", _collapse_unique_series),
            diseases=("disease", _collapse_unique_series),
            sexes=("sex", _collapse_sex_series),
            age_terms=("development_stage", _summarize_development_stage),
            suspension_types=("suspension_type", _collapse_unique_series),
        )
        .reset_index()
    )

    dataset_cols = [col for col in DATASET_COLS if col in datasets.columns]

    dataset_summary = dataset_summary.merge(
        datasets[dataset_cols],
        on="dataset_id",
        how="left",
    )

    front_cols = [
        col
        for col in [
            "collection_name",
            "dataset_id",
            "n_cells",
            "n_donors",
            "n_cell_types",
            "assays",
            "tissues",
            "diseases",
            "sexes",
            "age_terms",
            "suspension_types",
        ]
        if col in dataset_summary.columns
    ]

    dataset_summary = dataset_summary[
        front_cols
        + [col for col in dataset_summary.columns if col not in front_cols]
    ]

    dataset_summary = dataset_summary.sort_values(
        ["n_cells", "n_donors"],
        ascending=False,
    )

    return dataset_summary


def main() -> None:
    datasets, obs = _read_cxg_census_metadata()
    obs = _filter_datasets_with_relevant_disease(obs)
    aggregated_dataset = _summarize_dataset(datasets=datasets, obs=obs)
    aggregated_dataset.to_csv("cellxgene_human.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
