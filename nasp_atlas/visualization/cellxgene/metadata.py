"""Visualize CELLxGENE tissue and disease makeup."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


AnnotationType: TypeAlias = Literal["tissue", "disease"]
CategoryRules: TypeAlias = Mapping[str, Sequence[str]]


def _set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 5,
            "axes.titlesize": 5,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "figure.titlesize": 5,
            "figure.dpi": 450,
            "font.sans-serif": ["Arial", "Nimbus Sans"],
            "axes.linewidth": 0.25,
            "xtick.major.width": 0.25,
            "ytick.major.width": 0.25,
            "xtick.minor.width": 0.25,
            "ytick.minor.width": 0.25,
        }
    )


def default_tissue_rules() -> dict[str, list[str]]:
    """Return broad tissue annotation rules."""
    return {
        "Brain & nervous system": [
            "brain",
            "cortex",
            "cortical",
            "neocortex",
            "forebrain",
            "hindbrain",
            "midbrain",
            "telencephalon",
            "cerebell",
            "hippocamp",
            "hypothalamus",
            "thalamus",
            "spinal cord",
            "neural",
            "ganglion",
            "pons",
            "medulla",
            "striatum",
            "amygdala",
        ],
        "heart": [
            "heart",
            "cardiac",
            "atrium",
            "ventricle",
            "interventricular",
            "sinoatrial",
            "atrioventricular",
            "coronary",
        ],
        "blood_immune": [
            "blood",
            "bone marrow",
            "lymph",
            "spleen",
            "thymus",
            "tonsil",
            "cerebrospinal fluid",
        ],
        "lung_airway": [
            "lung",
            "bronch",
            "trachea",
            "alveol",
            "airway",
            "nasal",
            "nasopharynx",
            "oropharynx",
            "pleural",
        ],
        "gut": [
            "intestine",
            "colon",
            "caecum",
            "cecum",
            "ileum",
            "jejunum",
            "duodenum",
            "rectum",
            "stomach",
            "esophagus",
            "gastric",
            "pylor",
            "appendix",
        ],
        "liver_biliary": [
            "liver",
            "hepatic",
            "bile",
            "biliary",
            "gallbladder",
        ],
        "kidney_urinary": [
            "kidney",
            "renal",
            "ureter",
            "urinary bladder",
            "bladder organ",
        ],
        "eye": [
            "eye",
            "retina",
            "macula",
            "fovea",
            "cornea",
            "sclera",
            "iris",
            "lens",
            "choroid",
            "uvea",
            "optic",
        ],
        "skin": [
            "skin",
            "dermis",
            "epidermis",
            "scalp",
        ],
        "reproductive": [
            "ovary",
            "uterus",
            "endometrium",
            "myometrium",
            "fallopian",
            "placenta",
            "decidua",
            "testis",
            "prostate",
            "mammary",
            "breast",
            "gonad",
        ],
        "endocrine_pancreas": [
            "adrenal",
            "pancreas",
            "islet",
            "thyroid",
        ],
        "musculoskeletal": [
            "muscle",
            "tendon",
            "bone",
            "skeletal",
            "cartilage",
            "rib",
        ],
        "adipose": [
            "adipose",
            "fat",
        ],
        "vascular": [
            "aorta",
            "artery",
            "vein",
            "vasculature",
            "vessel",
        ],
        "oral_salivary": [
            "oral",
            "tongue",
            "gingiva",
            "dental",
            "salivary",
            "parotid",
            "submandibular",
            "sublingual",
            "palate",
            "lip",
        ],
        "developmental_model": [
            "embryoid",
            "embryo",
            "trophoblast",
            "yolk sac",
            "neural tube",
        ],
    }


def default_disease_rules() -> dict[str, list[str]]:
    """Return broad disease annotation rules."""
    return {
        "normal": [
            r"^normal$",
        ],
        "cancer": [
            "cancer",
            "carcinoma",
            "adenocarcinoma",
            "melanoma",
            "neoplasm",
            "tumor",
            "blastoma",
            "glioblastoma",
            "leukemia",
            "lymphoma",
            "myeloma",
        ],
        "neurodegeneration": [
            "alzheimer",
            "dementia",
            "parkinson",
            "lewy",
            "frontotemporal",
            "tauopathy",
            "amyotrophic lateral sclerosis",
            "multiple sclerosis",
            "progressive supranuclear palsy",
        ],
        "cardiovascular": [
            "atherosclerosis",
            "cardiomyopathy",
            "heart failure",
            "heart disorder",
            "heart valve",
            "myocardial",
            "coronary",
            "atrial fibrillation",
            "ischemia",
        ],
        "metabolic": [
            "diabetes",
            "prediabetes",
        ],
        "renal": [
            "kidney failure",
            "chronic kidney",
            "obstructive nephropathy",
        ],
        "respiratory": [
            "chronic obstructive pulmonary disease",
            "emphysema",
            "pulmonary fibrosis",
            "interstitial lung",
            "respiratory failure",
            "pneumonia",
            "cystic fibrosis",
            "sarcoidosis",
            "hypersensitivity pneumonitis",
        ],
        "infection": [
            "covid",
            "influenza",
            "hiv",
            "leishmaniasis",
            "malaria",
            "listeriosis",
            "toxoplasmosis",
            "cytomegalovirus",
        ],
        "inflammatory_autoimmune": [
            "inflammatory",
            "inflammation",
            "crohn",
            "colitis",
            "celiac",
            "gastritis",
            "arthritis",
            "lupus",
            "sjogren",
            "dermatomyositis",
            "scleroderma",
            "fibrosis",
            "keloid",
        ],
        "hematologic": [
            "clonal hematopoiesis",
            "myelodysplastic",
            "myeloproliferative",
            "hematologic disorder",
        ],
        "eye": [
            "macular degeneration",
            "glaucoma",
            "cataract",
        ],
        "developmental_genetic": [
            "down syndrome",
            "trisomy",
            "anencephaly",
            "congenital",
        ],
        "injury": [
            "injury",
            "head injury",
        ],
        "psychiatric_neurologic_other": [
            "schizophrenia",
            "bipolar",
            "depressive",
            "epilepsy",
            "post-traumatic stress",
            "obsessive-compulsive",
            "opiate dependence",
        ],
    }


def _build_category_lookup(
    unique_terms: Sequence[str],
    rules: CategoryRules,
    fallback: str = "other",
) -> dict[str, str]:
    """Build a term -> broad category lookup using precompiled regexes.

    Patterns within a category are combined into a single alternation regex
    so each unique term incurs at most one regex search per category.

    Args:
      unique_terms: Unique annotation terms to classify.
      rules: Broad category name to pattern list mapping.
      fallback: Category to assign when no pattern matches.

    Returns:
      Mapping from each input term to its broad category.
    """
    compiled_rules = [
        (
            category,
            re.compile("|".join(f"(?:{pattern})" for pattern in patterns)),
        )
        for category, patterns in rules.items()
    ]

    lookup: dict[str, str] = {}
    for term in unique_terms:
        normalized = term.strip().lower()
        if normalized in ("", "nan"):
            lookup[term] = "unknown"
            continue

        matched_category = fallback
        for category, pattern in compiled_rules:
            if pattern.search(normalized):
                matched_category = category
                break
        lookup[term] = matched_category

    return lookup


def _format_dataset_labels(
    collection_names: pd.Series[Any],
    dataset_ids: pd.Series[Any],
    max_name_length: int = 75,
) -> pd.Series[Any]:
    """Vectorized dataset label formatting."""

    def _truncate(name: str) -> str:
        if len(name) <= max_name_length:
            return name
        return f"{name[:max_name_length]}..."

    names = (
        collection_names.astype(object)
        .fillna("unknown collection")
        .astype(str)
        .map(_truncate)
    )
    ids = (
        dataset_ids.astype(object)
        .fillna("unknown dataset")
        .astype(str)
        .str.slice(0, 8)
    )
    return names.str.cat(ids, sep=" | ")


def _summarize_fine_terms(
    term_counts: pd.DataFrame,
    max_terms_per_category: int,
) -> pd.DataFrame:
    """Summarize annotation terms within each broad category."""
    term_counts = term_counts.sort_values(
        ["dataset_id", "broad_category", "cell_count"],
        ascending=[True, True, False],
    ).copy()

    term_counts["term_count"] = (
        str(term_counts["term"]) + " (" + str(term_counts["cell_count"]) + ")"
    )
    term_counts["rank"] = term_counts.groupby(
        ["dataset_id", "broad_category"], observed=True
    ).cumcount()

    top_terms = term_counts.loc[term_counts["rank"] < max_terms_per_category]

    return (
        top_terms.groupby(["dataset_id", "broad_category"], observed=True)
        .agg(fine_terms=("term_count", " | ".join))
        .reset_index()
    )


def _count_annotation_makeup(
    obs: pd.DataFrame,
    annotation_column: str,
    annotation_type: AnnotationType,
    rules: CategoryRules,
    max_terms_per_category: int,
) -> pd.DataFrame:
    """Count broad and fine annotation makeup per dataset."""
    annotation = obs.loc[:, ["dataset_id", annotation_column]].copy()
    annotation["term"] = (
        annotation[annotation_column]
        .astype(object)
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )
    annotation.loc[annotation["term"] == "", "term"] = "unknown"

    category_lookup = _build_category_lookup(
        unique_terms=annotation["term"].unique().tolist(),
        rules=rules,
    )
    annotation["broad_category"] = annotation["term"].map(category_lookup)

    term_counts = (
        annotation.groupby(
            ["dataset_id", "broad_category", "term"],
            observed=True,
        )
        .size()
        .rename("cell_count")
        .reset_index()
    )

    category_counts = (
        term_counts.groupby(["dataset_id", "broad_category"], observed=True)
        .agg(category_cell_count=("cell_count", "sum"))
        .reset_index()
    )

    dataset_counts = (
        category_counts.groupby("dataset_id", observed=True)[
            "category_cell_count"
        ]
        .sum()
        .rename("dataset_cell_count")
        .reset_index()
    )

    fine_terms = _summarize_fine_terms(
        term_counts=term_counts,
        max_terms_per_category=max_terms_per_category,
    )

    category_counts = category_counts.merge(
        dataset_counts,
        on="dataset_id",
        how="left",
    )
    category_counts = category_counts.merge(
        fine_terms,
        on=["dataset_id", "broad_category"],
        how="left",
    )

    category_counts["fraction_cells"] = category_counts[
        "category_cell_count"
    ].astype(float) / category_counts["dataset_cell_count"].astype(float)
    category_counts["annotation_type"] = annotation_type

    return category_counts


def build_dataset_makeup_table(
    obs: pd.DataFrame,
    aggregated_dataset: pd.DataFrame,
    tissue_rules: CategoryRules | None = None,
    disease_rules: CategoryRules | None = None,
    max_terms_per_category: int = 12,
) -> pd.DataFrame:
    """Build a dataset-level tissue and disease makeup table.

    Args:
      obs: Cell-level Census obs metadata.
      aggregated_dataset: Dataset-level summary table.
      tissue_rules: Broad tissue annotation rules.
      disease_rules: Broad disease annotation rules.
      max_terms_per_category: Maximum fine terms to include per category.

    Returns:
      Long table with one row per dataset, annotation type, and broad category.
    """
    tissue_rules = tissue_rules or default_tissue_rules()
    disease_rules = disease_rules or default_disease_rules()

    tissue_makeup = _count_annotation_makeup(
        obs=obs,
        annotation_column="tissue",
        annotation_type="tissue",
        rules=tissue_rules,
        max_terms_per_category=max_terms_per_category,
    )
    disease_makeup = _count_annotation_makeup(
        obs=obs,
        annotation_column="disease",
        annotation_type="disease",
        rules=disease_rules,
        max_terms_per_category=max_terms_per_category,
    )

    makeup = pd.concat([tissue_makeup, disease_makeup], ignore_index=True)

    metadata_columns = [
        column
        for column in [
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
            "dataset_h5ad_path",
        ]
        if column in aggregated_dataset.columns
    ]

    metadata = (
        aggregated_dataset.loc[:, metadata_columns]
        .drop_duplicates("dataset_id")
        .copy()
    )

    if "n_cells" in metadata.columns:
        metadata = metadata.rename(columns={"n_cells": "summary_cell_count"})

    metadata["dataset_label"] = _format_dataset_labels(
        collection_names=metadata["collection_name"],
        dataset_ids=metadata["dataset_id"],
    )

    makeup = makeup.merge(metadata, on="dataset_id", how="left")

    return makeup.sort_values(
        [
            "annotation_type",
            "dataset_cell_count",
            "collection_name",
            "broad_category",
        ],
        ascending=[True, False, True, True],
    )


def _scale_marker_sizes(
    counts: pd.Series[Any],
    min_marker_size: float,
    max_marker_size: float,
) -> list[float]:
    """Scale cell counts into marker sizes."""
    count_values = counts.astype(float).tolist()
    max_count = max(count_values)

    if max_count == 0:
        return [min_marker_size for _ in count_values]

    return [
        min_marker_size
        + (max_marker_size - min_marker_size) * count / max_count
        for count in count_values
    ]


def _plot_makeup_page(
    frame: pd.DataFrame,
    category_order: list[str],
    title: str,
    output_pdf: PdfPages,
    min_marker_size: float,
    max_marker_size: float,
) -> None:
    """Plot one page of the dataset-level makeup dotplot."""
    frame = frame.copy()

    category_positions = {
        category: index for index, category in enumerate(category_order)
    }
    frame["x_position"] = [
        category_positions[category]
        for category in frame["broad_category"].astype(str).tolist()
    ]

    dataset_order = (
        frame.loc[:, ["dataset_label", "dataset_cell_count"]]
        .drop_duplicates()
        .sort_values("dataset_cell_count", ascending=True)["dataset_label"]
        .astype(str)
        .tolist()
    )
    dataset_positions = {
        dataset_label: index
        for index, dataset_label in enumerate(dataset_order)
    }
    frame["y_position"] = [
        dataset_positions[dataset_label]
        for dataset_label in frame["dataset_label"].astype(str).tolist()
    ]

    marker_sizes = _scale_marker_sizes(
        counts=frame["category_cell_count"],
        min_marker_size=min_marker_size,
        max_marker_size=max_marker_size,
    )

    figure_width = max(10.0, 0.55 * len(category_order))
    figure_height = max(8.0, 0.22 * len(dataset_order))

    figure, axes = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    scatter = axes.scatter(
        frame["x_position"],
        frame["y_position"],
        s=marker_sizes,
        c=frame["fraction_cells"],
        alpha=1.0,
    )

    axes.set_title(title)
    axes.set_xlabel("Broad annotation")
    axes.set_ylabel("Dataset")
    axes.set_xticks(range(len(category_order)))
    axes.set_xticklabels(category_order, rotation=90)
    axes.set_yticks(range(len(dataset_order)))
    axes.set_yticklabels(dataset_order, fontsize=6)
    axes.grid(True, linewidth=0.3, alpha=0.4)

    colorbar = figure.colorbar(scatter, ax=axes)
    colorbar.set_label("Fraction of dataset cells")

    output_pdf.savefig(figure)
    plt.close(figure)


def plot_dataset_makeup_dotplot_pdf(
    makeup: pd.DataFrame,
    output_path: str | Path,
    annotation_type: AnnotationType,
    rows_per_page: int = 50,
    min_fraction: float = 0.0,
    min_cell_count: int = 1,
    min_marker_size: float = 10.0,
    max_marker_size: float = 100.0,
) -> None:
    """Write a paginated static dataset-level dotplot PDF.

    Args:
      makeup: Long dataset-level makeup table.
      output_path: PDF path for the static plot.
      annotation_type: Annotation type to plot.
      rows_per_page: Number of dataset rows per PDF page.
      min_fraction: Minimum dataset fraction to display.
      min_cell_count: Minimum category cell count to display.
      min_marker_size: Minimum dot size.
      max_marker_size: Maximum dot size.
    """
    frame = makeup.loc[
        (makeup["annotation_type"] == annotation_type)
        & (makeup["fraction_cells"] >= min_fraction)
        & (makeup["category_cell_count"] >= min_cell_count),
        :,
    ].copy()

    if frame.empty:
        raise ValueError(
            f"No rows to plot for annotation_type={annotation_type}"
        )

    category_order = (
        frame.groupby("broad_category", observed=True)["category_cell_count"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )

    dataset_order = (
        frame.loc[:, ["dataset_label", "dataset_cell_count"]]
        .drop_duplicates()
        .sort_values("dataset_cell_count", ascending=False)["dataset_label"]
        .astype(str)
        .tolist()
    )

    n_pages = math.ceil(len(dataset_order) / rows_per_page)

    with PdfPages(output_path) as output_pdf:
        for page_index in range(n_pages):
            start = page_index * rows_per_page
            stop = start + rows_per_page
            page_datasets = dataset_order[start:stop]
            page_frame = frame[frame["dataset_label"].isin(page_datasets)]

            title = (
                f"Dataset-level {annotation_type} makeup "
                f"({page_index + 1}/{n_pages})"
            )

            _plot_makeup_page(
                frame=page_frame,
                category_order=category_order,
                title=title,
                output_pdf=output_pdf,
                min_marker_size=min_marker_size,
                max_marker_size=max_marker_size,
            )
