"""Utilities for categorizing CELLxGENE metadata into broader groupings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import field
from functools import cache
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
import yaml


_DEFAULT_SCHEMA_PACKAGE = "nasp_atlas.cellxgene.configs"
_DEFAULT_SCHEMA_FILENAME = "category_schema.yaml"


@dataclass(frozen=True)
class CategorySchema:
    """Category schema for disease and tissue metadata labels."""

    disease_patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tissue_patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    disease_overrides: dict[str, str] = field(default_factory=dict)
    tissue_overrides: dict[str, str] = field(default_factory=dict)
    display_names: dict[str, str] = field(default_factory=dict)
    disease_match_normal_exactly: bool = False
    tissue_match_normal_exactly: bool = False

    @classmethod
    def from_mapping(cls, raw_schema: dict[str, object]) -> CategorySchema:
        """Create a category schema from parsed YAML."""
        disease = _as_mapping(raw_schema.get("disease"))
        tissue = _as_mapping(raw_schema.get("tissue"))

        return cls(
            disease_patterns=_coerce_patterns(disease.get("patterns")),
            tissue_patterns=_coerce_patterns(tissue.get("patterns")),
            disease_overrides=_coerce_string_mapping(disease.get("overrides")),
            tissue_overrides=_coerce_string_mapping(tissue.get("overrides")),
            display_names=_coerce_string_mapping(
                raw_schema.get("display_names")
            ),
            disease_match_normal_exactly=bool(
                disease.get("match_normal_exactly", False)
            ),
            tissue_match_normal_exactly=bool(
                tissue.get("match_normal_exactly", False)
            ),
        )

    def categorize_disease(self, raw: object) -> str:
        """Map a raw disease label to a configured broad category."""
        return _categorize_raw_label(
            raw,
            patterns=self.disease_patterns,
            match_normal_exactly=self.disease_match_normal_exactly,
            overrides=self.disease_overrides,
        )

    def categorize_tissue(self, raw: object) -> str:
        """Map a raw tissue label to a configured broad category."""
        return _categorize_raw_label(
            raw,
            patterns=self.tissue_patterns,
            match_normal_exactly=self.tissue_match_normal_exactly,
            overrides=self.tissue_overrides,
        )


def load_category_schema(source: str | Path | None = None) -> CategorySchema:
    """Load category schema from packaged YAML, local YAML, or public URL."""
    raw_schema = yaml.safe_load(_load_category_schema_text(source)) or {}

    if not isinstance(raw_schema, dict):
        raise ValueError("Category schema YAML must contain a mapping.")

    return CategorySchema.from_mapping(raw_schema)


@cache
def _load_default_category_schema() -> CategorySchema:
    """Load the packaged default category schema once per Python session."""
    return load_category_schema()


def categorize_disease(raw: object) -> str:
    """Map a raw disease label to the default broad category."""
    return _load_default_category_schema().categorize_disease(raw)


def categorize_tissue(raw: object) -> str:
    """Map a raw tissue label to the default broad category."""
    return _load_default_category_schema().categorize_tissue(raw)


def _load_category_schema_text(source: str | Path | None = None) -> str:
    """Load category schema YAML text."""
    if source is None:
        return (
            resources.files(_DEFAULT_SCHEMA_PACKAGE)
            .joinpath(_DEFAULT_SCHEMA_FILENAME)
            .read_text(encoding="utf-8")
        )

    source_string = str(source)
    parsed_source = urlparse(source_string)

    if parsed_source.scheme in {"http", "https"}:
        with urlopen(source_string, timeout=30) as response:
            return response.read().decode("utf-8")

    return Path(source).read_text(encoding="utf-8")


def _as_mapping(value: object) -> dict[str, object]:
    """Return value as a mapping or an empty mapping."""
    return value if isinstance(value, dict) else {}


def _coerce_patterns(raw_patterns: object) -> dict[str, tuple[str, ...]]:
    """Coerce parsed YAML category patterns into string tuples."""
    if not isinstance(raw_patterns, dict):
        return {}

    patterns: dict[str, tuple[str, ...]] = {}
    for category, raw_terms in raw_patterns.items():
        category_string = str(category)

        if raw_terms is None:
            patterns[category_string] = ()
            continue

        if isinstance(raw_terms, str):
            patterns[category_string] = (raw_terms.lower().strip(),)
            continue

        patterns[category_string] = tuple(
            str(term).lower().strip() for term in raw_terms if str(term).strip()
        )

    return patterns


def _coerce_string_mapping(raw_mapping: object) -> dict[str, str]:
    """Coerce parsed YAML mapping into string keys and values."""
    if not isinstance(raw_mapping, dict):
        return {}

    return {
        str(key).lower().strip(): str(value)
        for key, value in raw_mapping.items()
        if value is not None and str(key).strip()
    }


def _is_missing(value: object) -> bool:
    """Return True for None, NaN, or blank strings."""
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return isinstance(value, str) and not value.strip()


def _match_first(
    value: str,
    patterns: dict[str, tuple[str, ...]],
    *,
    match_normal_exactly: bool = False,
) -> str:
    """Return the first category whose terms appear in value."""
    for category, terms in patterns.items():
        if category == "normal" and match_normal_exactly:
            if value == "normal":
                return "normal"

            continue

        if any(term in value for term in terms):
            return category

    return "other"


def _categorize_raw_label(
    raw: object,
    patterns: dict[str, tuple[str, ...]],
    *,
    match_normal_exactly: bool = False,
    overrides: dict[str, str] | None = None,
) -> str:
    """Map a raw label to a broad category."""
    if _is_missing(raw):
        return "other"

    label = str(raw).lower().strip()

    if overrides and label in overrides:
        return overrides[label]

    return _match_first(
        label,
        patterns,
        match_normal_exactly=match_normal_exactly,
    )


def _collapse_sex_series(values: pd.Series) -> str:
    """Collapse sex annotations, merging male and female when both are
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

    if match := re.search(r"carnegie stage\s+(\d+)", value):
        return -1.0 + int(match[1]) / 100.0

    if match := re.search(
        r"(\d+)(?:st|nd|rd|th)? week post-fertilization", value
    ):
        return -0.75 + int(match[1]) / 100.0

    if match := re.search(r"(\w+) lmp month", value):
        month_map = {
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
        }
        return -0.5 + month_map.get(match[1], 0) / 100.0

    if (
        "embryonic" in value
        or "organogenesis" in value
        or "blastula" in value
        or "prenatal" in value
    ):
        return -0.25

    if "newborn" in value:
        return 0.0

    if match := re.search(r"(\d+)-month-old", value):
        return int(match[1]) / 12.0

    if match := re.search(r"(\d+)-year-old", value):
        return float(match[1])

    if match := re.search(r"(\d+)\s*year-old and over", value):
        return float(match[1])

    if match := re.search(r"(\d+)-(\d+)\s*year-old", value):
        return float(match[1])

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

    return next(
        (age for key, age in stage_map.items() if key in value),
        float("inf") - 1.0,
    )


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


def categorize_development_stage(stage: object) -> str:
    """Map a raw development-stage label to an approximate age-range label."""
    if _is_missing(stage):
        return "unknown"

    stage_string = str(stage).strip()
    age = _stage_age_value(stage_string)

    if age in {float("inf"), float("inf") - 1.0}:
        return "unknown"

    return _stage_range_label(age, stage_string)


def _summarize_development_stage(
    values: pd.Series,
    delimiter: str = ", ",
) -> str:
    """Summarize development stages as range plus sorted stage counts."""
    counts = values.dropna().astype(str).str.strip().value_counts()
    counts = counts[counts.index != ""]

    if counts.empty:
        return ""

    ordered = sorted(
        counts.index,
        key=lambda stage: (_stage_age_value(stage), stage),
    )

    if known := [
        (stage, _stage_age_value(stage))
        for stage in ordered
        if _stage_age_value(stage) not in [float("inf"), float("inf") - 1.0]
    ]:
        min_stage, min_age = min(known, key=lambda item: item[1])
        max_stage, max_age = max(known, key=lambda item: item[1])
        range_prefix = f"{_stage_range_label(min_age, min_stage)}"
        range_prefix += f"-{_stage_range_label(max_age, max_stage)}"
    else:
        range_prefix = "unknown"

    terms = [f"{stage} ({counts[stage]})" for stage in ordered]

    return delimiter.join([range_prefix, *terms])
