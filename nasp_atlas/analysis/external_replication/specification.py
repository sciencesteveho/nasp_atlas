"""Explicit cohort selection and expression-source requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import]


__all__ = [
    "CohortSpec",
    "ComparisonSpec",
    "PopulationSpec",
    "read_cohort_spec",
]


@dataclass(frozen=True, kw_only=True)
class ComparisonSpec:
    """Two contexts and required support for one donor-paired estimand."""

    level_key: Literal["tissue", "population"]
    target: str
    reference: str
    minimum_cells: int = 10
    minimum_pairs: int = 6

    def __post_init__(self) -> None:
        """Reject comparisons that cannot define complete biological pairs."""
        if self.level_key not in ("tissue", "population"):
            raise ValueError(
                "Comparison level_key must be tissue or population"
            )
        if (
            not self.target
            or not self.reference
            or self.target == self.reference
        ):
            raise ValueError("Comparison requires distinct nonempty levels")
        if self.minimum_cells < 1 or self.minimum_pairs < 2:
            raise ValueError(
                "Require minimum_cells >= 1 and minimum_pairs >= 2"
            )


@dataclass(frozen=True, kw_only=True)
class PopulationSpec:
    """Published labels belonging to one declared analysis population."""

    name: str
    column: str
    labels: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class CohortSpec:
    """Source, metadata crosswalk and selection passed through intake stages."""

    cohort_id: str
    input_path: Path
    counts_source: str
    gene_id_column: str
    gene_symbol_column: str
    obs_columns: tuple[tuple[str, str], ...]
    obs_constants: tuple[tuple[str, str], ...]
    filters: tuple[tuple[str, tuple[str | bool, ...]], ...]
    populations: tuple[PopulationSpec, ...]
    library_keys: tuple[str, ...]
    donor_metadata: tuple[str, ...]
    comparison: ComparisonSpec
    source_notes: str

    def __post_init__(self) -> None:
        """Validate source paths and canonical metadata ownership."""
        if not self.cohort_id or not self.source_notes.strip():
            raise ValueError("Cohort identity and source_notes are required")
        if self.counts_source not in ("X", "raw/X") and not (
            self.counts_source.startswith("layers/")
            and self.counts_source.count("/") == 1
            and self.counts_source.removeprefix("layers/")
        ):
            raise ValueError("counts_source must be X, raw/X or layers/<name>")
        columns = [key for key, _ in self.obs_columns]
        constants = [key for key, _ in self.obs_constants]
        keys = [*columns, *constants]
        if len(keys) != len(set(keys)):
            raise ValueError("Canonical obs columns/constants must be disjoint")
        required = {"donor_id", "tissue", "assay", "modality"}
        if missing := required.difference(keys):
            raise ValueError(f"Missing canonical metadata: {sorted(missing)}")
        if {"cohort_id", "population"}.intersection(keys):
            raise ValueError("cohort_id and population are assigned by intake")
        if "donor_id" not in columns:
            raise ValueError("Donor identity must come from a source column")
        if not self.populations or len(
            {p.name for p in self.populations}
        ) != len(self.populations):
            raise ValueError("Specify distinct named analysis populations")
        if any(
            not p.name or not p.column or not p.labels for p in self.populations
        ):
            raise ValueError("Each population needs a name, column and labels")


def read_cohort_spec(path: str | Path) -> CohortSpec:
    """Read a strict YAML request, resolving input paths relative to the YAML.

    This parses the cohort request only; hypotheses and scoring settings have
    their own owners. Unknown fields and missing source descriptions fail
    rather than silently becoming defaults. Inputs are never evaluated as code.
    """
    source = Path(path).resolve()
    with source.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    record = dict(raw)
    try:
        population_records = _string_mapping(record.pop("populations"))
        populations = []
        for name, value in population_records.items():
            population = _string_mapping(value)
            populations.append(
                PopulationSpec(
                    name=name,
                    column=_text(population["column"]),
                    labels=_strings(population["labels"]),
                )
            )
            if set(population) != {"column", "labels"}:
                raise ValueError(f"Unknown population fields for {name}")

        comparison = _comparison_spec(record.pop("comparison"))
        columns = _string_mapping(record.pop("obs_columns"))
        constants = _string_mapping(record.pop("obs_constants", {}))
        filters = _string_mapping(record.pop("filters", {}))
        resolved_filters = []
        for column, values in filters.items():
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, (str, bool)) for value in values)
            ):
                raise ValueError(f"Filter {column} needs string/bool values")
            resolved_filters.append((column, tuple(values)))

        spec = CohortSpec(
            cohort_id=_text(record.pop("cohort_id")),
            counts_source=_text(record.pop("counts_source")),
            gene_id_column=_text(record.pop("gene_id_column")),
            gene_symbol_column=_text(record.pop("gene_symbol_column")),
            source_notes=_text(record.pop("source_notes")),
            input_path=(
                source.parent / _text(record.pop("input_path"))
            ).resolve(),
            obs_columns=tuple(
                (key, _text(value)) for key, value in columns.items()
            ),
            obs_constants=tuple(
                (key, _text(value)) for key, value in constants.items()
            ),
            filters=tuple(resolved_filters),
            populations=tuple(populations),
            library_keys=_strings(record.pop("library_keys")),
            donor_metadata=_strings(record.pop("donor_metadata", [])),
            comparison=comparison,
        )
        if record:
            raise ValueError(f"Unknown cohort fields: {sorted(record)}")
        return spec
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid cohort specification {source}: {error}"
        ) from error


def _string_mapping(value: object) -> dict[str, object]:
    """Read a YAML object with string keys without coercing invalid inputs."""
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError("Expected a mapping with string keys")
    return dict(value)


def _comparison_spec(value: object) -> ComparisonSpec:
    """Parse context identity and integral support thresholds."""
    record = _string_mapping(value)
    level_key = record.pop("level_key")
    if level_key not in ("tissue", "population"):
        raise ValueError("Comparison level_key must be tissue or population")
    minimum_cells = record.pop("minimum_cells", 10)
    minimum_pairs = record.pop("minimum_pairs", 6)
    if (
        not isinstance(minimum_cells, int)
        or isinstance(minimum_cells, bool)
        or not isinstance(minimum_pairs, int)
        or isinstance(minimum_pairs, bool)
    ):
        raise TypeError("Support thresholds must be integers")
    comparison = ComparisonSpec(
        level_key=level_key,
        target=_text(record.pop("target")),
        reference=_text(record.pop("reference")),
        minimum_cells=minimum_cells,
        minimum_pairs=minimum_pairs,
    )
    if record:
        raise ValueError(f"Unknown comparison fields: {sorted(record)}")
    return comparison


def _text(value: object) -> str:
    """Require a nonempty YAML string where a source name is needed."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError("Expected a nonempty string")
    return value


def _strings(value: object) -> tuple[str, ...]:
    """Require an explicit list of nonempty strings."""
    if not isinstance(value, list):
        raise TypeError("Expected a list of strings")
    return tuple(_text(item) for item in value)
