"""Strict YAML requests for one condition-cohort analysis population."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml  # type: ignore[import]


CovariateKind = Literal[
    "categorical",
    "numeric",
    "binned",
    "development_stage",
    "development_stage_top_coded",
]
ContrastRole = Literal["primary", "secondary"]
Modality = Literal["cells", "nuclei"]

CONDITION_LABEL_SEPARATOR = " | "


@dataclass(frozen=True, kw_only=True)
class ConditionSpec:
    """Source labels assigned to each analysis arm.

    Attributes:
      column: Source obs column holding the condition labels, or several
        columns whose values are joined with `CONDITION_LABEL_SEPARATOR`
        into one label, such as "Crohn disease | inflamed" from disease
        and inflammation status.
      levels: Arm names with their source labels, in reporting order.
      reference: Arm used as the comparison baseline.
      varies_within_donor: Whether a donor may contribute samples from
        different arms, such as inflamed and non-inflamed tissue.
    """

    column: str | tuple[str, ...]
    levels: tuple[tuple[str, tuple[str, ...]], ...]
    reference: str
    varies_within_donor: bool = False

    def __post_init__(self) -> None:
        """Reject arms that are ambiguous or cannot define a comparison."""
        if not isinstance(self.column, str) and (
            len(self.column) < 2 or len(self.column) != len(set(self.column))
        ):
            raise ValueError("Several condition columns must be distinct")

        arms = [arm for arm, _ in self.levels]
        labels = [label for _, values in self.levels for label in values]
        if len(arms) < 2 or len(arms) != len(set(arms)):
            raise ValueError("Condition needs at least two distinct arms")
        if any(not arm or not values for arm, values in self.levels):
            raise ValueError("Every condition arm needs a name and labels")
        if len(labels) != len(set(labels)):
            raise ValueError("A source label is assigned to several arms")
        if self.reference not in arms:
            raise ValueError(f"Reference arm {self.reference!r} is undefined")

    @property
    def columns(self) -> tuple[str, ...]:
        """Source columns that form each cell's condition label."""
        return (self.column,) if isinstance(self.column, str) else self.column

    @property
    def arms(self) -> tuple[str, ...]:
        """Arm names in reporting order."""
        return tuple(arm for arm, _ in self.levels)

    def arm_by_label(self) -> dict[str, str]:
        """Map each source label to its arm."""
        return {label: arm for arm, values in self.levels for label in values}


@dataclass(frozen=True, kw_only=True)
class CovariateSpec:
    """One adjustment variable read from a canonical obs column.

    Attributes:
      name: Canonical covariate name used in models and outputs.
      column: Canonical obs column (a key of `obs_columns`) to read.
      kind: "categorical", "numeric", "binned" for integer bins such as
        "60-69" or "50-59 ml/min/1.73m2", which become bin midpoints, or
        "development_stage" for CELLxGENE labels such as "45-year-old
        stage", which become years. Open-ended bins such as ">60" are not
        parsed. "development_stage_top_coded" also reads an open-ended top
        stage such as "90 year-old and over stage" as its lower bound
        (90), keeping the oldest donors at the cost of under-stating their
        age; declare it only where such donors are numerous.
    """

    name: str
    column: str
    kind: CovariateKind


@dataclass(frozen=True, kw_only=True)
class ContrastSpec:
    """One planned arm comparison and its multiplicity role.

    Attributes:
      estimand: Stable name, also the BH family identifier.
      target: Arm whose value is the minuend.
      reference: Arm whose value is subtracted.
      role: "primary" for prespecified claims, "secondary" otherwise.
    """

    estimand: str
    target: str
    reference: str
    role: ContrastRole


@dataclass(frozen=True, kw_only=True)
class ContinuousContrastSpec:
    """One planned slope on a donor-level measurement within some arms.

    Attributes:
      estimand: Stable name, also the BH family identifier.
      column: Canonical obs column holding the measurement.
      kind: "numeric" or "binned" (see `CovariateSpec`).
      arms: Arms whose donors enter the slope, such as CKD and AKI.
      step: Measurement increment that defines one unit of slope, such as
        10 ml/min/1.73 m2 of eGFR.
      role: "primary" or "secondary".

    Donors without a parsable measurement leave this slope only; they stay
    in every other analysis.
    """

    estimand: str
    column: str
    kind: Literal["numeric", "binned"]
    arms: tuple[str, ...]
    step: float
    role: ContrastRole


@dataclass(frozen=True, kw_only=True)
class CellStateSpec:
    """Altered and reference cell states compared within donor (AB5).

    Attributes:
      column: Canonical obs column holding each cell's state.
      reference: State label used as the within-donor baseline.
      altered: State labels pooled as the altered state.
    """

    column: str
    reference: str
    altered: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class IfnStrataSpec:
    """Pre-specified split of a disease arm by IFN-I output (AB2).

    Attributes:
      arm: Disease arm to split.
      cell_type: Cell type whose unit mean IFN_I_OUTPUT defines a donor's
        IFN state, for example classical monocytes.
      quantile: Reference-donor quantile used as the threshold.
      scorer: Scorer whose IFN_I_OUTPUT score is used.
    """

    arm: str
    cell_type: str
    quantile: float = 0.95
    scorer: str = "scanpy"


@dataclass(frozen=True, kw_only=True)
class ConditionCohortSpec:
    """Source, metadata crosswalk and planned contrasts for one population.

    Canonical obs columns must include `donor_id`, `cell_type`, `assay`,
    `tissue`, `sex` and `development_stage` (the last two label the scoring
    stage's metadata UMAPs), either mapped from a source column or set as a
    constant. The canonical `sample_id` joins `sample_columns`: the source
    columns that together identify one processed sample whose cells share
    every covariate value, such as a library, or a suspension within a
    processing batch. Condition arms always come from `condition`, never
    from a source column assumed to mean the same thing. The input is
    identified by its file name and SHA-256 and downloaded from
    `source_url`; each host keeps it in its own data directory. An internal
    input without a public download declares `source_url` as None and is
    placed in the data directory by hand. A filter keeps cells whose value
    is listed; a None value also keeps cells missing that column.
    `state_contrasts` compare the within-donor state difference of
    `cell_state` between two arms, such as the change with weight loss in
    remitters versus non-remitters.
    """

    cohort_id: str
    population_id: str
    source_notes: str
    input_filename: str
    source_url: str | None
    input_sha256: str
    counts_source: str
    gene_id_column: str
    gene_symbol_column: str
    modality: Modality
    sample_columns: tuple[str, ...]
    obs_columns: tuple[tuple[str, str], ...]
    obs_constants: tuple[tuple[str, str], ...]
    filters: tuple[tuple[str, tuple[str | bool | None, ...]], ...]
    condition: ConditionSpec
    covariates: tuple[CovariateSpec, ...]
    contrasts: tuple[ContrastSpec, ...]
    continuous_contrasts: tuple[ContinuousContrastSpec, ...] = ()
    cell_state: CellStateSpec | None = None
    ifn_strata: IfnStrataSpec | None = None
    state_contrasts: tuple[ContrastSpec, ...] = ()

    def __post_init__(self) -> None:
        """Validate identities, canonical columns and planned contrasts."""
        self._validate_input()
        keys = self._validate_canonical_columns()
        self._validate_covariates(keys)
        self._validate_contrasts(keys)
        self._validate_states(keys)

    def _validate_input(self) -> None:
        """Require a described source, a checksum and a count matrix."""
        if not self.cohort_id or not self.population_id:
            raise ValueError("cohort_id and population_id are required")
        if not self.source_notes.strip():
            raise ValueError("source_notes must describe the data source")
        if not re.fullmatch(r"[0-9a-f]{64}", self.input_sha256):
            raise ValueError(
                "input_sha256 must be the lowercase SHA-256 digest of "
                f"{self.input_filename} (sha256sum)"
            )
        if Path(self.input_filename).name != self.input_filename:
            raise ValueError("input_filename must be a file name, not a path")
        if self.source_url is not None and not self.source_url.startswith(
            "https://"
        ):
            raise ValueError("source_url must be an https URL or null")
        if self.counts_source not in ("X", "raw/X") and not (
            self.counts_source.startswith("layers/")
            and self.counts_source.count("/") == 1
            and self.counts_source.removeprefix("layers/")
        ):
            raise ValueError("counts_source must be X, raw/X or layers/<name>")
        if self.modality not in get_args(Modality):
            raise ValueError("modality must be cells or nuclei")

    def _validate_canonical_columns(self) -> list[str]:
        """Check canonical metadata and sample columns; return their names."""
        columns = [key for key, _ in self.obs_columns]
        constants = [key for key, _ in self.obs_constants]
        keys = [*columns, *constants]
        if len(keys) != len(set(keys)):
            raise ValueError("Canonical obs columns/constants must be disjoint")

        required = {
            "donor_id",
            "cell_type",
            "assay",
            "tissue",
            "sex",
            "development_stage",
        }
        if missing := required.difference(keys):
            raise ValueError(f"Missing canonical metadata: {sorted(missing)}")
        if "donor_id" not in columns:
            raise ValueError("Donor identity needs a source column")

        if not self.sample_columns or len(self.sample_columns) != len(
            set(self.sample_columns)
        ):
            raise ValueError("sample_columns needs distinct source columns")
        assigned = {"condition", "source_condition", "sample_id"}
        if reserved := assigned.intersection(keys):
            raise ValueError(f"Assigned by intake, not mapped: {reserved}")
        return keys

    def _validate_covariates(self, keys: list[str]) -> None:
        """Require unique covariates that read mapped columns."""
        names = [covariate.name for covariate in self.covariates]
        if len(names) != len(set(names)):
            raise ValueError("Covariate names must be unique")

        for covariate in self.covariates:
            if covariate.column not in keys:
                raise ValueError(
                    f"Covariate {covariate.name} reads unmapped column "
                    f"{covariate.column!r}"
                )
            if covariate.kind not in get_args(CovariateKind):
                raise ValueError(f"Unknown covariate kind {covariate.kind!r}")

    def _validate_contrasts(self, keys: list[str]) -> None:
        """Require unique, primary-bearing contrasts over declared arms."""
        estimands = [
            contrast.estimand
            for contrast in (
                *self.contrasts,
                *self.continuous_contrasts,
                *self.state_contrasts,
            )
        ]
        if len(estimands) != len(set(estimands)):
            raise ValueError("Contrast estimands must be unique")
        if not any(
            contrast.role == "primary"
            for contrast in (*self.contrasts, *self.state_contrasts)
        ):
            raise ValueError("Declare at least one primary contrast")
        if self.state_contrasts and (
            self.cell_state is None or self.condition.varies_within_donor
        ):
            raise ValueError(
                "state_contrasts need a cell_state and arms that share no donor"
            )

        for contrast in (*self.contrasts, *self.state_contrasts):
            if {contrast.target, contrast.reference} - set(self.condition.arms):
                raise ValueError(f"{contrast.estimand} uses an undefined arm")
            if contrast.target == contrast.reference:
                raise ValueError(
                    f"{contrast.estimand} compares an arm to itself"
                )
            if contrast.role not in get_args(ContrastRole):
                raise ValueError(f"Unknown contrast role {contrast.role!r}")

        for slope in self.continuous_contrasts:
            if slope.column not in keys:
                raise ValueError(f"{slope.estimand} reads unmapped column")
            if not slope.arms or set(slope.arms) - set(self.condition.arms):
                raise ValueError(f"{slope.estimand} needs declared arms")
            if slope.kind not in ("numeric", "binned") or slope.step <= 0:
                raise ValueError(f"{slope.estimand} needs a kind and step > 0")

    def _validate_states(self, keys: list[str]) -> None:
        """Check the optional cell-state contrast and IFN strata."""
        if self.cell_state is not None and (
            self.cell_state.column not in keys
            or self.cell_state.reference in self.cell_state.altered
            or not self.cell_state.altered
        ):
            raise ValueError(
                "cell_state needs a mapped column and distinct reference "
                "and altered labels"
            )

        if self.ifn_strata is not None:
            if self.ifn_strata.arm not in self.condition.arms:
                raise ValueError("ifn_strata.arm must be a declared arm")
            if self.ifn_strata.arm == self.condition.reference:
                raise ValueError("ifn_strata.arm cannot be the reference arm")
            if not 0.5 <= self.ifn_strata.quantile < 1:
                raise ValueError("ifn_strata.quantile must be in [0.5, 1)")

    def source_column(self, canonical: str) -> str:
        """Return the source obs column mapped to a canonical column."""
        return dict(self.obs_columns)[canonical]

    @property
    def arm_obs_columns(self) -> tuple[str, ...]:
        """Canonical columns read from a source column that defines the arms.

        Arms such as age groups are then a function of that measurement.
        """
        return tuple(
            canonical
            for canonical, source in self.obs_columns
            if source in self.condition.columns
        )


def read_condition_cohort_spec(path: str | Path) -> ConditionCohortSpec:
    """Read a strict YAML request.

    Unknown fields, missing source notes and ambiguous condition labels fail
    rather than becoming defaults. YAML content is never evaluated as code.
    """
    source = Path(path).resolve()
    with source.open() as handle:
        raw = yaml.safe_load(handle)

    try:
        record = dict(_mapping(raw))
        condition = _condition_spec(record.pop("condition"))
        covariates = tuple(
            _covariate_spec(name, value)
            for name, value in _mapping(record.pop("covariates", {})).items()
        )
        contrasts = tuple(
            _contrast_spec(value)
            for value in _sequence(record.pop("contrasts"))
        )
        filters = tuple(
            (column, _filter_values(column, values))
            for column, values in _mapping(record.pop("filters", {})).items()
        )

        # Required, but null for an internal input with no public download.
        source_url = record.pop("source_url")
        declared_states = record.pop("state_contrasts", None)
        state_contrasts = (
            tuple(_contrast_spec(value) for value in _sequence(declared_states))
            if declared_states is not None
            else ()
        )

        declared = record.pop("continuous_contrasts", None)
        slopes = (
            tuple(
                _continuous_contrast_spec(value)
                for value in _sequence(declared)
            )
            if declared is not None
            else ()
        )

        spec = ConditionCohortSpec(
            cohort_id=_text(record.pop("cohort_id")),
            population_id=_text(record.pop("population_id")),
            source_notes=_text(record.pop("source_notes")),
            input_filename=_text(record.pop("input_filename")),
            source_url=None if source_url is None else _text(source_url),
            input_sha256=_text(record.pop("input_sha256")),
            counts_source=_text(record.pop("counts_source")),
            gene_id_column=_text(record.pop("gene_id_column")),
            gene_symbol_column=_text(record.pop("gene_symbol_column")),
            modality=_text(record.pop("modality")),  # type: ignore[arg-type]
            sample_columns=_strings(record.pop("sample_columns")),
            obs_columns=_text_pairs(record.pop("obs_columns")),
            obs_constants=_text_pairs(record.pop("obs_constants", {})),
            filters=filters,
            condition=condition,
            covariates=covariates,
            contrasts=contrasts,
            continuous_contrasts=slopes,
            cell_state=_cell_state_spec(record.pop("cell_state", None)),
            ifn_strata=_ifn_strata_spec(record.pop("ifn_strata", None)),
            state_contrasts=state_contrasts,
        )

        if record:
            raise ValueError(f"Unknown cohort fields: {sorted(record)}")
        return spec
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid condition cohort specification {source}: {error}"
        ) from error


def _condition_spec(value: object) -> ConditionSpec:
    """Parse arm labels while preserving their declared order."""
    record = dict(_mapping(value))
    levels = tuple(
        (arm, _strings(labels))
        for arm, labels in _mapping(record.pop("levels")).items()
    )

    varies = record.pop("varies_within_donor", False)
    if not isinstance(varies, bool):
        raise TypeError("varies_within_donor must be true or false")

    column = record.pop("column")
    condition = ConditionSpec(
        column=_strings(column) if isinstance(column, list) else _text(column),
        levels=levels,
        reference=_text(record.pop("reference")),
        varies_within_donor=varies,
    )

    if record:
        raise ValueError(f"Unknown condition fields: {sorted(record)}")
    return condition


def _covariate_spec(name: str, value: object) -> CovariateSpec:
    """Parse one named covariate."""
    record = dict(_mapping(value))
    covariate = CovariateSpec(
        name=name,
        column=_text(record.pop("column")),
        kind=_text(record.pop("kind")),  # type: ignore[arg-type]
    )

    if record:
        raise ValueError(f"Unknown fields for covariate {name}: {record}")
    return covariate


def _continuous_contrast_spec(value: object) -> ContinuousContrastSpec:
    """Parse one planned slope."""
    record = dict(_mapping(value))
    step = record.pop("step")
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise TypeError("continuous contrast step must be a number")

    slope = ContinuousContrastSpec(
        estimand=_text(record.pop("estimand")),
        column=_text(record.pop("column")),
        kind=_text(record.pop("kind")),  # type: ignore[arg-type]
        arms=_strings(record.pop("arms")),
        step=float(step),
        role=_text(record.pop("role")),  # type: ignore[arg-type]
    )

    if record:
        raise ValueError(
            f"Unknown continuous contrast fields: {sorted(record)}"
        )
    return slope


def _cell_state_spec(value: object) -> CellStateSpec | None:
    """Parse the optional within-donor cell-state contrast."""
    if value is None:
        return None

    record = dict(_mapping(value))
    state = CellStateSpec(
        column=_text(record.pop("column")),
        reference=_text(record.pop("reference")),
        altered=_strings(record.pop("altered")),
    )

    if record:
        raise ValueError(f"Unknown cell_state fields: {sorted(record)}")
    return state


def _ifn_strata_spec(value: object) -> IfnStrataSpec | None:
    """Parse the optional IFN stratification."""
    if value is None:
        return None

    record = dict(_mapping(value))
    quantile = record.pop("quantile", 0.95)
    if not isinstance(quantile, float):
        raise TypeError("ifn_strata.quantile must be a number such as 0.95")

    strata = IfnStrataSpec(
        arm=_text(record.pop("arm")),
        cell_type=_text(record.pop("cell_type")),
        quantile=quantile,
        scorer=_text(record.pop("scorer", "scanpy")),
    )

    if record:
        raise ValueError(f"Unknown ifn_strata fields: {sorted(record)}")
    return strata


def _contrast_spec(value: object) -> ContrastSpec:
    """Parse one planned contrast."""
    record = dict(_mapping(value))
    contrast = ContrastSpec(
        estimand=_text(record.pop("estimand")),
        target=_text(record.pop("target")),
        reference=_text(record.pop("reference")),
        role=_text(record.pop("role")),  # type: ignore[arg-type]
    )

    if record:
        raise ValueError(f"Unknown contrast fields: {sorted(record)}")
    return contrast


def _filter_values(
    column: str,
    values: object,
) -> tuple[str | bool | None, ...]:
    """Require a nonempty list of string, boolean or null filter values.

    A null value keeps cells missing `column`, such as cells of a study that
    never recorded it.
    """
    if (
        not isinstance(values, list)
        or not values
        or any(
            value is not None and not isinstance(value, (str, bool))
            for value in values
        )
    ):
        raise ValueError(f"Filter {column} needs string/bool/null values")
    return tuple(values)


def _mapping(value: object) -> Mapping[str, object]:
    """Read a YAML mapping with string keys without coercion."""
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError("Expected a mapping with string keys")
    return value


def _sequence(value: object) -> list[object]:
    """Read a nonempty YAML list."""
    if not isinstance(value, list) or not value:
        raise TypeError("Expected a nonempty list")
    return value


def _text_pairs(value: object) -> tuple[tuple[str, str], ...]:
    """Read a mapping of canonical names to nonempty strings."""
    return tuple((key, _text(text)) for key, text in _mapping(value).items())


def _text(value: object) -> str:
    """Require a nonempty YAML string."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError("Expected a nonempty string")
    return value


def _strings(value: object) -> tuple[str, ...]:
    """Require a nonempty list of nonempty strings."""
    return tuple(_text(item) for item in _sequence(value))
