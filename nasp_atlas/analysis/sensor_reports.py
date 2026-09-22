"""Publish exploratory paired sensor reports without rescoring modules."""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nasp_atlas.analysis.external_replication import cohort
from nasp_atlas.analysis.external_replication.cohort import (
    prepare_cohort_counts,
)
from nasp_atlas.analysis.external_replication.cohort import read_cohort_metadata
from nasp_atlas.analysis.external_replication.cohort import select_cohort
from nasp_atlas.analysis.external_replication.gene_workflows import (
    validate_gene_sensitivity,
)
from nasp_atlas.analysis.external_replication.provenance import file_identity
from nasp_atlas.analysis.external_replication.provenance import request_identity
from nasp_atlas.analysis.external_replication.registered_inputs import (
    RegisteredPrimary,
)
from nasp_atlas.analysis.external_replication.result_artifacts import (
    validate_analysis_artifacts,
)
from nasp_atlas.analysis.external_replication.specification import CohortSpec
from nasp_atlas.analysis.external_replication.specification import (
    read_cohort_spec,
)
from nasp_atlas.single_cell import sensor_effects
from nasp_atlas.single_cell import sensor_features
from nasp_atlas.single_cell.associations import core
from nasp_atlas.single_cell.associations import paired
from nasp_atlas.single_cell.visualization import sensors


__all__ = ["analyze_replication_sensors", "analyze_sensor_reference"]


def analyze_sensor_reference(
    *,
    spec_path: Path,
    panel_path: Path,
    output_dir: Path,
    h5ad_path: Path | None = None,
) -> Path:
    """Run a focused TS sensor comparison from its declared count source.

    The existing cohort spec owns populations, assay selection and support.
    An optional input-path override makes the same spec usable on the cluster.
    This independent stage does not fit the full atlas or change module scores.
    Existing output directories are rejected, including completed reports.
    """
    if output_dir.exists():
        raise FileExistsError(
            f"Use a new sensor output directory: {output_dir}"
        )
    spec = read_cohort_spec(spec_path)
    if h5ad_path is not None:
        spec = replace(spec, input_path=h5ad_path.resolve())
    obs, _ = read_cohort_metadata(spec)
    intake = select_cohort(obs, spec)
    catalog = sensor_features.sensor_catalog(panel_path=panel_path)
    expression, mapping = _reference_measurements(
        spec, intake.selected_obs, catalog
    )
    request = {
        "source": file_identity(spec.input_path),
        "spec_file": file_identity(spec_path),
        "specification": {**asdict(spec), "input_path": str(spec.input_path)},
        "panel": file_identity(panel_path),
        "normalization": "log1p(full source counts / library size * 10000)",
    }
    return _publish_sensor_report(
        {"Tabula Sapiens": (spec, expression)},
        catalog,
        output_dir=output_dir,
        request=request,
        extra_tables={
            "reference_feature_mapping.csv": mapping,
            "reference_selected_cells.csv.gz": (
                intake.selected_obs.reset_index()
            ),
            "reference_donor_support.csv": intake.donor_support,
        },
    )


def analyze_replication_sensors(
    prepared: RegisteredPrimary,
    *,
    primary_analysis_dir: Path,
    gene_dir: Path,
    output_dir: Path,
) -> Path:
    """Reuse external gene tables and extract the exact matched TS cells.

    Raw sensor expression is independent of Scanpy/AUCell. Only the Scanpy
    copy of the expression table is used. Frozen primary decisions are neither
    recalculated nor interpreted as sensor-level hypotheses.
    """
    if output_dir.exists():
        raise FileExistsError(
            f"Use a new sensor output directory: {output_dir}"
        )
    primary = validate_analysis_artifacts(primary_analysis_dir)
    genes = validate_gene_sensitivity(gene_dir)
    primary_request = primary["request"]
    gene_request = genes["request"]
    if not isinstance(gene_request, dict):
        raise ValueError("Gene manifest requires a request object")
    if not isinstance(primary_request, dict) or (
        primary_request["registration_identity"]
        != prepared.registration.identity
        or primary_request["cohort_id"] != prepared.spec.cohort_id
        or gene_request["primary_analysis"]
        != file_identity(primary_analysis_dir / "analysis_manifest.json")
        or gene_request["registration_identity"]
        != prepared.registration.identity
    ):
        raise ValueError(
            "Sensor inputs must link to this frozen primary analysis"
        )
    if prepared.target_sum != 10_000:
        raise ValueError(
            "Sensor report currently requires counts normalized to 10,000"
        )
    catalog = sensor_features.sensor_catalog(panel_path=prepared.panel_path)
    saved = pd.read_csv(gene_dir / "scanpy_gene_donor_expression.csv.gz")
    keys = ["cohort_id", "donor_id", "tissue", "population", "assay"]
    external = sensor_effects.unique_sensor_measurements(
        saved, catalog, context_keys=keys
    )
    actual = (
        external[[*keys, "n_cells"]]
        .drop_duplicates()
        .sort_values(keys)
        .reset_index(drop=True)
    )
    expected = (
        prepared.intake.selected_obs.groupby(keys, observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    if not actual.astype(str).equals(expected.astype(str)):
        raise ValueError(
            "Saved sensor expression does not match the selected primary cells"
        )
    reference, mapping = _reference_measurements(
        prepared.reference_spec, prepared.reference_intake.selected_obs, catalog
    )
    request = {
        "registration_identity": prepared.registration.identity,
        "primary_analysis": file_identity(
            primary_analysis_dir / "analysis_manifest.json"
        ),
        "gene_manifest": file_identity(gene_dir / "gene_manifest.json"),
        "external_expression": file_identity(
            gene_dir / "scanpy_gene_donor_expression.csv.gz"
        ),
        "panel": file_identity(prepared.panel_path),
        "reference_specification": {
            **asdict(prepared.reference_spec),
            "input_path": str(prepared.reference_spec.input_path),
        },
        "reference_source": file_identity(prepared.reference_spec.input_path),
        "normalization": "log1p(full source counts / library size * 10000)",
    }
    return _publish_sensor_report(
        {
            "Tabula Sapiens": (prepared.reference_spec, reference),
            prepared.spec.cohort_id: (prepared.spec, external),
        },
        catalog,
        output_dir=output_dir,
        request=request,
        extra_tables={
            "reference_feature_mapping.csv": mapping,
            "reference_selected_cells.csv.gz": (
                prepared.reference_intake.selected_obs.reset_index()
            ),
        },
    )


def _reference_measurements(
    spec: CohortSpec,
    selected: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read selected cells with the full normalization denominator."""
    if selected.empty:
        raise ValueError(
            f"No supported paired cells for sensor comparison: {spec.cohort_id}"
        )
    expression = prepare_cohort_counts(spec, selected)
    return sensor_effects.sensor_donor_expression(
        expression,
        catalog,
        group_keys=["cohort_id", "donor_id", "tissue", "population", "assay"],
        gene_symbol_column=spec.gene_symbol_column,
    )


def _publish_sensor_report(
    datasets: Mapping[str, tuple[CohortSpec, pd.DataFrame]],
    catalog: pd.DataFrame,
    *,
    output_dir: Path,
    request: dict[str, object],
    extra_tables: Mapping[str, pd.DataFrame],
) -> Path:
    """Calculate all sensor pairs, write figures and publish completion last."""
    output_dir.mkdir(parents=True, exist_ok=False)
    tables, comparisons = _sensor_tables(datasets, catalog)
    for name, table in {**tables, **extra_tables}.items():
        table.to_csv(output_dir / name, index=False)
    fixed_contexts = []
    for spec, expression in datasets.values():
        fixed_axis = (
            "population" if spec.comparison.level_key == "tissue" else "tissue"
        )
        fixed_contexts.extend(expression[fixed_axis].unique().astype(str))
    context_label = " / ".join(
        value.replace("_", " ") for value in dict.fromkeys(fixed_contexts)
    )
    comparison_labels = " / ".join(
        f"{label}: {values['target']} minus {values['reference']}"
        for label, values in comparisons.items()
    )
    title = f"{context_label}\n{comparison_labels}"
    _render_sensor_report(tables["sensor_effects.csv"], title, output_dir)

    request.update(
        analysis_role="exploratory_sensor_expression",
        comparisons=comparisons,
        family="all catalog sensors per dataset/comparison",
        alpha=0.05,
        detection_threshold=0,
        independent_unit="person",
        covariates="none beyond pairing",
        implementation={
            str(path): file_identity(path)["sha256"]
            for path in (
                Path(__file__),
                Path(sensor_effects.__file__),
                Path(sensor_features.__file__),
                Path(paired.__file__),
                Path(core.__file__),
                Path(cohort.__file__),
                Path(sensors.__file__),
            )
        },
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "anndata",
                "matplotlib",
                "nasp_compendium",
            )
        },
    )
    manifest = {
        "status": "completed_sensor_report",
        "request": request,
        "identity": request_identity(request),
        "artifacts": {
            path.name: file_identity(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    partial = output_dir / "sensor_manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    partial.replace(output_dir / "sensor_manifest.json")
    return output_dir


def _sensor_tables(
    datasets: Mapping[str, tuple[CohortSpec, pd.DataFrame]],
    catalog: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    """Collect paired effects and their source measurements without scoring."""
    effects, pairs, donors = [], [], []
    comparisons = {}
    for label, (spec, expression) in datasets.items():
        comparison = spec.comparison
        result = sensor_effects.paired_sensor_effects(
            expression,
            catalog,
            pair_keys=["cohort_id", "donor_id"],
            level_key=comparison.level_key,
            target=comparison.target,
            reference=comparison.reference,
            minimum_cells=comparison.minimum_cells,
            minimum_pairs=comparison.minimum_pairs,
        )
        effects.append(
            result.estimates.assign(
                dataset=label, cohort_spec_id=spec.cohort_id
            )
        )
        pairs.append(result.donor_differences.assign(dataset=label))
        donors.append(expression.assign(dataset=label))
        comparisons[label] = asdict(comparison)
    estimates = pd.concat(effects, ignore_index=True)
    tables = {
        "sensor_catalog.csv": catalog,
        "sensor_effects.csv": estimates,
        "sensor_donor_differences.csv": pd.concat(pairs, ignore_index=True),
        "sensor_donor_expression.csv.gz": pd.concat(donors, ignore_index=True),
    }
    return tables, comparisons


def _render_sensor_report(
    estimates: pd.DataFrame,
    title: str,
    output_dir: Path,
) -> None:
    """Export paired and measurement figures with their interpretation."""
    for filename, plot in (
        ("paired_sensor_effects", sensors.plot_sensor_effects),
        ("sensor_expression_detection", sensors.plot_sensor_measurements),
    ):
        figure, _ = plot(estimates, title=title)
        for suffix in ("png", "pdf"):
            figure.savefig(
                output_dir / f"{filename}.{suffix}",
                dpi=200,
                bbox_inches="tight",
            )
        plt.close(figure)
    (output_dir / "report.md").write_text(
        f"# Exploratory individual-sensor report\n\n{title}\n\n"
        f"All {estimates.gene.nunique()} annotated sensors from the panel; "
        "one measurement per sensor/person/context, independent of module "
        "membership and scorer. The effect is the equal-person mean paired "
        "difference in mean log1p(counts per 10,000), not a log fold change. "
        "Counts are normalized on the full source gene universe. The cohort "
        "specifications supply cell selection and support floors. One assay "
        "per person is required; between-person assays are not adjusted.\n\n"
        "Expression uses paired t tests and pointwise 95% intervals. BH "
        "correction includes the entire sensor family separately for each "
        "dataset/comparison; unavailable tests count as p=1 in the denominator "
        "and retain missing reported p-values. Detection differences are "
        "descriptive on the same complete pairs. Zero detection differs from "
        "missing measurement; see support reasons and finite-cell counts.\n\n"
        "These post hoc comparisons do not change registered decisions, "
        "estimate sensor activation, attribute module-score contributions, "
        "or establish a common downstream cascade. TS and external magnitudes "
        "are not calibrated across assays.\n\n"
        "![Paired effects](paired_sensor_effects.png)\n\n"
        "![Expression and detection](sensor_expression_detection.png)\n"
    )
