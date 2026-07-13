# CELLxGENE metadata

[Back to the project README](../../README.md)

`nasp_atlas.cellxgene` reads CELLxGENE Census metadata, maps raw disease and
tissue labels into configured broad categories, filters observations, creates
metadata summaries, and writes composition plots.

## Quick start

```python
from pathlib import Path

from nasp_atlas.cellxgene import CXGMetadata


output_dir = Path("results/cellxgene")
output_dir.mkdir(parents=True, exist_ok=True)

metadata = CXGMetadata.from_census()
metadata.annotate_default_categories()

metadata.plot_disease_makeup(output_dir / "disease_makeup.png")
metadata.plot_tissue_makeup(output_dir / "tissue_makeup.png")
metadata.plot_age_ranges(output_dir / "age_ranges.png")

metadata.to_csv(output_dir / "dataset_summary.tsv")
```

The default category schema is packaged at
`nasp_atlas/cellxgene/configs/category_schema.yaml`.

## Data model

`CXGMetadata` keeps two tables:

| Attribute | Unit |
| --- | --- |
| `datasets` | One row per CELLxGENE dataset. |
| `obs` | Primary-cell observation metadata read from Census. |

Annotation and filtering methods update `metadata.obs` and return the same
`CXGMetadata` object for chaining. Plotting methods write files. `to_csv`
writes a dataset-level summary derived from the current, possibly filtered,
observation table.

## Public API

### `CXGMetadata`

| API | Purpose |
| --- | --- |
| `CXGMetadata(datasets=..., obs=..., config=...)` | Construct a workflow from existing metadata frames. |
| `CXGMetadata.from_census(organism="homo_sapiens", config=None)` | Read primary-cell and dataset metadata from the configured Census release. |
| `annotate_obs_categories(...)` | Add one category column using a supplied categorizer. |
| `annotate_default_categories()` | Add `disease_category` and `tissue_category`. |
| `filter_by_category(...)` | Retain selected values from any category column. |
| `filter_diseases(...)` | Retain selected configured disease categories. |
| `filter_tissues(...)` | Retain selected configured tissue categories. |
| `plot_category_makeup(...)` | Plot dataset composition for any category column. |
| `plot_disease_makeup(...)` | Plot dataset composition by disease category. |
| `plot_tissue_makeup(...)` | Plot dataset composition by tissue category. |
| `metadata_barplot(...)` | Plot the current observation distribution, optionally for one dataset. |
| `metadata_sankey(...)` | Plot one dataset or the current observations as a Sankey view. |
| `plot_age_ranges(...)` | Plot development-stage age ranges. |
| `plot_age_makeup(...)` | Plot dataset composition across age ranges. |
| `summarize_datasets()` | Return one summary row per dataset. |
| `to_csv(...)` | Write `summarize_datasets()` as a delimited table. |

### Configuration and category schema

| API | Purpose |
| --- | --- |
| `CXGMetadataConfig(...)` | Immutable Census columns, release, dataset columns, and category schema. |
| `CXGMetadataConfig.from_category_schema(source, **kwargs)` | Load a local or public YAML schema and apply optional config overrides. |
| `CXGMetadataConfig.categorize_disease(raw)` | Categorize a raw disease label with the configured schema. |
| `CXGMetadataConfig.categorize_tissue(raw)` | Categorize a raw tissue label with the configured schema. |
| `CXGMetadataConfig.display_name(label)` | Resolve a configured display name or human-readable fallback. |
| `CategorySchema(...)` | Immutable disease, tissue, override, and display-name mappings. |
| `CategorySchema.from_mapping(raw_schema)` | Construct a schema from parsed YAML data. |
| `CategorySchema.categorize_disease(raw)` | Categorize a raw disease label. |
| `CategorySchema.categorize_tissue(raw)` | Categorize a raw tissue label. |

### Package-level helpers

These names are exported from `nasp_atlas.cellxgene`:

| Function | Purpose |
| --- | --- |
| `load_category_schema(source=None)` | Load the packaged schema, a local YAML file, or a public YAML URL. |
| `categorize_disease(raw)` | Categorize one disease label with the packaged default schema. |
| `categorize_tissue(raw)` | Categorize one tissue label with the packaged default schema. |
| `categorize_development_stage(stage)` | Convert one development-stage label to an approximate age-range label. |
| `stage_age_value(stage)` | Convert one stage label to an approximate numeric age used for ordering. |
| `summarize_development_stage(values)` | Summarize a series as an age range and sorted stage counts. |
| `collapse_sex_series(values)` | Collapse unique sex labels, combining male and female when both occur. |
| `add_development_stage_age_obs(adata, ...)` | Add approximate numeric age to an AnnData observation column in place. |
| `category_color_map_from_uns(adata, obs_key)` | Build a category-to-color mapping from Scanpy-style `uns` colors. |

## Configure categories

Use the packaged schema:

```python
from nasp_atlas.cellxgene import CXGMetadataConfig


config = CXGMetadataConfig()
metadata = CXGMetadata.from_census(config=config)
```

Or provide a local file or public URL:

```python
config = CXGMetadataConfig.from_category_schema(
    "path/to/category_schema.yaml",
)
metadata = CXGMetadata.from_census(config=config)
```

Apply both configured categories at once:

```python
metadata.annotate_default_categories()
```

Apply one categorizer to explicit columns:

```python
metadata.annotate_obs_categories(
    source_column="disease",
    target_column="disease_category",
    categorizer=metadata.config.categorize_disease,
)

metadata.annotate_obs_categories(
    source_column="tissue",
    target_column="tissue_category",
    categorizer=metadata.config.categorize_tissue,
)
```

Broad categories are operational metadata groupings. Preserve raw labels and
the schema source when exporting or interpreting grouped results.

## Filter metadata

Category values must match the configured category labels:

```python
metadata.filter_diseases(
    keep=(
        "normal",
        "metabolic",
        "cardiovascular",
        "neurodegeneration",
    )
)

metadata.filter_tissues(
    keep=(
        "adipose",
        "liver_biliary",
        "brain",
        "heart",
    )
)
```

Equivalent chained use:

```python
metadata = (
    CXGMetadata.from_census()
    .annotate_default_categories()
    .filter_diseases(keep=("normal", "metabolic"))
    .filter_tissues(keep=("adipose", "liver_biliary"))
)
```

## Plot dataset composition

Dataset-level makeup plots are chunked when many datasets are present. Use
`datasets_per_plot` to control each page and `front` or `back` to make category
ordering explicit.

```python
metadata.plot_disease_makeup(
    output_dir / "disease_makeup.png",
    front=("normal",),
    back=("inflammatory_autoimmune", "cancer"),
    datasets_per_plot=35,
)

metadata.plot_tissue_makeup(
    output_dir / "tissue_makeup.png",
    front=("adipose", "liver_biliary", "brain"),
    back=("blood_immune",),
    datasets_per_plot=35,
)

metadata.plot_age_makeup(
    output_dir / "age_makeup.png",
    datasets_per_plot=35,
)
```

## Plot current observations

Without `dataset_id`, these methods summarize the current `metadata.obs`,
including any preceding filters:

```python
metadata.metadata_barplot(
    label_column="disease_category",
    outpath=output_dir / "disease_categories.png",
)

metadata.metadata_barplot(
    label_column="tissue",
    grouped=True,
    outpath=output_dir / "tissue_labels_grouped.png",
)

metadata.plot_age_ranges(
    output_dir / "age_ranges.png",
)
```

`grouped=True` supports raw `disease` and `tissue` labels and groups them under
the configured broad categories.

## Plot one dataset

```python
dataset_id = "53d208b0-2cfd-4366-9866-c3c6114081bc"

metadata.metadata_barplot(
    dataset_id=dataset_id,
    label_column="tissue",
    grouped=True,
    outpath=output_dir / "dataset_tissues.png",
)

metadata.metadata_sankey(
    dataset_id=dataset_id,
    label_column="disease",
    outpath=output_dir / "dataset_disease_sankey.png",
)

metadata.plot_age_ranges(
    dataset_id=dataset_id,
    outpath=output_dir / "dataset_age_ranges.png",
)
```

## Use helpers with AnnData

```python
from nasp_atlas.cellxgene import add_development_stage_age_obs
from nasp_atlas.cellxgene import category_color_map_from_uns


adata = add_development_stage_age_obs(
    adata,
    stage_column="development_stage",
    age_column="age_years",
)

cell_type_colors = category_color_map_from_uns(
    adata,
    "cell_type",
)
```

Approximate numeric ages support ordering and exploratory associations. They
do not recover exact donor ages when Census supplies only broad development
stages.

## Save the current summary

```python
summary = metadata.summarize_datasets()
metadata.to_csv(
    output_dir / "dataset_summary.tsv",
    sep="\t",
    index=False,
)
```
