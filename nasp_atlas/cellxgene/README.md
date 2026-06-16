# CELLxGENE metadata usage

## Load metadata

Set up an output directory.

```python
from pathlib import Path

from nasp_atlas.cellxgene import CXGMetadata
from nasp_atlas.cellxgene import CXGMetadataConfig

outdir = Path("results/")
outdir.mkdir(parents=True, exist_ok=True)
```

Load metadata with the packaged category schema at
`nasp_atlas.cellxgene.configs.category_schema.yaml`.

```python
config = CXGMetadataConfig()
metadata = CXGMetadata.from_census(config=config)
```

Or load metadata with a custom category schema.

```python
config = CXGMetadataConfig.from_category_schema("/path/to/custom_yaml.yaml")
metadata = CXGMetadata.from_census(config=config)
```

## Categorize tissue and disease annotations

Create explicit broad-category columns from the configured YAML schema.

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

## Plot dataset-level makeup

These methods generate chunked stacked barplots with one bar per dataset.
Use `datasets_per_plot` to control how many datasets are shown per figure.

```python
metadata.plot_disease_makeup(
    outdir / "disease_makeup.png",
    category_column="disease_category",
)

metadata.plot_tissue_makeup(
    outdir / "tissue_makeup.png",
    category_column="tissue_category",
    datasets_per_plot=35,
)

metadata.plot_age_makeup(
    outdir / "age_makeup.png",
    datasets_per_plot=35,
)
```

Category ordering is an analysis choice. Pass `front` and `back` when specific
categories should be placed at the start or end of the stacked bar.

```python
metadata.plot_disease_makeup(
    outdir / "disease_makeup_ordered.png",
    category_column="disease_category",
    front=("normal",),
    back=("inflammatory_autoimmune", "cancer"),
)

metadata.plot_tissue_makeup(
    outdir / "tissue_makeup_ordered.png",
    category_column="tissue_category",
    front=(
        "adipose",
        "liver_biliary",
        "brain",
        "heart",
        "skeletal_muscle",
        "pancreas",
        "vasculature",
    ),
    back=("blood_immune",),
)
```

## Plot current obs as a single distribution

These methods use the current `metadata.obs`. If no `dataset_id` is supplied,
they summarize all rows currently present in `metadata.obs` as one distribution.
This is useful after filtering.

```python
metadata.metadata_barplot(
    label_column="disease_category",
    outpath=outdir / "obs_disease_categories.png",
)

metadata.metadata_barplot(
    label_column="tissue_category",
    outpath=outdir / "obs_tissue_categories.png",
)

metadata.plot_age_ranges(outdir / "obs_age_ranges.png")
```

## Filter metadata

The `keep` values should match labels in the category columns created above.
Filtering mutates `metadata.obs`, so downstream plots summarize the filtered
metadata.

```python
metadata.filter_diseases(
    keep=(
        "normal",
        "metabolic",
        "cardiovascular",
        "neurodegeneration",
        "renal",
        "respiratory",
        "fibrosis_injury",
        "developmental_genetic",
        "eye_disease",
    )
)

metadata.filter_tissues(
    keep=(
        "adipose",
        "liver_biliary",
        "brain",
        "heart",
        "skeletal_muscle",
        "pancreas",
        "vasculature",
        "kidney",
        "lung",
        "gut",
        "skin",
        "reproductive",
        "eye",
        "oral",
        "endocrine",
        "urinary",
        "serosal",
    )
)
```

After filtering, call the same plotting methods to visualize the filtered obs.

```python
metadata.metadata_barplot(
    label_column="disease_category",
    outpath=outdir / "filtered_disease_categories.png",
)

metadata.metadata_barplot(
    label_column="disease",
    grouped=True,
    outpath=outdir / "filtered_disease_labels_grouped.png",
)

metadata.metadata_barplot(
    label_column="tissue",
    grouped=True,
    outpath=outdir / "filtered_tissue_labels_grouped.png",
)


metadata.plot_disease_makeup(
    outdir / "filtered_disease_makeup.png",
    category_column="disease_category",
)

metadata.plot_age_ranges(outdir / "filtered_age_ranges.png")
metadata.plot_age_makeup(outdir / "filtered_age_makeup.png")
```
Use `grouped=True` with raw `disease` or `tissue` labels to plot detailed
raw-label composition while grouping labels under the configured broad category.

```python
metadata.metadata_barplot(
    label_column="disease",
    grouped=True,
    outpath=outdir / "obs_disease_labels_grouped.png",
)

metadata.metadata_barplot(
    label_column="tissue",
    grouped=True,
    outpath=outdir / "obs_tissue_labels_grouped.png",
)
```

## Plot one dataset

Pass `dataset_id` to make a single-dataset version of the obs-level plots.

```python
dataset_id = "53d208b0-2cfd-4366-9866-c3c6114081bc"

metadata.metadata_barplot(
    dataset_id=dataset_id,
    label_column="tissue",
    grouped=True,
    outpath=outdir / "dataset_tissues_grouped.png",
)

metadata.metadata_sankey(
    dataset_id=dataset_id,
    label_column="tissue",
    outpath=outdir / "dataset_tissue_sankey.png",
)

metadata.metadata_sankey(
    dataset_id=dataset_id,
    label_column="disease",
    outpath=outdir / "dataset_disease_sankey.png",
)

metadata.plot_age_ranges(
    dataset_id=dataset_id,
    outpath=outdir / "dataset_age_ranges.png",
)
```

## Save metadata

Save the current summarized metadata table at any time.

```python
metadata.to_csv(outdir / "dataset_summary.tsv")
```
