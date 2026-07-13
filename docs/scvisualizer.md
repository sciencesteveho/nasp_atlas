# `SCVisualizer`

[Back to the project README](../README.md)

`SCVisualizer` is the primary visualization engine for single-cell
embeddings, heatmaps, dot plots, association plots, and NASP summaries. It
creates `output_dir` during construction and produces publication-ready
figures.

## Quick start

```python
from nasp_atlas.single_cell import SCUtils
from nasp_atlas.single_cell import SCVisualizer
from nasp_compendium import GeneModules

adata = SCUtils.load_h5ad("atlas_subset.h5ad")
viz = SCVisualizer(output_dir="results/figures")

gene_modules = GeneModules()
nucleic_acid_sensors = gene_modules.sensors(
    "nucleic_acid_sensors",
    adata=adata,
    gene_symbol_column="feature_name",
    output="symbols",
)

viz.plot_multi_gene_umap_panel(
    adata,
    genes=nucleic_acid_sensors,
    filename="nucleic_acid_sensors",
    gene_symbol_column="feature_name",
    expression_layer=None,
    ncols=6,
)
```

## Public API

`SCVisualizer(output_dir)` initialize the visualization and create output directory.

### Embeddings

| Method | Purpose |
| --- | --- |
| `plot_embedding(...)` | Plot one embedding colored by observation columns or genes. |
| `plot_umap_panel(...)` | Combine categorical and numeric observation panels with explicit panel specifications. |
| `plot_multi_gene_umap_panel(...)` | Plot expression of several genes on a shared embedding layout. |
| `plot_multi_obs_umap_panel(...)` | Plot several numeric observation columns, such as module scores. |

### Heatmaps, dot plots, and violins

| Method | Purpose |
| --- | --- |
| `plot_multi_gene_expression_heatmap(...)` | Plot group-level expression for a requested gene list. |
| `plot_grouped_obs_score_heatmap(...)` | Plot group means for observation-level scores. |
| `plot_grouped_obs_score_barplot(...)` | Plot one score by group and return the per-group summary. |
| `plot_marker_dotplot(...)` | Plot expression and detection for genes or named marker groups. |
| `plot_rank_genes_dotplot(...)` | Plot top genes from an existing Scanpy rank result. |
| `plot_annotation_violins(...)` | Plot annotation-score distributions across Leiden clusters. |

### General associations

| Method | Purpose |
| --- | --- |
| `plot_feature_regression(...)` | Plot an aggregated feature against a continuous predictor with an optional fitted result. |
| `plot_feature_group_boxplot(...)` | Plot group central values, uncertainty, and individual analysis-unit points. |

### NASP summaries

| Method | Purpose |
| --- | --- |
| `plot_module_coupling_heatmap(...)` | Plot pairwise module correlations. |
| `plot_competence_output_state_map(...)` | Plot relative competence and output evidence by context. |
| `plot_ranked_nasp_hypotheses(...)` | Plot highest-ranked contexts for each hypothesis class. |
| `plot_sensor_output_mismatch(...)` | Plot sensor-expression and output-module coupling. |
| `plot_age_effect_dotplot(...)` | Plot context-specific age-effect estimates. |
| `plot_age_effect_consistency(...)` | Plot cross-stratum age-effect stability. |
| `plot_mechanistic_edge_network(...)` | Plot prespecified module-pair correlations as an undirected network. |

### Style helpers

| API | Purpose |
| --- | --- |
| `pastelize_cmap(...)` | Blend a Matplotlib colormap toward white. |
| `umap_expression_cmap(...)` | Build an expression colormap with light gray at zero. |
| `zero_gray_cmap(...)` | Set the low, center, high, or numeric zero position to light gray. |
| `ColorbarStyle(...)` | Configure inset colorbar geometry and tick styling. |
| `ColorbarStyle.with_overrides(...)` | Return an immutable style copy with selected geometry changes. |

## Embedding examples

### One categorical embedding

```python
viz.plot_embedding(
    adata,
    color="cell_type",
    basis="X_umap",
    filename="cell_types",
    color_map={
        "B cell": "#4c78a8",
        "T cell": "#f58518",
    },
)
```

### Mixed observation panel

```python
panels = [
    {
        "obs_key": "cell_type",
        "title": "Cell type",
        "kind": "categorical",
        "legend_loc": "bottom",
    },
    {
        "obs_key": "age_years",
        "title": "Age",
        "kind": "numeric",
        "cmap": "viridis",
    },
]

viz.plot_umap_panel(
    adata,
    panels,
    filename="cell_type_and_age",
    ncols=2,
)
```

### Module-score panels

```python
viz.plot_multi_obs_umap_panel(
    adata,
    obs_keys=[
        "NASP_DNA_SENSING_score",
        "NASP_RNA_SENSING_score",
    ],
    filename="nasp_module_scores",
    cmap="RdBu_r",
    center_zero=True,
    vmin=None,
    vmax=None,
)
```

## Heatmap and dot-plot examples

```python
viz.plot_multi_gene_expression_heatmap(
    adata,
    genes=["AIM2", "CGAS", "DDX58"],
    groupby="cell_type",
    filename="sensor_expression_by_cell_type",
    gene_symbol_column="feature_name",
    expression_layer="log1p",
)

viz.plot_marker_dotplot(
    adata,
    groupby="cell_type",
    filename="sensor_marker_dotplot",
    marker_groups={
        "DNA sensors": ["AIM2", "CGAS"],
        "RNA sensors": ["DDX58", "IFIH1"],
    },
    gene_symbol_column="feature_name",
    expression_layer="log1p",
)
```

Plot grouped module scores:

```python
score_keys = [
    "NASP_DNA_SENSING_score",
    "NASP_RNA_SENSING_score",
]

viz.plot_grouped_obs_score_heatmap(
    adata,
    score_keys=score_keys,
    groupby="cell_type",
    filename="module_scores_by_cell_type",
    center_zero=True,
)

group_summary = viz.plot_grouped_obs_score_barplot(
    adata,
    score_key="NASP_DNA_SENSING_score",
    groupby="cell_type",
    filename="dna_sensing_by_cell_type",
    stat_test="kruskal",
)
```

`plot_rank_genes_dotplot` expects an existing Scanpy ranking key. For example:

```python
viz.plot_rank_genes_dotplot(
    adata,
    groupby="cell_type",
    rank_key="rank_genes_groups",
    filename="ranked_markers",
    n_genes_per_group=5,
)
```

## Association examples

Association plots expect a frame already aggregated to the declared analysis
unit. One point must represent one donor or other explicitly chosen unit, not
an unlabelled collection of dependent cells.

```python
viz.plot_feature_regression(
    unit_frame,
    feature_id="NASP_DNA_SENSING_score",
    predictor_key="age_years",
    filename="dna_sensing_age",
    result_row=regression_result,
    color_key="tissue_in_publication",
)

viz.plot_feature_group_boxplot(
    unit_frame,
    feature_id="NASP_DNA_SENSING_score",
    group_key="sex",
    filename="dna_sensing_by_sex",
    result_row=group_test_result,
)
```

## NASP summary examples

The analysis workflow can render its complete fixed figure set:

```python
from nasp_atlas.analysis import plot_nasp_association_visualizations


plot_nasp_association_visualizations(
    output_dir="results/association_plots/nasp",
    module_coupling=module_coupling,
    context_summary=context_summary,
    hypothesis_priorities=hypothesis_priorities,
    sensor_output_coupling=sensor_output_coupling,
    regression_results=regression_results,
    age_stability=age_stability,
    mechanistic_edges=mechanistic_edges,
)
```

Use direct methods when only one view is needed:

```python
viz.plot_module_coupling_heatmap(
    module_coupling,
    filename="module_coupling",
    show_fdr=True,
)

viz.plot_competence_output_state_map(
    context_summary,
    filename="competence_output_states",
    label_columns=["tissue_in_publication", "cell_type"],
    min_donors=2,
)
```



## Colorbar styling

```python
from nasp_atlas.single_cell import ColorbarStyle

colorbar = ColorbarStyle(
    height="35%",
    width="3%",
    pad=0.04,
)

viz.plot_umap_panel(
    adata,
    panels=["age_years"],
    filename="age",
    colorbar_style=colorbar,
)

compact_colorbar = colorbar.with_overrides(
    height="25%",
    pad=0.02,
)
```

Color limits should be shared across panels intended for comparison. Missing
or non-estimable values must remain distinct from numeric zero.
