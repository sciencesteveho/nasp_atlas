# Visualization plotters

[Back to the project README](../README.md)

The visualization package exposes focused plotters for embeddings, heatmaps,
dot plots, score summaries, associations, mixed-model inference, and NASP
interpretation. Each plotter owns one coherent method family and creates its
`output_dir` during construction. The former `SCVisualizer` aggregate is no
longer part of the API; choose the plotter that owns the requested figure.

## Quick start

```python
from nasp_atlas.single_cell import SCUtils
from nasp_atlas.single_cell.visualization import UmapPlotter
from nasp_compendium import GeneModules

adata = SCUtils.load_h5ad("atlas_subset.h5ad")
umap_plotter = UmapPlotter(output_dir="results/figures")

gene_modules = GeneModules()
nucleic_acid_sensors = gene_modules.sensors(
    "nucleic_acid_sensors",
    adata=adata,
    gene_symbol_column="feature_name",
    output="symbols",
)

umap_plotter.plot_multi_gene_umap_panel(
    adata,
    genes=nucleic_acid_sensors,
    filename="nucleic_acid_sensors",
    gene_symbol_column="feature_name",
    expression_layer=None,
    ncols=6,
    shared_colorbar=True,
)
```

With `shared_colorbar=True`, every gene panel uses one color scale from 0.0 to
the exact highest finite expression value across the displayed genes. This
makes color directly comparable between panels; leave the option disabled when
each gene needs its own dynamic range. If no displayed gene has positive finite
expression, the shared 0-to-maximum range is undefined and the method raises an
actionable error instead of inventing a color scale.

## Public API

Import plotters from `nasp_atlas.single_cell.visualization`. Every constructor
accepts `output_dir` and an optional keyword-only `dpi`.

### `UmapPlotter`

| Method | Purpose |
| --- | --- |
| `plot_embedding(...)` | Plot one embedding colored by observation columns or genes. |
| `plot_umap_panel(...)` | Combine categorical and numeric observation panels with explicit panel specifications. |
| `plot_multi_gene_umap_panel(...)` | Plot expression of several genes on a shared embedding layout. |
| `plot_multi_obs_umap_panel(...)` | Plot several numeric observation columns, such as module scores. |

### `HeatmapPlotter`

| Method | Purpose |
| --- | --- |
| `summarize_gene_expression_by_obs(...)` | Prepare sparse-aware grouped means reusable across heatmaps. |
| `plot_multi_gene_expression_heatmap(...)` | Plot group-level expression for a requested gene list. |
| `plot_grouped_obs_score_heatmap(...)` | Plot group means for observation-level scores. |

### `DotplotPlotter`

| Method | Purpose |
| --- | --- |
| `plot_marker_dotplot(...)` | Plot expression and detection for genes or named marker groups. |
| `plot_rank_genes_dotplot(...)` | Plot top genes from an existing Scanpy rank result. |

### `SummaryPlotter`

| Method | Purpose |
| --- | --- |
| `plot_grouped_obs_score_barplot(...)` | Plot one score by group and return the per-group summary. |
| `plot_annotation_violins(...)` | Plot annotation-score distributions across Leiden clusters. |
| `plot_scorer_concordance_heatmap(...)` | Plot cross-scorer module-score correlations. |

### `AssociationPlotter`

| Method | Purpose |
| --- | --- |
| `plot_feature_regression(...)` | Plot an aggregated feature against a continuous predictor with an optional fitted result. |
| `plot_feature_group_boxplot(...)` | Plot unit-level distributions as boxplots, optionally with adjacent categorical strata. |
| `plot_feature_group_barplot(...)` | Plot grouped central values and uncertainty, optionally with adjacent categorical strata. |

### `MixedModelPlotter`

| Method | Purpose |
| --- | --- |
| `plot_mixed_model_effects(...)` | Plot one planned mixed-model estimand as effect estimates with 95% confidence intervals, FDR, and independent-unit support. |
| `plot_mixed_model_variance(...)` | Plot conditional variance fractions while keeping missing or unallocated components distinct from numeric zero. |

### `NaspPlotter`

| Method | Purpose |
| --- | --- |
| `plot_module_coupling_heatmap(...)` | Plot pairwise module correlations. |
| `plot_competence_output_state_map(...)` | Plot relative competence and output evidence by context. |
| `plot_ranked_nasp_hypotheses(...)` | Plot highest-ranked contexts for each hypothesis class. |
| `plot_sensor_output_mismatch(...)` | Plot sensor-expression and output-module coupling. |
| `plot_mechanistic_edge_barplot(...)` | Rank every curated mechanistic edge by Spearman correlation. |
| `plot_mechanistic_edge_network(...)` | Plot prespecified directional hypotheses with a symmetric correlation overlay. |

The following NASP examples use:

```python
from nasp_atlas.single_cell.visualization import NaspPlotter

nasp_plotter = NaspPlotter(output_dir="results/figures")
```

`plot_ranked_nasp_hypotheses(...)` labels each panel with its exact priority
formula. The inputs are zero-to-one relative evidence axes:

| Hypothesis | Priority formula |
| --- | --- |
| Active-like | `min(competence, output)` |
| Responsive-like | `max(output - competence, 0) * output` |
| Restricted/buffered | `max(restriction - output, 0) * mean(restriction, competence)` |
| Feedback-dominant | `max(feedback - output, 0) * feedback` |
| Post without NASP | `max(post - max(competence, output), 0) * post` |

`plot_sensor_output_mismatch(...)` accepts `column_spacing` and `row_spacing`
as independent axis multipliers. Values below 1.0 compact an axis and values
above 1.0 spread it out. Use `max_dot_size` to set the largest marker area in
points squared. The method also accepts `cbar_height`, `cbar_width`, and
`figsize`:

```python
nasp_plotter.plot_sensor_output_mismatch(
    sensor_output_coupling,
    filename="sensor_output_mismatch",
    column_spacing=0.65,
    row_spacing=1.1,
    max_dot_size=24.0,
)
```

`plot_mechanistic_edge_barplot(...)` places the median Spearman correlation on
the x axis and every curated source-to-target edge on the y axis. The shared
-1-to-1 scale and diverging colors distinguish negative from positive
association. Edges without a finite estimate remain visible as "not
estimable"; arrow-like labels denote curated direction, not causal evidence.

```python
nasp_plotter.plot_mechanistic_edge_barplot(
    mechanistic_edges,
    filename="mechanistic_edge_correlations",
    row_spacing=0.8,
    bar_height=0.56,
    tick_label_pad=1.0,
    figsize=(3.2, 3.8),
)
```

`plot_mechanistic_edge_network(...)` automatically wraps 5-point module names
inside equal-sized rectangles arranged in top-down mechanistic bands. Occupied
layers run from upstream ligand sources through sensing, proximal signaling,
pathway output, and post-NASP feedback. Band labels occupy the left gutter.
Straight arrows meet each rectangle along their direction of travel.

`node_size=(width, height)` sets the exact rectangle dimensions in layout
units. The horizontal layout expands when necessary to keep the busiest layer
from overlapping; the requested size is never silently clamped. Increase the
requested width or height when a 5-point label cannot fit.
`node_corner_radius` uses the same units. `label_gutter` reserves a fraction of
panel width for band labels. `layer_spacing` scales panel height and
`within_layer_spacing` scales panel width. `figsize=(width, height)` overrides
the adaptive total figure size in inches; shrinking it can require taller nodes
so wrapped 5-point labels still fit. Set `max_layer_span=1` to retain only
adjacent-layer edges. Filtering reports dropped edges and any modules that
consequently disappear.

```python
nasp_plotter.plot_mechanistic_edge_network(
    mechanistic_edges,
    filename="mechanistic_network",
    max_layer_span=1,
    node_size=(0.20, 0.46),
    node_corner_radius=0.02,
    label_gutter=0.23,
    layer_spacing=1.25,
    within_layer_spacing=1.0,
    figsize=(4.0, 3.5),
)
```

## Embedding examples

These examples use `umap_plotter = UmapPlotter(output_dir="results/figures")`.

### One categorical embedding

```python
umap_plotter.plot_embedding(
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

umap_plotter.plot_umap_panel(
    adata,
    panels,
    filename="cell_type_and_age",
    ncols=2,
)
```

### Module-score panels

Plot raw module scores with separate colorbars to preserve each module's native
score range:

```python
umap_plotter.plot_multi_obs_umap_panel(
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

The Tabula Sapiens workflow uses this native-scale, separate-colorbar view for
both Scanpy and AUCell. Colors are interpreted within a panel against its own
colorbar; they are not directly comparable between modules. This preserves the
score values used in exported tables and downstream inference without adding a
cohort-relative visualization transform.

## Heatmap and dot-plot examples

```python
from nasp_atlas.single_cell.visualization import DotplotPlotter
from nasp_atlas.single_cell.visualization import HeatmapPlotter
from nasp_atlas.single_cell.visualization import SummaryPlotter

heatmap_plotter = HeatmapPlotter(output_dir="results/figures")
dotplot_plotter = DotplotPlotter(output_dir="results/figures")
summary_plotter = SummaryPlotter(output_dir="results/figures")

heatmap_plotter.plot_multi_gene_expression_heatmap(
    adata,
    genes=["AIM2", "CGAS", "DDX58"],
    groupby="cell_type",
    filename="sensor_expression_by_cell_type",
    gene_symbol_column="feature_name",
    expression_layer="log1p",
)

dotplot_plotter.plot_marker_dotplot(
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

heatmap_plotter.plot_grouped_obs_score_heatmap(
    adata,
    score_keys=score_keys,
    groupby="cell_type",
    filename="module_scores_by_cell_type",
    center_zero=True,
)

group_summary = summary_plotter.plot_grouped_obs_score_barplot(
    adata,
    score_key="NASP_DNA_SENSING_score",
    groupby="cell_type",
    filename="dna_sensing_by_cell_type",
    stat_test="kruskal",
)
```

`plot_rank_genes_dotplot` expects an existing Scanpy ranking key. For example:

```python
dotplot_plotter.plot_rank_genes_dotplot(
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
an unlabelled collection of dependent cells. These direct plots are
descriptive unless the supplied `result_row` comes from a model appropriate to
that unit and dependence structure; the Tabula Sapiens workflow uses the mixed
model plots below for primary inference.

```python
from nasp_atlas.single_cell.visualization import AssociationPlotter

association_plotter = AssociationPlotter(output_dir="results/figures")

association_plotter.plot_feature_regression(
    unit_frame,
    feature_id="NASP_DNA_SENSING_score",
    predictor_key="age_years",
    filename="dna_sensing_age",
    result_row=regression_result,
    color_key="tissue_in_publication",
)

association_plotter.plot_feature_group_boxplot(
    unit_frame,
    feature_id="NASP_DNA_SENSING_score",
    group_key="cell_type",
    filename="dna_sensing_by_cell_type",
    stratify_key="sex",
    stratify_colors={"male": "#d2e7ef", "female": "#f9bebc"},
)
```

## Mixed-model inference examples

The two public methods have the following keyword interfaces and return
`None`; each writes `<filename>.png` under the plotter's `output_dir`:

```python
from nasp_atlas.single_cell.visualization import MixedModelPlotter

mixed_model_plotter = MixedModelPlotter(output_dir="results/figures")
```

```text
plot_mixed_model_effects(
    effects: pd.DataFrame,
    *,
    estimand: str,
    filename: str,
    analysis: str | None = None,
    feature_type: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    max_effects: int = 30,
    figure_width: float = 3.2,
    row_height: float = 0.44,
    minimum_height: float = 1.8,
    point_size: float = 11.0,
    interval_linewidth: float = 0.7,
    tick_label_pad: float = 1.5,
    label_wrap_width: int = 42,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
) -> None

plot_mixed_model_variance(
    variance: pd.DataFrame,
    *,
    analysis: str,
    filename: str,
    feature_type: str | None = None,
    component_order: Sequence[str] | None = None,
    component_colors: Mapping[str, ColorType] | None = None,
    title: str | None = None,
    max_features: int = 30,
    figure_width: float = 3.2,
    row_height: float = 0.25,
    minimum_height: float = 1.7,
    bar_height: float = 0.68,
    tick_label_pad: float = 1.5,
    figsize: tuple[float, float] | None = None,
) -> None
```

`plot_mixed_model_effects` requires the tidy columns `analysis`,
`feature_type`, `feature_id`, `feature_label`, `estimand`, `contrast`,
`estimate`, `ci_low`, `ci_high`, `pvalue_fdr`, `estimable`, and `status`.
It filters to the exact requested estimand and optional analysis or feature
type, omits rows not explicitly estimable, and selects at most `max_effects`
by FDR then absolute effect size with deterministic tie breaking. Filled
markers meet `fdr_threshold`; open markers do not. Reference, conditioning
level, donor support, and exact FDR are retained in the row label.

```python
mixed_model_plotter.plot_mixed_model_effects(
    contrasts,
    estimand="age_by_cell_type",
    analysis="age_by_cell_type",
    feature_type="module_score",
    filename="nasp_mixed_age_slopes_by_cell_type",
    max_effects=24,
)
```

`plot_mixed_model_variance` requires `analysis`, `feature_type`, `feature_id`,
`feature_label`, `component`, `variance_fraction`, `estimable`, and `status`.
It stacks finite estimable fractions exactly as supplied and never
renormalizes them. Pass `component_order` to declare expected components;
absent, non-estimable, and unallocated fractions are hatched and annotated,
whereas an estimable numeric zero remains zero.

```python
mixed_model_plotter.plot_mixed_model_variance(
    variance_components,
    analysis="adjusted_context",
    feature_type="module_score",
    component_order=("study", "donor", "context", "residual"),
    filename="nasp_mixed_variance_decomposition",
)
```

Both methods skip an empty selected subset. An estimable effect with a
non-finite or malformed confidence interval, or an estimable variance component
with an invalid fraction, raises an actionable error instead of plotting the
value as zero.

### Fixed Tabula Sapiens plot set

The Tabula Sapiens helper requests all five planned contrast families and the
variance decomposition. Effect families without estimable rows are skipped;
the variance view preserves unavailable components when rows exist:

```python
from nasp_atlas.analysis import plot_tabula_sapiens_mixed_model_inference


paths = plot_tabula_sapiens_mixed_model_inference(
    output_dir="results/association_plots/mixed_models",
    contrasts=contrasts,
    variance_components=variance_components,
)
```

The helper returns the paths it actually produced. Its fixed filename stems
are `nasp_mixed_adjusted_cell_type_effects`,
`nasp_mixed_condition_effects_by_cell_type`,
`nasp_mixed_age_slopes_by_cell_type`,
`nasp_mixed_paired_tissue_effects`,
`nasp_mixed_assay_batch_effects`, and
`nasp_mixed_variance_decomposition`.
It displays `feature_type="module_score"` by default; pass
`feature_type="gene_expression"` to render the corresponding sensor-gene
results.

## NASP summary examples

Use direct methods when only one view is needed:

```python
nasp_plotter.plot_module_coupling_heatmap(
    module_coupling,
    filename="module_coupling",
    show_fdr=True,
)

nasp_plotter.plot_competence_output_state_map(
    context_summary,
    filename="competence_output_states",
    label_columns=["tissue_in_publication", "cell_type"],
    min_donors=2,
    adjust_labels=True,
    label_force=(0.1, 0.2),
    label_static_force=(0.1, 0.2),
    label_explode_force=(0.1, 0.5),
    label_expand=(1.05, 1.2),
    label_max_move=(10, 10),
    figsize=(4.0, 4.0),
)
```



## Colorbar styling

```python
from nasp_atlas.single_cell.visualization import ColorbarStyle

colorbar = ColorbarStyle(
    height="35%",
    width="3%",
    pad=0.04,
)

umap_plotter.plot_umap_panel(
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
