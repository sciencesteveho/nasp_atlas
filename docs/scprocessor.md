# `SCProcessor`

[Back to the project README](../README.md)

`SCProcessor` owns Scanpy preprocessing, embedding generation, clustering, and
small processing operations that transform AnnData objects. `EmbeddingConfig`
keeps the full embedding recipe serializable and reproducible.

## Quick start

```python
from nasp_atlas.single_cell import EmbeddingConfig
from nasp_atlas.single_cell import SCProcessor
from nasp_atlas.single_cell import SCUtils


adata = SCUtils.load_h5ad("atlas_subset.h5ad")

config = EmbeddingConfig(
    name="standard",
    pipeline="standard",
    n_top_genes=3_000,
    n_pcs=30,
    n_neighbors=15,
    umap_min_dist=0.3,
)
processor = SCProcessor(output_dir="results", random_seed=42)

adata = processor.generate_embeddings(
    adata,
    config=config,
    out_h5ad="atlas_embedded.h5ad",
)
adata = processor.cluster(
    adata,
    resolution=0.5,
)
```

`generate_embeddings` stores the config and seed in
`adata.uns["embedding_config"]`. With `save_h5ad=True`, it also writes the
processed object under `output_dir`.

## Processing pipelines

| Pipeline | Main operations | Stored expression |
| --- | --- | --- |
| `standard` | Total-count normalization, `log1p`, highly variable genes, optional regression and scaling, PCA, neighbors, UMAP | Original input in `adata.raw`; log-normalized values in `adata.layers["log1p"]` |
| `pearson_residuals` | Pearson-residual highly variable genes and normalization, PCA, neighbors, UMAP | Original input in `adata.raw`; log-normalized values in `adata.layers["log1p"]`; residuals in `adata.layers["pearson_residuals"]` |

Set `EmbeddingConfig.harmony_key` to an observation column to run Harmony after
PCA. This path requires `harmonypy`. Set `force_directed` to `"fa"` or `"fr"`
to add a force-directed layout.

## Mutation and output behavior

| Operation | AnnData behavior | Disk behavior |
| --- | --- | --- |
| `generate_embeddings` | Mutates and returns the supplied object. Sets `raw`, expression layers, PCA, neighbors, UMAP, and config metadata. | Writes `out_h5ad` by default. |
| `cluster` | Mutates and returns the supplied object. Adds `obs["leiden_<resolution>"]`. | No file output. |
| `recompute_umap` | Mutates the supplied object. Replaces its neighbor graph and UMAP coordinates using the selected representation. | No file output. |
| `subset_to_raw_counts` | Returns a new subset with raw counts in `X`. | No file output. |
| `auto_annotate_from_scores` | Adds per-cell score columns and automated labels to the supplied object. | No file output. |

## Public API

### `SCProcessor`

| API | Purpose |
| --- | --- |
| `SCProcessor(output_dir, random_seed=42)` | Create a processor and output directory with a deterministic seed. |
| `generate_embeddings(adata, *, config, save_h5ad=True, out_h5ad="embedded.h5ad")` | Run the configured normalization and embedding pipeline. |
| `cluster(adata, *, resolution=0.5, neighbors_key="neighborhood")` | Run deterministic Leiden clustering on an existing neighbor graph. |
| `recompute_umap(adata, *, use_rep=None, n_neighbors=15, random_state=42, min_dist=0.5)` | Rebuild neighbors and UMAP coordinates from `X` or an `obsm` representation. |
| `subset_to_raw_counts(adata, *, obs_key, values)` | Select one or more observation labels and return a raw-count copy. |
| `auto_annotate_from_scores(adata, *, leiden_key, score_key="score_ulm", prefix="ctscore", ambiguity_margin=0.1)` | Assign each cluster the highest mean precomputed annotation score, marking close calls as `-like`. |

### `EmbeddingConfig`

| API | Purpose |
| --- | --- |
| `EmbeddingConfig(...)` | Immutable configuration for normalization, PCA, neighbor, UMAP, Harmony, and force-directed settings. |
| `to_dict()` | Serialize every config field to a plain dictionary. |
| `from_dict(data)` | Reconstruct a config from a dictionary. |
| `to_json(indent=2)` | Serialize the config to JSON. |
| `from_json(raw)` | Reconstruct a config from JSON. |

## Examples

### Pearson-residual embedding

```python
config = EmbeddingConfig(
    name="pearson_residuals",
    pipeline="pearson_residuals",
    n_top_genes=3_000,
    pearson_residuals_kwargs={"theta": 100},
)

adata = processor.generate_embeddings(
    adata,
    config=config,
    save_h5ad=False,
)
```

### Harmony integration

```python
config = EmbeddingConfig(
    name="harmony",
    harmony_key="assay",
    n_top_genes=3_000,
)

adata = processor.generate_embeddings(
    adata,
    config=config,
)
```

The configured `harmony_key` must exist in `adata.obs`.

### Subset and restore raw counts

```python
liver = SCProcessor.subset_to_raw_counts(
    adata,
    obs_key="tissue_in_publication",
    values="Liver",
)

immune = SCProcessor.subset_to_raw_counts(
    adata,
    obs_key="cell_type",
    values=["T cell", "B cell"],
)
```

This operation requires `adata.raw`; `generate_embeddings` sets it before
normalization.

### Annotate clusters from precomputed scores

```python
cluster_means, cluster_labels, score_adata = (
    SCProcessor.auto_annotate_from_scores(
        adata,
        leiden_key="leiden_0.5",
        score_key="score_ulm",
        ambiguity_margin=0.1,
    )
)
```

This operation requires `decoupler` and a compatible score matrix in
`adata.obsm[score_key]`. Automated labels are computational annotations and
should be validated against marker expression and biological context.
