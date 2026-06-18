<div align="center">
  <h1>Single-cell analysis of NASP in atlas data</h1>
</div>

Curation and analyses of **N**ucleic **A**cid **S**ensing **P**athways (**NASP**) in publicly-available single-cell atlases.

<br>

## Installation
```sh
# prepare a fresh conda environment
conda create -n nasp_atlas python=3.11 -y
conda activate nasp_atlas
python -m pip install -U pip setuptools wheel

# download and install from source
git clone https://github.com/sciencesteveho/nasp_atlas.git
cd nasp_atlas
pip install -e .
```
<br>

## Data requirements

Lorem ipsum for now.

<br>

## Modules

### Single-cell

Generalizable, reproducible single-cell utilities and visualizations.

`SCProcessor` handles common Scanpy processing steps like normalization, HVG PCA, neighbor graphs, and clustering. `SCVisualizer` centralizes stylistic choices for embedding viz, multi-gene panels, and dotplots.

```python
from nasp_atlas.single_cell import SCUtils
from nasp_atlas.single_cell import SCVisualizer

sc_utils = SCUtils(output_dir="results")
adata = SCUtils.load_h5ad("atlas_subset.h5ad")

sc_utils.viz.plot_multi_gene_umap_panel(
    adata,
    genes=["AIM2", "CGAS", "ZBP1"],
    filename="dna_sensing_genes",
    gene_symbol_column="feature_name",
    expression_layer=None,
)
```

For NASP module marker panels, resolve symbols through the given anndata's gene-symbol column:

```python
from nasp_compendium import GeneModules

viz = SCVisualizer(output_dir="results/tabula_sapiens_scoring_dev")
genes = GeneModules.genes(
    "NASP_DNA_SENSING",
    adata=adata,
    gene_symbol_column="feature_name",
    output="symbols",
)

viz.plot_multi_gene_umap_panel(
    adata,
    genes=genes,
    filename="NASP_DNA_SENSING_gene_expression_umaps",
    gene_symbol_column="feature_name",
    expression_layer=None,
    ncols=6,
)
```

</br>

### CELLxGENE metadata

The CELLxGENE module reads Census metadata, collapses raw disease and tissue
labels into broader categories, filters metadata tables, and generates metadata visualizations.

```python
from nasp_atlas.cellxgene import CXGMetadata

query = CXGMetadata.from_census()
query.annotate_default_categories()

query.plot_disease_makeup("disease_makeup.png")
query.plot_tissue_makeup("tissue_makeup.png")
query.plot_age_ranges("age_ranges.png")

query.to_csv("cellxgene_metadata.tsv")
```

See [`nasp_atlas/cellxgene/README.md`](nasp_atlas/cellxgene/README.md) for
specific examples.
