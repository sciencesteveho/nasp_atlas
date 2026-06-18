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

Also lorem ipsum.

<br>

## Modules

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