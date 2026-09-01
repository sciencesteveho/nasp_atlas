<div align="center">
  <h1>NASP Atlas</h1>
  <p>Single-cell analysis of nucleic-acid-sensing pathways.</p>
</div>

Reusable tools for processing atlas-scale single-cell data, scoring
nucleic-acid-sensing pathway modules, testing donor-aware associations, and
visualizing results. Curated marker and sensor definitions come from
[`nasp_compendium`](https://github.com/sciencesteveho/nasp_compendium).

## Installation

This package supports Python 3.11 and 3.12.

```bash
conda create -n nasp_atlas python=3.11 -y
conda activate nasp_atlas
python -m pip install --upgrade pip setuptools wheel

git clone https://github.com/sciencesteveho/nasp_atlas.git
cd nasp_atlas
python -m pip install -e .
```

For development, install the extra dependencies:

```bash
python -m pip install -e ".[dev]"
```

Enable automatic checks before commits and pushes:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```


## Quick start

Load an atlas subset, resolve nucleic-acid sensors from the compendium, and
plot their expression via UMAP:

```python
from nasp_compendium import GeneModules
from nasp_atlas.single_cell import SCUtils
from nasp_atlas.single_cell.visualization import UmapPlotter

single_cell = SCUtils(output_dir="results")
adata = single_cell.load_h5ad("atlas_subset.h5ad")
umap_plotter = UmapPlotter(output_dir="results")

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
)
```

## Modules

| Module | Purpose | Documentation |
| --- | --- | --- |
| `nasp_atlas.single_cell` | Processing, module scoring, associations, and reusable utilities | [SCProcessor](docs/scprocessor.md) |
| `nasp_atlas.single_cell.visualization` | Focused plotters for embeddings, heatmaps, associations, inference, and NASP summaries | [Visualization plotters](docs/visualization.md) |
| `nasp_atlas.cellxgene` | CELLxGENE Census metadata querying, categorization, filtering, and plots | [CELLxGENE](nasp_atlas/cellxgene/README.md) |
| `nasp_atlas.analysis` | Tabula Sapiens workflows, donor-aware inference, and atlas summaries | [Outputs and interpretation](docs/analysis_outputs.md) |

## Further documentation

- [Analysis outputs and biological interpretation](docs/analysis_outputs.md)
- [PBS and command-line workflows](run_scripts/README.md)
