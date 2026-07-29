# Tabula Sapiens scoring and PBS workflows

[Back to the project README](../README.md)

`score_modules.py` runs a complete Tabula Sapiens h5ad or one tissue from
module scoring through donor-aware association analysis. `score_modules.sh`
adapts that worker for PBS without placing scientific logic in the scheduler
script.

## Workflow

For each input h5ad, the worker:

1. optionally subsets one tissue and recomputes its embedding;
2. scores compendium modules with Scanpy, AUCell, or both;
3. writes one shared cell-level score table;
4. runs an independent association analysis for each scorer; and
5. writes tables, general association plots, and optional NASP summary plots.

Scanpy and AUCell remain separate sensitivity analyses downstream. See
[Analysis outputs and interpretation](../docs/analysis_outputs.md) for output
contracts and scientific caveats.

## Input contract

Pass one `.h5ad` file. Omit `TISSUE_LABEL` or `--tissue-label` to analyze the
entire input and retain its existing embedding. Set the tissue label only to
select an exact observation label and recompute a tissue-specific embedding.

Default metadata names are:

| Role | AnnData field |
| --- | --- |
| Donor | `obs["donor_id"]` |
| Tissue | `obs["tissue_in_publication"]` |
| Cell type | `obs["cell_type"]` |
| Sex | `obs["sex"]` |
| Assay | `obs["assay"]` |
| Development stage | `obs["development_stage"]` |
| Numeric age | `obs["age_years"]` |
| Gene symbol | `var["feature_name"]` |

Donor-aware inference requires donor identifiers. Missing optional metadata
reduces the analyses that can be estimated and should be checked in the
skipped-feature and result tables.

## Run locally

Analyze the complete atlas:

```bash
python run_scripts/score_modules.py \
  --h5ad-path data/tabula_sapiens.h5ad \
  --output-path results/nasp_atlas_analysis \
  --run-name tabula_sapiens
```

Analyze a pre-split tissue file:

```bash
python run_scripts/score_modules.py \
  --h5ad-path data/liver_tabula_sapiens.h5ad \
  --output-path results/nasp_tissue_analysis \
  --run-name liver
```

Inspect every worker option with:

```bash
python run_scripts/score_modules.py --help
```

Common options:

| Option | Meaning |
| --- | --- |
| `--scorers scanpy aucell` | Scorers calculated and analyzed independently. |
| `--resume` | Reuse a score table only when its metadata records every requested scorer. |
| `--tissue-label Liver` | Subset an exact tissue label and recompute its embedding; omit for the complete atlas. |
| `--single-tissue-use-rep X_scvi` | AnnData representation used for the recomputed neighbors and UMAP. |
| `--single-tissue-use-x` | Use `adata.X` instead of a named representation. |
| `--statistical-unit donor` | Independent or observational unit used for the main association frame. |
| `--aggregation mean` | Cell-to-unit aggregation. |
| `--subset-fraction 0.1` | Deterministic exploratory cell subset. |
| `--no-plot-modules` | Skip per-module marker plots. |
| `--no-nasp-visualizations` | Skip the fixed NASP summary figure set. |
| `--max-plots 200` | Cap general association plots per scorer. |

## Submit with PBS

Review the `#PBS` resource and log directives in `score_modules.sh` for the
target cluster. Submit the complete atlas without `TISSUE_LABEL`:

```bash
qsub -v H5AD_NAME=tabula_sapiens.h5ad,RUN_NAME=tabula_sapiens \
  run_scripts/score_modules.sh
```

Or submit a pre-split tissue file:

```bash
qsub -v H5AD_NAME=liver_tabula_sapiens.h5ad,RUN_NAME=liver \
  run_scripts/score_modules.sh
```

Subset a larger file and recompute the tissue UMAP with `X_scvi`:

```bash
qsub -v H5AD_NAME=all.h5ad,TISSUE_LABEL=Liver,RUN_NAME=liver \
  run_scripts/score_modules.sh
```

Resume from a complete score table or run one scorer:

```bash
qsub -v H5AD_NAME=liver.h5ad,RUN_NAME=liver,RESUME=1 \
  run_scripts/score_modules.sh

qsub -v H5AD_NAME=liver.h5ad,RUN_NAME=liver,SCORERS=scanpy \
  run_scripts/score_modules.sh
```

### Analysis variables

Values are passed with `qsub -v` as comma-separated `NAME=value` pairs.

| Variable | Default | Meaning |
| --- | --- | --- |
| `H5AD_NAME` | required | Absolute h5ad path or filename relative to `DATA_DIR`. |
| `RUN_NAME` | tissue label or input stem | Unique output-directory label. |
| `TISSUE_LABEL` | unset | Exact tissue value to subset; omit to analyze the complete h5ad. |
| `SCORERS` | `scanpy:aucell` | Colon-separated scorer list. |
| `RESUME` | `0` | Reuse a complete compatible score table. |
| `PLOT_MODULES` | `1` | Generate module marker plots. |
| `PLOT_NASP_VISUALIZATIONS` | `1` | Generate fixed mechanistic summary plots. |
| `EXPRESSION_LAYER` | unset | Expression layer; unset uses `adata.X`. |
| `SINGLE_TISSUE_USE_REP` | `X_scvi` | Representation for a recomputed tissue embedding; use `none` for `X`. |
| `STATISTICAL_UNIT` | `donor` | Main association unit. |
| `AGGREGATION` | `mean` | Cell-to-unit aggregation. |
| `MAX_PLOTS` | `200` | Maximum general association plots per scorer. |
| `RANDOM_STATE` | `42` | Seed for subsetting, embedding, and scoring. |
| `AUCELL_CHUNK_SIZE` | `1000` | Cells processed in each AUCell block. |
| `AUCELL_NUM_WORKERS` | `8` | AUCell worker processes per block. |

Boolean variables accept `0` or `1` and common `false` or `true` spellings.

### Site paths

The shell script assigns installation and project paths directly. Edit these
assignments when adapting the worker to another cluster:

| Variable | Meaning |
| --- | --- |
| `CONDA_ROOT` | Conda installation root used by the PBS helper. |
| `CONDA_ENV` | NASP Atlas environment path. |
| `PROJECT_DIR` | Project root containing data, results, and logs. |
| `REPO_DIR` | NASP Atlas checkout. |
| `DATA_DIR` | Base directory for relative `H5AD_NAME` values. |
| `OUT_DIR` | Root for tissue run directories. |
| `PBS_COMMON` | Shell helper defining `pbs::setup_logging`, `pbs::activate_conda_environment`, and `pbs::log_job_context`. |
| `LOG_DIR` | Live-log directory. |

`PBS_COMMON` and the input h5ad must exist before work begins. The job exits
non-zero when either is unavailable.

## Outputs and resume behavior

Each run is isolated under `<OUT_DIR>/<run-name>`:

```text
<run-name>/
├── scoring/
│   └── tabula_sapiens_module_scores.csv.gz
└── associations/
    ├── scanpy/
    └── aucell/
```

Use distinct run names for concurrent inputs. `RESUME=1` reuses scoring only
when the existing score table records all requested scorers; otherwise scoring
runs again. Association analyses then run independently for every requested
scorer.

## Validation before a large submission

```bash
bash -n run_scripts/score_modules.sh
python run_scripts/score_modules.py --help
```

Start with a deterministic subset run. Confirm resolved paths, requested
resources, scorer selection, random seed, output location, and resulting table
schemas before submitting a full atlas. A complete-atlas run pools one row per
donor for the primary association estimand and estimates tissue- and
cell-type-specific age effects within donor strata; it does not treat cells as
independent biological replicates.
