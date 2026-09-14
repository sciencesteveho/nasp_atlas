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
4. runs the selected combined-input and/or per-tissue mixed models for each
   scorer, reusing the same scores; and
5. writes model tables and, when NASP visualizations are enabled, up to five
   estimable contrast figures, a variance figure, and descriptive NASP
   summaries.

Optional donor sensitivity, gene diagnostics and gene-removal sensitivity run
after the models, independently for each scorer. These produce tables under
`associations/<scorer>/robustness/` without changing compendium definitions.

The PBS script also enables mechanism diagnostics and bundled reference sets
by default (`MECHANISM_DIAGNOSTICS=1`, `REFERENCE_SETS=1`). Both use the same
full scored population unless you explicitly request a subset. Python/CLI
callers opt in with `--mechanism-diagnostics --reference-sets`. Either PBS
setting can independently be set to `0`; no separate analysis job is needed.

Reference scoring adds the shipped Reactome/Hallmark sets to the requested
Scanpy/AUCell pass, with one separate UMAP per reference and scorer under
`scoring/`. Gene coverage is exported alongside scores. There are no runtime
downloads or gene-list caches. Adding/changing references invalidates older
score checkpoints, so the first reference-enabled run must rescore.

Mechanism tables and compact PNG/PDF figures live under
`associations/<scorer>/mechanisms/`: regulator-output coupling and two
exploratory donor-context scatterplots, IFN components, the OAS/RNase L pilot,
and reference agreement/overlap. Figures obey `PLOT_NASP_VISUALIZATIONS`;
tables are still generated when plotting is disabled. See the
[bundled-source notes](../nasp_atlas/data/README.md) for releases, licensing,
and the corrected official name of requested pathway `R-HSA-975138`.

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
| Condition | `obs["disease"]` |
| Study | `obs["dataset_id"]` |
| Development stage | `obs["development_stage"]` |
| Numeric age | `obs["age_years"]` |
| Gene symbol | `var["feature_name"]` |

Mixed-model inference requires donor, tissue, and cell-type identifiers.
Missing optional metadata reduces the fixed or random structure that can be
estimated and is recorded in the availability and diagnostic tables. In the
current Tabula Sapiens input, `disease` contains only `normal` and
`dataset_id` is absent, so condition contrasts and the multi-study structure
are expected to be unavailable rather than estimated.

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
| `--resume` | Reuse completed scores only with a matching input/settings manifest and file checksum. |
| `--mixed-models-combined` / `--no-mixed-models-combined` | Fit all scored cells together; enabled by default. |
| `--mixed-models-per-tissue` / `--no-mixed-models-per-tissue` | Independently fit every tissue using all its scored cells; disabled by default. |
| `--donor-sensitivity` | Write donor scores, leave-one-donor-out rankings and directly paired tissue differences within matching cell types. |
| `--gene-diagnostics` | Write module-gene expression, detection and descriptive gene-score correlations by donor, tissue, cell type and assay. |
| `--gene-removal-sensitivity` | Rescore dominant-gene deletions and shared-gene exclusions for the prespecified mechanistic module pairs; also runs gene diagnostics. |
| `--sensitivity-dominant-genes 1` | Number of candidate driver genes removed separately per module. |
| `--mechanism-diagnostics` | Regulator/output coupling, IFN components, and OAS/RNase L pilot; reuses gene diagnostics when available. |
| `--reference-sets` | Add offline Reactome/Hallmark signatures, individual score UMAPs, and curated/reference comparisons. |
| `--dry-run` | Print resolved worker options without loading data or writing outputs. |
| `--tissue-label Liver` | Subset an exact tissue label and recompute its embedding; omit for the complete atlas. |
| `--single-tissue-use-rep X_scvi` | AnnData representation used for the recomputed neighbors and UMAP. |
| `--single-tissue-use-x` | Use `adata.X` instead of a named representation. |
| `--statistical-unit donor` | Unit used by retained non-mixed stages, including optional eQTL analysis; mixed-model rows use their prespecified unit below. |
| `--aggregation mean` | Cell-to-unit aggregation. |
| `--detection-threshold 0.0` | Finite per-cell expression floor for fraction/percent aggregations. |
| `--condition-key disease` | Observation column identifying biological condition. |
| `--condition-reference normal` | Reference condition for effects estimated within each cell type. |
| `--study-key dataset_id` | Observation column identifying source studies in combined inputs. |
| `--mixed-model-min-cells 10` | Minimum cells supporting each donor–study–tissue–cell-type–assay observation. |
| `--mixed-model-min-donors 3` | Minimum independent donors required overall and for each reported focal level or pair. |
| `--mixed-model-min-studies 3` | Minimum studies required for a study random intercept. |
| `--mixed-model-min-repeated-contexts 3` | Minimum donor-contexts with repeated assay observations required for a context random effect. |
| `--subset-fraction 0.1` | Deterministic exploratory cell subset. |
| `--no-plot-modules` | Skip per-module marker plots. |
| `--no-nasp-visualizations` | Skip the fixed mixed-model and descriptive NASP figure sets; model tables are still written. |
| `--max-plots 200` | Cap general association plots per scorer. |

`--detection-threshold` must be finite. `--mixed-model-min-cells` must be at
least 1. The donor, study, and repeated-context thresholds must each be at
least 3 and independently govern their documented support checks.

## Submit with PBS

Review the `#PBS` resource and log directives in `score_modules.sh` for the
target cluster. Submit the complete atlas without `TISSUE_LABEL`:

For both mixed-model scopes and all three new robustness analyses:

```bash
qsub -v H5AD_NAME=tabula_sapiens.h5ad,RUN_NAME=atlas_robustness,MIXED_MODELS_COMBINED=1,MIXED_MODELS_PER_TISSUE=1,DONOR_SENSITIVITY=1,GENE_DIAGNOSTICS=1,GENE_REMOVAL_SENSITIVITY=1 \
  run_scripts/score_modules.sh
```

Keep `TISSUE_LABEL` and `SUBSET_FRACTION` unset to use **all cells of the input
file**. Combined means all scored cells together; it cannot recover other
tissues from a pre-split file. Per-tissue mode iterates over every observed
tissue, with no cell subsampling or rescoring within tissues. Thus signed-score
standardization and Scanpy/AUCell references remain shared between model scopes.
Each tissue still has only its own donors, regardless of the total atlas size.

Set either model switch to `0` to run only the other scope; setting both to `0`
retains descriptive associations and any requested robustness analyses. The
three robustness switches default to `0`, and removal enables its required
gene diagnostics even when `GENE_DIAGNOSTICS=0`. Both scopes run sequentially in
one PBS job, not as new submissions or arrays. Budget more wall time for all
tissues and gene rescoring; the existing 12-hour request is not a benchmark for
this expanded workload.

Preview the exact PBS worker command locally, without activating the HPC
environment or requiring HPC paths to exist:

```bash
DRY_RUN=1 H5AD_NAME=tabula_sapiens.h5ad MIXED_MODELS_PER_TISSUE=1 \
  bash run_scripts/score_modules.sh
```

The original combined-only invocation remains:

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
| `MIXED_MODELS_COMBINED` | `1` | Fit all scored cells together. |
| `MIXED_MODELS_PER_TISSUE` | `0` | Also fit each observed tissue separately. |
| `DONOR_SENSITIVITY` | `0` | Donor deletion rankings and matched-cell-type tissue differences. |
| `GENE_DIAGNOSTICS` | `0` | Donor/assay gene expression and detection diagnostics. |
| `GENE_REMOVAL_SENSITIVITY` | `0` | Dominant-gene and shared-gene removal rescoring. |
| `SENSITIVITY_DOMINANT_GENES` | `1` | Candidate genes removed separately per module. |
| `SUBSET_FRACTION` | unset | Optional exploratory cell fraction; unset uses all cells. |
| `DRY_RUN` | `0` | Preview the worker command without running it. |
| `PLOT_MODULES` | `1` | Generate module marker plots. |
| `PLOT_NASP_VISUALIZATIONS` | `1` | Generate fixed mixed-model inference and descriptive NASP summary plots. |
| `EXPRESSION_LAYER` | unset | Expression layer; unset uses `adata.X`. |
| `SINGLE_TISSUE_USE_REP` | `X_scvi` | Representation for a recomputed tissue embedding; use `none` for `X`. |
| `STATISTICAL_UNIT` | `donor` | Main association unit. |
| `AGGREGATION` | `mean` | Cell-to-unit aggregation. |
| `DETECTION_THRESHOLD` | `0.0` | Finite expression floor for fraction/percent aggregations. |
| `CONDITION_KEY` | `disease` | Observation column identifying biological condition. |
| `CONDITION_REFERENCE` | `normal` | Reference condition for within-cell-type contrasts. |
| `STUDY_KEY` | `dataset_id` | Observation column identifying studies in combined inputs. |
| `MIXED_MODEL_MIN_CELLS` | `10` | Minimum cells supporting one mixed-model aggregate. |
| `MIXED_MODEL_MIN_DONORS` | `3` | Minimum independent donors supporting the model and each focal level or pair. |
| `MIXED_MODEL_MIN_STUDIES` | `3` | Minimum studies supporting a study random intercept. |
| `MIXED_MODEL_MIN_REPEATED_CONTEXTS` | `3` | Minimum repeated assay contexts supporting a context random effect. |
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
    │   ├── association_tables/
    │   └── association_plots/
    │       ├── mixed_models/
    │       └── nasp/
    └── aucell/
        ├── association_tables/
        └── association_plots/
            ├── mixed_models/
            └── nasp/
```

Use distinct run names for concurrent inputs. `RESUME=1` validates a score
completion sidecar (`<score-table>.manifest.json`) against input identity,
module-panel hash, scoring settings, tissue/fraction selection and the score
file checksum. Incompatible or incomplete scores are regenerated. Older score
tables without this sidecar are regenerated once; they are never assumed to
represent the full atlas just because both scorers are present. Changing only
model scopes or robustness switches reuses compatible scores.

Inference and robustness stages rerun on resume. Combined model tables keep
their original locations. Tissue model outputs live under
`associations/<scorer>/mixed_models_by_tissue/<tissue-token>/`, each with its
own six model tables, provenance and optional figures. Tissue tokens include
a label hash to avoid collisions. The table
`association_tables/association_mixed_model_scopes.csv` lists all observed
scopes, whether enabled, completion status, scored-cell counts and current
artifact paths. A completed scope may still have non-estimable model results;
inspect its model diagnostics.

`robustness/robustness_manifest.csv` records requested stages and actual
artifacts. Consult these manifests rather than globbing old output directories:
files from disabled scopes/stages are retained but are not current results.
An interrupted run leaves its stage running/pending or failed, never completed.

## Validation before a large submission

```bash
bash -n run_scripts/score_modules.sh
python run_scripts/score_modules.py --help
```

Start with a deterministic subset run. Confirm resolved paths, requested
resources, scorer selection, random seed, output location, and resulting table
schemas before submitting a full atlas. Mixed models aggregate cells to one
equally weighted row per donor, optional study, tissue, cell type, optional
assay, and feature, with donor as the independent biological unit and
repeated-measures grouping factor. They estimate adjusted cell-type
deviations, paired tissue
differences, condition effects within cell type, age slopes per 10 years within
cell type, assay deviations, and conditional variance components. Cells count
toward measurement support only; they are not independent biological
replicates.

The six model tables are
`association_mixed_model_contrasts.csv`,
`association_mixed_model_fixed_effects.csv`,
`association_mixed_model_term_tests.csv`,
`association_mixed_model_variance_components.csv`,
`association_mixed_model_diagnostics.csv`, and
`association_mixed_model_availability.csv`. Always inspect availability,
support, `status`, `reason`, convergence, and optimizer warnings. Planned
contrasts use Benjamini–Hochberg FDR within each analysis and estimand family;
unsupported or failed models remain explicit and are not interpreted as zero
or as evidence for no association. `association_provenance.csv` ties these
outputs to the input files, score-generation metadata, resolved features,
compendium marker-panel hash, and analysis configuration.
