# Analysis outputs and interpretation

[Back to the project README](../README.md)

`association_analysis` joins a saved module-score table to the matching cells
in an AnnData file, builds donor-aware analysis units, and writes tabular and
visual summaries. Baseline module scores are not recomputed. Optional
gene-removal sensitivity analysis scores derived signatures separately.

## Run from Python

```python
from nasp_atlas.analysis import association_analysis


association_analysis(
    h5ad_path="atlas_subset.h5ad",
    score_csv_path="results/tabula_sapiens_module_scores.csv.gz",
    output_dir="results/tabula_sapiens_associations",
    statistical_unit="donor",
)
```

For a complete tissue run, including scoring and one association analysis per
scorer, use the [command-line or PBS workflow](../run_scripts/README.md).

## Output layout

The tissue workflow keeps Scanpy and AUCell results separate:

```text
<output-root>/<run-name>/
├── scoring/
│   ├── tabula_sapiens_module_scores.csv.gz
│   ├── tabula_sapiens_scorer_concordance.csv
│   ├── tabula_sapiens_cross_scorer_module_correlations.csv
│   ├── tabula_sapiens_scorer_concordance.png
│   └── *.png
└── associations/
    ├── scanpy/
    │   ├── association_tables/
    │   └── association_plots/
    │       ├── mixed_models/
    │       ├── regressions/
    │       └── nasp/
    └── aucell/
        ├── association_tables/
        └── association_plots/
```

Plot generation and scorer concordance are conditional on the requested
options and available inputs.

The sensor and optional per-module gene-expression UMAPs use one shared scale
per figure from 0.0 to the exact highest finite expression value among that
figure's genes. A shared scale makes color comparable within the figure; each
separate sensor or module figure resolves its own maximum.

Each enabled scorer writes one module-score UMAP with separate, native-scale
colorbars (`tabula_sapiens_scanpy_module_umaps.png` or
`tabula_sapiens_aucell_module_umaps.png`). No plot-only standardization is
applied: each panel preserves its module's score scale, and its colors must be
interpreted against that panel's own colorbar.

`tabula_sapiens_scorer_concordance.csv` retains same-module scorer diagnostics.
`tabula_sapiens_cross_scorer_module_correlations.csv` contains every available
Scanpy-module/AUCell-module Spearman correlation used by the square concordance
heatmap. Off-diagonal correlations can reflect shared genes or biological
covariation and are not, by themselves, evidence of scorer disagreement.

## Association tables

Every table lives under `association_tables/`. Core result tables are still
written when no feature can be analyzed; populated rows depend on the requested
stage, available inputs, and model support.

| File | Contents |
| --- | --- |
| `association_provenance.csv` | One-row identity of the h5ad, score table, optional eQTL input, resolved scorer and feature sources, compendium marker-panel hash, copied score-generation provenance, and inferential/plotting configuration. |
| `association_skipped_features.csv` | Requested features that could not be resolved, with feature type and skip reason. |
| `association_mixed_model_contrasts.csv` | Primary planned cell-type, tissue, condition, age, and assay estimates with 95% confidence intervals, donor support, raw p-values, and Benjamini–Hochberg FDR within each `analysis` and `estimand` family. |
| `association_mixed_model_fixed_effects.csv` | Raw fixed-effect coefficients retained for model audit; planned contrasts, not treatment-coded coefficients, are the primary biological results. |
| `association_mixed_model_term_tests.csv` | Wald tests of the coefficient block assigned to each Patsy fixed-effect term. With interactions, a main-effect block is conditional on Patsy's reference levels; planned marginal contrasts remain the primary results. |
| `association_mixed_model_variance_components.csv` | Conditional study, donor, repeated-context, and residual variance estimates and fractions when estimable. |
| `association_mixed_model_diagnostics.csv` | Per-feature formulas, support, exclusions, design rank, optimizer warnings, convergence, software version, and fit status. |
| `association_mixed_model_availability.csv` | Dataset-level availability of all five planned contrast families, variance decomposition, and the multi-study random-effects structure, including explicit non-estimability reasons. |
| `association_regression_results.csv` | Retained optional eQTL regressions; this stable table is empty when no eQTL input is requested. |
| `association_eqtl_annotations.csv` | Optional canonical eQTL counts joined to analyzed units, including match status, merge mode, source identifiers, count units, predictor transformation, and input path. |
| `association_plot_manifest.csv` | Generated association plots and the analysis settings represented by each file. |

The former stratum-wise OLS age, age-stability, tissue-adjusted partial
correlation, and unpaired categorical-test outputs are not called by the
Tabula Sapiens workflow. Their questions are covered by the repeated-donor
models below without treating multiple rows from one donor as independent.

### Planned mixed-model analyses

Every family is fitted separately for each module score or sensor-gene
expression feature. The adjusted-context model includes the available
cell-type, tissue, assay, sex, and condition main effects plus centered age in
decades. Study is also a fixed adjustment when multiple studies are present
but too few exist for a study random intercept. The age model adds
age-by-cell-type interactions. A separate two-arm model is fitted for each
condition-versus-reference comparison, adding condition-by-cell-type
interactions while retaining the other available covariates. Sparse cell-type
strata are excluded before constructing either interaction design, but remain
explicit non-estimable rows in the outputs. The paired-tissue model uses the
adjusted fixed-effect structure but is fitted only to donors observed in at
least two tissues; single-tissue donors cannot influence its tissue
coefficients. Unsupported predictors and interaction levels are not silently
substituted.

| `estimand` | Primary comparison |
| --- | --- |
| `adjusted_cell_type` | Each supported cell type versus the equally weighted mean of the supported cell types, adjusted for the other available model covariates. |
| `paired_tissue` | Within-donor tissue contrasts estimated from the connected multi-tissue-donor network and standardized over donors directly observed in each reported pair. Each pair requires direct paired-donor support; other multi-tissue donors can inform the shared adjusted coefficients and variance, while tissue-only donors are excluded. |
| `condition_by_cell_type` | Each supported condition versus `condition_reference` within each cell type, from a separate two-arm adjusted fit. If no reference is configured, the first donor-supported condition is used. |
| `age_by_cell_type` | Adjusted score change per 10 years within each supported cell type; age is centered at the donor-weighted mean before fitting. |
| `assay_batch_effects` | Each supported assay versus the equally weighted mean of supported assays; this is a technical sensitivity estimate, not a biological tissue or cell-type effect. |

Benjamini–Hochberg correction includes only finite, estimable p-values and is
performed within each `analysis` and planned `estimand` family. Raw fixed
coefficients and coefficient-block tests use separate `analysis`-by-`term`
families and are secondary audit outputs. Scanpy and AUCell runs remain separate
sensitivity analyses rather than sharing one multiplicity family.

The adjusted-context model also decomposes its conditional variability into
the random and residual components supported by the data. A donor random
intercept is always requested. With at least the configured minimum number of
studies, study becomes the top-level random intercept and donor is fitted as a
nested variance component; a repeated donor–tissue–cell-type context component
is included only when supported by repeated assay observations.

The current Tabula Sapiens input has only the `normal` disease level and no
`dataset_id` study column. Consequently, `condition_by_cell_type` and the
multi-study structure are expected to be `not_estimable`; the workflow records
these facts in `association_mixed_model_availability.csv` instead of inventing
a comparison or treating missing structure as zero. A tissue-specific input
similarly makes `paired_tissue` unavailable because only one tissue level is
present.

### Combined-input and per-tissue scopes

`mixed_models_combined=True` retains the combined-input analysis.
Independently, `mixed_models_per_tissue=True` fits each observed tissue using
all its scored cells. Both scopes reuse one set of scores generated over the
selected input: per-tissue inference does not change the scoring reference.
Omit tissue and fraction filters to use the full input atlas. These options
are also exposed by the [PBS wrapper](../run_scripts/README.md).

Combined tables retain their existing locations. Per-tissue results live in
`mixed_models_by_tissue/<tissue-label>_<hash>/`, with their own association
tables and plots. `association_tables/association_mixed_model_scopes.csv`
records cell counts, scope locations, and pending/running/completed/failed or
disabled status. A completed scope may still contain non-estimable contrasts;
inspect its diagnostics and availability tables. In particular, a per-tissue
fit cannot estimate a between-tissue contrast.

FDR is calculated separately within each scope and its existing analysis and
estimand families, **not jointly across tissues**. Per-tissue findings are
exploratory; do not pool their adjusted p-values as atlas-wide FDR control.
Disabled outputs from earlier runs are retained; consult the current manifests
and provenance before using them.

### Optional donor and gene robustness checks

Each scorer writes these tables under `robustness/`. All are descriptive
sensitivity checks, not additional causal or significance tests. The
`robustness_manifest.csv` records enabled stages and their completion;
`robustness_provenance.csv` records inputs and analysis settings. Gene removal
automatically enables its prerequisite gene diagnostics.

With `plot_nasp_visualizations=True` (PBS: `PLOT_NASP_VISUALIZATIONS=1`),
enabled robustness stages also write compact PNG/PDF summaries and displayed
data under `robustness/figures/`. No separate plotting command is needed.
The stage manifest lists current figures; unsupported comparisons are not
plotted as zero. See [visualization guidance](visualization.md#robustness-diagnostics)
for selection rules and interpretation.

| Option | Main outputs and use |
| --- | --- |
| `run_donor_sensitivity` | `nasp_donor_scores.csv`, `nasp_context_ranks.csv`, `nasp_leave_one_donor_out.csv`, `nasp_matched_tissue_differences.csv`: donor-level scores, context rankings, donor-deletion stability, and within-donor tissue differences for matching cell types. |
| `run_gene_diagnostics` | `nasp_gene_donor_expression.csv`, `nasp_gene_context_diagnostics.csv`: signed module membership, expression and detection per donor/context/assay, support, and gene–score correlations. Missing, ambiguous and undetected genes remain distinct. |
| `run_gene_removal_sensitivity` | `nasp_gene_removal_variants.csv`, `nasp_gene_removal_donor_scores.csv`, `nasp_gene_removal_context_changes.csv`, `nasp_overlap_removed_coupling.csv`: actual rescoring after individual dominant-gene deletions and shared-gene removal from mechanistic module pairs. |

Rankings use medians of donor mean scores, pooling assays within each donor
and context. Matching-cell-type ranks compare tissues without mixing cell
identities. Leave-one-donor-out analysis removes that donor from all contexts
together; a study-qualified donor identifier is used when available. Too few
remaining donors yields an unsupported result, not an artificial rank. Tissue
differences use only donors observed in both tissues and the same cell type.

Gene diagnostics retain assays separately. Driver selection is a heuristic:
maximum, across supported contexts, of median absolute expression divided by
the number of genes in that score arm. It is not an exact Scanpy or AUCell
contribution. Gene–score correlations include a part–whole relationship when
the gene belongs to the score; they are not independent validation. By default
the top candidate per module is removed; `sensitivity_dominant_genes` requests
more candidates, each removed separately.

Derived signatures preserve scoring direction, including inverse arms, and
never edit the compendium. Losing a signed arm is explicitly unscorable.
Scanpy reruns its control selection with the original seed and full background;
AUCell reranks the same expression source. Signed arms are standardized over
the same scored cells, not separately by tissue. Changes therefore measure
the complete scoring procedure's sensitivity, not a subtraction from a score.
Shared-gene coupling comparisons use identical complete donor/context rows,
both across contexts and after context centering; repeated contexts are not
independent donors and no p-values are assigned. Lack of shared genes and
insufficient support are recorded explicitly.

### Reusable inference API

`mixed_model_inference` is dataset-agnostic: callers provide an already
aggregated long frame, identify the repeated-measures group and independent
biological unit, and declare predictor columns rather than constructing a
Patsy formula. Arbitrary source column names are internally aliased. Each
feature must retain the canonical `feature_type`, `feature_id`, and
`feature_label` identity columns so all result tables share one stable schema.

```python
from nasp_atlas.single_cell import MixedModelContrast
from nasp_atlas.single_cell import MixedModelSpec
from nasp_atlas.single_cell import mixed_model_inference


spec = MixedModelSpec(
    analysis="age_by_cell_type",
    group_key="donor_id",
    independent_unit_key="donor_id",
    continuous_keys=("age_decades_centered",),
    categorical_keys=("cell_type", "tissue", "assay", "sex"),
    interactions=(("age_decades_centered", "cell_type"),),
    variance_component_keys=("donor_context_id",),
)
result = mixed_model_inference(
    aggregated_features,
    spec=spec,
    contrasts=(
        MixedModelContrast(
            mode="continuous_slope",
            predictor_key="age_decades_centered",
            by_key="cell_type",
            estimand="age_by_cell_type",
            effect_scale="score_change_per_10_years",
        ),
    ),
)
```

The generic result keeps planned contrasts, raw fixed effects,
coefficient-block term tests, conditional variance components, and fit
diagnostics separate. It does
not silently fall back to OLS or a simpler random structure when the requested
model is unsupported or fails.

The eQTL stage accepts canonical tables and the NASP sensor-count long or wide
schemas. Gene mode uses wide or long gene totals for a descriptive comparison
across the curated sensor genes. Gene-tissue mode uses the long table for both
per-gene comparisons across tissues and within-tissue comparisons across
genes. Tissue mode sums significant pairs over source sensor genes, then
compares each module score or sensor-gene expression value across tissues.
Module-keyed input remains annotation-only because a single fixed count per
module cannot support a within-module regression.

Gene and tissue comparisons first average donor-level values at the count's
actual unit; repeated donor rows therefore do not inflate support. Tissue joins
normalize spelling and case only and retain both source and atlas labels rather
than silently applying a biological tissue crosswalk. The supplied counts are
significant gene-variant pairs, not unique causal variants, and depend on eQTL
discovery power. These associations are exploratory descriptions of selected
genes and overlapping tissues, not evidence of regulatory direction or
mechanism. Counts are tested on their reported raw scale; Spearman results
provide the rank-based companion to Pearson correlation and the raw-count OLS
slope.

### NASP-focused tables

| File | Contents |
| --- | --- |
| `nasp_evidence_profiles.csv` | Donor-tissue-cell-type percentile axes and non-exclusive relative flags for competent, active-like, responsive-like, restricted, feedback-high, and post-NASP states. |
| `nasp_context_summary.csv` | Tissue-cell-type summaries of relative evidence axes, mismatch gaps, independent-unit counts, and donor support. |
| `nasp_hypothesis_priorities.csv` | Ranked active-like, responsive-like, restricted or buffered, feedback-dominant, and post-without-NASP hypotheses with cautious experiment-oriented follow-ups. |
| `nasp_mechanistic_edges.csv` | A compact, prespecified subset of sensing-to-signaling or output and output-to-feedback or phenotype module pairs, extracted from the full coupling table. |
| `nasp_module_context_ranking.csv` | Per-module context ranks with finite-score coverage, donor support, and contributing-cell counts; under-supported contexts remain visible but unranked. |
| `nasp_module_coupling.csv` | Raw and within-context-centered module correlations, FDR values, support, and signed marker-gene overlap annotations. |
| `nasp_sensor_output_coupling.csv` | Individual sensor-expression correlations with canonical output modules, including whether each sensor is itself an output-module member. |

## Figures

When NASP visualizations are enabled, the workflow requests the following
inference figures under `association_plots/mixed_models/`. Effect figures need
estimable contrasts; the variance figure preserves unavailable components when
variance rows exist.

| Filename stem | View |
| --- | --- |
| `nasp_mixed_adjusted_cell_type_effects` | Adjusted cell-type deviations with 95% confidence intervals. |
| `nasp_mixed_condition_effects_by_cell_type` | Condition-versus-reference effects within cell type. |
| `nasp_mixed_age_slopes_by_cell_type` | Adjusted score change per 10 years within cell type. |
| `nasp_mixed_paired_tissue_effects` | Adjusted tissue differences among paired donors. |
| `nasp_mixed_assay_batch_effects` | Adjusted assay deviations. |
| `nasp_mixed_variance_decomposition` | Conditional study, donor, repeated-context, and residual variance fractions. |

`plot_nasp_visualizations=False` (or `--no-nasp-visualizations`) skips both the
mixed-model and descriptive NASP figure sets without suppressing their result
tables. Effect figures show only rows marked estimable after support and fit
checks. Filled points meet FDR 0.05 and open points do not. The variance plot
preserves the saved fractions without renormalizing and marks missing or
unallocated components separately from numeric zero. A requested effect figure
is skipped when its analysis has no estimable rows. The workflow's fixed plots
show `module_score` features; gene-expression mixed-model results remain in the
same tables and can be replotted by selecting that feature type.

When NASP visualizations are enabled, `association_plots/nasp/` may contain:

| Filename stem | View |
| --- | --- |
| `nasp_module_coupling_heatmap` | Pairwise module coupling. |
| `nasp_competence_output_state_map` | Relative competence and output evidence by context. |
| `nasp_ranked_hypotheses` | Highest-priority context-hypothesis combinations. |
| `nasp_sensor_output_mismatch` | Sensor-expression and output-module coupling. |
| `nasp_mechanistic_edge_network` | Prespecified directional hypotheses overlaid with symmetric module-pair correlations. |

The regression directory is reserved for explicitly requested cell-level
descriptive plots. Such plots are QC views, not mixed-model inference.

## Scientific interpretation

These outputs are hypothesis-generating transcriptomic summaries. Keep four
evidence levels distinct:

1. Sensor expression and module scores are observed or derived measurements.
2. Relative NASP profiles and context ranks summarize those measurements
   within the analyzed atlas.
3. Planned mixed-model contrasts estimate adjusted associations while
   accounting for repeated contexts from the same donor; optional eQTL
   regressions retain their separately declared unit.
4. Mechanistic activation, sender-receiver relationships, direction, and
   causality require independent experimental evidence.

### Statistical unit and estimand

The mixed-model observational row is one equally weighted aggregate per donor,
study when available, tissue, cell type, assay when available, and feature,
after requiring the configured minimum number of contributing cells. The
independent biological unit is the donor, qualified by study when study
metadata are available; donor is also the repeated-measures grouping factor
unless the supported multi-study structure places study above nested donors.
Cell counts measure precision and filtering support, not biological
replication.

`statistical_unit` continues to configure retained non-mixed stages such as
optional eQTL analysis; it does not change this prespecified mixed-model unit.
The primary estimands are the marginal contrasts and simple slopes in
`association_mixed_model_contrasts.csv`, not raw treatment-coded coefficients
or cell-level differences.

### Relative states and rankings

Evidence axes and flags are relative to the analyzed atlas. Flags are
non-exclusive summaries, not discrete biological states:

- An active-like profile combines relatively high competence and output
  evidence; it is not proof of pathway activation.
- A responsive-like profile can reflect paracrine signaling, technical
  dropout, an unmeasured sensor, or other biology. It is not proof of a
  sender-receiver link.
- Restricted, feedback-high, and post-NASP profiles identify expression
  patterns compatible with those interpretations, not confirmed mechanisms.

Recompute percentiles, context ranks, and hypothesis priorities after changing
the atlas or combining tissues. Per-tissue ranks are not absolute quantities
and must not be treated as global ranks by concatenation.

### Coupling and mixed-model inference

Module coupling and mechanistic-edge tables contain correlations. Prespecified
edge labels provide biological context but do not make a symmetric correlation
directional or causal. Signed marker overlap is an annotation of shared module
definitions, not independent evidence of coupling.

Mixed-model estimates remain observational associations. Adjustment and a
donor random intercept reduce confounding and pseudoreplication under the
specified linear model, but they do not establish pathway activation,
mechanism, or causality. Variance fractions are conditional on the fitted
fixed and random structure and should not be interpreted as immutable
biological proportions.

### Missingness and support

Use the skipped-feature, availability, diagnostic, support, `status`, and
`reason` fields when interpreting an absent estimate. The default thresholds
require at least 10 cells per aggregate observation and at least 3 independent
donors per modeled level or pair. Unmeasured, undetected, filtered,
non-estimable, non-converged, and numeric zero have different meanings; a
failed or rank-deficient model is never replaced by a simpler unreported
model. Review `converged`, optimizer warnings, exclusions, and design rank
before interpreting estimates. Confidence intervals and p-values rely on the
specified linear mixed model and its large-sample approximations.

## Replot mixed-model inference

The workflow calls the fixed Tabula Sapiens plot set. It can also be reproduced
from the two primary saved inference tables:

```python
import pandas as pd

from nasp_atlas.analysis import plot_tabula_sapiens_mixed_model_inference


tables = "results/tabula_sapiens_associations/association_tables"
plot_tabula_sapiens_mixed_model_inference(
    output_dir="results/tabula_sapiens_associations/association_plots/mixed_models",
    contrasts=pd.read_csv(f"{tables}/association_mixed_model_contrasts.csv"),
    variance_components=pd.read_csv(
        f"{tables}/association_mixed_model_variance_components.csv"
    ),
)
```

## Reproducibility checklist

Retain these with exported results:

- input h5ad identity and matching score table;
- compendium source and revision;
- scorer, expression layer, gene-symbol mapping, and module selection;
- observational and independent units, aggregation, fixed and random effects,
  condition reference, age center, support thresholds, and exclusions;
- random seed and relevant software versions;
- skipped-feature, availability, support, convergence, warning, and
  non-estimable diagnostics.
## Mechanism and reference companions

Enabled mechanism diagnostics write to `associations/<scorer>/mechanisms/`.
They use the complete saved scoring population, the same expression source,
and current compendium definitions. A stage manifest and provenance accompany
the tables; unsupported results remain missing, not zero.

- `regulator_output_coupling.csv`: individual feedback/restriction genes versus
  IFN-I, NF-kappa-B/cytokine, ISR, inflammasome and SASP outputs when scored.
  Cell means are aggregated by study-qualified donor, tissue, cell type and
  assay. Spearman correlations use context-centered means after minimum-cell
  and minimum-donor filtering. Whole-donor deletion ranges are sensitivity
  summaries, not confidence intervals. A regulator present in the output gene
  set is excluded as circular. These descriptive associations are not adjusted
  causal effects; age, donor state and other covariates may explain them.
  Scanpy control-gene subtraction can also introduce coupling; compare the
  separate AUCell results before prioritizing a mechanism.
- `regulator_output_points.csv`: supported donor-context points for the two
  largest absolute non-circular correlations. Selection is exploratory;
  repeated contexts do not increase the number of independent donors.
- `mechanism_components.csv`: donor-median expression and detection for IFN
  ligands (including IFNG separately), IFN-I receptor context, response genes,
  and ordered OAS1/2/3 plus RNASEL. This is a companion view, not a replacement
  IFN score. RNA abundance cannot establish ligand secretion, RNase L activity,
  signaling direction or temporal order; lack of detected ligand is not proof
  of paracrine signaling. Unmeasured genes and measured-but-undetected genes
  retain distinct states.
- `reference_comparisons.csv`: donor/context-centered reference-versus-curated
  score correlations and signed-arm gene overlap. Nested pathways/shared genes
  prevent interpreting agreement as independent validation. Reference scores
  do not enter the curated NASP-state classification or mixed-model feature
  selection; their purpose is a separate expression-based sensitivity view.

The compact figures in `mechanisms/figures/` export PNG, vector PDF and their
displayed data. Component overviews prioritize donor-supported contexts and
use within-gene z-scores across displayed contexts, not cross-gene expression
comparisons. The complete unscaled values remain in the table.
