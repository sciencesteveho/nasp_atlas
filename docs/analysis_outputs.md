# Analysis outputs and interpretation

[Back to the project README](../README.md)

`association_analysis` joins a saved module-score table to the matching cells
in an AnnData file, builds donor-aware analysis units, and writes tabular and
visual summaries. It does not recompute module scores.

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
    │       ├── regressions/
    │       ├── boxplots/
    │       ├── barplots/
    │       └── nasp/
    └── aucell/
        ├── association_tables/
        └── association_plots/
```

Plot generation and scorer concordance are conditional on the requested
options and available inputs.

`tabula_sapiens_scorer_concordance.csv` retains same-module scorer diagnostics.
`tabula_sapiens_cross_scorer_module_correlations.csv` contains every available
Scanpy-module/AUCell-module Spearman correlation used by the square concordance
heatmap. Off-diagonal correlations can reflect shared genes or biological
covariation and are not, by themselves, evidence of scorer disagreement.

## Association tables

Every table lives under `association_tables/`. Core result tables are still
written when no feature can be analyzed; diagnostics and conditional tables
appear only when their stage and inputs apply.

| File | Contents |
| --- | --- |
| `association_skipped_features.csv` | Requested features that could not be resolved, with feature type and skip reason. |
| `association_regression_results.csv` | Continuous-predictor estimates, including age and eQTL associations, support, analysis scope, raw p-values, and adjusted p-values where estimable. |
| `association_age_stability.csv` | Directional consistency of within-tissue and within-tissue-cell-type age slopes across tested strata. |
| `association_partial_correlation_age.csv` | Age associations adjusted for tissue when the required tissue variation exists. |
| `association_group_test_results.csv` | Donor-aware categorical tests for available sex, tissue, and cell-type comparisons. |
| `association_group_summary.csv` | Descriptive group summaries accompanying the categorical tests. |
| `association_eqtl_annotations.csv` | Optional canonical eQTL counts joined to analyzed units, including match status, merge mode, source identifiers, count units, predictor transformation, and input path. |
| `association_plot_manifest.csv` | Generated association plots and the analysis settings represented by each file. |

Sex associations remain available in the group-test and summary tables. Group
figures encode donor sex within each tissue or cell-type category as adjacent
bars or boxplots rather than writing a separate sex-only figure.

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

When NASP visualizations are enabled, `association_plots/nasp/` may contain:

| Filename stem | View |
| --- | --- |
| `nasp_module_coupling_heatmap` | Pairwise module coupling. |
| `nasp_competence_output_state_map` | Relative competence and output evidence by context. |
| `nasp_ranked_hypotheses` | Highest-priority context-hypothesis combinations. |
| `nasp_sensor_output_mismatch` | Sensor-expression and output-module coupling. |
| `nasp_age_effects_by_cell_type` | Context-specific age effects on NASP module scores. |
| `nasp_sensor_age_effects_by_cell_type` | Context-specific age effects on nucleic-acid sensor-gene expression. |
| `nasp_age_effect_consistency_across_cell_types` | Cross-context stability of age effects on module scores. |
| `nasp_sensor_age_effect_consistency_across_cell_types` | Cross-context stability of age effects on nucleic-acid sensor-gene expression. |
| `nasp_mechanistic_edge_network` | Prespecified directional hypotheses overlaid with symmetric module-pair correlations. |

General regression, boxplot, and barplot files live in their corresponding
directories. Their exact set depends on input metadata, estimability, and
`max_plots`.

## Scientific interpretation

These outputs are hypothesis-generating transcriptomic summaries. Keep four
evidence levels distinct:

1. Sensor expression and module scores are observed or derived measurements.
2. Relative NASP profiles and context ranks summarize those measurements
   within the analyzed atlas.
3. Regression and group-test tables estimate associations at the declared
   statistical unit.
4. Mechanistic activation, sender-receiver relationships, direction, and
   causality require independent experimental evidence.

### Statistical unit and estimand

The default pooled inferential unit is one aggregated row per donor. The
workflow also estimates age slopes within tissue and tissue-cell-type strata,
using donors as independent observations within each stratum. Cell-level plots
are descriptive QC only unless a model explicitly accounts for dependence
between cells from the same donor.

Changing `statistical_unit` changes the estimand. Finer pooled units such as
donor-tissue or donor-tissue-cell-type rows can repeat donors and therefore
need a repeated-measures model for formal pooled inference. Report cell counts
as observational support, not as the inferential sample size.

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

### Coupling and stability

Module coupling and mechanistic-edge tables contain correlations. Prespecified
edge labels provide biological context but do not make a symmetric correlation
directional or causal. Signed marker overlap is an annotation of shared module
definitions, not independent evidence of coupling.

`association_age_stability.csv` is a directional replication screen across
tested strata. It is not a repeated-measures effect estimate. A summary with
too few estimable strata remains under-supported rather than becoming a null
or zero result.

### Missingness and support

Use the skipped-feature, support, and status fields when interpreting an absent
estimate. Unmeasured, undetected, filtered, non-estimable, and numeric zero have
different meanings. Under-supported contexts remain visible where possible.

## Atlas-wide visualization

After tissue tables are combined, recompute global context percentiles,
hypothesis priorities, and age stability before plotting:

```python
from nasp_atlas.analysis import plot_global_nasp_visualizations


plot_global_nasp_visualizations(
    output_dir="results/global_nasp_plots",
    module_coupling=module_coupling,
    context_summary=context_summary,
    hypothesis_priorities=hypothesis_priorities,
    sensor_output_coupling=sensor_output_coupling,
    regression_results=regression_results,
    age_stability=age_stability,
    mechanistic_edges=mechanistic_edges,
)
```

## Reproducibility checklist

Retain these with exported results:

- input h5ad identity and matching score table;
- compendium source and revision;
- scorer, expression layer, gene-symbol mapping, and module selection;
- statistical unit, aggregation, covariates, strata, and exclusions;
- random seed and relevant software versions;
- skipped-feature, support, and non-estimable diagnostics.
