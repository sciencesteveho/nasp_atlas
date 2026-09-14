"""Run end-to-end NASP scoring and analysis for a Tabula Sapiens h5ad."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nasp_atlas.analysis import tabula_sapiens_tissue_analysis


logger = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Parse Tabula Sapiens analysis command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Score NASP modules in a complete or tissue-subset Tabula Sapiens "
            "h5ad, then run donor-aware analysis once per scorer."
        ),
    )
    parser.add_argument(
        "--h5ad-path",
        type=Path,
        required=True,
        help="Input complete-atlas or tissue-specific Tabula Sapiens h5ad.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results"),
        help="Root directory for isolated scoring and analysis runs.",
    )
    parser.add_argument(
        "--tissue-label",
        "--single-tissue",
        dest="tissue_label",
        default=None,
        help=(
            "Exact tissue obs label to subset and re-embed. Omit to analyze "
            "all cells in the input h5ad and retain its existing embedding."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output directory label; defaults to tissue label or h5ad stem.",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        choices=("scanpy", "aucell"),
        default=("scanpy", "aucell"),
        help="Scorers to calculate and analyze independently.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse scores when the completion manifest and request match.",
    )
    for option, default, description in (
        ("mixed-models-combined", True, "Fit all scored cells together."),
        (
            "mixed-models-per-tissue",
            False,
            "Independently fit every scored tissue.",
        ),
        (
            "donor-sensitivity",
            False,
            "Write donor deletion ranks and paired tissue differences.",
        ),
        (
            "gene-diagnostics",
            False,
            "Write gene expression/detection by donor and assay.",
        ),
        (
            "gene-removal-sensitivity",
            False,
            "Rescore dominant-gene and shared-gene deletions.",
        ),
    ):
        parser.add_argument(
            f"--{option}",
            action=argparse.BooleanOptionalAction,
            default=default,
            help=description,
        )
    parser.add_argument(
        "--sensitivity-dominant-genes",
        type=int,
        default=1,
        help="Driver genes removed separately per module.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print worker options without loading data or writing files.",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=None,
        help="Optional cell fraction for an exploratory run.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--gene-symbol-column",
        default="feature_name",
    )
    parser.add_argument(
        "--expression-layer",
        default=None,
        help="Expression layer. Omit to use adata.X.",
    )
    parser.add_argument(
        "--module-ids",
        nargs="+",
        default=None,
        help="Optional module subset; defaults to every compendium module.",
    )
    parser.add_argument(
        "--sensor-group",
        default="nucleic_acid_sensors",
    )
    parser.add_argument(
        "--plot-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate per-module marker plots.",
    )
    parser.add_argument("--aucell-chunk-size", type=int, default=1_000)
    parser.add_argument("--aucell-num-workers", type=int, default=1)
    representation = parser.add_mutually_exclusive_group()
    representation.add_argument(
        "--single-tissue-use-rep",
        default="X_scvi",
        help="Representation used to recompute the tissue UMAP.",
    )
    representation.add_argument(
        "--single-tissue-use-x",
        dest="single_tissue_use_rep",
        action="store_const",
        const=None,
        help="Use adata.X to recompute the tissue UMAP.",
    )
    parser.add_argument(
        "--statistical-unit",
        choices=(
            "cell",
            "metacell",
            "donor",
            "donor_tissue",
            "donor_tissue_cell_type",
            "donor_tissue_sex",
        ),
        default="donor",
    )
    parser.add_argument(
        "--aggregation",
        choices=(
            "mean",
            "median",
            "sum",
            "fraction_expressing",
            "percent_expressing",
        ),
        default="mean",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.0,
        help="Finite expression floor for fraction/percent aggregations.",
    )
    parser.add_argument(
        "--condition-key",
        default="disease",
        help="Obs column used for condition-by-cell-type mixed models.",
    )
    parser.add_argument(
        "--condition-reference",
        default="normal",
        help="Reference condition for within-cell-type contrasts.",
    )
    parser.add_argument(
        "--study-key",
        default="dataset_id",
        help="Obs column identifying studies in combined-atlas models.",
    )
    parser.add_argument(
        "--mixed-model-min-cells",
        type=int,
        default=10,
        help="Minimum finite cell values in each donor-context-assay row.",
    )
    parser.add_argument(
        "--mixed-model-min-donors",
        type=int,
        default=3,
        help="Minimum independent donors supporting a modeled level.",
    )
    parser.add_argument(
        "--mixed-model-min-studies",
        type=int,
        default=3,
        help="Minimum studies required for a study random intercept.",
    )
    parser.add_argument(
        "--mixed-model-min-repeated-contexts",
        type=int,
        default=3,
        help="Minimum repeated assay contexts for a context random effect.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=200,
        help="Maximum association plots per scorer.",
    )
    parser.add_argument(
        "--nasp-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render fixed mixed-model inference and NASP summary figures, "
            "faceted by tissue where applicable."
        ),
    )
    parser.add_argument(
        "--mechanism-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regulator branches, IFN components, and OAS/RNase L diagnostics.",
    )
    parser.add_argument(
        "--reference-sets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Score shipped Reactome/Hallmark sets and compare with curated "
            "scores."
        ),
    )
    parser.add_argument(
        "--score-table-filename",
        default="tabula_sapiens_module_scores.csv.gz",
    )
    return parser.parse_args()


def main() -> None:
    """Run a complete atlas or tissue analysis through associations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_arguments()
    if args.dry_run:
        for name, value in vars(args).items():
            logger.info("%s = %s", name, value)
        return
    outputs = tabula_sapiens_tissue_analysis(
        h5ad_path=args.h5ad_path,
        output_dir=args.output_path,
        tissue_label=args.tissue_label,
        run_name=args.run_name,
        scorers=args.scorers,
        resume_from_scores=args.resume,
        subset_fraction=args.subset_fraction,
        random_state=args.random_state,
        gene_symbol_column=args.gene_symbol_column,
        expression_layer=args.expression_layer,
        module_ids=args.module_ids,
        sensor_group=args.sensor_group,
        plot_modules=args.plot_modules,
        aucell_chunk_size=args.aucell_chunk_size,
        aucell_num_workers=args.aucell_num_workers,
        score_table_filename=args.score_table_filename,
        single_tissue_use_rep=args.single_tissue_use_rep,
        statistical_unit=args.statistical_unit,
        aggregation=args.aggregation,
        detection_threshold=args.detection_threshold,
        condition_key=args.condition_key,
        condition_reference=args.condition_reference,
        study_key=args.study_key,
        mixed_model_min_cells=args.mixed_model_min_cells,
        mixed_model_min_donors=args.mixed_model_min_donors,
        mixed_model_min_studies=args.mixed_model_min_studies,
        mixed_model_min_repeated_contexts=(
            args.mixed_model_min_repeated_contexts
        ),
        max_plots=args.max_plots,
        plot_nasp_visualizations=args.nasp_visualizations,
        mixed_models_combined=args.mixed_models_combined,
        mixed_models_per_tissue=args.mixed_models_per_tissue,
        run_donor_sensitivity=args.donor_sensitivity,
        run_gene_diagnostics=args.gene_diagnostics,
        run_gene_removal_sensitivity=args.gene_removal_sensitivity,
        sensitivity_dominant_genes=args.sensitivity_dominant_genes,
        run_mechanism_diagnostics=args.mechanism_diagnostics,
        include_reference_sets=args.reference_sets,
    )
    for output_name, output_path in outputs.items():
        logger.info("%s -> %s", output_name, output_path)


if __name__ == "__main__":
    main()
