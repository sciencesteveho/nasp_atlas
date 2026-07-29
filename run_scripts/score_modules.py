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
        help="Reuse an existing score table only when all scorers completed.",
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
            "Render dynamically sized NASP summaries, faceted by tissue for "
            "complete-atlas inputs."
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
        max_plots=args.max_plots,
        plot_nasp_visualizations=args.nasp_visualizations,
    )
    for output_name, output_path in outputs.items():
        logger.info("%s -> %s", output_name, output_path)


if __name__ == "__main__":
    main()
