"""Query CELLxGENE for dataset metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Self

import cellxgene_census  # type: ignore
import pandas as pd

from nasp_atlas.cellxgene.categorize import _collapse_sex_series
from nasp_atlas.cellxgene.categorize import _summarize_development_stage
from nasp_atlas.cellxgene.config import CXGMetadataConfig
from nasp_atlas.cellxgene.filter import _annotate_obs_categories
from nasp_atlas.cellxgene.filter import _filter_obs_by_category
from nasp_atlas.cellxgene.visualization.age import _plot_age_makeup
from nasp_atlas.cellxgene.visualization.age import _plot_age_ranges
from nasp_atlas.cellxgene.visualization.composition import _metadata_barplot
from nasp_atlas.cellxgene.visualization.composition import _plot_category_makeup
from nasp_atlas.cellxgene.visualization.sankey import _metadata_sankey
from nasp_atlas.common import _collapse_unique_series
from nasp_atlas.visualization import _set_matplotlib_publication_parameters


class CXGMetadata:
    """Stateful workflow for CELLxGENE metadata querying and visualization."""

    def __init__(
        self,
        datasets: pd.DataFrame,
        obs: pd.DataFrame,
        config: CXGMetadataConfig | None = None,
    ) -> None:
        """Initialize with dataset-level and cell-level metadata."""
        self.datasets = datasets
        self.obs = obs
        self.config = config or CXGMetadataConfig()

        _set_matplotlib_publication_parameters()

    def annotate_obs_categories(
        self,
        *,
        source_column: str,
        target_column: str,
        categorizer: Callable[[object], str],
    ) -> Self:
        """Add a broad-category column to obs."""
        self.obs = _annotate_obs_categories(
            self.obs,
            source_column=source_column,
            target_column=target_column,
            categorizer=categorizer,
        )
        return self

    def annotate_default_categories(self) -> Self:
        """Add default disease and tissue category columns to obs."""
        self.annotate_obs_categories(
            source_column="disease",
            target_column="disease_category",
            categorizer=self.config.categorize_disease,
        )
        self.annotate_obs_categories(
            source_column="tissue",
            target_column="tissue_category",
            categorizer=self.config.categorize_tissue,
        )
        return self

    def _ensure_default_category_column(self, category_column: str) -> None:
        """Create a default category column when a default plot needs it."""
        if category_column in self.obs.columns:
            return

        if category_column == "disease_category":
            self.annotate_obs_categories(
                source_column="disease",
                target_column=category_column,
                categorizer=self.config.categorize_disease,
            )
            return

        if category_column == "tissue_category":
            self.annotate_obs_categories(
                source_column="tissue",
                target_column=category_column,
                categorizer=self.config.categorize_tissue,
            )

    def _categorizer_for_label_column(
        self,
        label_column: str,
    ) -> Callable[[object], str] | None:
        """Return the configured categorizer for a metadata label column."""
        if label_column == "disease":
            return self.config.categorize_disease

        if label_column == "tissue":
            return self.config.categorize_tissue

        return (
            str
            if label_column in {"disease_category", "tissue_category"}
            else None
        )

    def filter_by_category(
        self,
        *,
        column: str,
        keep: Iterable[str],
    ) -> Self:
        """Restrict obs to rows whose category value is in keep."""
        self.obs = _filter_obs_by_category(
            self.obs,
            column=column,
            keep=keep,
        )
        return self

    def filter_diseases(
        self,
        keep: Iterable[str],
        column: str = "disease_category",
    ) -> Self:
        """Restrict obs to selected disease categories."""
        return self._filter_by_category(column, keep)

    def filter_tissues(
        self,
        keep: Iterable[str],
        column: str = "tissue_category",
    ) -> Self:
        """Restrict obs to selected tissue categories."""
        return self._filter_by_category(column, keep)

    def _filter_by_category(self, column, keep):
        """Ensure the category column exists and filter obs by selected
        values.
        """
        self._ensure_default_category_column(column)
        self.filter_by_category(column=column, keep=keep)
        return self

    def plot_category_makeup(
        self,
        *,
        category_column: str,
        outpath: str | Path,
        front: Sequence[str] = (),
        back: Sequence[str] = (),
        datasets_per_plot: int = 35,
    ) -> None:
        """Plot dataset makeup along one categorical axis."""
        _plot_category_makeup(
            obs=self.obs,
            datasets=self.datasets,
            category_column=category_column,
            outpath=outpath,
            front=tuple(front),
            back=tuple(back),
            datasets_per_plot=datasets_per_plot,
        )

    def plot_disease_makeup(
        self,
        outpath: str | Path,
        *,
        category_column: str = "disease_category",
        front: Sequence[str] = (),
        back: Sequence[str] = (),
        datasets_per_plot: int = 35,
    ) -> None:
        """Plot dataset makeup by disease category.

        Args:
          outpath: File path to save the plot.
          category_column: The column containing the category values.
          front: Categories to display at the front of the plots.
          back: Categories to display at the end of the plots.
          datasets_per_plot: # of datasets per plot.
        """
        self._ensure_default_category_column(category_column)
        self.plot_category_makeup(
            category_column=category_column,
            outpath=outpath,
            front=front,
            back=back,
            datasets_per_plot=datasets_per_plot,
        )

    def plot_tissue_makeup(
        self,
        outpath: str | Path,
        *,
        category_column: str = "tissue_category",
        front: Sequence[str] = (),
        back: Sequence[str] = (),
        datasets_per_plot: int = 35,
    ) -> None:
        """Plot dataset makeup by tissue category.

        Args:
          outpath: File path to save the plot.
          category_column: The column containing the category values.
          front: Categories to display at the front of the plots.
          back: Categories to display at the end of the plots.
          datasets_per_plot: # of datasets per plot.
        """
        self._ensure_default_category_column(category_column)
        self.plot_category_makeup(
            category_column=category_column,
            outpath=outpath,
            front=front,
            back=back,
            datasets_per_plot=datasets_per_plot,
        )

    def metadata_barplot(
        self,
        *,
        label_column: str,
        outpath: str | Path,
        dataset_id: str | None = None,
        grouped: bool = False,
        bar_height_in: float = 0.18,
        fig_width_in: float = 2.75,
        cmap: str = "tab20",
    ) -> None:
        """Plot metadata barplot for current obs or one dataset."""
        _metadata_barplot(
            obs=self.obs,
            dataset_id=dataset_id,
            label_column=label_column,
            outpath=outpath,
            grouped=grouped,
            bar_height_in=bar_height_in,
            fig_width_in=fig_width_in,
            cmap=cmap,
            categorizer=self._categorizer_for_label_column(label_column),
        )

    def metadata_sankey(
        self,
        *,
        label_column: str,
        outpath: str | Path,
        dataset_id: str | None = None,
        node_height_in: float = 0.22,
        fig_width_in: float = 4.5,
        cmap: str = "tab20",
    ) -> Path:
        """Plot metadata Sankey for current obs or one dataset."""
        return _metadata_sankey(
            obs=self.obs,
            dataset_id=dataset_id,
            label_column=label_column,
            outpath=outpath,
            node_height_in=node_height_in,
            fig_width_in=fig_width_in,
            cmap=cmap,
            categorizer=self._categorizer_for_label_column(label_column),
        )

    def plot_age_ranges(
        self,
        outpath: str | Path,
        *,
        dataset_id: str | None = None,
        stage_column: str = "development_stage",
        collapsed: bool = True,
        bar_height_in: float = 0.18,
        fig_width_in: float = 3.5,
        cmap: str = "tab20",
    ) -> Path:
        """Plot development-stage age ranges for current obs or one dataset."""
        return _plot_age_ranges(
            obs=self.obs,
            outpath=outpath,
            dataset_id=dataset_id,
            stage_column=stage_column,
            collapsed=collapsed,
            bar_height_in=bar_height_in,
            fig_width_in=fig_width_in,
            cmap=cmap,
        )

    def plot_age_makeup(
        self,
        outpath: str | Path,
        *,
        stage_column: str = "development_stage",
        collapsed: bool = True,
        datasets_per_plot: int = 35,
        bar_height_in: float = 0.15,
        fig_width_in: float = 2.85,
        cmap: str = "tab20c",
    ) -> None:
        """Plot dataset makeup across development-stage age ranges."""
        _plot_age_makeup(
            obs=self.obs,
            datasets=self.datasets,
            outpath=outpath,
            stage_column=stage_column,
            collapsed=collapsed,
            datasets_per_plot=datasets_per_plot,
            bar_height_in=bar_height_in,
            fig_width_in=fig_width_in,
            cmap=cmap,
        )

    def to_csv(
        self,
        outpath: str | Path,
        *,
        sep: str = "\t",
        index: bool = False,
    ) -> Path:
        """Write summarized dataset metadata to disk."""
        output_path = Path(outpath)
        self.summarize_datasets().to_csv(output_path, sep=sep, index=index)
        return output_path

    def summarize_datasets(self) -> pd.DataFrame:
        """Summarize primary-cell metadata by dataset.

        Args:
        datasets: Dataset-level Census metadata from census_info["datasets"].
        obs: Primary-cell metadata from census_data[organism].obs.
        dataset_cols: Dataset-level columns to preserve when present.

        Returns:
        Dataset-level summary table with one row per dataset.
        """
        dataset_summary = (
            self.obs.groupby("dataset_id", observed=True)
            .agg(
                n_cells=("dataset_id", "size"),
                n_donors=("donor_id", "nunique"),
                n_cell_types=("cell_type", "nunique"),
                assays=("assay", _collapse_unique_series),
                tissues=("tissue", _collapse_unique_series),
                diseases=("disease", _collapse_unique_series),
                sexes=("sex", _collapse_sex_series),
                age_terms=(
                    "development_stage",
                    _summarize_development_stage,
                ),
                suspension_types=("suspension_type", _collapse_unique_series),
            )
            .reset_index()
        )

        available_dataset_cols = [
            col
            for col in self.config.dataset_cols
            if col in self.datasets.columns
        ]

        dataset_summary = dataset_summary.merge(
            self.datasets[available_dataset_cols],
            on="dataset_id",
            how="left",
        )

        front_cols = [
            col
            for col in [
                "collection_name",
                "dataset_id",
                "n_cells",
                "n_donors",
                "n_cell_types",
                "assays",
                "tissues",
                "diseases",
                "sexes",
                "age_terms",
                "suspension_types",
            ]
            if col in dataset_summary.columns
        ]

        dataset_summary = dataset_summary[
            front_cols
            + [col for col in dataset_summary.columns if col not in front_cols]
        ]

        return dataset_summary.sort_values(
            ["n_cells", "n_donors"],
            ascending=False,
        )

    @classmethod
    def from_census(
        cls,
        organism: str = "homo_sapiens",
        config: CXGMetadataConfig | None = None,
    ) -> Self:
        """Read CELLxGENE Census metadata and return a query object."""
        query_config = config or CXGMetadataConfig()
        datasets, obs = _read_cxg_census_metadata(
            organism=organism,
            census_version=query_config.census_version,
            obs_cols=query_config.obs_cols,
        )
        return cls(datasets=datasets, obs=obs, config=query_config)


def _read_cxg_census_metadata(
    *,
    organism: str = "homo_sapiens",
    census_version: str,
    obs_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read CELLxGENE Census metadata at dataset and cell levels."""
    census = cellxgene_census.open_soma(census_version=census_version)

    try:
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()

        obs = (
            census["census_data"][organism]
            .obs.read(
                value_filter="is_primary_data == True",
                column_names=list(obs_cols),
            )
            .concat()
            .to_pandas()
        )

    finally:
        census.close()

    return datasets, obs
