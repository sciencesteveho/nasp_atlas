"""Mixed-model contrast and variance-component visualizations."""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.typing import ColorType

from nasp_atlas.single_cell.visualization.style import _PlotterBase
from nasp_atlas.visualization import darken_color
from nasp_atlas.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


class MixedModelPlotter(_PlotterBase):
    """Render mixed-model effects and variance components.

    Example Usage:
      >>> plotter = MixedModelPlotter(output_dir="path/to/output")
      >>> plotter.plot_mixed_model_effects(
      ...     contrasts,
      ...     estimand="age_by_cell_type",
      ...     filename="age_effects",
      ... )
    """

    def plot_mixed_model_effects(
        self,
        effects: pd.DataFrame,
        *,
        estimand: str,
        filename: str,
        analysis: str | None = None,
        feature_type: str | None = None,
        title: str | None = None,
        x_label: str | None = None,
        max_effects: int = 30,
        figure_width: float = 3.2,
        row_height: float = 0.44,
        minimum_height: float = 1.8,
        point_size: float = 11.0,
        interval_linewidth: float = 0.25,
        tick_label_pad: float = 1.5,
        label_wrap_width: int = 42,
        compact_labels: bool = False,
        compact_label_support: bool = False,
        fdr_threshold: float = 0.05,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot planned mixed-model contrasts with 95% confidence intervals.

        Only rows explicitly marked estimable with a plot-worthy status are
        rendered. Rows are selected by strongest FDR evidence, then absolute
        effect size, with stable label-based tie breaking. The displayed order
        groups the retained rows by feature and contrast. Full row labels
        preserve the contrast reference, conditioning level, best available
        independent-unit support, and exact FDR value. Compact labels retain
        the feature and focal level while relying on the axis and marker-fill
        legend for the shared estimand and FDR threshold. Independent-unit
        support can be added to compact labels explicitly. Filled markers meet
        `fdr_threshold`; open markers do not.

        Args:
          effects: Tidy planned-contrast table from mixed-model inference.
          estimand: Exact `estimand` value to display.
          filename: Output filename stem under `output_dir`, without an
            extension.
          analysis: Optional exact `analysis` value used to further filter rows.
          feature_type: Optional exact feature type, such as "module_score".
          title: Optional figure title. Defaults to the humanized estimand.
          x_label: Optional effect-axis label. Defaults to the single available
            `effect_scale`, when present, followed by "(95% CI)".
          max_effects: Maximum number of planned contrasts displayed after
            deterministic evidence-based selection.
          figure_width: Adaptive figure width in inches.
          row_height: Vertical inches allocated per displayed effect.
          minimum_height: Minimum adaptive figure height in inches.
          point_size: Estimate marker area in points squared.
          interval_linewidth: Confidence-interval line width in points.
          tick_label_pad: Gap in points between row labels and the axis.
          label_wrap_width: Target character width for wrapped contrast labels.
          compact_labels: Whether to omit redundant contrast, reference, and
            exact-FDR prose from row labels while retaining the focal level.
          compact_label_support: Whether compact labels include the best
            available independent-unit support. Has no effect on full labels.
          fdr_threshold: Adjusted p-value threshold encoded by marker fill.
          figsize: Optional exact figure width and height in inches, overriding
            adaptive dimensions.

        Example Usage:
          >>> plotter.plot_mixed_model_effects(
          ...     contrasts,
          ...     estimand="age_slope_by_cell_type",
          ...     analysis="age_by_cell_type",
          ...     feature_type="module_score",
          ...     filename="nasp_mixed_age_slopes_by_cell_type",
          ...     max_effects=24,
          ...     row_height=0.44,
          ... )
        """
        set_matplotlib_publication_parameters()
        self._validate_filename_stem(filename)
        self._validate_effect_geometry(
            max_effects=max_effects,
            figure_width=figure_width,
            row_height=row_height,
            minimum_height=minimum_height,
            point_size=point_size,
            interval_linewidth=interval_linewidth,
            tick_label_pad=tick_label_pad,
            label_wrap_width=label_wrap_width,
            fdr_threshold=fdr_threshold,
            figsize=figsize,
        )

        required = {
            "analysis",
            "feature_type",
            "feature_id",
            "feature_label",
            "estimand",
            "contrast",
            "estimate",
            "ci_low",
            "ci_high",
            "pvalue_fdr",
            "estimable",
            "status",
        }
        self._require_columns(effects, required, table_name="effects")
        selected = effects.loc[
            effects["estimand"].astype(str).eq(estimand)
        ].copy()
        if analysis is not None:
            selected = selected.loc[
                selected["analysis"].astype(str).eq(analysis)
            ]
        if feature_type is not None:
            selected = selected.loc[
                selected["feature_type"].astype(str).eq(feature_type)
            ]

        estimable_mask = self._boolean_values(
            selected["estimable"],
            column="estimable",
        )
        status_mask = self._plot_worthy_status(selected["status"])
        omitted = ~(estimable_mask & status_mask)
        if omitted.any():
            logger.info(
                "[plot] Omitting %d non-estimable mixed-model effects from %s",
                int(omitted.sum()),
                filename,
            )
        selected = selected.loc[~omitted].copy()
        if selected.empty:
            logger.warning(
                "[plot] No estimable mixed-model effects for %s. Skipping.",
                filename,
            )
            return

        for column in ("estimate", "ci_low", "ci_high", "pvalue_fdr"):
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
        invalid_interval_values = np.atleast_1d(
            ~np.isfinite(
                selected[["estimate", "ci_low", "ci_high"]].to_numpy(
                    dtype=float
                )
            ).all(axis=1)
        ).astype(bool)
        if invalid_interval_values.any():
            invalid_features = sorted(
                selected.iloc[np.flatnonzero(invalid_interval_values)][
                    "feature_id"
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                "estimable effects contain non-finite estimate or confidence "
                f"interval values for features: {invalid_features}"
            )
        malformed_interval = (
            selected["ci_low"].gt(selected["estimate"])
            | selected["ci_high"].lt(selected["estimate"])
            | selected["ci_low"].gt(selected["ci_high"])
        )
        if malformed_interval.any():
            invalid_features = sorted(
                selected.loc[malformed_interval, "feature_id"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                "confidence intervals must contain their effect estimates for "
                f"features: {invalid_features}"
            )
        finite_fdr = selected["pvalue_fdr"].dropna()
        if ((finite_fdr < 0.0) | (finite_fdr > 1.0)).any():
            raise ValueError("pvalue_fdr values must be between 0 and 1")

        selected = self._select_effect_rows(selected, maximum=max_effects)
        selected["_row_label"] = selected.apply(
            lambda row: self._effect_row_label(
                row,
                wrap_width=label_wrap_width,
                compact=compact_labels,
                compact_support=compact_label_support,
            ),
            axis="columns",
        )
        selected = selected.assign(
            _feature_sort=selected["feature_label"].astype(str).str.casefold(),
            _contrast_sort=selected["contrast"].astype(str).str.casefold(),
            _by_sort=selected.get(
                "by_level",
                pd.Series("", index=selected.index),
            )
            .astype(str)
            .str.casefold(),
        ).sort_values(
            ["_feature_sort", "_contrast_sort", "_by_sort", "feature_id"],
            kind="stable",
        )

        resolved_figsize = figsize or (
            figure_width,
            max(minimum_height, 0.7 + selected.shape[0] * row_height),
        )
        fig, ax = plt.subplots(figsize=resolved_figsize)
        y_positions = np.arange(selected.shape[0], dtype=float)
        estimates = selected["estimate"].to_numpy(dtype=float)
        ci_low = selected["ci_low"].to_numpy(dtype=float)
        ci_high = selected["ci_high"].to_numpy(dtype=float)
        effect_limit = float(
            np.nanmax(np.abs(np.concatenate((ci_low, ci_high, estimates))))
        )
        effect_limit = max(effect_limit, np.finfo(float).eps)
        cmap = plt.get_cmap("RdBu_r")
        colors = cmap((estimates / effect_limit + 1.0) / 2.0)

        ax.hlines(
            y_positions,
            ci_low,
            ci_high,
            color="black",
            linewidth=interval_linewidth,
            zorder=1,
        )
        significant = (
            selected["pvalue_fdr"].notna()
            & selected["pvalue_fdr"].le(fdr_threshold)
        ).to_numpy(dtype=bool)
        if significant.any():
            ax.scatter(
                estimates[significant],
                y_positions[significant],
                s=point_size,
                facecolors=colors[significant],
                edgecolors=[
                    darken_color(color, factor=0.70)
                    for color in colors[significant]
                ],
                linewidth=0.3,
                zorder=2,
            )
        if (~significant).any():
            ax.scatter(
                estimates[~significant],
                y_positions[~significant],
                s=point_size,
                facecolors="white",
                edgecolors=[
                    (
                        float(color[0]),
                        float(color[1]),
                        float(color[2]),
                        float(color[3]),
                    )
                    for color in colors[~significant]
                ],
                linewidth=0.5,
                zorder=2,
            )

        ax.axvline(0.0, color="#555555", linewidth=0.5, linestyle="--")
        ax.set_xlim(-effect_limit * 1.08, effect_limit * 1.08)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(selected["_row_label"].tolist())
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", pad=tick_label_pad)
        ax.set_xlabel(x_label or self._effect_axis_label(selected))
        resolved_title = title or self._display_label(estimand)
        ax.set_title(resolved_title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#777777",
                    markeredgecolor="#555555",
                    markersize=3.2,
                    label=f"FDR ≤ {fdr_threshold:g}",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="white",
                    markeredgecolor="#777777",
                    markersize=3.2,
                    label=f"FDR > {fdr_threshold:g} or unadjusted",
                ),
            ],
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] mixed-model effects -> %s",
        )

    def plot_mixed_model_variance(
        self,
        variance: pd.DataFrame,
        *,
        analysis: str,
        filename: str,
        feature_type: str | None = None,
        component_order: Sequence[str] | None = None,
        component_colors: Mapping[str, ColorType] | None = None,
        title: str | None = None,
        max_features: int = 30,
        figure_width: float = 3.2,
        row_height: float = 0.25,
        minimum_height: float = 1.7,
        bar_height: float = 0.68,
        tick_label_pad: float = 1.5,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot variance fractions without converting missing values to zero.

        Finite estimable component fractions are stacked exactly as supplied;
        they are never renormalized. A hatched background and row annotation
        distinguish absent, non-estimable, or unallocated variance from numeric
        zero. Pass `component_order` to declare components expected across every
        feature, including components that may be unavailable in a dataset.

        Args:
          variance: Tidy mixed-model variance-component table.
          analysis: Exact `analysis` value to display.
          filename: Output filename stem under `output_dir`, without an
            extension.
          feature_type: Optional exact feature type, such as "module_score".
          component_order: Optional expected component order. Components absent
            for a feature are rendered as missing rather than numeric zero.
          component_colors: Optional component-to-color mapping.
          title: Optional figure title. Defaults to the humanized analysis name.
          max_features: Maximum number of alphabetically ordered features shown.
          figure_width: Adaptive figure width in inches.
          row_height: Vertical inches allocated per feature.
          minimum_height: Minimum adaptive figure height in inches.
          bar_height: Stacked bar height in categorical-axis units, between zero
            and one.
          tick_label_pad: Gap in points between row labels and the axis.
          figsize: Optional exact figure width and height in inches, overriding
            adaptive dimensions.

        Example Usage:
          >>> plotter.plot_mixed_model_variance(
          ...     variance_components,
          ...     analysis="variance_decomposition",
          ...     feature_type="module_score",
          ...     component_order=("donor", "study", "context", "residual"),
          ...     filename="nasp_mixed_variance_decomposition",
          ...     row_height=0.25,
          ... )
        """
        set_matplotlib_publication_parameters()
        self._validate_filename_stem(filename)
        self._validate_variance_geometry(
            max_features=max_features,
            figure_width=figure_width,
            row_height=row_height,
            minimum_height=minimum_height,
            bar_height=bar_height,
            tick_label_pad=tick_label_pad,
            figsize=figsize,
        )

        required = {
            "analysis",
            "feature_type",
            "feature_id",
            "feature_label",
            "component",
            "variance_fraction",
            "estimable",
            "status",
        }
        self._require_columns(variance, required, table_name="variance")
        selected = variance.loc[
            variance["analysis"].astype(str).eq(analysis)
        ].copy()
        if feature_type is not None:
            selected = selected.loc[
                selected["feature_type"].astype(str).eq(feature_type)
            ]
        if selected.empty:
            logger.warning(
                "[plot] No mixed-model variance rows for %s. Skipping.",
                filename,
            )
            return

        duplicate_keys = selected.duplicated(["feature_id", "component"])
        if duplicate_keys.any():
            duplicates = sorted(
                selected.loc[duplicate_keys, "feature_id"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                "variance contains duplicate feature/component rows for "
                f"features: {duplicates}"
            )
        selected["variance_fraction"] = pd.to_numeric(
            selected["variance_fraction"], errors="coerce"
        )
        selected["_estimable"] = self._boolean_values(
            selected["estimable"],
            column="estimable",
        )
        selected["_status_ok"] = self._plot_worthy_status(selected["status"])
        available = selected["_estimable"] & selected["_status_ok"]
        selected["_component_available"] = available
        invalid_available = available & ~np.isfinite(
            selected["variance_fraction"]
        )
        if invalid_available.any():
            invalid_features = sorted(
                selected.loc[invalid_available, "feature_id"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                "estimable variance components contain non-finite fractions "
                f"for features: {invalid_features}"
            )
        finite_fractions = selected.loc[available, "variance_fraction"]
        if ((finite_fractions < 0.0) | (finite_fractions > 1.0)).any():
            raise ValueError("variance_fraction values must be between 0 and 1")

        components = self._variance_component_order(
            selected["component"],
            requested=component_order,
        )
        feature_table = (
            selected.groupby(
                ["feature_id", "feature_label"],
                observed=True,
                dropna=False,
                sort=False,
            )["_component_available"]
            .any()
            .rename("_has_estimable_component")
            .reset_index()
            .assign(
                _sort=lambda frame: (
                    frame["feature_label"].astype(str).str.casefold()
                )
            )
            .sort_values(
                ["_has_estimable_component", "_sort", "feature_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(max_features)
        )
        feature_ids = feature_table["feature_id"].astype(str).tolist()
        selected = selected.loc[
            selected["feature_id"].astype(str).isin(feature_ids)
        ].copy()
        rows_by_key = {
            (str(row["feature_id"]), str(row["component"])): row
            for _, row in selected.iterrows()
        }

        resolved_figsize = figsize or (
            figure_width,
            max(minimum_height, 0.7 + len(feature_ids) * row_height),
        )
        fig, ax = plt.subplots(figsize=resolved_figsize)
        y_positions = np.arange(len(feature_ids), dtype=float)
        ax.barh(
            y_positions,
            np.ones(len(feature_ids), dtype=float),
            height=bar_height,
            color="#eeeeee",
            edgecolor="#b8b8b8",
            linewidth=0.3,
            hatch="////",
            zorder=0,
        )

        colors = self._variance_component_colors(
            components,
            requested=component_colors,
        )
        row_labels: list[str] = []
        missing_any = False
        for position, feature_row in enumerate(feature_table.itertuples()):
            feature_id = str(feature_row.feature_id)
            feature_rows = selected.loc[
                selected["feature_id"].astype(str).eq(feature_id)
            ]
            left = 0.0
            missing: list[str] = []
            for component in components:
                row = rows_by_key.get((feature_id, component))
                if (
                    row is None
                    or not bool(row["_estimable"])
                    or not bool(row["_status_ok"])
                ):
                    missing.append(component)
                    continue
                fraction = float(row["variance_fraction"])
                ax.barh(
                    position,
                    fraction,
                    left=left,
                    height=bar_height,
                    color=colors[component],
                    edgecolor=darken_color(colors[component], factor=0.75),
                    linewidth=0.35,
                    zorder=1,
                )
                left += fraction

            if left > 1.0 + 1e-6:
                raise ValueError(
                    "variance fractions exceed one for feature "
                    f"{feature_id}: {left:.6g}"
                )
            unallocated = max(0.0, 1.0 - left)
            if missing or unallocated > 1e-6:
                missing_any = True
                notes = []
                if missing:
                    names = ", ".join(
                        self._display_label(component) for component in missing
                    )
                    notes.append(f"Missing: {names}")
                if unallocated > 1e-6:
                    notes.append(f"Unallocated: {unallocated:.0%}")
                ax.text(
                    min(left + 0.015, 0.98),
                    position,
                    "; ".join(notes),
                    ha="left" if left < 0.8 else "right",
                    va="center",
                    color="#555555",
                    fontsize=5,
                    clip_on=False,
                    zorder=2,
                )
            row_labels.append(
                self._variance_row_label(
                    feature_rows, feature_row.feature_label
                )
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(row_labels)
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", pad=tick_label_pad)
        ax.set_xlabel("Variance fraction")
        ax.set_title(title or self._display_label(analysis))
        ax.spines[["top", "right"]].set_visible(False)
        handles = [
            Patch(
                facecolor=colors[component],
                edgecolor=darken_color(colors[component], factor=0.75),
                linewidth=0.35,
                label=self._display_label(component),
            )
            for component in components
        ]
        if missing_any:
            handles.append(
                Patch(
                    facecolor="#eeeeee",
                    edgecolor="#b8b8b8",
                    hatch="////",
                    linewidth=0.3,
                    label="Missing or unallocated",
                )
            )
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        self._save_figure_and_log(
            fig,
            self.output_dir / filename,
            "[plot] mixed-model variance -> %s",
        )

    @staticmethod
    def _select_effect_rows(
        effects: pd.DataFrame,
        *,
        maximum: int,
    ) -> pd.DataFrame:
        """Return deterministic top effects ranked by FDR and magnitude."""
        ranked = effects.assign(
            _fdr_sort=effects["pvalue_fdr"].fillna(np.inf),
            _magnitude_sort=effects["estimate"].abs(),
            _feature_sort=effects["feature_label"].astype(str).str.casefold(),
            _contrast_sort=effects["contrast"].astype(str).str.casefold(),
        ).sort_values(
            [
                "_fdr_sort",
                "_magnitude_sort",
                "_feature_sort",
                "_contrast_sort",
                "feature_id",
            ],
            ascending=[True, False, True, True, True],
            kind="stable",
        )
        return ranked.head(maximum).copy()

    @classmethod
    def _effect_row_label(
        cls,
        row: pd.Series,
        *,
        wrap_width: int,
        compact: bool,
        compact_support: bool,
    ) -> str:
        """Return a feature, contrast, reference, support, and FDR label."""
        feature = cls._display_feature_label(row["feature_label"])
        if compact:
            detail = cls._compact_effect_detail(
                row,
                include_support=compact_support,
            )
            return f"{feature} — {detail}"

        raw_contrast = cls._optional_text(row["contrast"])
        contrast = cls._display_label(raw_contrast)
        by_key = cls._optional_text(row.get("by_key", row.get("by")))
        by_level = cls._optional_text(row.get("by_level"))
        if by_level and by_level.casefold() not in raw_contrast.casefold():
            context = (
                f"{cls._display_label(by_key)}: {cls._display_label(by_level)}"
                if by_key
                else cls._display_label(by_level)
            )
            contrast += f" within {context}"
        reference = cls._optional_text(row.get("reference"))
        if reference and reference.casefold() not in raw_contrast.casefold():
            contrast += f"; reference: {cls._display_label(reference)}"
        contrast = textwrap.fill(contrast, width=wrap_width)

        annotations = []
        if support := cls._best_support_label(row):
            annotations.append(support)
        annotations.append(cls._fdr_label(row.get("pvalue_fdr")))
        return f"{feature}\n{contrast}\n{'; '.join(annotations)}"

    @classmethod
    def _compact_effect_detail(
        cls,
        row: pd.Series,
        *,
        include_support: bool,
    ) -> str:
        """Return a concise focal comparison with optional unit support."""
        by_level = cls._optional_text(row.get("by_level"))
        level = cls._optional_text(row.get("level"))
        reference = cls._optional_text(row.get("reference"))
        if by_level:
            focal = cls._display_label(by_level)
        elif level and reference and reference.casefold() != "marginal_mean":
            focal = (
                f"{cls._display_label(level)} vs "
                f"{cls._display_label(reference)}"
            )
        elif level:
            focal = cls._display_label(level)
        else:
            focal = cls._display_label(cls._optional_text(row.get("contrast")))

        support = cls._compact_support_label(row) if include_support else ""
        return f"{focal}; {support}" if support else focal

    @classmethod
    def _compact_support_label(cls, row: pd.Series) -> str:
        """Return the shortest unambiguous available support count."""
        paired = cls._finite_count(row.get("n_paired_units"))
        if paired is not None:
            return f"n={cls._format_count(paired)} paired"

        level = cls._finite_count(row.get("n_level_units"))
        reference = cls._finite_count(row.get("n_reference_units"))
        if level is not None and reference is not None:
            return (
                f"n={cls._format_count(level)}/{cls._format_count(reference)}"
            )
        if level is not None:
            return f"n={cls._format_count(level)}"

        independent = cls._finite_count(row.get("n_independent_units"))
        return (
            f"n={cls._format_count(independent)}"
            if independent is not None
            else ""
        )

    @classmethod
    def _variance_row_label(
        cls,
        rows: pd.DataFrame,
        feature_label: object,
    ) -> str:
        """Return a feature label with independent-unit support."""
        support = ""
        if "n_independent_units" in rows:
            numeric = pd.to_numeric(
                rows["n_independent_units"], errors="coerce"
            ).dropna()
            if not numeric.empty:
                support = (
                    f"n={cls._format_count(float(numeric.max()))} independent"
                )
        feature = cls._display_feature_label(feature_label)
        return f"{feature}\n{support}" if support else feature

    @classmethod
    def _best_support_label(cls, row: pd.Series) -> str:
        """Return the most specific available independent-unit support."""
        paired = cls._finite_count(row.get("n_paired_units"))
        if paired is not None:
            return f"n={cls._format_count(paired)} paired"

        level = cls._finite_count(row.get("n_level_units"))
        reference = cls._finite_count(row.get("n_reference_units"))
        if level is not None and reference is not None:
            return (
                f"n={cls._format_count(level)}/"
                f"{cls._format_count(reference)} units"
            )
        if level is not None:
            return f"n={cls._format_count(level)} level units"

        independent = cls._finite_count(row.get("n_independent_units"))
        if independent is not None:
            return f"n={cls._format_count(independent)} independent"
        return ""

    @classmethod
    def _effect_axis_label(cls, effects: pd.DataFrame) -> str:
        """Return an effect axis label from a unique saved effect scale."""
        if "effect_scale" in effects:
            scales = effects["effect_scale"].dropna().astype(str)
            scales = scales.loc[scales.str.strip().ne("")].unique().tolist()
            if len(scales) == 1:
                return f"{cls._display_label(scales[0])} (95% CI)"
        return "Effect estimate (95% CI)"

    @staticmethod
    def _variance_component_order(
        components: pd.Series,
        *,
        requested: Sequence[str] | None,
    ) -> list[str]:
        """Return validated expected variance components in stable order."""
        if requested is not None:
            order = [str(component) for component in requested]
            if not order:
                raise ValueError("component_order must not be empty")
            if len(order) != len(set(order)):
                raise ValueError("component_order must not contain duplicates")
            return order

        available = set(components.dropna().astype(str))
        preferred = [
            "donor",
            "study",
            "context",
            "tissue",
            "cell_type",
            "residual",
        ]
        order = [component for component in preferred if component in available]
        order.extend(sorted(available.difference(order)))
        if not order:
            raise ValueError("variance contains no component labels")
        return order

    @staticmethod
    def _variance_component_colors(
        components: Sequence[str],
        *,
        requested: Mapping[str, ColorType] | None,
    ) -> dict[str, ColorType]:
        """Return caller colors with deterministic categorical fallbacks."""
        fallback = plt.get_cmap("tab10")
        default_colors: dict[str, ColorType] = {
            "donor": "#8fb9d0",
            "study": "#f2b880",
            "context": "#91c7a3",
            "tissue": "#b9a6d3",
            "cell_type": "#e5a6b8",
            "residual": "#c9c9c9",
        }
        colors: dict[str, ColorType] = {}
        for index, component in enumerate(components):
            if requested is not None and component in requested:
                colors[component] = requested[component]
            elif component in default_colors:
                colors[component] = default_colors[component]
            else:
                colors[component] = fallback(index % 10)
        return colors

    @staticmethod
    def _plot_worthy_status(values: pd.Series) -> pd.Series:
        """Return rows whose inference status supports visualization."""
        return (
            values.astype("string")
            .fillna("")
            .str.casefold()
            .isin(("ok", "estimable"))
        )

    @staticmethod
    def _boolean_values(values: pd.Series, *, column: str) -> pd.Series:
        """Return strict boolean values for a serialized indicator column."""
        if pd.api.types.is_bool_dtype(values.dtype):
            return values.fillna(False).astype(bool)

        normalized = values.astype("string").fillna("").str.casefold()
        valid = normalized.isin({"true", "false", "1", "0", "yes", "no"})
        if not valid.all():
            invalid = sorted(normalized.loc[~valid].unique().tolist())
            raise ValueError(
                f"{column} contains unsupported boolean values: {invalid}"
            )
        return normalized.isin({"true", "1", "yes"})

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame,
        required: set[str],
        *,
        table_name: str,
    ) -> None:
        """Raise an actionable error for missing plotting columns."""
        if missing := sorted(required.difference(frame.columns)):
            raise KeyError(
                f"{table_name} missing mixed-model columns: {missing}"
            )

    @staticmethod
    def _validate_filename_stem(filename: str) -> None:
        """Reject a PNG extension because the shared saver appends it."""
        if not filename.strip():
            raise ValueError("filename must not be empty")
        if filename.casefold().endswith(".png"):
            raise ValueError("filename must be a stem without a .png extension")

    @classmethod
    def _validate_effect_geometry(
        cls,
        *,
        max_effects: int,
        figure_width: float,
        row_height: float,
        minimum_height: float,
        point_size: float,
        interval_linewidth: float,
        tick_label_pad: float,
        label_wrap_width: int,
        fdr_threshold: float,
        figsize: tuple[float, float] | None,
    ) -> None:
        """Validate mixed-effect selection and presentation controls."""
        if max_effects < 1:
            raise ValueError("max_effects must be at least 1")
        for name, value in (
            ("figure_width", figure_width),
            ("row_height", row_height),
            ("minimum_height", minimum_height),
            ("point_size", point_size),
            ("interval_linewidth", interval_linewidth),
        ):
            cls._require_positive_finite(name, value)
        if not np.isfinite(tick_label_pad) or tick_label_pad < 0.0:
            raise ValueError("tick_label_pad must be nonnegative and finite")
        if label_wrap_width < 1:
            raise ValueError("label_wrap_width must be at least 1")
        if not np.isfinite(fdr_threshold) or not 0.0 < fdr_threshold <= 1.0:
            raise ValueError("fdr_threshold must be finite and in (0, 1]")
        cls._validate_figsize(figsize)

    @classmethod
    def _validate_variance_geometry(
        cls,
        *,
        max_features: int,
        figure_width: float,
        row_height: float,
        minimum_height: float,
        bar_height: float,
        tick_label_pad: float,
        figsize: tuple[float, float] | None,
    ) -> None:
        """Validate variance-component selection and presentation controls."""
        if max_features < 1:
            raise ValueError("max_features must be at least 1")
        for name, value in (
            ("figure_width", figure_width),
            ("row_height", row_height),
            ("minimum_height", minimum_height),
        ):
            cls._require_positive_finite(name, value)
        if not np.isfinite(bar_height) or not 0.0 < bar_height <= 1.0:
            raise ValueError("bar_height must be finite and in (0, 1]")
        if not np.isfinite(tick_label_pad) or tick_label_pad < 0.0:
            raise ValueError("tick_label_pad must be nonnegative and finite")
        cls._validate_figsize(figsize)

    @staticmethod
    def _require_positive_finite(name: str, value: float) -> None:
        """Raise when a presentation dimension is nonpositive or non-finite."""
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    @staticmethod
    def _validate_figsize(figsize: tuple[float, float] | None) -> None:
        """Validate an optional exact figure size."""
        if figsize is None:
            return
        if len(figsize) != 2 or any(
            not np.isfinite(value) or value <= 0.0 for value in figsize
        ):
            raise ValueError("figsize dimensions must be positive and finite")

    @staticmethod
    def _display_feature_label(value: object) -> str:
        """Return a readable feature label while preserving biological case."""
        return str(value).replace("_", " ")

    @staticmethod
    def _display_label(value: object) -> str:
        """Return a readable sentence-style result label."""
        label = str(value).replace("_", " ").strip()
        return label[:1].upper() + label[1:] if label else ""

    @staticmethod
    def _optional_text(value: object) -> str:
        """Return stripped optional text, treating missing values as empty."""
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.casefold() in {"", "nan", "nat", "<na>"} else text

    @staticmethod
    def _finite_count(value: object) -> float | None:
        """Return a finite nonnegative support count when available."""
        try:
            count = float(str(value))
        except (TypeError, ValueError):
            return None
        return count if np.isfinite(count) and count >= 0.0 else None

    @staticmethod
    def _format_count(value: float) -> str:
        """Format an integer-like or fractional support count compactly."""
        return f"{int(value)}" if value.is_integer() else f"{value:g}"

    @staticmethod
    def _fdr_label(value: object) -> str:
        """Return an exact compact FDR label or an unavailable marker."""
        try:
            fdr = float(str(value))
        except (TypeError, ValueError):
            return "FDR unavailable"
        if not np.isfinite(fdr):
            return "FDR unavailable"
        return f"FDR={fdr:.1e}" if fdr < 0.001 else f"FDR={fdr:.3f}"
