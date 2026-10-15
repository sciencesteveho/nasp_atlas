"""Donor-level linear models within each stratum, with robust errors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from nasp_atlas.single_cell.associations import benjamini_hochberg


FocalKind = Literal["indicator", "continuous"]


def donor_regression(
    frame: pd.DataFrame,
    *,
    focal: str,
    focal_kind: FocalKind,
    numeric_covariates: Sequence[str] = (),
    categorical_covariates: Sequence[str] = (),
    stratum_key: str = "cell_type",
    donor_key: str = "donor_id",
    feature_keys: Sequence[str] = (
        "feature_type",
        "feature_id",
        "feature_label",
    ),
    response_key: str = "feature_value",
    minimum_donors: int = 3,
    minimum_residual_df: int = 2,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit `response ~ focal + covariates` on donor means, per stratum.

    Rows of `frame` are observational units, such as one donor, sample and
    cell type, for one feature. Within each stratum and feature, a donor's
    units are averaged into one row, so the donor is the independent unit
    and each stratum keeps its own residual variance and covariate slopes.
    Numeric covariates are averaged; categorical covariates become level
    indicators whose donor mean is the share of the donor's units at that
    level. A level held by fewer than `minimum_donors` donors in the stratum
    is dropped and reported, because its indicator would fit those donors
    exactly. Covariates that are constant in the stratum are dropped too.

    The focal coefficient is estimated by ordinary least squares with HC3
    standard errors, which stay valid when arms or donors differ in
    variance, and a t reference with the residual degrees of freedom. An
    "indicator" focal must be 0 or 1 and constant within donor, and each
    side needs `minimum_donors` donors; a "continuous" focal needs
    `minimum_donors` donors and some spread. Fits that cannot be estimated
    are returned with a status and reason, never dropped. P-values are raw;
    the caller defines the multiplicity family.

    Returns:
      One row per stratum and feature with `estimate`, `standard_error`,
      `ci_low`, `ci_high`, `statistic`, `pvalue`, donor and unit support,
      `df_residual`, `residual_sd`, `max_leverage`, the covariates used and
      dropped, `estimable`, `status` and `reason`.
    """
    required = {
        focal,
        stratum_key,
        donor_key,
        response_key,
        *feature_keys,
        *numeric_covariates,
        *categorical_covariates,
    }
    if missing := sorted(required.difference(frame.columns)):
        raise KeyError(f"Missing donor-regression columns: {missing}")
    if minimum_donors < 2 or minimum_residual_df < 1:
        raise ValueError(
            "Require minimum_donors >= 2, minimum_residual_df >= 1"
        )

    records = []
    # Missing keys, such as the scorer of a sensor feature, form groups too.
    for keys, units in frame.groupby(
        [stratum_key, *feature_keys], observed=True, sort=True, dropna=False
    ):
        identity = dict(zip([stratum_key, *feature_keys], keys, strict=True))
        units = units.loc[np.isfinite(units[response_key].astype(float))]

        donors, columns, dropped = _donor_design(
            units,
            focal=focal,
            focal_kind=focal_kind,
            numeric_covariates=numeric_covariates,
            categorical_covariates=categorical_covariates,
            donor_key=donor_key,
            response_key=response_key,
            minimum_donors=minimum_donors,
        )

        fit = _fit_focal(
            donors,
            focal=focal,
            focal_kind=focal_kind,
            columns=columns,
            response_key=response_key,
            minimum_donors=minimum_donors,
            minimum_residual_df=minimum_residual_df,
            alpha=alpha,
        )

        records.append(
            {
                **identity,
                **fit,
                "n_units": len(units),
                "covariates": ";".join(columns),
                "dropped_covariates": ";".join(dropped),
            }
        )

    return fit_table(records)


def fit_table(records: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Build an effect table whose `ols_hc3` columns exist even if none fit.

    A record that could not be estimated carries no fit values, so a table
    of only such records would otherwise lack the columns that readers and
    figures expect. Missing fit values are NaN.
    """
    table = pd.DataFrame.from_records(records)

    fit_columns = [
        "estimate",
        "standard_error",
        "ci_low",
        "ci_high",
        "statistic",
        "pvalue",
        "df_residual",
        "residual_sd",
        "max_leverage",
        "reason",
    ]
    return table.reindex(
        columns=[*table.columns, *(c for c in fit_columns if c not in table)]
    )


def adjust_fdr_within(
    table: pd.DataFrame,
    family_columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy with `pvalue_fdr`: BH within each family of rows.

    A family is one combination of `family_columns`; missing values form
    their own family, such as the absent scorer of sensor features. Rows
    without a raw p-value, which were not tested, keep a missing FDR.
    """
    table = table.copy()
    if "pvalue" not in table:
        table["pvalue"] = np.nan
    table["pvalue_fdr"] = np.nan

    tested = table.loc[table["pvalue"].notna()]
    for _, index in tested.groupby(
        list(family_columns), dropna=False, observed=True
    ).groups.items():
        table.loc[index, "pvalue_fdr"] = benjamini_hochberg(
            table.loc[index, "pvalue"].tolist()
        )

    return table


def _donor_design(
    units: pd.DataFrame,
    *,
    focal: str,
    focal_kind: FocalKind,
    numeric_covariates: Sequence[str],
    categorical_covariates: Sequence[str],
    donor_key: str,
    response_key: str,
    minimum_donors: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Average units into donor rows and choose estimable covariate columns.

    Returns the donor table (response, focal and design columns), the design
    columns kept, and the covariates or levels dropped with the reason.
    """
    table = units[[donor_key, response_key, focal, *numeric_covariates]].copy()
    table[focal] = table[focal].astype(float)
    dropped: list[str] = []
    candidates = list(numeric_covariates)

    for name in categorical_covariates:
        indicators, notes = _level_indicators(
            units, name, donor_key=donor_key, minimum_donors=minimum_donors
        )
        for column, values in indicators.items():
            table[column] = values
        candidates.extend(indicators)
        dropped.extend(notes)

    if focal_kind == "indicator":
        spread = units.groupby(donor_key, observed=True)[focal].nunique()
        if spread.gt(1).any():
            raise ValueError(
                f"{focal} varies within donor; donor-level contrasts need "
                "donor-disjoint arms"
            )

    donors = table.groupby(donor_key, observed=True).mean()
    columns = [name for name in candidates if donors[name].nunique() > 1]
    dropped.extend(
        f"{name}:constant" for name in candidates if name not in columns
    )

    return donors, columns, dropped


def _level_indicators(
    units: pd.DataFrame,
    name: str,
    *,
    donor_key: str,
    minimum_donors: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return a categorical covariate's level indicators and dropped levels.

    A level held by fewer than `minimum_donors` donors is dropped (it would
    fit those donors exactly) and shares the reference, which is the most
    common kept level. A covariate with fewer than two kept levels has no
    indicator.
    """
    levels = units[name].astype(str)
    donors_per_level = units.groupby(levels, observed=True)[donor_key].nunique()
    kept = donors_per_level.index[donors_per_level.ge(minimum_donors)]
    notes = [
        f"{name}={level}:rare"
        for level in donors_per_level.index.difference(kept)
    ]

    if len(kept) < 2:
        return {}, [*notes, f"{name}:single_level"]

    reference = donors_per_level.loc[kept].idxmax()
    indicators = {
        f"{name}[{level}]": levels.eq(level).astype(float).to_numpy()
        for level in sorted(set(kept) - {reference})
    }
    return indicators, notes


def _fit_focal(
    donors: pd.DataFrame,
    *,
    focal: str,
    focal_kind: FocalKind,
    columns: Sequence[str],
    response_key: str,
    minimum_donors: int,
    minimum_residual_df: int,
    alpha: float,
) -> dict[str, object]:
    """Estimate the focal coefficient by OLS with HC3 errors, or say why not."""
    focal_values = donors[focal].to_numpy(dtype=float)
    response = donors[response_key].to_numpy(dtype=float)
    support: dict[str, object] = {"n_donors": len(donors)}

    if focal_kind == "indicator":
        n_level = int(np.sum(focal_values == 1.0))
        support |= {
            "n_level_donors": n_level,
            "n_reference_donors": len(donors) - n_level,
        }
        supported = min(n_level, len(donors) - n_level) >= minimum_donors
    else:
        supported = len(donors) >= minimum_donors and np.ptp(focal_values) > 0

    if not supported:
        return {**support, "estimable": False, "status": "insufficient_donors"}

    design = np.column_stack(
        [np.ones(len(donors)), focal_values, donors[list(columns)].to_numpy()]
    )

    return {
        **support,
        **ols_hc3(
            design,
            response,
            coefficient=1,
            minimum_residual_df=minimum_residual_df,
            alpha=alpha,
        ),
    }


def ols_hc3(
    design: np.ndarray,
    response: np.ndarray,
    *,
    coefficient: int,
    minimum_residual_df: int = 2,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Estimate one OLS coefficient with HC3 errors and a t interval.

    `design` already holds every column, including any intercept. HC3
    inflates each squared residual by its leverage, which keeps the error
    valid under unequal variances in small samples. Returns the estimate,
    standard error, interval, raw p-value, residual degrees of freedom,
    residual SD and maximum leverage, or `estimable` False with a status:
    "constant_response", "rank_deficient", "insufficient_residual_degrees"
    or "leverage_one" (an observation fitted exactly, where HC3 is
    undefined).
    """
    df_residual = design.shape[0] - design.shape[1]
    status = "ok"
    if np.ptp(response) == 0:
        status = "constant_response"
    elif np.linalg.matrix_rank(design) < design.shape[1]:
        status = "rank_deficient"
    elif df_residual < minimum_residual_df:
        status = "insufficient_residual_degrees"
    if status != "ok":
        return {
            "df_residual": df_residual,
            "estimable": False,
            "status": status,
            "reason": status,
        }

    gram_inverse = np.linalg.inv(design.T @ design)
    coefficients = gram_inverse @ design.T @ response
    residuals = response - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, gram_inverse, design)
    if leverage.max() > 1.0 - 1e-8:
        return {
            "df_residual": df_residual,
            "max_leverage": float(leverage.max()),
            "estimable": False,
            "status": "leverage_one",
            "reason": "an observation is fitted exactly; HC3 is undefined",
        }

    weights = (residuals / (1.0 - leverage)) ** 2
    covariance = gram_inverse @ (design.T * weights) @ design @ gram_inverse
    estimate = float(coefficients[coefficient])
    error = float(np.sqrt(covariance[coefficient, coefficient]))
    statistic = estimate / error if error > 0 else np.nan
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df_residual))

    return {
        "estimate": estimate,
        "standard_error": error,
        "ci_low": estimate - critical * error,
        "ci_high": estimate + critical * error,
        "statistic": statistic,
        "pvalue": float(2.0 * stats.t.sf(abs(statistic), df_residual)),
        "df_residual": df_residual,
        "residual_sd": float(np.sqrt(residuals @ residuals / df_residual)),
        "max_leverage": float(leverage.max()),
        "estimable": True,
        "status": "ok",
        "reason": "",
    }
