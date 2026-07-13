"""Common and shared utility functions."""

import pandas as pd


def collapse_unique_series(values: pd.Series, delimiter: str = ", ") -> str:
    """Collapse unique series values into a delimiter-separated string."""
    vals = sorted(values.dropna().astype(str).unique())
    return delimiter.join(vals)
