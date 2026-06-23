"""Filter and order CELLxGENE obs metadata by analytic categories."""

from collections.abc import Callable, Iterable, Mapping, Sequence

import pandas as pd


def _annotate_obs_categories(
    obs: pd.DataFrame,
    source_column: str,
    target_column: str,
    categorizer: Callable[[object], str],
) -> pd.DataFrame:
    """Add a broad-category column derived from a raw obs annotation.

    Args:
      obs: Cell-level metadata.
      source_column: Existing column with raw labels.
      target_column: New column name to write.
      categorizer: Maps a raw value to category labels.

    Returns:
      A copy of obs with target_column appended.
    """
    annotated = obs.copy()
    annotated[target_column] = annotated[source_column].map(categorizer)
    return annotated


def _filter_obs_by_category(
    obs: pd.DataFrame,
    column: str,
    keep: Iterable[str],
) -> pd.DataFrame:
    """Restrict obs to rows whose `column` value is in `keep`.

    Args:
      obs: Cell-level metadata.
      column: Category column to filter on.
      keep: Allowed category values.

    Returns:
      Filtered obs
    """
    keep_set = set(keep)
    return obs.loc[obs[column].isin(keep_set)].copy()


def _order_categories(
    categories: Iterable[str],
    front: Sequence[str] = (),
    back: Sequence[str] = (),
) -> list[str]:
    """Order categories with priority entries at the start and end.

    Entries in `front` or `back` not present in `categories` are skipped.
    Categories that appear in neither are sorted alphabetically.

    Args:
      categories: Categories to order.
      front: Categories to place first, in the given order.
      back: Categories to place last, in the given order.

    Returns:
      An ordered list of unique category labels.
    """
    seen = list(dict.fromkeys(categories))
    front_set = set(front)
    back_set = set(back)
    middle = sorted(c for c in seen if c not in front_set and c not in back_set)
    head = [c for c in front if c in seen]
    tail = [c for c in back if c in seen]
    return head + middle + tail


def _humanize_label(
    text: object,
    display_names: Mapping[str, str] | None = None,
) -> str:
    """Convert a category value to a readable axis or legend label."""
    label = str(text)
    if display_names and label in display_names:
        return display_names[label]
    return label.replace("_", " ").capitalize()
