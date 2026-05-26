"""Visualization utilities."""

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt


def _set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 5,
            "axes.titlesize": 5,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "figure.titlesize": 5,
            "figure.dpi": 450,
            "font.sans-serif": ["Arial", "Nimbus Sans"],
            "axes.linewidth": 0.25,
            "xtick.major.width": 0.25,
            "ytick.major.width": 0.25,
            "xtick.minor.width": 0.25,
            "ytick.minor.width": 0.25,
        }
    )


def _darken_color(
    color: object, factor: float = 0.975
) -> tuple[float, float, float]:
    """Return a darker RGB version of a matplotlib-compatible color."""
    red, green, blue = mcolors.to_rgb(color)  # type: ignore
    return red * factor, green * factor, blue * factor
