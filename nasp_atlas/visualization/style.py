"""Shared publication styling for NASP figures."""

from pathlib import Path

import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.typing import ColorType


__all__ = [
    "darken_color",
    "set_matplotlib_publication_parameters",
]


def set_matplotlib_publication_parameters() -> None:
    """Set publication parameters with Graphik and sans-serif fallback."""
    font_families = _publication_font_families()

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
            "font.family": font_families,
            "font.sans-serif": [
                "Arial",
                "Nimbus Sans",
                "DejaVu Sans",
            ],
            "axes.linewidth": 0.25,
            "xtick.major.width": 0.25,
            "ytick.major.width": 0.25,
            "xtick.minor.width": 0.25,
            "ytick.minor.width": 0.25,
        }
    )


def darken_color(
    color: ColorType,
    factor: float = 0.975,
) -> tuple[float, float, float]:
    """Return a darker RGB version of a matplotlib-compatible color."""
    red, green, blue = mcolors.to_rgb(color)
    return red * factor, green * factor, blue * factor


def _publication_font_families() -> list[str]:
    """Return an unambiguous Graphik family with sans-serif fallback."""
    alias = "Graphik NASP"
    manager = font_manager.fontManager
    if any(font.name == alias for font in manager.ttflist):
        return [alias, "sans-serif"]

    system_graphik = [
        path
        for path in font_manager.findSystemFonts()
        if "graphik" in Path(path).stem.casefold()
    ]
    registered_paths = {font.fname for font in manager.ttflist}
    for path in system_graphik:
        if path in registered_paths:
            continue
        try:
            manager.addfont(path)
        except (OSError, RuntimeError):
            continue

    graphik_fonts = [font for font in manager.ttflist if font.name == "Graphik"]
    otf_fonts = [
        font
        for font in graphik_fonts
        if Path(font.fname).suffix.casefold() == ".otf"
    ]
    selected_fonts = otf_fonts or graphik_fonts
    if not selected_fonts:
        return ["sans-serif"]

    manager.ttflist.extend(
        font_manager.FontEntry(
            fname=font.fname,
            name=alias,
            style=font.style,
            variant=font.variant,
            weight=font.weight,
            stretch=font.stretch,
            size=font.size,
        )
        for font in selected_fonts
    )
    return [alias, "sans-serif"]
