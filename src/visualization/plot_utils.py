"""
plot_utils.py
=============
Shared plotting infrastructure: styles, color palettes, figure saving, and
annotation helpers.  Every EDA and reporting module imports from here to
guarantee visual consistency across all outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette — derived from config, supplemented with a full categorical set
# ---------------------------------------------------------------------------

BLUE    = CFG.plotting.palette_main
RED     = CFG.plotting.palette_accent
GREEN   = CFG.plotting.palette_monsoon
ORANGE  = CFG.plotting.palette_dry
GRAY    = CFG.plotting.palette_neutral

# IMD rainfall category colours for consistent use across all rainfall charts
RAIN_CATEGORY_COLORS = {
    "No rain":        "#E8E8E8",
    "Light":          "#AEC6CF",
    "Moderate":       "#5B9BD5",
    "Heavy":          "#2E75B6",
    "Very Heavy":     "#1A4E8A",
    "Extremely Heavy":"#0D2137",
}

# Categorical palette for multi-series plots (up to 8 series)
CATEGORICAL_PALETTE = [
    "#2E86AB",  # blue
    "#E84855",  # red
    "#3BB273",  # green
    "#F4A261",  # orange
    "#8B5E83",  # purple
    "#5C6BC0",  # indigo
    "#00ACC1",  # cyan
    "#FF7043",  # deep orange
]

# Season colours — consistent across all seasonal plots
SEASON_COLORS = {
    "Monsoon":      GREEN,
    "Pre-Monsoon":  "#FFB74D",
    "Post-Monsoon": "#4DB6AC",
    "Winter":       "#90CAF9",
}


# ---------------------------------------------------------------------------
# Global style application
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """
    Apply the project-wide matplotlib style.
    Called once at the top of every EDA script.
    """
    try:
        plt.style.use(CFG.plotting.style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update({
        "figure.dpi":           CFG.plotting.dpi,
        "savefig.dpi":          CFG.plotting.dpi,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.labelsize":       12,
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      10,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
        "font.family":          "sans-serif",
        "lines.linewidth":      1.5,
        "patch.edgecolor":      "none",
    })


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    filename: str,
    subdir: str = "eda",
    tight: bool = True,
    formats: Sequence[str] = ("png",),
) -> Path:
    """
    Save a matplotlib figure to the outputs/figures directory.

    Parameters
    ----------
    fig       : The figure to save.
    filename  : Filename without extension (extension added per `formats`).
    subdir    : Subdirectory under outputs/figures (e.g. 'eda', 'preprocessing').
    tight     : Apply tight_layout before saving.
    formats   : File formats to export.  Default is PNG only.

    Returns
    -------
    Path to the primary (first format) saved file.
    """
    from config.config_loader import abs_path

    out_dir = abs_path(f"outputs/figures/{subdir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if tight:
        fig.tight_layout()

    primary_path: Optional[Path] = None
    for fmt in formats:
        out_path = out_dir / f"{filename}.{fmt}"
        fig.savefig(
            out_path,
            dpi=CFG.plotting.dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        if primary_path is None:
            primary_path = out_path
        logger.info(f"Figure saved → {out_path}")

    return primary_path


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def annotate_monsoon_bands(
    ax: plt.Axes,
    df: pd.DataFrame,
    alpha: float = 0.08,
    color: str = GREEN,
) -> None:
    """
    Shade the Jun–Sep monsoon period on a time-series axes.

    Parameters
    ----------
    ax  : Target axes with a datetime x-axis.
    df  : DataFrame with DatetimeIndex — used to determine year range.
    """
    years = range(df.index.year.min(), df.index.year.max() + 2)
    for year in years:
        ax.axvspan(
            pd.Timestamp(f"{year}-06-01"),
            pd.Timestamp(f"{year}-09-30"),
            alpha=alpha,
            color=color,
            zorder=0,
            label="_nolegend_",
        )


def add_stat_annotations(
    ax: plt.Axes,
    stats: dict[str, Any],
    x: float = 0.97,
    y_start: float = 0.97,
    spacing: float = 0.07,
    fontsize: int = 9,
    ha: str = "right",
) -> None:
    """
    Add a text box of statistics to an axes corner.

    Parameters
    ----------
    stats    : Ordered dict of {label: value} pairs.
    x, y_start : Axes-fraction coordinates.
    spacing  : Vertical spacing between lines (axes fraction).
    """
    y = y_start
    for label, value in stats.items():
        if isinstance(value, float):
            text = f"{label}: {value:.4f}"
        elif isinstance(value, int):
            text = f"{label}: {value:,}"
        else:
            text = f"{label}: {value}"
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            ha=ha, va="top",
            fontsize=fontsize,
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.85,
            ),
        )
        y -= spacing


def format_date_axis(
    ax: plt.Axes,
    rotation: int = 30,
    date_format: str = "%Y",
) -> None:
    """Apply consistent date axis formatting."""
    ax.xaxis.set_major_formatter(
        mpl.dates.DateFormatter(date_format)
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha="right")


def rainfall_category_legend(ax: plt.Axes) -> None:
    """Add an IMD rainfall category legend to an axes."""
    patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in RAIN_CATEGORY_COLORS.items()
        if cat != "No rain"
    ]
    ax.legend(
        handles=patches,
        title="IMD category",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
    )


# ---------------------------------------------------------------------------
# Reusable figure templates
# ---------------------------------------------------------------------------

def make_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, Any]:
    """Thin wrapper around plt.subplots that applies the project style."""
    apply_style()
    if figsize is None:
        figsize = (
            CFG.plotting.figsize_wide
            if ncols > 1
            else CFG.plotting.figsize_tall
        )
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)


def add_figure_title(
    fig: plt.Figure,
    title: str,
    subtitle: str = "",
    fontsize: int = 15,
) -> None:
    """Add a bold title and optional subtitle to a figure."""
    if subtitle:
        full_title = f"{title}\n{subtitle}"
        fig.suptitle(full_title, fontsize=fontsize, fontweight="bold", y=1.01)
    else:
        fig.suptitle(title, fontsize=fontsize, fontweight="bold")
