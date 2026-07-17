"""
prediction_intervals.py
=======================
Visualization of MC Dropout prediction intervals and uncertainty statistics.

Figures produced
----------------
U1. Time-series of GRU mean prediction with 70%, 80%, 90% confidence bands
U2. Uncertainty vs actual rainfall (heteroscedasticity analysis)
U3. Interval width distribution by season and rainfall intensity
U4. Comparison of interval widths: monsoon vs non-monsoon
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG
from src.visualization.plot_utils import (
    apply_style, save_figure, add_figure_title,
    annotate_monsoon_bands,
    BLUE, RED, GREEN, ORANGE, GRAY, CATEGORICAL_PALETTE,
)

logger = logging.getLogger(__name__)

MONSOON_MONTHS  = {6, 7, 8, 9}
INTERVAL_LEVELS = [70, 80, 90]
INTERVAL_ALPHAS = [0.40, 0.25, 0.12]   # visual transparency per band (outermost lightest)
INTERVAL_COLORS = [BLUE, BLUE, BLUE]


def plot_all_uncertainty_figures(
    pred_df: pd.DataFrame,
    save: bool = True,
) -> Dict[str, float]:
    """
    Generate all prediction-interval figures and return summary statistics.

    Parameters
    ----------
    pred_df : DataFrame from mc_dropout.run_mc_dropout() with columns:
              actual, mean_mm, std_mm, lower_70/80/90, upper_70/80/90.
    save    : Persist figures to disk.

    Returns
    -------
    dict of summary statistics (coverage, interval widths, etc.)
    """
    apply_style()

    stats = _compute_interval_stats(pred_df)
    _plot_interval_timeseries(pred_df, save)
    _plot_uncertainty_vs_actual(pred_df, save)
    _plot_interval_width_distribution(pred_df, save)
    _plot_seasonal_uncertainty(pred_df, save)

    return stats


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _compute_interval_stats(pred_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute coverage, mean width, and MPIW for each interval level.
    """
    actual = pred_df["actual"].values
    stats  = {}

    for level in INTERVAL_LEVELS:
        lo  = pred_df[f"lower_{level}"].values
        hi  = pred_df[f"upper_{level}"].values
        covered    = ((actual >= lo) & (actual <= hi)).mean()
        mean_width = (hi - lo).mean()
        # Mean Prediction Interval Width normalised by std of actuals
        mpiw_norm  = mean_width / (actual.std() + 1e-8)

        stats[f"coverage_{level}pct"]    = float(covered)
        stats[f"mean_width_{level}pct"]  = float(mean_width)
        stats[f"mpiw_norm_{level}pct"]   = float(mpiw_norm)

        logger.info(
            f"[Intervals] {level}% interval: "
            f"empirical coverage={covered*100:.1f}% | "
            f"mean width={mean_width:.3f} mm | "
            f"MPIW (norm)={mpiw_norm:.3f}"
        )

    stats["mean_std_mm"]  = float(pred_df["std_mm"].mean())
    stats["max_std_mm"]   = float(pred_df["std_mm"].max())
    stats["std_monsoon"]  = float(pred_df.loc[
        pred_df.index.month.isin(MONSOON_MONTHS), "std_mm"
    ].mean())
    stats["std_nonmonsoon"] = float(pred_df.loc[
        ~pred_df.index.month.isin(MONSOON_MONTHS), "std_mm"
    ].mean())

    return stats


# ---------------------------------------------------------------------------
# Figure U1: Interval time-series
# ---------------------------------------------------------------------------

def _plot_interval_timeseries(pred_df: pd.DataFrame, save: bool) -> None:
    """
    Nested fan chart: 70 / 80 / 90 % prediction bands around GRU mean.
    Shows full test period (top) and monsoon season zoom (bottom).
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=False)

    for ax_idx, ax in enumerate(axes):
        if ax_idx == 0:
            df = pred_df
            title = "(a) Full Test Period — GRU Prediction Intervals"
        else:
            # First monsoon season in test
            zoom_start = None
            for ts in pred_df.index:
                if ts.month == 6:
                    zoom_start = ts
                    break
            if zoom_start is None:
                ax.set_visible(False)
                continue
            zoom_end = zoom_start + pd.DateOffset(months=4)
            df = pred_df.loc[
                (pred_df.index >= zoom_start) & (pred_df.index <= zoom_end)
            ]
            title = (
                f"(b) Monsoon Zoom — "
                f"{zoom_start.strftime('%b %Y')} to {zoom_end.strftime('%b %Y')}"
            )

        # Shade bands from widest to narrowest
        for level, alpha in zip([90, 80, 70], INTERVAL_ALPHAS):
            lo = df[f"lower_{level}"].values
            hi = df[f"upper_{level}"].values
            ax.fill_between(df.index, lo, hi, alpha=alpha, color=BLUE,
                            label=f"{level}% PI" if ax_idx == 0 else "_nolegend_")

        # Mean prediction
        ax.plot(df.index, df["mean_mm"], lw=1.2, color=BLUE,
                label="GRU mean" if ax_idx == 0 else "_nolegend_")

        # Actual rainfall
        ax.fill_between(df.index, df["actual"], alpha=0.22, color=RED)
        ax.plot(df.index, df["actual"], lw=0.7, color=RED,
                label="Actual" if ax_idx == 0 else "_nolegend_")

        # Mark actual values that exceed the 90% upper bound
        exceed = df["actual"] > df["upper_90"]
        if exceed.any():
            ax.scatter(
                df.index[exceed], df.loc[exceed, "actual"],
                marker="*", s=60, color=RED, zorder=5,
                label=f"Outside 90% PI ({exceed.sum()} days)" if ax_idx == 0 else "_nolegend_"
            )

        if ax_idx == 0:
            annotate_monsoon_bands(ax, pred_df, alpha=0.06)

        ax.set_ylabel("Rainfall (mm/day)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=20)
        if ax_idx == 0:
            ax.legend(fontsize=9, ncol=5, loc="upper right")

    add_figure_title(
        fig,
        "GRU MC Dropout Prediction Intervals (T=100 passes)",
        "Nested bands: 70% / 80% / 90% | ★ = actual exceeded 90% upper bound"
    )
    if save:
        save_figure(fig, "uncertainty_interval_timeseries", subdir="uncertainty")
    plt.close(fig)
    logger.info("[PI] Interval time-series figure saved")


# ---------------------------------------------------------------------------
# Figure U2: Uncertainty vs actual (heteroscedasticity)
# ---------------------------------------------------------------------------

def _plot_uncertainty_vs_actual(pred_df: pd.DataFrame, save: bool) -> None:
    """
    Scatter of GRU predictive std vs actual rainfall.
    Reveals whether uncertainty correctly escalates for heavy-rain events.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    monsoon = pred_df.index.month.isin(MONSOON_MONTHS)

    # Panel (a): std_mm vs actual
    ax = axes[0]
    ax.scatter(pred_df.loc[~monsoon, "actual"],
               pred_df.loc[~monsoon, "std_mm"],
               s=4, alpha=0.20, color=ORANGE, label="Non-Monsoon")
    ax.scatter(pred_df.loc[monsoon, "actual"],
               pred_df.loc[monsoon, "std_mm"],
               s=4, alpha=0.25, color=GREEN, label="Monsoon")

    # LOWESS trend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    si = np.argsort(pred_df["actual"].values)
    sm = lowess(
        pred_df["std_mm"].values[si],
        pred_df["actual"].values[si],
        frac=0.25, return_sorted=True
    )
    ax.plot(sm[:, 0], sm[:, 1], color=RED, lw=2.0, label="LOWESS trend")
    ax.set_xlabel("Actual Rainfall (mm/day)")
    ax.set_ylabel("GRU Predictive Std (mm/day)")
    ax.set_title(
        "(a) Predictive Uncertainty vs Actual Rainfall\n"
        "(ideal: std rises with actual → calibrated heteroscedasticity)",
        fontweight="bold"
    )
    ax.legend(fontsize=8)

    # Panel (b): interval width vs actual for each level
    ax = axes[1]
    for level, color in zip(INTERVAL_LEVELS, [GREEN, BLUE, ORANGE]):
        widths = (pred_df[f"upper_{level}"] - pred_df[f"lower_{level}"]).values
        si2 = np.argsort(pred_df["actual"].values)
        sm2 = lowess(widths[si2], pred_df["actual"].values[si2],
                     frac=0.25, return_sorted=True)
        ax.plot(sm2[:, 0], sm2[:, 1], lw=1.8, color=color,
                label=f"{level}% PI width (smoothed)")
    ax.set_xlabel("Actual Rainfall (mm/day)")
    ax.set_ylabel("Prediction Interval Width (mm)")
    ax.set_title(
        "(b) PI Width vs Actual Rainfall\n"
        "(wider intervals at high rainfall = honest uncertainty)",
        fontweight="bold"
    )
    ax.legend(fontsize=8)

    add_figure_title(
        fig,
        "Uncertainty Heteroscedasticity Analysis — GRU MC Dropout"
    )
    if save:
        save_figure(fig, "uncertainty_heteroscedasticity", subdir="uncertainty")
    plt.close(fig)
    logger.info("[PI] Heteroscedasticity figure saved")


# ---------------------------------------------------------------------------
# Figure U3: Interval width distribution
# ---------------------------------------------------------------------------

def _plot_interval_width_distribution(pred_df: pd.DataFrame, save: bool) -> None:
    """
    Histograms of interval widths, split by rainfall intensity bin.
    Shows whether the model is more/less confident on different rainfall regimes.
    """
    bins_def = [(0, 0.1, "Dry"), (0.1, 5, "Light"), (5, 20, "Moderate"), (20, 999, "Heavy+")]
    level    = 90   # use 90% PI for this analysis

    fig, axes = plt.subplots(1, len(bins_def), figsize=(18, 5), sharey=True)
    colors_bin = [GRAY, BLUE, GREEN, RED]

    for ax, (lo, hi, label), color in zip(axes, bins_def, colors_bin):
        mask   = (pred_df["actual"] >= lo) & (pred_df["actual"] < hi)
        widths = (pred_df.loc[mask, f"upper_{level}"] -
                  pred_df.loc[mask, f"lower_{level}"]).values

        if len(widths) == 0:
            ax.set_visible(False)
            continue

        ax.hist(widths, bins=30, color=color, alpha=0.80, edgecolor="none", density=True)
        ax.axvline(widths.mean(), color="black", lw=1.5, linestyle="--",
                   label=f"Mean = {widths.mean():.1f} mm")
        ax.set_xlabel(f"{level}% PI Width (mm)")
        ax.set_ylabel("Density" if ax == axes[0] else "")
        ax.set_title(
            f"{label} Rain\n({lo}–{hi if hi < 999 else '∞'} mm) | N={mask.sum()}",
            fontweight="bold"
        )
        ax.legend(fontsize=8)

    add_figure_title(
        fig,
        f"GRU {level}% Prediction Interval Width Distribution by Rainfall Intensity"
    )
    if save:
        save_figure(fig, "uncertainty_width_distribution", subdir="uncertainty")
    plt.close(fig)
    logger.info("[PI] Width distribution figure saved")


# ---------------------------------------------------------------------------
# Figure U4: Seasonal uncertainty
# ---------------------------------------------------------------------------

def _plot_seasonal_uncertainty(pred_df: pd.DataFrame, save: bool) -> None:
    """Monthly mean std_mm and monthly 90% PI width — shows seasonal uncertainty profile."""
    months = range(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    level = 90
    monthly_std   = pred_df["std_mm"].groupby(pred_df.index.month).mean()
    monthly_width = (
        (pred_df[f"upper_{level}"] - pred_df[f"lower_{level}"])
        .groupby(pred_df.index.month).mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (series, title, ylabel) in zip(axes, [
        (monthly_std,   "Monthly Mean Predictive Std (mm/day)", "Std Dev (mm)"),
        (monthly_width, f"Monthly Mean {level}% PI Width (mm)", "PI Width (mm)"),
    ]):
        vals = [series.get(m, np.nan) for m in months]
        colors_m = [GREEN if m in MONSOON_MONTHS else BLUE for m in months]
        bars = ax.bar(month_labels, vals, color=colors_m, alpha=0.80, edgecolor="none")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v*1.03,
                        f"{v:.2f}", ha="center", fontsize=8)
        ax.axvspan(5, 9, alpha=0.06, color=GREEN)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=GREEN, alpha=0.7, label="Monsoon (JJAS)"),
            Patch(facecolor=BLUE,  alpha=0.7, label="Non-Monsoon"),
        ], fontsize=8)

    add_figure_title(
        fig,
        "Seasonal Uncertainty Profile — GRU MC Dropout",
        "Uncertainty highest during monsoon onset (Jun) and withdrawal (Sep)"
    )
    if save:
        save_figure(fig, "uncertainty_seasonal_profile", subdir="uncertainty")
    plt.close(fig)
    logger.info("[PI] Seasonal uncertainty figure saved")
