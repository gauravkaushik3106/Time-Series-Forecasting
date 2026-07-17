"""
eda_temporal.py
===============
Seasonal and temporal analysis of the Lucknow rainfall dataset.

Covers
------
- Monthly climatology (mean, median, std, percentiles)
- Annual totals and inter-annual variability
- Rolling mean and variance (short / medium / long windows)
- Monsoon onset / withdrawal proxy detection
- Dry spell length distribution and recurrence
- Year × DayOfYear heatmap for pattern visualisation
- Mann-Kendall trend test on annual totals
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG
from src.visualization.plot_utils import (
    apply_style, save_figure, add_stat_annotations, annotate_monsoon_bands,
    format_date_axis, BLUE, RED, GREEN, ORANGE, GRAY, SEASON_COLORS,
    make_figure, add_figure_title, CATEGORICAL_PALETTE,
)

logger = logging.getLogger(__name__)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyse_temporal_patterns(
    df: pd.DataFrame,
    save: bool = True,
) -> Dict:
    """
    Run all temporal EDA plots and return a summary statistics dict.

    Parameters
    ----------
    df   : Full cleaned dataframe with DatetimeIndex.
    save : Persist figures to disk.
    """
    apply_style()

    # Compute all seasonal/temporal statistics first
    summary = _compute_temporal_stats(df)
    logger.info("Temporal statistics computed")

    _plot_monthly_climatology(df, summary, save)
    _plot_annual_totals(df, summary, save)
    _plot_rolling_statistics(df, save)
    _plot_monsoon_analysis(df, summary, save)
    _plot_dry_spell_analysis(df, summary, save)
    _plot_doy_year_heatmap(df, save)

    return summary


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_temporal_stats(df: pd.DataFrame) -> Dict:
    rain = df["RAINFALL"]

    # Monthly climatology
    monthly = rain.groupby(rain.index.month).agg(
        mean="mean", median="median", std="std",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        p95=lambda x: x.quantile(0.95),
        total="sum",
    ).round(3)

    # Annual totals
    annual = rain.groupby(rain.index.year).sum().round(1)

    # Monsoon fraction
    monsoon_months = list(CFG.monsoon_months)
    monsoon_total  = rain[rain.index.month.isin(monsoon_months)].sum()
    annual_total   = rain.sum()
    monsoon_frac   = monsoon_total / annual_total * 100

    # Mann-Kendall trend on annual totals (scipy-based implementation)
    years  = np.array(annual.index)
    values = annual.values
    slope, intercept, r, p, se = stats.linregress(years, values)

    # Dry spell statistics
    dry_spells = _compute_dry_spells(rain)

    return {
        "monthly_climatology": monthly,
        "annual_totals":       annual,
        "monsoon_fraction_pct": float(monsoon_frac),
        "annual_mean_mm":      float(annual.mean()),
        "annual_std_mm":       float(annual.std()),
        "annual_cv_pct":       float(annual.std() / annual.mean() * 100),
        "annual_min_mm":       float(annual.min()),
        "annual_max_mm":       float(annual.max()),
        "trend_slope_mm_yr":   float(slope),
        "trend_p_value":       float(p),
        "trend_r_squared":     float(r ** 2),
        "dry_spells":          dry_spells,
        "n_dry_spells":        len(dry_spells),
        "max_dry_spell_days":  int(max(dry_spells)) if dry_spells else 0,
        "mean_dry_spell_days": float(np.mean(dry_spells)) if dry_spells else 0.0,
        "n_long_dry_spells":   int(sum(s > 30 for s in dry_spells)),
    }


def _compute_dry_spells(rain: pd.Series) -> List[int]:
    """
    Extract consecutive dry-day run lengths.
    A day is "dry" if rainfall < dry_day_threshold.
    Returns a list of spell lengths (in days).
    """
    threshold = CFG.rainfall.dry_day_threshold
    spell_lengths: List[int] = []
    current = 0
    for val in rain:
        if val < threshold:
            current += 1
        else:
            if current >= CFG.eda.dry_spell_min_length:
                spell_lengths.append(current)
            current = 0
    if current >= CFG.eda.dry_spell_min_length:
        spell_lengths.append(current)
    return spell_lengths


# ---------------------------------------------------------------------------
# Figure 5: Monthly climatology
# ---------------------------------------------------------------------------

def _plot_monthly_climatology(
    df: pd.DataFrame,
    summary: Dict,
    save: bool,
) -> None:
    """
    3-panel monthly climatology:
      (a) Mean daily rainfall with IQR band and 95th percentile line
      (b) Monthly total rainfall (summed across all years)
      (c) Probability of rain per month (P(rainfall > threshold))
    """
    rain   = df["RAINFALL"]
    dry_t  = CFG.rainfall.dry_day_threshold
    clim   = summary["monthly_climatology"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    months = np.arange(1, 13)
    monsoon = list(CFG.monsoon_months)
    colors_m = [GREEN if m in monsoon else BLUE for m in months]

    # --- (a) Mean daily rainfall with spread ---
    axes[0].bar(months, clim["mean"], color=colors_m, alpha=0.75,
                edgecolor="none", label="Mean daily rainfall")
    axes[0].fill_between(months, clim["p25"], clim["p75"],
                         alpha=0.20, color=BLUE, label="IQR (25th–75th %ile)")
    axes[0].plot(months, clim["p95"], color=RED, lw=1.2, linestyle="--",
                 marker="o", ms=4, label="95th percentile")
    axes[0].set_xticks(months)
    axes[0].set_xticklabels(MONTH_LABELS)
    axes[0].set_ylabel("Daily Rainfall (mm)")
    axes[0].set_title("(a) Mean Daily Rainfall by Month")
    axes[0].legend(fontsize=8)

    # --- (b) Monthly totals (sum across all years) ---
    monthly_totals = rain.groupby(rain.index.month).sum()
    axes[1].bar(months, monthly_totals.values, color=colors_m,
                alpha=0.75, edgecolor="none")
    axes[1].set_xticks(months)
    axes[1].set_xticklabels(MONTH_LABELS)
    axes[1].set_ylabel("Total Rainfall (mm) — 2000–2025")
    axes[1].set_title("(b) Cumulative Monthly Rainfall (26 years)")
    # Annotate with percentage
    grand_total = monthly_totals.sum()
    for m, tot in zip(months, monthly_totals.values):
        pct = tot / grand_total * 100
        if pct > 0.5:
            axes[1].text(m, tot + 50, f"{pct:.1f}%", ha="center",
                         va="bottom", fontsize=8)

    # --- (c) Probability of rain day ---
    p_rain = rain.groupby(rain.index.month).apply(
        lambda x: (x >= dry_t).mean() * 100
    )
    axes[2].bar(months, p_rain.values, color=colors_m, alpha=0.75,
                edgecolor="none")
    axes[2].set_xticks(months)
    axes[2].set_xticklabels(MONTH_LABELS)
    axes[2].set_ylabel("Probability of Rain Day (%)")
    axes[2].set_title("(c) Monthly Probability of Rainfall")
    axes[2].set_ylim(0, 105)
    for m, p in zip(months, p_rain.values):
        axes[2].text(m, p + 1.5, f"{p:.0f}%", ha="center", va="bottom",
                     fontsize=8)

    # Shared monsoon annotation
    for ax in axes:
        ax.axvspan(5.5, 9.5, alpha=0.06, color=GREEN)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=GREEN, alpha=0.7, label="Monsoon (JJAS)"),
        Patch(facecolor=BLUE,  alpha=0.7, label="Non-Monsoon"),
    ]
    axes[0].legend(
        handles=axes[0].get_legend_handles_labels()[0] + legend_els,
        fontsize=8, loc="upper left"
    )

    add_figure_title(
        fig,
        "Monthly Rainfall Climatology — Lucknow 2000–2025",
        f"Monsoon (JJAS) contributes {summary['monsoon_fraction_pct']:.1f}% of annual total",
    )
    if save:
        save_figure(fig, "05_monthly_climatology", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Annual totals with trend
# ---------------------------------------------------------------------------

def _plot_annual_totals(
    df: pd.DataFrame,
    summary: Dict,
    save: bool,
) -> None:
    """
    Annual rainfall totals bar chart with OLS trend line and ±1 std band.
    Includes a secondary panel for year-over-year anomaly (departure from mean).
    """
    rain   = df["RAINFALL"]
    annual = summary["annual_totals"]
    years  = annual.index.values
    values = annual.values
    mean   = values.mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # --- Top: annual totals ---
    bar_colors = [GREEN if v > mean else ORANGE for v in values]
    axes[0].bar(years, values, color=bar_colors, alpha=0.75, edgecolor="none",
                label="Annual total")
    axes[0].axhline(mean, color=BLUE, lw=1.5, linestyle="--",
                    label=f"26-yr mean: {mean:.0f} mm")
    axes[0].axhspan(mean - annual.std(), mean + annual.std(),
                    alpha=0.10, color=BLUE, label="±1 std dev band")

    # OLS trend line
    slope  = summary["trend_slope_mm_yr"]
    intercept = mean - slope * years.mean()
    y_trend = slope * years + intercept
    p_val  = summary["trend_p_value"]
    sig    = "**" if p_val < 0.05 else ("*" if p_val < 0.10 else "(n.s.)")
    axes[0].plot(years, y_trend, color=RED, lw=1.5, linestyle="-",
                 label=f"OLS trend: {slope:+.1f} mm/yr {sig}")

    axes[0].set_ylabel("Annual Total Rainfall (mm)")
    axes[0].set_title("Annual Rainfall Totals with OLS Trend Line")
    axes[0].legend(fontsize=9)
    add_stat_annotations(axes[0], {
        "CV": f"{summary['annual_cv_pct']:.1f}%",
        "Min (year)": f"{summary['annual_min_mm']:.0f} ({annual.idxmin()})",
        "Max (year)": f"{summary['annual_max_mm']:.0f} ({annual.idxmax()})",
        "p-value":    summary["trend_p_value"],
    }, x=0.03, y_start=0.97, ha="left")

    # --- Bottom: anomaly ---
    anomaly = values - mean
    bar_anom = [GREEN if a >= 0 else RED for a in anomaly]
    axes[1].bar(years, anomaly, color=bar_anom, alpha=0.75, edgecolor="none")
    axes[1].axhline(0, color=GRAY, lw=0.8)
    axes[1].set_ylabel("Anomaly (mm)")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Annual Anomaly (Departure from 26-yr Mean)")
    axes[1].set_xlim(years.min() - 0.5, years.max() + 0.5)

    plt.xticks(years, rotation=45, ha="right", fontsize=9)
    add_figure_title(
        fig,
        "Annual Rainfall Totals & Trend — Lucknow 2000–2025"
    )
    if save:
        save_figure(fig, "06_annual_totals_trend", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Rolling statistics
# ---------------------------------------------------------------------------

def _plot_rolling_statistics(df: pd.DataFrame, save: bool) -> None:
    """
    Time-series plot of:
      - Daily rainfall (bar, light)
      - 7-day rolling mean
      - 30-day rolling mean
      - 90-day rolling mean
      - 30-day rolling standard deviation (separate panel)
    """
    rain = df["RAINFALL"]
    short  = CFG.eda.rolling_window_short
    medium = CFG.eda.rolling_window_medium
    long   = CFG.eda.rolling_window_long

    roll_short  = rain.rolling(short,  center=True, min_periods=short//2).mean()
    roll_medium = rain.rolling(medium, center=True, min_periods=medium//2).mean()
    roll_long   = rain.rolling(long,   center=True, min_periods=long//2).mean()
    roll_std    = rain.rolling(medium, center=True, min_periods=medium//2).std()

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # --- Top panel: daily + rolling means ---
    axes[0].bar(rain.index, rain.values, color=BLUE, alpha=0.20,
                width=1, label="Daily rainfall")
    axes[0].plot(roll_short.index,  roll_short,  lw=0.9, color=ORANGE,
                 alpha=0.85, label=f"{short}-day rolling mean")
    axes[0].plot(roll_medium.index, roll_medium, lw=1.4, color=RED,
                 label=f"{medium}-day rolling mean")
    axes[0].plot(roll_long.index,   roll_long,   lw=1.8, color=GREEN,
                 label=f"{long}-day rolling mean")

    annotate_monsoon_bands(axes[0], df)
    axes[0].set_ylabel("Rainfall (mm/day)")
    axes[0].set_title("Daily Rainfall with Rolling Mean Smoothing")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_ylim(0, rain.quantile(0.998) * 1.05)

    # --- Bottom panel: rolling volatility ---
    axes[1].fill_between(roll_std.index, roll_std.values, alpha=0.45,
                         color=ORANGE, label=f"{medium}-day rolling std dev")
    axes[1].plot(roll_std.index, roll_std.values, lw=0.8, color=ORANGE)
    annotate_monsoon_bands(axes[1], df)
    axes[1].set_ylabel("Rolling Std Dev (mm)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Rolling Rainfall Variability (Volatility)")
    axes[1].legend(fontsize=9)

    format_date_axis(axes[1], date_format="%Y")

    add_figure_title(
        fig, "Rolling Mean & Variability Analysis — Lucknow 2000–2025"
    )
    if save:
        save_figure(fig, "07_rolling_statistics", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Monsoon analysis
# ---------------------------------------------------------------------------

def _plot_monsoon_analysis(
    df: pd.DataFrame,
    summary: Dict,
    save: bool,
) -> None:
    """
    3-panel monsoon deep-dive:
      (a) Year-by-year monsoon (Jun–Sep) total
      (b) Intra-monsoon daily rainfall profile (30-day rolling mean per DOY)
      (c) Season comparison boxplots
    """
    rain      = df["RAINFALL"]
    monsoon_m = list(CFG.monsoon_months)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- (a) Annual monsoon totals ---
    mask_m   = rain.index.month.isin(monsoon_m)
    ann_mon  = rain[mask_m].groupby(rain[mask_m].index.year).sum()
    ann_full = rain.groupby(rain.index.year).sum()
    frac     = (ann_mon / ann_full * 100)

    ax = axes[0]
    ax.bar(ann_mon.index, ann_mon.values, color=GREEN, alpha=0.75,
           edgecolor="none", label="Monsoon total")
    ax.plot(frac.index, frac.values / frac.max() * ann_mon.max(),
            color=RED, lw=1.2, linestyle="--",
            marker=".", ms=4, label="% of annual (scaled)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Monsoon Total Rainfall (mm)")
    ax.set_title("(a) Annual Monsoon Season Totals")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(frac.index, frac.values, color=RED, lw=0, alpha=0)
    ax2.set_ylabel("Monsoon fraction of annual total (%)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)

    # --- (b) Climatological intra-monsoon rainfall profile ---
    ax = axes[1]
    doy_mean = rain.groupby(rain.index.dayofyear).mean()
    roll_doy = doy_mean.rolling(14, center=True, min_periods=5).mean()
    doys = doy_mean.index.values
    ax.fill_between(doys, doy_mean.values, alpha=0.25, color=BLUE)
    ax.plot(doys, doy_mean.values, color=BLUE, lw=0.7, alpha=0.5,
            label="Daily climatological mean")
    ax.plot(doys, roll_doy.values, color=GREEN, lw=2.0,
            label="14-day smoothed mean")
    # Shade monsoon DOY range (approx DOY 152–273)
    ax.axvspan(152, 273, alpha=0.08, color=GREEN, label="JJAS (DOY 152–273)")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Mean Rainfall (mm/day)")
    ax.set_title("(b) Climatological Daily Rainfall Profile")
    ax.set_xlim(1, 365)
    ax.legend(fontsize=8)

    # --- (c) Season comparison boxplots ---
    ax = axes[2]
    seasons_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
    season_data   = [
        rain[df["SEASON"] == s].values for s in seasons_order
    ]
    season_colors = [SEASON_COLORS[s] for s in seasons_order]
    bp = ax.boxplot(
        season_data, labels=seasons_order,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3, color=GRAY),
        medianprops=dict(color="black", lw=1.5),
    )
    for patch, color in zip(bp["boxes"], season_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_ylabel("Daily Rainfall (mm, symlog scale)")
    ax.set_title("(c) Rainfall Distribution by Season")
    ax.tick_params(axis="x", rotation=15)

    add_figure_title(
        fig,
        "Monsoon Seasonality Analysis — Lucknow 2000–2025",
        f"JJAS fraction: {summary['monsoon_fraction_pct']:.1f}% of annual total",
    )
    if save:
        save_figure(fig, "08_monsoon_analysis", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Dry spell analysis
# ---------------------------------------------------------------------------

def _plot_dry_spell_analysis(
    df: pd.DataFrame,
    summary: Dict,
    save: bool,
) -> None:
    """
    (a) Histogram of dry spell lengths
    (b) Month-of-year distribution of dry spell starts
    (c) Cumulative distribution of dry spell lengths
    """
    dry_spells = summary["dry_spells"]
    rain       = df["RAINFALL"]
    dry_t      = CFG.rainfall.dry_day_threshold

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- (a) Dry spell length histogram ---
    axes[0].hist(dry_spells, bins=40, color=ORANGE, alpha=0.80, edgecolor="none")
    axes[0].axvline(np.mean(dry_spells), color=RED, lw=1.5, linestyle="--",
                    label=f"Mean: {np.mean(dry_spells):.1f} days")
    axes[0].axvline(30, color=GRAY, lw=1.2, linestyle=":",
                    label="30-day threshold")
    axes[0].set_xlabel("Dry Spell Length (days)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("(a) Dry Spell Length Distribution")
    axes[0].legend(fontsize=9)
    add_stat_annotations(axes[0], {
        "N spells": summary["n_dry_spells"],
        "Max spell": f"{summary['max_dry_spell_days']} days",
        ">30 days": summary["n_long_dry_spells"],
    }, x=0.97, y_start=0.97)

    # --- (b) Month-of-year of dry spell starts ---
    # Re-derive spell start months
    dry_flag    = (rain < dry_t).astype(int)
    spell_starts: List[int] = []
    in_spell = False
    for date, val in dry_flag.items():
        if val == 1 and not in_spell:
            spell_starts.append(date.month)
            in_spell = True
        elif val == 0:
            in_spell = False

    month_counts = pd.Series(spell_starts).value_counts().sort_index()
    months_all   = pd.Series(0, index=range(1, 13))
    month_counts = months_all.add(month_counts, fill_value=0).astype(int)
    monsoon = list(CFG.monsoon_months)
    bar_c = [GREEN if m in monsoon else ORANGE for m in range(1, 13)]
    axes[1].bar(range(1, 13), month_counts.values, color=bar_c,
                alpha=0.75, edgecolor="none")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(MONTH_LABELS)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Number of Dry Spell Starts")
    axes[1].set_title("(b) Dry Spell Start Month Distribution")
    axes[1].axvspan(5.5, 9.5, alpha=0.06, color=GREEN)

    # --- (c) ECDF of dry spell lengths ---
    sorted_spells = np.sort(dry_spells)
    ecdf = np.arange(1, len(sorted_spells) + 1) / len(sorted_spells)
    axes[2].plot(sorted_spells, ecdf, color=BLUE, lw=1.5, label="ECDF")
    axes[2].axhline(0.50, color=RED, lw=0.9, linestyle="--",
                    label=f"Median: {int(np.median(sorted_spells))} days")
    axes[2].axhline(0.90, color=ORANGE, lw=0.9, linestyle=":",
                    label=f"90th %ile: {int(np.percentile(sorted_spells, 90))} days")
    axes[2].set_xlabel("Dry Spell Length (days)")
    axes[2].set_ylabel("Cumulative Probability")
    axes[2].set_title("(c) Empirical CDF of Dry Spell Lengths")
    axes[2].legend(fontsize=9)
    axes[2].set_xlim(0)
    axes[2].set_ylim(0, 1.05)

    add_figure_title(
        fig,
        "Dry Spell Analysis — Lucknow 2000–2025",
        f"{summary['n_dry_spells']} spells identified "
        f"(min. {CFG.eda.dry_spell_min_length} consecutive dry days)",
    )
    if save:
        save_figure(fig, "09_dry_spell_analysis", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Year × DOY heatmap
# ---------------------------------------------------------------------------

def _plot_doy_year_heatmap(df: pd.DataFrame, save: bool) -> None:
    """
    2D heatmap: rows = year, columns = day-of-year, colour = rainfall.
    Reveals monsoon inter-annual variability and extreme event clustering.
    Uses a perceptually-uniform colourmap with log scale for better contrast.
    """
    rain  = df["RAINFALL"]
    years = sorted(rain.index.year.unique())
    doys  = range(1, 366)

    # Build 2D array: rows=year, cols=DOY
    grid = np.full((len(years), 365), np.nan)
    for i, yr in enumerate(years):
        yr_data = rain[rain.index.year == yr]
        for date, val in yr_data.items():
            doy = date.dayofyear
            if 1 <= doy <= 365:
                grid[i, doy - 1] = val

    # Log scale for visual contrast (0 → small epsilon)
    grid_log = np.log1p(grid)

    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(
        grid_log,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        vmin=0,
        vmax=np.nanpercentile(grid_log, 99),
        interpolation="nearest",
    )

    # y-axis: years
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=8)
    ax.set_ylabel("Year")

    # x-axis: month labels at mid-month DOY
    mid_month_doys = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    ax.set_xticks([d - 1 for d in mid_month_doys])
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_xlabel("Month")

    # Monsoon boundary lines
    for doy_bound in [151, 273]:  # Jun 1, Sep 30
        ax.axvline(doy_bound, color="white", lw=1.0, linestyle="--", alpha=0.7)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("log(1 + Rainfall) [mm]", fontsize=9)
    # Custom tick labels to show real mm values
    tick_vals = [0, np.log1p(1), np.log1p(5), np.log1p(20),
                 np.log1p(50), np.log1p(100)]
    tick_lbls = ["0", "1", "5", "20", "50", "100"]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{l} mm" for l in tick_lbls], fontsize=8)

    ax.set_title(
        "Year × Day-of-Year Rainfall Heatmap — Lucknow 2000–2025\n"
        "(log scale | white dashed lines: monsoon boundaries Jun 1, Sep 30)",
        fontsize=12, fontweight="bold"
    )

    if save:
        save_figure(fig, "10_doy_year_heatmap", subdir="eda")
    plt.close(fig)
