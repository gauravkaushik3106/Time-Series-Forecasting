"""
eda_rainfall_distribution.py
============================
Rainfall distribution analysis: histogram, log-transform, KDE, boxplot,
zero-inflation, skewness/kurtosis, extreme event characterisation.

This is the first step of EDA because understanding the target variable's
statistical nature drives every subsequent modelling decision:
  - Zero-inflation (55.2% dry days) → two-stage / zero-inflated model
  - Heavy right skew (6.8) → log transform, specialised loss functions
  - Fat tails (kurtosis 73) → extreme value attention during evaluation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, expon

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG
from src.visualization.plot_utils import (
    apply_style, save_figure, add_stat_annotations,
    BLUE, RED, GREEN, ORANGE, GRAY, RAIN_CATEGORY_COLORS,
    make_figure, add_figure_title,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IMD rainfall classification
# ---------------------------------------------------------------------------

def classify_rainfall(series: pd.Series) -> pd.Series:
    """
    Apply India Meteorological Department (IMD) daily rainfall classification.
    Thresholds sourced from IMD guidelines for sub-divisional forecasting.
    """
    trace = CFG.rainfall.trace_threshold
    light = CFG.rainfall.light_threshold
    mod   = CFG.rainfall.moderate_threshold
    heavy = CFG.rainfall.heavy_threshold
    vhvy  = CFG.rainfall.very_heavy_threshold
    extrm = CFG.rainfall.extreme_threshold

    conditions = [
        series < trace,
        (series >= trace) & (series < light),
        (series >= light) & (series < mod),
        (series >= mod) & (series < heavy),
        (series >= heavy) & (series < vhvy),
        series >= vhvy,
    ]
    choices = ["No rain", "Light", "Moderate", "Heavy", "Very Heavy", "Extremely Heavy"]
    return pd.Series(
        np.select(conditions, choices, default="Unknown"),
        index=series.index,
        name="RAINFALL_CATEGORY",
    )


# ---------------------------------------------------------------------------
# Main distribution analysis function
# ---------------------------------------------------------------------------

def analyse_rainfall_distribution(
    df: pd.DataFrame,
    save: bool = True,
) -> Dict[str, float | int]:
    """
    Comprehensive rainfall distribution analysis.

    Parameters
    ----------
    df   : Full cleaned dataframe with DatetimeIndex.
    save : Whether to write figures to disk.

    Returns
    -------
    dict of computed statistics for downstream reporting.
    """
    apply_style()
    rain = df["RAINFALL"]
    rain_nonzero = rain[rain > CFG.rainfall.trace_threshold]

    # --- Compute key statistics ---
    stats_dict = _compute_distribution_stats(rain)
    logger.info("Rainfall distribution statistics computed")
    _log_stats(stats_dict)

    # --- Figure 1: Multi-panel distribution overview ---
    _plot_distribution_overview(rain, rain_nonzero, stats_dict, save)

    # --- Figure 2: IMD category frequency ---
    _plot_imd_categories(rain, save)

    # --- Figure 3: Extreme rainfall characterisation ---
    _plot_extreme_rainfall(rain, save)

    # --- Figure 4: Q-Q plot vs candidate distributions ---
    _plot_qq_distributions(rain_nonzero, save)

    return stats_dict


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _compute_distribution_stats(rain: pd.Series) -> Dict:
    """Return a dictionary of descriptive and distributional statistics."""
    dry_threshold = CFG.rainfall.dry_day_threshold
    heavy_t       = CFG.rainfall.heavy_threshold
    extreme_t     = CFG.rainfall.extreme_threshold

    n_total    = len(rain)
    n_dry      = int((rain < dry_threshold).sum())
    n_rainy    = n_total - n_dry
    n_heavy    = int((rain >= heavy_t).sum())
    n_extreme  = int((rain >= extreme_t).sum())

    rain_nonzero = rain[rain > dry_threshold]

    # Skewness and kurtosis (Fisher's definition, excess kurtosis)
    skew = float(stats.skew(rain))
    kurt = float(stats.kurtosis(rain))         # excess (normal = 0)
    skew_nz = float(stats.skew(rain_nonzero))  # for wet days only

    # Log-transform statistics (add 0.01 to handle zeros)
    log_rain = np.log1p(rain)
    log_skew = float(stats.skew(log_rain))

    # Annual statistics
    annual = rain.groupby(rain.index.year).sum()

    return {
        "n_total":          n_total,
        "n_dry":            n_dry,
        "n_rainy":          n_rainy,
        "pct_dry":          n_dry / n_total * 100,
        "pct_rainy":        n_rainy / n_total * 100,
        "n_heavy":          n_heavy,
        "n_extreme":        n_extreme,
        "mean":             float(rain.mean()),
        "median":           float(rain.median()),
        "std":              float(rain.std()),
        "max":              float(rain.max()),
        "p95":              float(rain.quantile(0.95)),
        "p99":              float(rain.quantile(0.99)),
        "skewness":         skew,
        "kurtosis_excess":  kurt,
        "skewness_nonzero": skew_nz,
        "log_skewness":     log_skew,
        "annual_mean":      float(annual.mean()),
        "annual_std":       float(annual.std()),
        "annual_cv_pct":    float(annual.std() / annual.mean() * 100),
        "annual_min":       float(annual.min()),
        "annual_max":       float(annual.max()),
    }


def _log_stats(s: Dict) -> None:
    logger.info(
        f"Rainfall stats: mean={s['mean']:.2f} mm, skew={s['skewness']:.2f}, "
        f"dry_days={s['pct_dry']:.1f}%, max={s['max']:.1f} mm"
    )


# ---------------------------------------------------------------------------
# Figure 1: Distribution overview (4-panel)
# ---------------------------------------------------------------------------

def _plot_distribution_overview(
    rain: pd.Series,
    rain_nonzero: pd.Series,
    stats_dict: Dict,
    save: bool,
) -> None:
    """
    4-panel figure:
      (a) Raw histogram (all days)
      (b) Log-scale histogram (all days)
      (c) Wet-day KDE with fitted distributions
      (d) Monthly boxplots
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # --- Panel (a): Raw histogram ---
    bins_raw = np.concatenate([
        [0, 0.1],                              # dry-day bin
        np.arange(0.1, 10, 0.5),              # light rain (fine)
        np.arange(10, 50, 2),                  # moderate
        np.arange(50, 200, 10),                # heavy
    ])
    ax1.hist(
        rain, bins=bins_raw, color=BLUE, alpha=0.80, edgecolor="none",
        label="All days"
    )
    ax1.axvline(
        CFG.rainfall.dry_day_threshold, color=RED, lw=1.2,
        linestyle="--", label=f"Dry threshold ({CFG.rainfall.dry_day_threshold} mm)"
    )
    ax1.axvline(
        CFG.rainfall.heavy_threshold, color=ORANGE, lw=1.2,
        linestyle=":", label=f"Heavy rain ({CFG.rainfall.heavy_threshold} mm)"
    )
    ax1.set_xlabel("Daily Rainfall (mm)")
    ax1.set_ylabel("Frequency (days)")
    ax1.set_title("(a) Raw Rainfall Distribution")
    ax1.set_xlim(-1, rain.max() * 1.02)
    ax1.legend(fontsize=8)
    add_stat_annotations(ax1, {
        "N": stats_dict["n_total"],
        "Dry days": f"{stats_dict['pct_dry']:.1f}%",
        "Skewness": stats_dict["skewness"],
        "Kurtosis": stats_dict["kurtosis_excess"],
    }, x=0.97, y_start=0.97)

    # --- Panel (b): Log1p-transformed histogram ---
    log_rain = np.log1p(rain)
    ax2.hist(
        log_rain, bins=60, color=GREEN, alpha=0.80, edgecolor="none",
        label="log(1 + rainfall)"
    )
    # Overlay fitted normal
    mu, sigma = log_rain.mean(), log_rain.std()
    x_fit = np.linspace(log_rain.min(), log_rain.max(), 300)
    y_fit = norm.pdf(x_fit, mu, sigma) * len(log_rain) * (log_rain.max() - log_rain.min()) / 60
    ax2.plot(x_fit, y_fit, color=RED, lw=1.8, linestyle="--", label="Fitted normal")
    ax2.set_xlabel("log(1 + Rainfall)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("(b) Log-Transformed Distribution")
    ax2.legend(fontsize=8)
    add_stat_annotations(ax2, {
        "Log-skewness": stats_dict["log_skewness"],
        "μ (log)": mu,
        "σ (log)": sigma,
    }, x=0.97, y_start=0.97)

    # --- Panel (c): Wet-day KDE ---
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(rain_nonzero, bw_method="scott")
    x_kde = np.linspace(0, rain_nonzero.quantile(0.995), 400)
    ax3.fill_between(
        x_kde, kde(x_kde), alpha=0.35, color=BLUE, label="KDE (wet days)"
    )
    ax3.plot(x_kde, kde(x_kde), color=BLUE, lw=1.5)

    # Overlay exponential fit (common null model for rainfall amounts)
    exp_scale = rain_nonzero.mean()
    ax3.plot(
        x_kde, expon.pdf(x_kde, scale=exp_scale),
        color=RED, lw=1.5, linestyle="--", label="Exponential fit"
    )

    for thresh, label, col in [
        (CFG.rainfall.light_threshold,    "Light",    "#AEC6CF"),
        (CFG.rainfall.moderate_threshold, "Moderate", "#5B9BD5"),
        (CFG.rainfall.heavy_threshold,    "Heavy",    "#2E75B6"),
    ]:
        ax3.axvline(thresh, color=col, lw=0.9, linestyle=":", alpha=0.7)
        ax3.text(
            thresh + 0.5, ax3.get_ylim()[1] * 0.01,
            label, fontsize=7, color=col, rotation=90, va="bottom"
        )

    ax3.set_xlabel("Daily Rainfall (mm)")
    ax3.set_ylabel("Density")
    ax3.set_title("(c) Wet-Day KDE with Exponential Fit")
    ax3.set_xlim(0, rain_nonzero.quantile(0.995))
    ax3.legend(fontsize=8)

    # --- Panel (d): Monthly boxplots ---
    monthly_data = [
        rain[rain.index.month == m].values for m in range(1, 13)
    ]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bp = ax4.boxplot(
        monthly_data, labels=month_labels,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3, color=GRAY),
        medianprops=dict(color="white", lw=1.5),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
    )
    # Colour monsoon boxes differently
    monsoon = set(CFG.monsoon_months)
    for i, (patch, m) in enumerate(zip(bp["boxes"], range(1, 13))):
        patch.set_facecolor(GREEN if m in monsoon else BLUE)
        patch.set_alpha(0.7)

    ax4.set_xlabel("Month")
    ax4.set_ylabel("Daily Rainfall (mm)")
    ax4.set_title("(d) Monthly Distribution (Boxplots)")
    ax4.set_yscale("symlog", linthresh=1)
    # Add monsoon annotation
    ax4.axvspan(5.5, 9.5, alpha=0.06, color=GREEN, label="Monsoon (JJAS)")
    ax4.legend(fontsize=8)

    add_figure_title(
        fig,
        "Rainfall Distribution Analysis — Lucknow 2000–2025",
        "Daily meteorological records | IMD station",
    )

    if save:
        save_figure(fig, "01_rainfall_distribution_overview", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: IMD category bar chart
# ---------------------------------------------------------------------------

def _plot_imd_categories(rain: pd.Series, save: bool) -> None:
    """Bar chart of day-count by IMD rainfall category."""
    categories = classify_rainfall(rain)
    order = ["No rain", "Light", "Moderate", "Heavy", "Very Heavy", "Extremely Heavy"]
    counts = categories.value_counts().reindex(order, fill_value=0)
    pcts   = counts / len(rain) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = [RAIN_CATEGORY_COLORS[c] for c in order]

    # Left: absolute counts
    bars = axes[0].bar(order, counts.values, color=colors, edgecolor="#AAAAAA",
                       linewidth=0.5)
    axes[0].set_xlabel("IMD Category")
    axes[0].set_ylabel("Number of Days")
    axes[0].set_title("Day Count by IMD Rainfall Category")
    axes[0].tick_params(axis="x", rotation=30)
    # Annotate bars with counts
    for bar, cnt in zip(bars, counts.values):
        if cnt > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{cnt:,}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    # Right: percentage
    bars2 = axes[1].bar(order, pcts.values, color=colors, edgecolor="#AAAAAA",
                        linewidth=0.5)
    axes[1].set_xlabel("IMD Category")
    axes[1].set_ylabel("Percentage of All Days (%)")
    axes[1].set_title("Frequency Distribution by IMD Category")
    axes[1].tick_params(axis="x", rotation=30)
    for bar, pct in zip(bars2, pcts.values):
        if pct > 0:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                pct + 0.3,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    add_figure_title(fig, "IMD Daily Rainfall Classification — Lucknow 2000–2025")
    if save:
        save_figure(fig, "02_imd_category_distribution", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Extreme rainfall events
# ---------------------------------------------------------------------------

def _plot_extreme_rainfall(rain: pd.Series, save: bool) -> None:
    """
    Characterise extreme rainfall events:
      - Annual maximum series
      - Exceedance probability curve (empirical CDF complement)
      - Top-20 extreme events timeline
    """
    heavy_t = CFG.rainfall.heavy_threshold

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Annual maximum series ---
    annual_max = rain.groupby(rain.index.year).max()
    axes[0].bar(
        annual_max.index, annual_max.values,
        color=BLUE, alpha=0.75, edgecolor="none"
    )
    # Linear trend line
    slope, intercept, *_ = stats.linregress(annual_max.index, annual_max.values)
    x_trend = np.array([annual_max.index.min(), annual_max.index.max()])
    axes[0].plot(x_trend, slope * x_trend + intercept, color=RED, lw=1.5,
                 linestyle="--", label=f"Trend: {slope:+.2f} mm/yr")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Annual Maximum Daily Rainfall (mm)")
    axes[0].set_title("Annual Maximum Rainfall Series")
    axes[0].legend(fontsize=9)

    # --- Panel 2: Exceedance probability (return period approximation) ---
    sorted_rain = np.sort(rain[rain > 0].values)[::-1]
    n = len(sorted_rain)
    # Weibull plotting position: P(X > x) = rank / (n+1)
    exceedance_prob = np.arange(1, n + 1) / (n + 1)
    axes[1].semilogy(sorted_rain, exceedance_prob, color=BLUE, lw=1.2,
                     alpha=0.8, label="Empirical")
    # Mark IMD thresholds
    for thresh, label, col in [
        (heavy_t, "Heavy", RED),
        (CFG.rainfall.very_heavy_threshold, "V.Heavy", ORANGE),
    ]:
        p_exceed = (rain >= thresh).mean()
        axes[1].axvline(thresh, color=col, lw=0.9, linestyle=":",
                        label=f"{label}: {thresh} mm ({p_exceed*100:.2f}%)")
    axes[1].set_xlabel("Daily Rainfall (mm)")
    axes[1].set_ylabel("Exceedance Probability P(X ≥ x)")
    axes[1].set_title("Empirical Exceedance Probability Curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    # --- Panel 3: Top-20 extreme events timeline ---
    top20 = rain.nlargest(20).sort_index()
    monsoon_flag = top20.index.month.isin(list(CFG.monsoon_months))
    colors_top = [GREEN if m else ORANGE for m in monsoon_flag]
    axes[2].barh(
        range(len(top20)),
        top20.values,
        color=colors_top, alpha=0.8, edgecolor="none"
    )
    axes[2].set_yticks(range(len(top20)))
    axes[2].set_yticklabels(
        [f"{d.strftime('%Y-%m-%d')}" for d in top20.index],
        fontsize=8
    )
    axes[2].set_xlabel("Daily Rainfall (mm)")
    axes[2].set_title("Top-20 Extreme Rainfall Events")
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=GREEN, label="Monsoon (JJAS)"),
        Patch(facecolor=ORANGE, label="Non-Monsoon"),
    ]
    axes[2].legend(handles=legend_els, fontsize=8)

    add_figure_title(
        fig, "Extreme Rainfall Analysis — Lucknow 2000–2025"
    )
    if save:
        save_figure(fig, "03_extreme_rainfall_analysis", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Q-Q plots against candidate distributions
# ---------------------------------------------------------------------------

def _plot_qq_distributions(rain_nonzero: pd.Series, save: bool) -> None:
    """
    Q-Q plots comparing wet-day rainfall against:
      1. Normal distribution
      2. Log-normal distribution
      3. Gamma distribution
      4. Exponential distribution

    The best-fitting distribution informs the choice of probabilistic
    forecast output distribution in the uncertainty module.
    """
    from scipy.stats import probplot, gamma as gamma_dist

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    log_rain = np.log(rain_nonzero)

    from scipy.stats import gamma as gamma_dist_fit, expon as expon_dist_fit

    # Fit distribution parameters from data for shape-parameterised dists
    gamma_params = gamma_dist_fit.fit(rain_nonzero, floc=0)
    expon_params  = expon_dist_fit.fit(rain_nonzero, floc=0)

    distributions = [
        ("Normal (raw rainfall)",      rain_nonzero, "norm",  {}),
        ("Normal (log-rainfall)",      log_rain,     "norm",  {}),
        ("Gamma (raw rainfall)",       rain_nonzero, "gamma", {"sparams": (gamma_params[0],)}),
        ("Exponential (raw rainfall)", rain_nonzero, "expon", {}),
    ]

    for ax, (title, data, dist_name, extra_kwargs) in zip(axes, distributions):
        res = probplot(data, dist=dist_name, plot=None, **extra_kwargs)
        theoretical_q, ordered_vals = res[0]
        slope, intercept, r = res[1]

        ax.scatter(theoretical_q, ordered_vals, s=4, alpha=0.3, color=BLUE)
        x_line = np.array([theoretical_q.min(), theoretical_q.max()])
        ax.plot(x_line, slope * x_line + intercept, color=RED, lw=1.5,
                linestyle="--", label=f"R² = {r**2:.4f}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Ordered Values")
        ax.set_title(f"Q-Q Plot: {title}")
        ax.legend(fontsize=9)

    add_figure_title(
        fig,
        "Q-Q Distribution Fitting — Wet Days (Rainfall > 0.1 mm)",
        "Best-fitting distribution guides probabilistic forecast output",
    )
    if save:
        save_figure(fig, "04_qq_distribution_fitting", subdir="eda")
    plt.close(fig)
