"""
eda_timeseries.py
=================
Time-series analysis: autocorrelation structure, STL decomposition,
and stationarity testing with automatic interpretation generation.

Why each component exists
--------------------------
ACF / PACF
  Directly informs SARIMAX order selection (p, q) and the required lookback
  window for LSTM/GRU.  Strong lag-1 ACF (0.49) confirms AR(1) at minimum;
  the slow seasonal decay pattern confirms the need for seasonal differencing
  or explicit seasonal dummies.

STL Decomposition
  Separates the time series into trend, seasonal, and residual components.
  The residual component is the hardest part to predict and reveals whether
  significant unexplained variance remains after seasonal adjustment.
  Important: if residuals are non-random, additional structure is exploitable.

ADF Test (Augmented Dickey-Fuller)
  Tests H₀: unit root present (non-stationary).
  Rejection (p < 0.05) → stationary.
  SARIMAX requires stationarity; the degree of differencing (d, D) is
  determined here.

KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)
  Tests H₀: stationary.
  Non-rejection → stationary.
  Used alongside ADF: ADF rejects + KPSS doesn't reject → strong stationarity evidence.
  Both tests together avoid the power limitations of either alone.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG
from src.visualization.plot_utils import (
    apply_style, save_figure, add_stat_annotations, annotate_monsoon_bands,
    format_date_axis, BLUE, RED, GREEN, ORANGE, GRAY,
    make_figure, add_figure_title,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyse_timeseries_structure(
    df: pd.DataFrame,
    save: bool = True,
) -> Dict:
    """
    Full time-series structure analysis with auto-generated interpretation text.

    Parameters
    ----------
    df   : Cleaned dataframe with DatetimeIndex.
    save : Persist figures to disk.

    Returns
    -------
    dict containing test results, interpretation strings, and ARIMA order hints.
    """
    apply_style()
    rain = df["RAINFALL"]

    # --- Run all analyses ---
    acf_pacf_stats = _plot_acf_pacf(rain, save)
    stl_results    = _run_stl_decomposition(rain, df, save)
    adf_result     = _run_adf_test(rain)
    kpss_result    = _run_kpss_test(rain)

    # --- Stationarity plots ---
    _plot_stationarity_diagnostics(rain, df, adf_result, kpss_result, save)

    # --- Auto-generate interpretation text ---
    interpretation = _generate_interpretation(
        acf_pacf_stats, stl_results, adf_result, kpss_result
    )
    _log_interpretation(interpretation)

    return {
        "acf_pacf":      acf_pacf_stats,
        "stl":           stl_results,
        "adf":           adf_result,
        "kpss":          kpss_result,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Figure 15: ACF and PACF
# ---------------------------------------------------------------------------

def _plot_acf_pacf(rain: pd.Series, save: bool) -> Dict:
    """
    Plot ACF and PACF for:
      (a) Raw daily rainfall
      (b) log(1 + rainfall) — reduces right-skew influence on ACF
      (c) Monthly aggregated rainfall (reveals seasonal structure)
    Also plot lag-scatter for lag 1, 2, 7 to visualise autocorrelation shape.
    """
    n_lags = CFG.eda.acf_lags

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.30)

    # Row 1: raw daily
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # Row 2: log-transformed daily
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    # Row 3: monthly aggregated
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Compute manual ACF for statistics
    acf_vals_raw  = [rain.autocorr(lag=l) for l in range(1, 31)]
    log_rain      = np.log1p(rain)
    acf_vals_log  = [log_rain.autocorr(lag=l) for l in range(1, 31)]

    # --- Raw ACF/PACF ---
    plot_acf( rain.values,     ax=ax1, lags=n_lags, alpha=0.05,
              title="(a) ACF — Raw Daily Rainfall",
              color=BLUE, zero=False)
    plot_pacf(rain.values,     ax=ax2, lags=min(n_lags, 40), alpha=0.05,
              title="(b) PACF — Raw Daily Rainfall",
              color=BLUE, zero=False, method="ywm")
    # Mark significant lag-1
    ax1.annotate(
        f"lag-1: {acf_vals_raw[0]:.3f}", xy=(1, acf_vals_raw[0]),
        xytext=(5, acf_vals_raw[0] + 0.05),
        arrowprops=dict(arrowstyle="->", color=RED), fontsize=8, color=RED
    )

    # --- Log-transformed ACF/PACF ---
    plot_acf( log_rain.values, ax=ax3, lags=n_lags, alpha=0.05,
              title="(c) ACF — log(1+Rainfall)",
              color=GREEN, zero=False)
    plot_pacf(log_rain.values, ax=ax4, lags=min(n_lags, 40), alpha=0.05,
              title="(d) PACF — log(1+Rainfall)",
              color=GREEN, zero=False, method="ywm")

    # --- Monthly aggregated ACF/PACF ---
    monthly_rain = rain.resample("ME").sum()
    n_monthly_lags = min(36, len(monthly_rain) // 2 - 1)
    plot_acf( monthly_rain.values, ax=ax5, lags=n_monthly_lags, alpha=0.05,
              title="(e) ACF — Monthly Aggregated Rainfall",
              color=ORANGE, zero=False)
    plot_pacf(monthly_rain.values, ax=ax6, lags=min(24, n_monthly_lags), alpha=0.05,
              title="(f) PACF — Monthly Aggregated Rainfall",
              color=ORANGE, zero=False, method="ywm")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axhline(0, color=GRAY, lw=0.5)

    add_figure_title(
        fig,
        "Autocorrelation Structure — Lucknow Rainfall",
        "Blue shading: 95% confidence interval for white noise"
    )
    if save:
        save_figure(fig, "15_acf_pacf", subdir="eda")
    plt.close(fig)

    # --- Lag scatter plots ---
    _plot_lag_scatters(rain, save)

    return {
        "lag1_acf_raw":  float(acf_vals_raw[0]),
        "lag2_acf_raw":  float(acf_vals_raw[1]),
        "lag7_acf_raw":  float(acf_vals_raw[6]),
        "lag1_acf_log":  float(acf_vals_log[0]),
        "lag7_acf_log":  float(acf_vals_log[6]),
        "n_significant_lags_raw": int(sum(
            abs(v) > 2 / np.sqrt(len(rain)) for v in acf_vals_raw
        )),
    }


def _plot_lag_scatters(rain: pd.Series, save: bool) -> None:
    """Lag-scatter plots for lags 1, 2, 7, 14 — shows linearity of autocorrelation."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    lags = [1, 2, 7, 14]
    log_rain = np.log1p(rain.values)

    for ax, lag in zip(axes, lags):
        y = log_rain[lag:]
        x = log_rain[:-lag]
        r, _ = pd.Series(y).corr(pd.Series(x)), None
        r = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, s=2, alpha=0.12, color=BLUE)
        ax.set_xlabel(f"log(1+R)(t-{lag})")
        ax.set_ylabel(f"log(1+R)(t)")
        ax.set_title(f"Lag-{lag} scatter\nr = {r:.3f}")
        # Diagonal reference line
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.7, linestyle="--")

    add_figure_title(fig, "Lag Scatter Plots — log(1+Rainfall)")
    if save:
        save_figure(fig, "16_lag_scatter_plots", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# STL Decomposition
# ---------------------------------------------------------------------------

def _run_stl_decomposition(
    rain: pd.Series,
    df: pd.DataFrame,
    save: bool,
) -> Dict:
    """
    Apply STL (Seasonal and Trend decomposition using Loess) to rainfall.

    STL advantages over classical decomposition:
      - Robust to outliers (important given our extreme rain events)
      - Handles non-constant seasonal shapes
      - Can use any seasonal period (365 for daily data)

    We apply STL to log1p(rainfall) for better numerical stability with
    the zero-inflated, heavily skewed raw series.
    """
    log_rain = np.log1p(rain.copy())
    log_rain.index = pd.DatetimeIndex(log_rain.index, freq="D")

    stl = STL(
        log_rain,
        period=CFG.eda.stl_period,      # annual seasonality
        robust=True,                     # down-weights outlier influence
        seasonal=13,                     # seasonal smoother window (must be odd)
        trend=None,                      # auto-determined from period
    )
    result = stl.fit()

    trend    = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # STL residual variance explained (how much is in residuals vs trend+seasonal)
    total_var    = float(np.var(log_rain))
    resid_var    = float(np.var(residual))
    seasonal_var = float(np.var(seasonal))
    trend_var    = float(np.var(trend))
    resid_frac   = resid_var / total_var if total_var > 0 else 0.0

    # Seasonal strength (Wang et al. 2006 definition)
    seasonal_strength = max(0, 1 - np.var(residual) / np.var(seasonal + residual))
    trend_strength    = max(0, 1 - np.var(residual) / np.var(trend + residual))

    # Residual normality test (Jarque-Bera)
    jb_stat, jb_pvalue, skew_resid, kurt_resid = jarque_bera(residual)

    # --- Plot STL decomposition ---
    _plot_stl(log_rain, trend, seasonal, residual, df, save)

    return {
        "seasonal_strength":  float(seasonal_strength),
        "trend_strength":     float(trend_strength),
        "residual_var_frac":  float(resid_frac),
        "jb_pvalue":          float(jb_pvalue),
        "residual_skew":      float(skew_resid),
        "residual_kurtosis":  float(kurt_resid),
        "trend_component":    trend,
        "seasonal_component": seasonal,
        "residual_component": residual,
    }


def _plot_stl(
    log_rain: pd.Series,
    trend: pd.Series,
    seasonal: pd.Series,
    residual: pd.Series,
    df: pd.DataFrame,
    save: bool,
) -> None:
    """4-panel STL decomposition plot with monsoon shading."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={"hspace": 0.35})

    components = [
        (log_rain, "Observed log(1+Rainfall)",  BLUE,   None),
        (trend,    "Trend Component",            RED,    None),
        (seasonal, "Seasonal Component (annual)", GREEN,  None),
        (residual, "Residual Component",          ORANGE, None),
    ]

    for ax, (comp, title, color, _) in zip(axes, components):
        if title.startswith("Observed"):
            ax.fill_between(comp.index, comp.values, alpha=0.20, color=color)
        ax.plot(comp.index, comp.values, lw=0.8 if title.startswith("Obs") else 1.2,
                color=color)
        ax.axhline(0, color=GRAY, lw=0.5, linestyle="--")
        ax.set_ylabel(title, fontsize=9)
        annotate_monsoon_bands(ax, df, alpha=0.06, color=GREEN)

    axes[3].set_xlabel("Date")
    format_date_axis(axes[3], date_format="%Y")

    add_figure_title(
        fig,
        "STL Decomposition — log(1+Rainfall) — Lucknow 2000–2025",
        "Robust STL | Annual period (365 days) | Green shading: Jun–Sep monsoon"
    )
    if save:
        save_figure(fig, "17_stl_decomposition", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def _run_adf_test(rain: pd.Series) -> Dict:
    """
    Augmented Dickey-Fuller test on raw and log-transformed series.

    H₀: Unit root (non-stationary)
    H₁: Stationary (no unit root)

    We test on:
      1. Raw rainfall (original scale)
      2. log(1+rainfall) (variance stabilised)
      3. Monthly aggregated (smoother, stronger seasonal signal)
    """
    results = {}

    for label, series in [
        ("raw_daily",       rain),
        ("log_daily",       np.log1p(rain)),
        ("monthly_total",   rain.resample("ME").sum()),
    ]:
        # 'AIC' lag selection is standard; maxlags auto-determined by formula
        adf_stat, p_value, n_lags_used, n_obs, crit_vals, icbest = adfuller(
            series.dropna(), autolag="AIC"
        )
        results[label] = {
            "statistic": float(adf_stat),
            "p_value":   float(p_value),
            "n_lags":    int(n_lags_used),
            "critical_values": {k: float(v) for k, v in crit_vals.items()},
            "is_stationary": p_value < 0.05,
        }
        logger.info(
            f"ADF ({label}): stat={adf_stat:.4f}, p={p_value:.4f}, "
            f"stationary={'YES' if p_value < 0.05 else 'NO'}"
        )

    return results


def _run_kpss_test(rain: pd.Series) -> Dict:
    """
    KPSS test on raw and log-transformed series.

    H₀: Stationary (level or trend stationarity)
    H₁: Unit root (non-stationary)

    Note: KPSS and ADF test opposing hypotheses. The combination:
      ADF rejects H₀ (p<0.05)  AND  KPSS doesn't reject H₀ (p>0.05)
      → strong evidence of stationarity.
    """
    results = {}

    for label, series, regression in [
        ("raw_daily_level",    rain,             "c"),   # constant (level)
        ("log_daily_level",    np.log1p(rain),   "c"),
        ("log_daily_trend",    np.log1p(rain),   "ct"),  # constant + trend
    ]:
        try:
            kpss_stat, p_value, n_lags_used, crit_vals = kpss(
                series.dropna(), regression=regression, nlags="auto"
            )
            results[label] = {
                "statistic": float(kpss_stat),
                "p_value":   float(p_value),
                "n_lags":    int(n_lags_used),
                "critical_values": {k: float(v) for k, v in crit_vals.items()},
                "is_stationary": p_value > 0.05,
                "regression": regression,
            }
            logger.info(
                f"KPSS ({label}, reg='{regression}'): stat={kpss_stat:.4f}, "
                f"p={p_value:.4f}, stationary={'YES' if p_value > 0.05 else 'NO'}"
            )
        except Exception as e:
            logger.warning(f"KPSS ({label}) failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Figure 18: Stationarity diagnostic plots
# ---------------------------------------------------------------------------

def _plot_stationarity_diagnostics(
    rain: pd.Series,
    df: pd.DataFrame,
    adf_result: Dict,
    kpss_result: Dict,
    save: bool,
) -> None:
    """
    Visual summary of stationarity:
      (a) Rolling mean and std of monthly totals (visual stationarity check)
      (b) First-differenced series
      (c) Residuals after seasonal adjustment (log-transformed monthly mean subtracted)
      (d) Summary table of test results
    """
    monthly  = rain.resample("ME").sum()
    log_rain = np.log1p(rain)

    roll_win = 12  # 12-month rolling window
    roll_mean = monthly.rolling(roll_win).mean()
    roll_std  = monthly.rolling(roll_win).std()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- (a) Rolling statistics ---
    axes[0, 0].plot(monthly.index, monthly.values, color=BLUE, lw=0.8,
                    alpha=0.6, label="Monthly total")
    axes[0, 0].plot(roll_mean.index, roll_mean.values, color=RED, lw=1.8,
                    label=f"{roll_win}-month rolling mean")
    ax_twin = axes[0, 0].twinx()
    ax_twin.plot(roll_std.index, roll_std.values, color=ORANGE, lw=1.2,
                 linestyle="--", label="Rolling std")
    ax_twin.set_ylabel("Rolling Std (mm)", color=ORANGE)
    axes[0, 0].set_ylabel("Monthly Rainfall (mm)")
    axes[0, 0].set_title("(a) Rolling Mean & Std — Monthly Rainfall")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    ax_twin.legend(fontsize=8, loc="upper right")

    # --- (b) First-differenced monthly series ---
    diff1 = monthly.diff().dropna()
    axes[0, 1].plot(diff1.index, diff1.values, color=GREEN, lw=0.8, alpha=0.7)
    axes[0, 1].axhline(0, color=GRAY, lw=0.8)
    axes[0, 1].fill_between(diff1.index, diff1.values, alpha=0.25, color=GREEN)
    adf_stat = adf_result.get("monthly_total", {}).get("statistic", None)
    adf_p    = adf_result.get("monthly_total", {}).get("p_value", None)
    if adf_stat is not None:
        axes[0, 1].set_title(
            f"(b) First-Differenced Monthly Rainfall\n"
            f"ADF (monthly): stat={adf_stat:.3f}, p={adf_p:.4f}"
        )
    axes[0, 1].set_ylabel("Δ Monthly Rainfall (mm)")

    # --- (c) Deseasonalised daily (subtract monthly climatological mean) ---
    monthly_clim = rain.groupby(rain.index.month).mean()
    deseason = rain - rain.index.map(lambda d: monthly_clim[d.month])
    roll_des = deseason.rolling(30, center=True, min_periods=15).mean()
    axes[1, 0].plot(deseason.index, deseason.values, color=BLUE, lw=0.5,
                    alpha=0.30, label="Deseasonalised (daily)")
    axes[1, 0].plot(roll_des.index, roll_des.values, color=RED, lw=1.5,
                    label="30-day rolling mean")
    axes[1, 0].axhline(0, color=GRAY, lw=0.8)
    annotate_monsoon_bands(axes[1, 0], df, alpha=0.05)
    axes[1, 0].set_ylabel("Rainfall Anomaly (mm)")
    axes[1, 0].set_title("(c) Deseasonalised Daily Rainfall (Subtract Monthly Clim.)")
    axes[1, 0].legend(fontsize=8)

    # --- (d) Test results table ---
    axes[1, 1].axis("off")
    table_data = []
    table_cols = ["Test", "Series", "Statistic", "p-value", "Verdict"]

    adf_series_map = {
        "raw_daily":     "Daily (raw)",
        "log_daily":     "Daily (log)",
        "monthly_total": "Monthly",
    }
    for key, label in adf_series_map.items():
        r = adf_result.get(key, {})
        if r:
            verdict = "✓ Stationary" if r["is_stationary"] else "✗ Non-Stationary"
            table_data.append([
                "ADF", label, f"{r['statistic']:.3f}",
                f"{r['p_value']:.4f}", verdict
            ])

    kpss_series_map = {
        "raw_daily_level": "Daily (raw, level)",
        "log_daily_level": "Daily (log, level)",
        "log_daily_trend": "Daily (log, trend)",
    }
    for key, label in kpss_series_map.items():
        r = kpss_result.get(key, {})
        if r:
            verdict = "✓ Stationary" if r["is_stationary"] else "✗ Non-Stationary"
            table_data.append([
                "KPSS", label, f"{r['statistic']:.3f}",
                f"{r['p_value']:.4f}", verdict
            ])

    if table_data:
        tbl = axes[1, 1].table(
            cellText=table_data,
            colLabels=table_cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.1, 1.8)
        # Colour code by verdict
        for i, row in enumerate(table_data):
            colour = "#D4EDDA" if "✓" in row[-1] else "#F8D7DA"
            for j in range(len(table_cols)):
                tbl[(i + 1, j)].set_facecolor(colour)
        axes[1, 1].set_title("(d) Stationarity Test Results Summary",
                              fontweight="bold", fontsize=10, pad=20)

    add_figure_title(
        fig,
        "Stationarity Analysis — Lucknow Rainfall",
        "ADF: H₀=unit root | KPSS: H₀=stationary"
    )
    format_date_axis(axes[1, 0], date_format="%Y")
    if save:
        save_figure(fig, "18_stationarity_tests", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Automatic interpretation generation
# ---------------------------------------------------------------------------

def _generate_interpretation(
    acf_stats: Dict,
    stl_stats: Dict,
    adf: Dict,
    kpss: Dict,
) -> Dict[str, str]:
    """
    Generate human-readable interpretation text based on statistical test results.
    This text is included verbatim in the preprocessing report.
    """
    interp = {}

    # --- ACF interpretation ---
    lag1 = acf_stats["lag1_acf_raw"]
    lag7 = acf_stats["lag7_acf_raw"]
    n_sig = acf_stats["n_significant_lags_raw"]

    if lag1 > 0.4:
        acf_text = (
            f"Strong positive autocorrelation at lag-1 (r={lag1:.3f}) indicates "
            f"significant day-to-day persistence in rainfall. "
        )
    elif lag1 > 0.2:
        acf_text = f"Moderate autocorrelation at lag-1 (r={lag1:.3f}). "
    else:
        acf_text = f"Weak autocorrelation at lag-1 (r={lag1:.3f}). "

    acf_text += (
        f"By lag-7 the autocorrelation decays to {lag7:.3f}, suggesting "
        f"an effective memory window of approximately 7–14 days. "
        f"{n_sig} lags exceed the 95% significance bound for white noise. "
        "ARIMA p-order should start at 1–2; LSTM lookback window should be "
        "at minimum 14 days, ideally 30–60 days."
    )
    interp["acf"] = acf_text

    # --- STL interpretation ---
    ss = stl_stats["seasonal_strength"]
    ts = stl_stats["trend_strength"]
    rv = stl_stats["residual_var_frac"]

    stl_text = (
        f"STL decomposition reveals a strong seasonal component "
        f"(seasonal strength = {ss:.3f}; 1.0 = perfect seasonality). "
        f"Trend strength = {ts:.3f} — "
        f"{'a modest but present long-term trend' if ts > 0.3 else 'no dominant long-term trend'}. "
        f"The residual component accounts for {rv*100:.1f}% of total variance, "
    )
    if rv > 0.50:
        stl_text += (
            "indicating that a large fraction of variability is not explained by "
            "trend and seasonal components alone. This motivates the use of "
            "exogenous meteorological predictors (SARIMAX, LSTM) rather than "
            "univariate models."
        )
    else:
        stl_text += (
            "indicating that trend and seasonal patterns explain most variance. "
            "A well-specified seasonal model should capture the bulk of predictable signal."
        )
    interp["stl"] = stl_text

    # --- ADF interpretation ---
    adf_raw = adf.get("raw_daily", {})
    adf_log = adf.get("log_daily", {})

    if adf_raw.get("is_stationary") and adf_log.get("is_stationary"):
        adf_text = (
            f"The ADF test rejects the unit root hypothesis for both raw "
            f"(p={adf_raw.get('p_value', 'N/A'):.4f}) and log-transformed "
            f"(p={adf_log.get('p_value', 'N/A'):.4f}) daily series. "
            "Both series are stationary, consistent with bounded meteorological variables. "
            "No differencing is required before SARIMAX fitting."
        )
    elif adf_log.get("is_stationary"):
        adf_text = (
            f"The raw series may have borderline stationarity "
            f"(ADF p={adf_raw.get('p_value', 'N/A'):.4f}), but the "
            f"log-transformed series is clearly stationary "
            f"(p={adf_log.get('p_value', 'N/A'):.4f}). "
            "Use log(1+rainfall) as the SARIMAX target; d=0 is appropriate after transformation."
        )
    else:
        adf_text = (
            "Unit root evidence detected. First-order differencing (d=1) should be "
            "applied before ARIMA/SARIMAX fitting. Verify after differencing."
        )
    interp["adf"] = adf_text

    # --- KPSS interpretation ---
    kpss_log = kpss.get("log_daily_level", {})

    if kpss_log.get("is_stationary"):
        kpss_text = (
            f"KPSS test (log-transformed, level) does not reject H₀ of stationarity "
            f"(p={kpss_log.get('p_value', 'N/A'):.4f}). Combined with ADF evidence, "
            "this strongly supports level-stationarity of the log-transformed series."
        )
    else:
        kpss_text = (
            f"KPSS test (log-transformed, level) rejects H₀ of stationarity "
            f"(p={kpss_log.get('p_value', 'N/A'):.4f}), suggesting possible "
            "non-stationarity not captured by ADF. Consider seasonal differencing."
        )
    interp["kpss"] = kpss_text

    # --- ARIMA order recommendations ---
    interp["arima_recommendations"] = (
        "Based on ACF/PACF and stationarity analysis:\n"
        "  SARIMAX non-seasonal order: (p=1–2, d=0, q=0–1) — try (1,0,0) as baseline\n"
        "  SARIMAX seasonal order: (P=1, D=1, Q=1, s=365) for annual seasonality\n"
        "  (Monthly model: s=12, lighter computational cost for initial benchmarking)\n"
        "  LSTM lookback window: 30–60 days recommended\n"
        "  GRU lookback window: same as LSTM for fair comparison"
    )

    return interp


def _log_interpretation(interp: Dict[str, str]) -> None:
    logger.info("=" * 70)
    logger.info("AUTO-GENERATED TIME-SERIES INTERPRETATION")
    logger.info("=" * 70)
    for section, text in interp.items():
        logger.info(f"\n[{section.upper()}]")
        for line in textwrap.wrap(text, width=70):
            logger.info(f"  {line}")
    logger.info("=" * 70)
