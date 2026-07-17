"""
calibration.py
==============
Calibration analysis for MC Dropout prediction intervals.

A perfectly calibrated probabilistic forecast satisfies:
  P(actual ≤ F^{-1}(α)) = α   for all α ∈ (0,1)

where F^{-1}(α) is the α-quantile of the predictive distribution.

In plain terms: if a model says "I am 90% confident the true value lies
within this interval", then across many predictions, exactly 90% of the
true values should fall inside those intervals.

Calibration metrics
-------------------
Expected Calibration Error (ECE)
  ECE = Σ_k |coverage_k − nominal_k| × n_k / N
  where the sum is over K probability levels.

Sharpness
  Average interval width — a well-calibrated model should also be as sharp
  (narrow intervals) as possible.  Two models with equal ECE: prefer the one
  with narrower intervals.

Continuous Ranked Probability Score (CRPS)
  CRPS rewards both calibration and sharpness simultaneously.
  CRPS = E[|Y - y|] - 0.5 × E[|Y - Y'|]
  where Y, Y' are independent draws from the predictive distribution.
  Lower is better.

Outputs
-------
outputs/figures/uncertainty/calibration_reliability_diagram.png
outputs/figures/uncertainty/calibration_crps.png
outputs/uncertainty/calibration_metrics.csv
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

from config.config_loader import CFG, abs_path
from src.visualization.plot_utils import (
    apply_style, save_figure, add_figure_title,
    BLUE, RED, GREEN, ORANGE, GRAY,
)

logger = logging.getLogger(__name__)

NOMINAL_LEVELS = np.arange(0.10, 1.00, 0.10)   # 10% to 90% in steps of 10%
MONSOON_MONTHS = {6, 7, 8, 9}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_calibration_analysis(
    pred_df:     pd.DataFrame,
    mc_samples:  np.ndarray,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute calibration metrics and generate reliability diagrams.

    Parameters
    ----------
    pred_df    : DataFrame with actual, mean_mm, and lower/upper columns.
    mc_samples : Raw MC samples array shape (T, N) on log scale from
                 MCDropoutPredictor.predict_distribution()['samples'].
    save       : Persist figures and tables to disk.

    Returns
    -------
    calibration_df : DataFrame with one row per nominal level.
    """
    apply_style()

    actual_log = np.log1p(pred_df["actual"].values)
    actual_mm  = pred_df["actual"].values

    # ── Compute empirical coverage at each nominal level ──────────────────
    calibration_records = []
    for alpha in NOMINAL_LEVELS:
        lo_pct = 100 * alpha / 2
        hi_pct = 100 * (1 - alpha / 2)
        lo_log = np.percentile(mc_samples, lo_pct, axis=0)
        hi_log = np.percentile(mc_samples, hi_pct, axis=0)
        covered = ((actual_log >= lo_log) & (actual_log <= hi_log)).mean()

        lo_mm = np.clip(np.expm1(lo_log), 0.0, None)
        hi_mm = np.clip(np.expm1(hi_log), 0.0, None)
        width = (hi_mm - lo_mm).mean()

        calibration_records.append({
            "nominal_coverage":  1 - alpha,
            "empirical_coverage": float(covered),
            "calibration_error": float(abs(covered - (1 - alpha))),
            "mean_interval_width": float(width),
        })

    calib_df = pd.DataFrame(calibration_records)
    ece = (calib_df["calibration_error"] * (1 / len(calib_df))).sum() * len(calib_df)
    # ECE = mean absolute calibration error across levels
    ece = float(calib_df["calibration_error"].mean())

    # ── CRPS on mm scale (approximation via MC samples) ───────────────────
    crps_val = _compute_crps(actual_mm, mc_samples)

    logger.info(
        f"[Calibration] ECE={ece:.4f} | "
        f"CRPS={crps_val:.4f} mm | "
        f"Mean 90% width={calib_df.loc[calib_df['nominal_coverage'].round(1)==0.9, 'mean_interval_width'].values[0]:.2f} mm"
    )

    # ── Save table ─────────────────────────────────────────────────────────
    if save:
        out_dir = abs_path("outputs/uncertainty")
        out_dir.mkdir(parents=True, exist_ok=True)
        calib_df.to_csv(out_dir / "calibration_metrics.csv", index=False)
        # Summary metrics
        summary = pd.DataFrame([{
            "ECE":            ece,
            "CRPS_mm":        crps_val,
            "Mean_90pct_Width": calib_df.loc[
                calib_df["nominal_coverage"].round(2) == 0.90,
                "mean_interval_width"
            ].values[0],
        }])
        summary.to_csv(out_dir / "calibration_summary.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_reliability_diagram(calib_df, ece, save)
    _plot_crps_by_season(actual_mm, mc_samples, pred_df.index, save)

    return calib_df


# ---------------------------------------------------------------------------
# CRPS computation
# ---------------------------------------------------------------------------

def _compute_crps(
    actual_mm:   np.ndarray,
    mc_samples:  np.ndarray,
    subsample:   int = 500,
) -> float:
    """
    Compute the sample-based CRPS approximation.

    CRPS(F, y) = E[|X - y|] - 0.5 × E[|X - X'|]

    where X, X' are iid draws from the forecast distribution F.
    We approximate using T MC samples (already on mm scale after expm1).

    Only runs on a subsample of N for computational efficiency.
    """
    N, T = mc_samples.shape[1], mc_samples.shape[0]
    rng  = np.random.default_rng(CFG.project.random_seed)
    idx  = rng.choice(N, size=min(subsample, N), replace=False)

    # Convert samples to mm scale
    samples_mm = np.clip(np.expm1(mc_samples[:, idx]), 0.0, None)   # (T, subsample)
    actual_sub = actual_mm[idx]                                       # (subsample,)

    # E[|X - y|]: mean absolute deviation from actual
    term1 = np.abs(samples_mm - actual_sub[np.newaxis, :]).mean(axis=0)  # (subsample,)

    # E[|X - X'|]: mean pairwise spread (use subset of T pairs for speed)
    n_pairs = min(T, 50)
    idx1 = rng.choice(T, size=n_pairs, replace=False)
    idx2 = rng.choice(T, size=n_pairs, replace=False)
    term2 = np.abs(
        samples_mm[idx1, :] - samples_mm[idx2, :]
    ).mean(axis=0)   # (subsample,)

    crps_per_obs = term1 - 0.5 * term2
    return float(crps_per_obs.mean())


# ---------------------------------------------------------------------------
# Figure U5: Reliability diagram
# ---------------------------------------------------------------------------

def _plot_reliability_diagram(
    calib_df: pd.DataFrame,
    ece: float,
    save: bool,
) -> None:
    """
    Reliability diagram: nominal coverage (x) vs empirical coverage (y).
    Perfect calibration = identity line.
    Above line = under-confident (intervals too wide).
    Below line = over-confident (intervals too narrow).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    nominal  = calib_df["nominal_coverage"].values
    empirical = calib_df["empirical_coverage"].values
    calib_err = calib_df["calibration_error"].values

    # Panel (a): reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.0, linestyle="--", label="Perfect calibration")
    ax.plot(nominal, empirical, color=BLUE, lw=2.0, marker="o", ms=7,
            label="GRU MC Dropout")
    ax.fill_between(nominal, nominal, empirical,
                    where=(empirical > nominal), alpha=0.12, color=ORANGE,
                    label="Under-confident (too wide)")
    ax.fill_between(nominal, nominal, empirical,
                    where=(empirical < nominal), alpha=0.12, color=RED,
                    label="Over-confident (too narrow)")
    ax.set_xlabel("Nominal Coverage Probability")
    ax.set_ylabel("Empirical Coverage Probability")
    ax.set_title(
        f"(a) Reliability Diagram\nECE = {ece:.4f} "
        f"({'well-calibrated' if ece < 0.05 else 'moderate bias' if ece < 0.10 else 'poorly calibrated'})",
        fontweight="bold"
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # Panel (b): calibration error per level
    ax = axes[1]
    colors_bar = [RED if c > 0.05 else ORANGE if c > 0.02 else GREEN
                  for c in calib_err]
    bars = ax.bar(
        [f"{v:.0%}" for v in nominal],
        calib_err * 100,
        color=colors_bar, alpha=0.80, edgecolor="none"
    )
    ax.axhline(5, color=ORANGE, lw=1.0, linestyle="--", label="5% threshold")
    ax.axhline(2, color=GREEN,  lw=1.0, linestyle="--", label="2% threshold")
    ax.set_xlabel("Nominal Coverage Level")
    ax.set_ylabel("Calibration Error (%)")
    ax.set_title("(b) Absolute Calibration Error per Level", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)

    for bar, v in zip(bars, calib_err * 100):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                f"{v:.1f}%", ha="center", fontsize=8)

    add_figure_title(
        fig,
        "Calibration Analysis — GRU MC Dropout (T=100)",
        "Perfect calibration: all points on the diagonal | ECE = mean absolute deviation from diagonal"
    )
    if save:
        save_figure(fig, "calibration_reliability_diagram", subdir="uncertainty")
    plt.close(fig)
    logger.info("[Calibration] Reliability diagram saved")


# ---------------------------------------------------------------------------
# Figure U6: CRPS by season and rainfall bin
# ---------------------------------------------------------------------------

def _plot_crps_by_season(
    actual_mm:   np.ndarray,
    mc_samples:  np.ndarray,
    index:       pd.DatetimeIndex,
    save: bool,
) -> None:
    """CRPS broken down by month and rainfall intensity bin."""
    samples_mm = np.clip(np.expm1(mc_samples), 0.0, None)  # (T, N)
    T, N = samples_mm.shape

    # Per-observation CRPS (full computation)
    rng = np.random.default_rng(CFG.project.random_seed)
    n_pairs = min(T, 50)
    idx1 = rng.choice(T, size=n_pairs, replace=False)
    idx2 = rng.choice(T, size=n_pairs, replace=False)

    term1 = np.abs(samples_mm - actual_mm[np.newaxis, :]).mean(axis=0)
    term2 = np.abs(samples_mm[idx1, :] - samples_mm[idx2, :]).mean(axis=0)
    crps_per_obs = term1 - 0.5 * term2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Monthly mean CRPS
    months = range(1, 13)
    month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    monthly_crps = []
    for m in months:
        mask = (index.month == m)
        if mask.sum() == 0:
            monthly_crps.append(np.nan)
        else:
            monthly_crps.append(float(crps_per_obs[mask].mean()))

    bar_colors = [GREEN if m in MONSOON_MONTHS else BLUE for m in months]
    axes[0].bar(month_labels, monthly_crps, color=bar_colors, alpha=0.80, edgecolor="none")
    axes[0].axvspan(5, 9, alpha=0.06, color=GREEN)
    axes[0].set_ylabel("Mean CRPS (mm)")
    axes[0].set_title("(a) Monthly Mean CRPS\n(lower = better probabilistic forecast)", fontweight="bold")
    for i, v in enumerate(monthly_crps):
        if not np.isnan(v):
            axes[0].text(i, v*1.03, f"{v:.1f}", ha="center", fontsize=7)

    # Panel (b): CRPS by rainfall intensity bin
    bins_def = [(0, 0.1, "Dry"), (0.1, 5, "Light"), (5, 20, "Moderate"), (20, 999, "Heavy+")]
    bin_crps  = []
    bin_labels = []
    bin_n     = []
    for lo, hi, label in bins_def:
        mask = (actual_mm >= lo) & (actual_mm < hi)
        n = mask.sum()
        bin_n.append(n)
        bin_labels.append(f"{label}\n(N={n})")
        bin_crps.append(float(crps_per_obs[mask].mean()) if n > 0 else np.nan)

    colors_b = [GRAY, BLUE, GREEN, RED]
    bars = axes[1].bar(bin_labels, bin_crps, color=colors_b, alpha=0.80, edgecolor="none")
    axes[1].set_ylabel("Mean CRPS (mm)")
    axes[1].set_title("(b) CRPS by Rainfall Intensity\n(heavy rain = hardest to forecast)", fontweight="bold")
    for bar, v in zip(bars, bin_crps):
        if not np.isnan(v):
            axes[1].text(bar.get_x()+bar.get_width()/2, v*1.03,
                         f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    add_figure_title(
        fig,
        "CRPS Breakdown by Season and Rainfall Intensity",
        "CRPS rewards both calibration and sharpness | lower = better"
    )
    if save:
        save_figure(fig, "calibration_crps", subdir="uncertainty")
    plt.close(fig)
    logger.info("[Calibration] CRPS figure saved")
