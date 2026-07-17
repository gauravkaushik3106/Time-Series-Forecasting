"""
prediction_plots.py
===================
Visualisation of model predictions, residuals, and performance breakdowns.

Figures produced
----------------
For each model (Persistence, Climatology, SARIMAX, XGBoost):
  P1. Actual vs Predicted time-series (test period)
  P2. Scatter plot: actual vs predicted (with 1:1 line)
  P3. Residual time-series and histogram

Cross-model figures:
  P4. Model comparison bar chart (RMSE, MAE, NSE)
  P5. Seasonal performance comparison (monsoon vs non-monsoon)
  P6. Extreme event performance (≥50 mm days)
  P7. Taylor diagram (summary of all models)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

MODEL_COLORS = {
    "Persistence":  GRAY,
    "Climatology":  ORANGE,
    "SARIMAX":      BLUE,
    "XGBoost":      GREEN,
}

EXTREME_THRESHOLD = 50.0  # mm


# ---------------------------------------------------------------------------
# Per-model figures
# ---------------------------------------------------------------------------

def plot_model_predictions(
    pred_df: pd.DataFrame,
    model_name: str,
    save: bool = True,
) -> None:
    """
    Three-panel figure for one model:
      (a) Time-series of actual vs predicted (test period)
      (b) Scatter actual vs predicted
      (c) Residual time-series + histogram
    """
    apply_style()
    color = MODEL_COLORS.get(model_name, BLUE)

    actual    = pred_df["actual"].values
    predicted = pred_df["predicted"].values
    residuals = predicted - actual
    index     = pred_df.index

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, :])   # full width top
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # --- (a) Time-series ---
    ax1.fill_between(index, actual, alpha=0.25, color=BLUE, label="Actual")
    ax1.plot(index, actual,    lw=0.7, color=BLUE, alpha=0.7)
    ax1.plot(index, predicted, lw=1.1, color=color, label=f"{model_name} predicted")
    annotate_monsoon_bands(ax1, pred_df, alpha=0.07)
    ax1.set_ylabel("Rainfall (mm/day)")
    ax1.set_title(f"(a) {model_name} — Actual vs Predicted (Test Period)")
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    # --- (b) Scatter ---
    ax2.scatter(actual, predicted, s=4, alpha=0.25, color=color)
    lim = max(actual.max(), predicted.max()) * 1.05
    ax2.plot([0, lim], [0, lim], color=GRAY, lw=1.0, linestyle="--", label="1:1 line")
    ax2.set_xlabel("Actual Rainfall (mm/day)")
    ax2.set_ylabel("Predicted Rainfall (mm/day)")
    ax2.set_title(f"(b) Scatter — {model_name}")
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    # Correlation annotation
    r = np.corrcoef(actual, predicted)[0, 1]
    ax2.text(0.05, 0.92, f"r = {r:.3f}", transform=ax2.transAxes,
             fontsize=10, color=color, fontweight="bold")

    # --- (c) Residuals ---
    ax3.plot(index, residuals, lw=0.5, color=color, alpha=0.6)
    ax3.axhline(0, color=GRAY, lw=0.8)
    ax3.fill_between(index, residuals, alpha=0.18, color=color)
    # Inset histogram
    ax3_in = ax3.inset_axes([0.72, 0.55, 0.26, 0.40])
    ax3_in.hist(residuals, bins=40, color=color, alpha=0.8, edgecolor="none",
                density=True)
    ax3_in.axvline(0, color=GRAY, lw=0.7)
    ax3_in.set_xlabel("Residual", fontsize=7)
    ax3_in.tick_params(labelsize=6)
    ax3.set_ylabel("Residual (predicted − actual)")
    ax3.set_title(f"(c) Residual Series — {model_name}")
    ax3.text(0.03, 0.92,
             f"Bias={residuals.mean():+.2f} mm | std={residuals.std():.2f} mm",
             transform=ax3.transAxes, fontsize=9)

    add_figure_title(fig, f"{model_name} — Forecast Evaluation (Test Set)")
    if save:
        fname = f"model_{model_name.lower().replace(' ', '_')}_test_predictions"
        save_figure(fig, fname, subdir="models")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model comparison figures
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Side-by-side bar charts of RMSE, MAE, NSE across all models.
    Models ranked by RMSE (lowest = best = leftmost).
    """
    apply_style()

    models = comparison_df["Model"].tolist()
    colors = [MODEL_COLORS.get(m, BLUE) for m in models]

    metrics_to_plot = [
        ("RMSE",     "RMSE (mm/day)",           "lower is better"),
        ("MAE",      "MAE (mm/day)",             "lower is better"),
        ("NSE",      "Nash-Sutcliffe Efficiency","higher is better"),
        ("R2",       "R²",                       "higher is better"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for ax, (metric, ylabel, note) in zip(axes, metrics_to_plot):
        if metric not in comparison_df.columns:
            ax.set_visible(False)
            continue
        vals = comparison_df[metric].values
        bars = ax.bar(models, vals, color=colors, alpha=0.80, edgecolor="none")
        ax.set_title(f"{ylabel}\n({note})", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel(ylabel)
        # Annotate bar values
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")
        # Highlight best model
        if "lower" in note:
            best_idx = int(np.nanargmin(vals))
        else:
            best_idx = int(np.nanargmax(vals))
        bars[best_idx].set_edgecolor(RED)
        bars[best_idx].set_linewidth(2)

    add_figure_title(fig, "Model Comparison — Test Set Performance")
    if save:
        save_figure(fig, "model_comparison_metrics", subdir="models")
    plt.close(fig)


def plot_seasonal_comparison(
    comparison_df: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Grouped bar chart: monsoon vs non-monsoon RMSE for each model.
    """
    apply_style()

    if "RMSE_Monsoon" not in comparison_df.columns:
        logger.warning("Seasonal metrics not found; skipping seasonal comparison plot")
        return

    models  = comparison_df["Model"].tolist()
    x       = np.arange(len(models))
    width   = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (metric_m, metric_nm, title, ylabel) in zip(axes, [
        ("RMSE_Monsoon", "RMSE_NonMonsoon", "RMSE by Season", "RMSE (mm/day)"),
        ("NSE_Monsoon",  "NSE_NonMonsoon",  "NSE by Season",  "NSE"),
    ]):
        if metric_m not in comparison_df.columns:
            continue
        vals_m  = comparison_df[metric_m].values.astype(float)
        vals_nm = comparison_df[metric_nm].values.astype(float)

        bars1 = ax.bar(x - width/2, vals_m,  width, label="Monsoon (JJAS)",
                       color=GREEN, alpha=0.80, edgecolor="none")
        bars2 = ax.bar(x + width/2, vals_nm, width, label="Non-Monsoon",
                       color=ORANGE, alpha=0.80, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)

        for bar, v in zip(list(bars1) + list(bars2),
                          list(vals_m) + list(vals_nm)):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01 * abs(bar.get_height() or 1),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    add_figure_title(fig, "Seasonal Performance Comparison — Test Set")
    if save:
        save_figure(fig, "model_seasonal_comparison", subdir="models")
    plt.close(fig)


def plot_extreme_event_comparison(
    pred_dfs: Dict[str, pd.DataFrame],
    save: bool = True,
) -> None:
    """
    Scatter plots of actual vs predicted for extreme rainfall events (≥50mm)
    for each model, on a shared axes grid.
    """
    apply_style()
    models = list(pred_dfs.keys())
    n      = len(models)
    ncols  = min(n, 2)
    nrows  = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 6 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for i, (model_name, pred_df) in enumerate(pred_dfs.items()):
        ax    = axes[i]
        color = MODEL_COLORS.get(model_name, BLUE)

        extreme_mask = pred_df["actual"] >= EXTREME_THRESHOLD
        n_extreme    = extreme_mask.sum()

        if n_extreme == 0:
            ax.text(0.5, 0.5, "No extreme events\nin test set",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model_name} — Extreme Events (≥{EXTREME_THRESHOLD:.0f} mm)")
            continue

        act_e  = pred_df.loc[extreme_mask, "actual"].values
        pred_e = pred_df.loc[extreme_mask, "predicted"].values

        ax.scatter(act_e, pred_e, s=40, alpha=0.7, color=color,
                   edgecolors="white", linewidths=0.5)
        lim = max(act_e.max(), pred_e.max()) * 1.1
        ax.plot([0, lim], [0, lim], color=GRAY, lw=1.2, linestyle="--",
                label="1:1 line")

        mae_e  = float(np.mean(np.abs(act_e - pred_e)))
        bias_e = float(np.mean(pred_e - act_e))
        ax.text(0.05, 0.90,
                f"N={n_extreme} | MAE={mae_e:.1f} mm | Bias={bias_e:+.1f} mm",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.85))

        ax.set_xlabel("Actual Rainfall (mm/day)")
        ax.set_ylabel("Predicted Rainfall (mm/day)")
        ax.set_title(f"{model_name} — Extreme Events (≥{EXTREME_THRESHOLD:.0f} mm)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=8)

    # Hide unused panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    add_figure_title(
        fig,
        f"Extreme Rainfall Event Prediction (≥{EXTREME_THRESHOLD:.0f} mm)",
        "Each point = one heavy-rain day in the test set"
    )
    if save:
        save_figure(fig, "model_extreme_events", subdir="models")
    plt.close(fig)


def plot_taylor_diagram(
    pred_dfs: Dict[str, pd.DataFrame],
    save: bool = True,
) -> None:
    """
    Taylor diagram: summarises correlation and normalised std dev for all models.
    Normalised by observed std — reference point is the observation at (1.0, 1.0).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": "polar"})

    actual_ref = list(pred_dfs.values())[0]["actual"].values
    obs_std    = actual_ref.std()

    # Taylor diagram uses angle = arccos(r), radius = normalised std dev
    theta_max = np.pi / 2
    ax.set_thetamax(np.degrees(theta_max))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Reference point: observations (r=1, norm_std=1)
    ax.plot(0, 1, marker="*", ms=14, color=RED, zorder=10,
            label="Observations", linestyle="none")

    # RMS contours
    for rms in [0.5, 1.0, 1.5, 2.0]:
        theta_arc = np.linspace(0, theta_max, 200)
        r_arc = np.sqrt(rms ** 2 + 1 - 2 * rms * np.cos(theta_arc + np.pi))
        # Skip — simplified diagram using correlation angle only
        pass

    # Plot each model
    for (model_name, pred_df), marker in zip(
        pred_dfs.items(), ["o", "s", "^", "D"]
    ):
        act  = pred_df["actual"].values
        pred = pred_df["predicted"].values
        r    = np.corrcoef(act, pred)[0, 1]
        std_norm = pred.std() / (obs_std or 1.0)
        theta = np.arccos(np.clip(r, -1, 1))
        color = MODEL_COLORS.get(model_name, BLUE)
        ax.plot(theta, std_norm, marker=marker, ms=10, color=color,
                label=f"{model_name} (r={r:.3f})", linestyle="none")

    ax.set_ylabel("Normalised Std Dev", labelpad=30)
    ax.set_title("Taylor Diagram — Test Set\n(angle=1−r, radius=σ_model/σ_obs)",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    if save:
        save_figure(fig, "model_taylor_diagram", subdir="models")
    plt.close(fig)


def plot_all_predictions_overlay(
    pred_dfs: Dict[str, pd.DataFrame],
    save: bool = True,
) -> None:
    """
    Overlay all model predictions on a single time-series panel (test set).
    Shows a focused 6-month monsoon window and the full test period.
    """
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=False)

    # Full test period
    ax = axes[0]
    first_pred = list(pred_dfs.values())[0]
    ax.fill_between(first_pred.index, first_pred["actual"],
                    alpha=0.20, color=BLUE, label="Actual")
    ax.plot(first_pred.index, first_pred["actual"],
            lw=0.6, color=BLUE, alpha=0.7)
    for model_name, pred_df in pred_dfs.items():
        ax.plot(pred_df.index, pred_df["predicted"],
                lw=1.0, color=MODEL_COLORS.get(model_name, GRAY),
                alpha=0.80, label=model_name)
    annotate_monsoon_bands(ax, first_pred, alpha=0.06)
    ax.set_ylabel("Rainfall (mm/day)")
    ax.set_title("(a) All Models — Full Test Period")
    ax.legend(fontsize=9, ncol=3)
    ax.set_ylim(bottom=0)

    # Zoom: first monsoon season in test period
    ax2 = axes[1]
    zoom_start = first_pred.index[0]
    # Find first June in test set
    for ts in first_pred.index:
        if ts.month == 6:
            zoom_start = ts
            break
    zoom_end = zoom_start + pd.DateOffset(months=6)
    zoom_mask = (first_pred.index >= zoom_start) & (first_pred.index <= zoom_end)

    ax2.fill_between(
        first_pred.index[zoom_mask],
        first_pred["actual"].values[zoom_mask],
        alpha=0.22, color=BLUE, label="Actual"
    )
    ax2.plot(
        first_pred.index[zoom_mask],
        first_pred["actual"].values[zoom_mask],
        lw=0.8, color=BLUE, alpha=0.8
    )
    for model_name, pred_df in pred_dfs.items():
        pred_zoom = pred_df["predicted"].values[zoom_mask]
        ax2.plot(
            pred_df.index[zoom_mask], pred_zoom,
            lw=1.2, color=MODEL_COLORS.get(model_name, GRAY),
            alpha=0.85, label=model_name
        )
    ax2.set_ylabel("Rainfall (mm/day)")
    ax2.set_title(
        f"(b) Monsoon Season Zoom — "
        f"{zoom_start.strftime('%b %Y')} to {zoom_end.strftime('%b %Y')}"
    )
    ax2.legend(fontsize=9, ncol=3)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="x", rotation=20)

    add_figure_title(fig, "All Models — Prediction Overlay (Test Set)")
    if save:
        save_figure(fig, "model_all_overlay", subdir="models")
    plt.close(fig)
