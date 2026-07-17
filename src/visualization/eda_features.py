"""
eda_features.py
===============
Visualisation of feature engineering outputs: importance rankings,
distribution comparisons, SHAP beeswarm, and feature correlation
structure after engineering.

Figures produced
----------------
19_feature_importance_shap_xgb.png  — SHAP vs XGB importance comparison
20_feature_distributions.png        — Distribution grid for engineered features
21_feature_correlation_after.png    — Correlation heatmap post-engineering
22_dry_spell_counter.png            — Dry spell counter validation plot
23_cyclical_encoding.png            — Cyclical feature visualisation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG
from src.visualization.plot_utils import (
    apply_style, save_figure, add_figure_title,
    BLUE, RED, GREEN, ORANGE, GRAY, CATEGORICAL_PALETTE,
    annotate_monsoon_bands,
)
from src.features.feature_selector import FeatureSelectionResult

logger = logging.getLogger(__name__)


def plot_all_feature_figures(
    df: pd.DataFrame,
    selection_result: FeatureSelectionResult,
    save: bool = True,
) -> None:
    """Generate all feature engineering visualisation figures."""
    apply_style()
    _plot_importance_comparison(selection_result, save)
    _plot_engineered_distributions(df, save)
    _plot_feature_correlation_post(df, selection_result, save)
    _plot_dry_spell_validation(df, save)
    _plot_cyclical_encoding(df, save)


# ---------------------------------------------------------------------------
# Figure 19: SHAP vs XGB importance side-by-side
# ---------------------------------------------------------------------------

def _plot_importance_comparison(
    sr: FeatureSelectionResult,
    save: bool,
) -> None:
    """Horizontal bar charts: SHAP mean|value| and XGB gain, top 25 features."""
    top_n = 25
    shap_top = sr.shap_importance.head(top_n).sort_values("SHAP_MeanAbs")
    xgb_top  = sr.xgb_importance.head(top_n).sort_values("XGB_Gain")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # SHAP
    colors_shap = [GREEN if f in sr.features_ml else GRAY for f in shap_top["Feature"]]
    axes[0].barh(shap_top["Feature"], shap_top["SHAP_MeanAbs"],
                 color=colors_shap, alpha=0.82, edgecolor="none")
    axes[0].set_xlabel("Mean |SHAP Value|")
    axes[0].set_title(f"(a) SHAP Feature Importance — Top {top_n}")
    axes[0].axvline(0, color=GRAY, lw=0.5)
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=GREEN, alpha=0.8, label="In ML feature set"),
        Patch(facecolor=GRAY,  alpha=0.8, label="Not selected"),
    ]
    axes[0].legend(handles=legend_els, fontsize=8, loc="lower right")

    # XGBoost gain
    colors_xgb = [BLUE if f in sr.features_ml else GRAY for f in xgb_top["Feature"]]
    axes[1].barh(xgb_top["Feature"], xgb_top["XGB_Gain_Frac"] * 100,
                 color=colors_xgb, alpha=0.82, edgecolor="none")
    axes[1].set_xlabel("Gain Importance (% of total)")
    axes[1].set_title(f"(b) XGBoost Gain Importance — Top {top_n}")
    legend_els2 = [
        Patch(facecolor=BLUE, alpha=0.8, label="In ML feature set"),
        Patch(facecolor=GRAY, alpha=0.8, label="Not selected"),
    ]
    axes[1].legend(handles=legend_els2, fontsize=8, loc="lower right")

    add_figure_title(
        fig,
        "Feature Importance: SHAP vs XGBoost Gain",
        "Green/Blue = selected for ML models | Gray = below selection threshold",
    )
    if save:
        save_figure(fig, "19_feature_importance_shap_xgb", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 20: Distribution grid for key engineered features
# ---------------------------------------------------------------------------

def _plot_engineered_distributions(df: pd.DataFrame, save: bool) -> None:
    """
    4×3 grid of distribution plots for the most important engineered features.
    Each panel shows a histogram with monsoon vs non-monsoon overlay.
    """
    features_to_show = [
        "RAINFALL_lag1",        "RAINFALL_roll_mean_7",  "RAINFALL_roll_std_30",
        "SOIL_MOISTURE_GRADIENT","DEWPOINT_APPROX",       "TEMP_RANGE",
        "PRESSURE_RH_INTERACTION","DRY_SPELL_LENGTH",     "MONTH_SIN",
        "RH_lag1",              "CLOUD_lag1",             "PRESSURE_lag1",
    ]
    features_to_show = [f for f in features_to_show if f in df.columns]

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    axes = axes.flatten()

    monsoon_mask = df["IS_MONSOON"] == 1

    for i, feat in enumerate(features_to_show[:12]):
        ax = axes[i]
        series = df[feat].dropna()
        mon_data    = df.loc[monsoon_mask,  feat].dropna()
        nonmon_data = df.loc[~monsoon_mask, feat].dropna()

        bins = min(50, max(20, len(series) // 100))
        global_min = series.quantile(0.01)
        global_max = series.quantile(0.99)
        bin_edges = np.linspace(global_min, global_max, bins + 1)

        ax.hist(nonmon_data.clip(global_min, global_max), bins=bin_edges,
                color=BLUE, alpha=0.55, density=True, label="Non-Monsoon")
        ax.hist(mon_data.clip(global_min, global_max),    bins=bin_edges,
                color=GREEN, alpha=0.55, density=True, label="Monsoon")

        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density" if i % 4 == 0 else "")
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused panels
    for j in range(len(features_to_show), 12):
        axes[j].set_visible(False)

    add_figure_title(
        fig,
        "Engineered Feature Distributions",
        "Blue = Non-Monsoon | Green = Monsoon season (Jun–Sep)",
    )
    if save:
        save_figure(fig, "20_feature_distributions", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 21: Correlation heatmap after feature engineering
# ---------------------------------------------------------------------------

def _plot_feature_correlation_post(
    df: pd.DataFrame,
    sr: FeatureSelectionResult,
    save: bool,
) -> None:
    """
    Correlation heatmap of the ML feature set after engineering and selection.
    Demonstrates that the selected set has substantially reduced collinearity
    compared to the raw feature set shown in Figure 11.
    """
    ml_features = [f for f in sr.features_ml if f in df.columns]
    # Limit to 20 for visual legibility
    ml_features = ml_features[:20]
    if len(ml_features) < 3:
        logger.warning("Too few ML features for post-engineering heatmap; skipping")
        return

    corr = df[ml_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="#EEEEEE",
        annot_kws={"size": 7},
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title(
        "Post-Engineering Correlation Matrix — ML Feature Set (Top 20)\n"
        "Compare with Figure 11 (raw features) to confirm collinearity reduction",
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    if save:
        save_figure(fig, "21_feature_correlation_post", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 22: Dry spell counter validation
# ---------------------------------------------------------------------------

def _plot_dry_spell_validation(df: pd.DataFrame, save: bool) -> None:
    """
    Three-panel validation of the causal dry spell counter:
      (a) Sample 2-year window: rainfall bars + counter overlay
      (b) Counter value distribution
      (c) Counter vs P(rain the next day) — calibration check
    """
    if "DRY_SPELL_LENGTH" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- (a) Sample window: 2018 monsoon season ---
    sample = df.loc["2016-01-01":"2017-12-31"].copy()
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(sample.index, sample["RAINFALL"], color=BLUE, alpha=0.40,
           width=1, label="Rainfall (mm)")
    ax2.plot(sample.index, sample["DRY_SPELL_LENGTH"], color=RED, lw=1.3,
             label="Dry spell length")
    ax.set_ylabel("Rainfall (mm)", color=BLUE)
    ax2.set_ylabel("Consecutive dry days", color=RED)
    ax.set_title("(a) Dry Spell Counter vs Rainfall (2016–2017)")
    ax.tick_params(axis="x", rotation=30)
    annotate_monsoon_bands(ax, sample)

    # --- (b) Counter distribution ---
    axes[1].hist(df["DRY_SPELL_LENGTH"], bins=50, color=ORANGE,
                 alpha=0.80, edgecolor="none")
    axes[1].set_xlabel("Dry Spell Counter Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("(b) Distribution of Dry Spell Counter")
    axes[1].axvline(
        df["DRY_SPELL_LENGTH"].mean(), color=RED, lw=1.3, linestyle="--",
        label=f"Mean = {df['DRY_SPELL_LENGTH'].mean():.1f}"
    )
    axes[1].legend(fontsize=9)

    # --- (c) Counter vs P(rain) calibration ---
    # Bin the counter and compute P(RAIN_OCCURRENCE=1) within each bin
    if "RAIN_OCCURRENCE" in df.columns:
        df_cal = df[["DRY_SPELL_LENGTH", "RAIN_OCCURRENCE"]].dropna()
        bins = list(range(0, 31, 2)) + [df_cal["DRY_SPELL_LENGTH"].max() + 1]
        df_cal["SPELL_BIN"] = pd.cut(df_cal["DRY_SPELL_LENGTH"], bins=bins, right=False)
        calibration = df_cal.groupby("SPELL_BIN", observed=True).agg(
            P_rain=("RAIN_OCCURRENCE", "mean"),
            N=("RAIN_OCCURRENCE", "count"),
        ).reset_index()
        calibration["bin_mid"] = calibration["SPELL_BIN"].apply(
            lambda x: (x.left + x.right) / 2
        )

        axes[2].plot(calibration["bin_mid"], calibration["P_rain"] * 100,
                     color=GREEN, lw=1.8, marker="o", ms=5, label="P(rain | counter)")
        axes[2].fill_between(
            calibration["bin_mid"],
            (calibration["P_rain"] - 1.96 * np.sqrt(
                calibration["P_rain"] * (1 - calibration["P_rain"]) / calibration["N"].clip(1)
            )) * 100,
            (calibration["P_rain"] + 1.96 * np.sqrt(
                calibration["P_rain"] * (1 - calibration["P_rain"]) / calibration["N"].clip(1)
            )) * 100,
            alpha=0.20, color=GREEN
        )
        axes[2].set_xlabel("Dry Spell Counter Value")
        axes[2].set_ylabel("P(Rain next day) (%)")
        axes[2].set_title("(c) Counter Calibration: P(Rain) vs Dry Spell Length")
        axes[2].set_ylim(0, 100)
        axes[2].legend(fontsize=9)

    add_figure_title(
        fig,
        "Dry Spell Counter — Validation",
        "Counter is strictly causal: day-t value = dry days before day t",
    )
    if save:
        save_figure(fig, "22_dry_spell_counter_validation", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 23: Cyclical encoding visualisation
# ---------------------------------------------------------------------------

def _plot_cyclical_encoding(df: pd.DataFrame, save: bool) -> None:
    """
    Demonstrate the cyclical encoding of MONTH and DOY:
      (a) Unit circle plot of MONTH_SIN vs MONTH_COS
      (b) Monthly mean rainfall overlaid with sin/cos amplitude
      (c) DOY_SIN and DOY_COS time series over one year
    """
    if "MONTH_SIN" not in df.columns or "DOY_SIN" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- (a) Unit circle: month encoding ---
    ax = axes[0]
    months = np.arange(1, 13)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    # Build 12-color list by cycling the palette
    colors_12 = (CATEGORICAL_PALETTE * 3)[:12]
    ax.scatter(month_cos, month_sin, s=80, c=colors_12,
               zorder=5, edgecolors="white", linewidths=0.5)
    for m, ms, mc in zip(months, month_sin, month_cos):
        label = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
        monsoon = m in list(CFG.monsoon_months)
        ax.annotate(label, (mc, ms), xytext=(mc * 1.18, ms * 1.18),
                    ha="center", va="center", fontsize=8,
                    fontweight="bold" if monsoon else "normal",
                    color=GREEN if monsoon else "#333333")
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=0.8, alpha=0.5)
    ax.axhline(0, color=GRAY, lw=0.4)
    ax.axvline(0, color=GRAY, lw=0.4)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_xlabel("MONTH_COS")
    ax.set_ylabel("MONTH_SIN")
    ax.set_title("(a) Cyclical Month Encoding\n(green = monsoon months)")

    # --- (b) Monthly mean rainfall vs cyclical amplitude ---
    monthly_rain = df["RAINFALL"].groupby(df.index.month).mean()
    ax2 = axes[1].twinx()
    axes[1].bar(months, monthly_rain.values, color=BLUE, alpha=0.55,
                edgecolor="none", label="Mean rainfall (mm)")
    ax2.plot(months, month_sin, color=RED, lw=1.5, marker="o", ms=4,
             linestyle="--", label="MONTH_SIN")
    ax2.plot(months, month_cos, color=ORANGE, lw=1.5, marker="s", ms=4,
             linestyle="--", label="MONTH_COS")
    axes[1].set_xticks(months)
    axes[1].set_xticklabels(
        ["J","F","M","A","M","J","J","A","S","O","N","D"]
    )
    axes[1].set_ylabel("Mean Daily Rainfall (mm)", color=BLUE)
    ax2.set_ylabel("Encoding Value", color=RED)
    axes[1].set_title("(b) Monthly Rainfall vs Cyclical Encoding")
    axes[1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # --- (c) DOY sin/cos over one full year ---
    one_year = df.loc["2010-01-01":"2010-12-31"]
    axes[2].plot(one_year.index, one_year["DOY_SIN"], color=RED, lw=1.5,
                 label="DOY_SIN", alpha=0.85)
    axes[2].plot(one_year.index, one_year["DOY_COS"], color=BLUE, lw=1.5,
                 label="DOY_COS", alpha=0.85)
    axes[2].axhline(0, color=GRAY, lw=0.5)
    annotate_monsoon_bands(axes[2], one_year)
    axes[2].set_ylabel("Encoding Value")
    axes[2].set_xlabel("Date")
    axes[2].set_title("(c) DOY Cyclical Encoding — Year 2010")
    axes[2].legend(fontsize=9)
    axes[2].tick_params(axis="x", rotation=30)

    add_figure_title(
        fig,
        "Cyclical Temporal Feature Encoding",
        "sin/cos pairs encode periodicity without ordinal discontinuity at year boundary",
    )
    if save:
        save_figure(fig, "23_cyclical_encoding", subdir="eda")
    plt.close(fig)
