"""
shap_analysis.py
================
SHAP (SHapley Additive exPlanations) analysis for the XGBoost model.

SHAP is applied to the XGBoost regressor (wet-day amount model) because:
  1.  TreeExplainer is exact and computationally cheap for XGBoost.
  2.  The regressor captures the nonlinear feature contributions that
      drive the model's heavy-rain predictions — the most scientifically
      interesting component.
  3.  SHAP values satisfy the efficiency axiom: sum of SHAP values =
      (model prediction − expected value), making them interpretable as
      additive feature contributions.

Analyses generated
------------------
Global
  - Beeswarm plot (all features, all test-set predictions)
  - Bar plot (mean |SHAP| ranking)

Local (three representative day types)
  - Normal rainfall day (5–15 mm): typical monsoon behaviour
  - Extreme rainfall day (>50 mm): model reasoning under stress
  - Dry monsoon day (JJAS, < 0.1 mm): why the model predicts no rain

Feature-level
  - Dependence plots for: RH, CLOUD, SOIL_MOISTURE_GRADIENT, RAINFALL_lag1,
    RAINFALL_roll_mean_7, PRESSURE_RH_INTERACTION

All plots saved to outputs/figures/explainability/.
Summary table saved to outputs/explainability/shap_summary.csv.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.visualization.plot_utils import (
    apply_style, save_figure, add_figure_title,
    BLUE, RED, GREEN, ORANGE, GRAY,
)

logger = logging.getLogger(__name__)

DEPENDENCE_FEATURES = [
    "RH",
    "CLOUD",
    "SOIL_MOISTURE_GRADIENT",
    "RAINFALL_lag1",
    "RAINFALL_roll_mean_7",
    "PRESSURE_RH_INTERACTION",
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_shap_analysis(
    test_ml:  pd.DataFrame,
    train_ml: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the complete SHAP analysis suite.

    Parameters
    ----------
    test_ml  : Scaled ML feature set, test split.
    train_ml : Scaled ML feature set, training split (for background).
    save     : Persist figures and tables to disk.

    Returns
    -------
    shap_summary_df : DataFrame of mean |SHAP| per feature.
    """
    apply_style()

    # Load XGBoost model
    model_path = abs_path("outputs/models/xgboost_model.pkl")
    with open(model_path, "rb") as fh:
        xgb_data = pickle.load(fh)

    regressor   = xgb_data["regressor"]
    feature_cols = xgb_data["feature_cols"]

    # Filter test_ml to wet days (regressor only fires on wet days)
    wet_mask   = test_ml["RAIN_OCCURRENCE"] == 1
    X_test_wet = test_ml.loc[wet_mask, feature_cols]
    X_test_all = test_ml[feature_cols]

    # Background dataset: subsample of training wet days
    wet_train = train_ml[train_ml["RAIN_OCCURRENCE"] == 1]
    rng = np.random.default_rng(CFG.project.random_seed)
    bg_idx  = rng.choice(len(wet_train), size=min(500, len(wet_train)), replace=False)
    X_bg = wet_train.iloc[bg_idx][feature_cols]

    logger.info(
        f"[SHAP] Computing TreeExplainer on {len(X_test_wet):,} wet test days | "
        f"background: {len(X_bg)} training samples"
    )

    # TreeExplainer: exact SHAP for tree models, no approximation needed
    explainer   = shap.TreeExplainer(regressor, data=X_bg)
    shap_values = explainer.shap_values(X_test_wet)   # shape (n_wet, n_features)
    expected_val = float(explainer.expected_value)

    logger.info(
        f"[SHAP] SHAP values computed: shape={shap_values.shape} | "
        f"expected_value={expected_val:.4f} (log scale)"
    )

    # ── Global summary ────────────────────────────────────────────────
    shap_summary_df = _compute_shap_summary(shap_values, feature_cols, X_test_wet)
    if save:
        out_dir = abs_path("outputs/explainability")
        out_dir.mkdir(parents=True, exist_ok=True)
        shap_summary_df.to_csv(out_dir / "shap_summary.csv", index=False)
        logger.info("SHAP summary table saved")

    # ── Figures ───────────────────────────────────────────────────────
    _plot_beeswarm(shap_values, X_test_wet, feature_cols, save)
    _plot_bar_importance(shap_summary_df, save)
    _plot_dependence_grid(shap_values, X_test_wet, feature_cols, save)
    _plot_local_explanations(
        shap_values, X_test_wet, feature_cols,
        expected_val, test_ml, wet_mask, save
    )

    return shap_summary_df


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_shap_summary(
    shap_values: np.ndarray,
    feature_cols: List[str],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-feature mean |SHAP| and signed mean SHAP."""
    mean_abs  = np.abs(shap_values).mean(axis=0)
    mean_sign = shap_values.mean(axis=0)
    total     = mean_abs.sum() or 1.0
    df = pd.DataFrame({
        "Feature":        feature_cols,
        "SHAP_MeanAbs":   mean_abs,
        "SHAP_MeanSigned": mean_sign,
        "SHAP_Frac":      mean_abs / total,
    }).sort_values("SHAP_MeanAbs", ascending=False).reset_index(drop=True)
    df["SHAP_Rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Figure E1: Beeswarm plot
# ---------------------------------------------------------------------------

def _plot_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_cols: List[str],
    save: bool,
) -> None:
    """
    SHAP beeswarm: each dot is one wet-day prediction.
    Colour = feature value (red=high, blue=low).
    Vertical spread = density of predictions at that SHAP value.
    """
    top_n = 20
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    shap_top = shap_values[:, top_idx]
    X_top    = X.iloc[:, top_idx]
    feat_top = [feature_cols[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    # Manual beeswarm using scatter
    for j in range(top_n - 1, -1, -1):
        sv  = shap_top[:, j]
        fv  = X_top.iloc[:, j].values
        # Normalise feature value to [0,1] for colour map
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-10)
        # Vertical position = feature rank (j), jittered
        jitter  = np.random.default_rng(j).uniform(-0.3, 0.3, len(sv))
        y_pos   = np.full(len(sv), top_n - 1 - j) + jitter
        scatter = ax.scatter(
            sv, y_pos, c=fv_norm, cmap="coolwarm",
            s=4, alpha=0.35, vmin=0, vmax=1, linewidths=0,
        )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_top[::-1], fontsize=9)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("SHAP Value (log rainfall impact)", fontsize=11)
    ax.set_title(
        "SHAP Beeswarm — XGBoost Regressor (Wet Days Only)\n"
        "Each dot = one test-day prediction | Colour = feature value (red=high, blue=low)",
        fontweight="bold",
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Feature value (normalised)", fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    if save:
        save_figure(fig, "shap_beeswarm", subdir="explainability")
    plt.close(fig)
    logger.info("[SHAP] Beeswarm plot saved")


# ---------------------------------------------------------------------------
# Figure E2: Bar importance
# ---------------------------------------------------------------------------

def _plot_bar_importance(
    summary_df: pd.DataFrame,
    save: bool,
) -> None:
    """Horizontal bar chart of mean |SHAP| for top 20 features."""
    top = summary_df.head(20).sort_values("SHAP_MeanAbs")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = []
    for sign in top["SHAP_MeanSigned"]:
        colors.append(GREEN if sign > 0 else RED)

    bars = ax.barh(
        top["Feature"], top["SHAP_MeanAbs"],
        color=colors, alpha=0.82, edgecolor="none",
    )
    ax.set_xlabel("Mean |SHAP Value| (log rainfall contribution)")
    ax.set_title(
        "Global Feature Importance — SHAP (XGBoost Regressor)\n"
        "Green = positive average effect | Red = negative average effect",
        fontweight="bold",
    )
    for bar, v in zip(bars, top["SHAP_MeanAbs"]):
        ax.text(
            v + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=8,
        )

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=GREEN, alpha=0.8, label="Positive avg effect"),
        Patch(facecolor=RED,   alpha=0.8, label="Negative avg effect"),
    ], fontsize=8)

    if save:
        save_figure(fig, "shap_bar_importance", subdir="explainability")
    plt.close(fig)
    logger.info("[SHAP] Bar importance plot saved")


# ---------------------------------------------------------------------------
# Figure E3: Dependence plots grid
# ---------------------------------------------------------------------------

def _plot_dependence_grid(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_cols: List[str],
    save: bool,
) -> None:
    """
    2×3 grid of SHAP dependence plots for the six most hydrologically
    relevant features.  Each panel shows feature value on x-axis vs
    SHAP value on y-axis, coloured by the interaction feature with the
    highest interaction magnitude.
    """
    targets = [f for f in DEPENDENCE_FEATURES if f in feature_cols]
    n       = len(targets)
    ncols   = 3
    nrows   = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten() if nrows > 1 else axes

    feat_idx = {f: feature_cols.index(f) for f in targets if f in feature_cols}

    for i, feat in enumerate(targets):
        ax  = axes[i]
        idx = feat_idx[feat]
        fv  = X.iloc[:, idx].values
        sv  = shap_values[:, idx]

        # Colour by RAINFALL_lag1 if available, else by feature value
        colour_feat = "RAINFALL_lag1" if "RAINFALL_lag1" in feature_cols else feat
        cidx  = feature_cols.index(colour_feat)
        cv    = X.iloc[:, cidx].values
        cv_n  = (cv - cv.min()) / (cv.max() - cv.min() + 1e-10)

        sc = ax.scatter(fv, sv, c=cv_n, cmap="RdYlGn",
                        s=5, alpha=0.35, vmin=0, vmax=1)
        ax.axhline(0, color=GRAY, lw=0.7)

        # LOWESS trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            si = np.argsort(fv)
            sm = lowess(sv[si], fv[si], frac=0.2, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=RED, lw=1.8, label="LOWESS")
        except Exception:
            pass

        ax.set_xlabel(f"{feat} (scaled)", fontsize=9)
        ax.set_ylabel("SHAP Value", fontsize=9)
        ax.set_title(f"Dependence: {feat}", fontweight="bold", fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.04).set_label(
            f"{colour_feat} value", fontsize=7
        )

    for j in range(len(targets), len(axes)):
        axes[j].set_visible(False)

    add_figure_title(
        fig,
        "SHAP Dependence Plots — Key Meteorological Features",
        "x = feature value (scaled) | y = SHAP contribution to log-rainfall | Colour = lag-1 rainfall",
    )
    if save:
        save_figure(fig, "shap_dependence_grid", subdir="explainability")
    plt.close(fig)
    logger.info("[SHAP] Dependence grid saved")


# ---------------------------------------------------------------------------
# Figure E4: Local explanations (waterfall-style) for three day types
# ---------------------------------------------------------------------------

def _plot_local_explanations(
    shap_values: np.ndarray,
    X_wet:        pd.DataFrame,
    feature_cols: List[str],
    expected_val: float,
    test_ml:      pd.DataFrame,
    wet_mask:     pd.Series,
    save:         bool,
) -> None:
    """
    Waterfall plots for three representative day types:
      1. Normal monsoon day (5–15 mm actual)
      2. Extreme event day (>50 mm actual)
      3. Dry monsoon day (JJAS, < 0.1 mm)
    """
    actual_wet = test_ml.loc[wet_mask, "RAINFALL"].values
    monsoon_wet = test_ml.loc[wet_mask].index.month.isin([6,7,8,9])

    # Day type selection
    normal_candidates  = np.where((actual_wet >= 5) & (actual_wet <= 15))[0]
    extreme_candidates = np.where(actual_wet >= 50)[0]

    # Dry monsoon days require going back to the full test set
    monsoon_mask = test_ml.index.month.isin([6, 7, 8, 9])
    dry_monsoon  = test_ml.loc[monsoon_mask & (test_ml["RAINFALL"] < 0.1)]

    day_cases = []
    rng = np.random.default_rng(42)

    if len(normal_candidates) > 0:
        idx = rng.choice(normal_candidates)
        day_cases.append((idx, "Normal Monsoon Day",
                          f"Actual: {actual_wet[idx]:.1f} mm"))
    if len(extreme_candidates) > 0:
        idx = extreme_candidates[np.argmax(actual_wet[extreme_candidates])]
        day_cases.append((idx, "Extreme Rainfall Day",
                          f"Actual: {actual_wet[idx]:.1f} mm"))

    fig, axes = plt.subplots(1, len(day_cases), figsize=(8 * len(day_cases), 9))
    if len(day_cases) == 1:
        axes = [axes]

    for ax, (idx, label, subtitle) in zip(axes, day_cases):
        sv    = shap_values[idx]
        fv    = X_wet.iloc[idx]
        pred  = expected_val + sv.sum()

        # Top N features by absolute SHAP
        n_show = 12
        top_idx  = np.argsort(np.abs(sv))[::-1][:n_show]
        sv_show  = sv[top_idx]
        ft_show  = [f"{feature_cols[i]}\n= {fv.iloc[i]:.3f}" for i in top_idx]

        colors = [GREEN if v > 0 else RED for v in sv_show]
        y_pos  = range(len(sv_show))

        ax.barh(y_pos, sv_show, color=colors, alpha=0.82, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ft_show, fontsize=8)
        ax.axvline(0, color=GRAY, lw=0.8)
        ax.set_xlabel("SHAP Value", fontsize=10)
        ax.set_title(
            f"Local Explanation — {label}\n{subtitle}\n"
            f"Base={expected_val:.3f} → Pred={pred:.3f} (log scale)",
            fontweight="bold", fontsize=9,
        )
        ax.text(0.98, 0.02,
                f"Prediction (log)={pred:.3f}\n"
                f"Prediction (mm)={np.expm1(pred):.1f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=BLUE,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCC", alpha=0.85))

    add_figure_title(
        fig,
        "Local SHAP Explanations — Representative Day Types",
        "Green = feature increases prediction | Red = feature decreases prediction",
    )
    if save:
        save_figure(fig, "shap_local_waterfall", subdir="explainability")
    plt.close(fig)
    logger.info("[SHAP] Local waterfall plots saved")
