"""
eda_correlation.py
==================
Correlation structure, multicollinearity detection via VIF, and physically
motivated feature replacement recommendations.

The analysis here directly informs which features enter which model family:
- Linear models (SARIMAX): must remove highly collinear features
- Tree-based (XGBoost): handles collinearity natively but VIF still informative
- Deep learning (LSTM/GRU): handles collinearity but benefit from domain features

Key findings from Phase 1 analysis that this module confirms and documents:
- TMAX / TMIN / TAVG: r ~ 0.82–0.96 → remove TMAX, retain TMIN (higher rain corr)
- SOIL_WET_SURF / SOIL_WET_ROOT: r = 0.94 → replace with moisture gradient
- RH / SOIL_WET_SURF: r = 0.93 → flag, monitor VIF
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG
from src.visualization.plot_utils import (
    apply_style, save_figure, add_stat_annotations,
    BLUE, RED, GREEN, ORANGE, GRAY, CATEGORICAL_PALETTE,
    make_figure, add_figure_title,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyse_correlations(
    df: pd.DataFrame,
    save: bool = True,
) -> Dict:
    """
    Full correlation and multicollinearity analysis.

    Parameters
    ----------
    df   : Cleaned dataframe with DatetimeIndex.
    save : Persist figures to disk.

    Returns
    -------
    dict containing VIF scores, correlation matrix, and feature recommendations.
    """
    apply_style()

    feature_cols: List[str] = list(CFG.schema.feature_columns)
    target_col   = CFG.schema.target_column

    features_df = df[feature_cols + [target_col]].copy()

    # --- Correlation matrix ---
    corr_matrix = features_df.corr(method="pearson")
    _plot_correlation_heatmap(corr_matrix, target_col, save)

    # --- VIF analysis ---
    vif_df = _compute_vif(features_df[feature_cols])
    _plot_vif_barplot(vif_df, save)
    _log_vif_results(vif_df)

    # --- Pairwise scatter (top predictors) ---
    _plot_top_predictor_scatters(df, target_col, save)

    # --- Scatter matrix (high-VIF group) ---
    _plot_collinear_group(df, save)

    # --- Feature recommendations ---
    recommendations = _generate_feature_recommendations(corr_matrix, vif_df, target_col)
    _log_recommendations(recommendations)

    return {
        "correlation_matrix":   corr_matrix,
        "vif_scores":           vif_df,
        "feature_recommendations": recommendations,
        "rain_correlations":    corr_matrix[target_col].drop(target_col).sort_values(
                                    ascending=False
                                ),
    }


# ---------------------------------------------------------------------------
# VIF computation
# ---------------------------------------------------------------------------

def _compute_vif(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature column.

    VIF_i = 1 / (1 - R²_i), where R²_i is the R² from regressing
    feature i on all other features.

    Interpretation:
      VIF < 5   : No significant multicollinearity
      VIF 5–10  : Moderate multicollinearity — monitor
      VIF > 10  : Severe multicollinearity — action required
    """
    df_clean = features_df.dropna()

    # VIF requires an intercept column
    X = df_clean.copy()
    X.insert(0, "intercept", 1.0)

    vif_records = []
    feature_names = df_clean.columns.tolist()

    for i, col in enumerate(feature_names):
        # statsmodels variance_inflation_factor takes the matrix and column index
        # The intercept is at index 0, features start at index 1
        vif_val = variance_inflation_factor(X.values, i + 1)  # +1 for intercept offset
        vif_records.append({
            "Feature": col,
            "VIF": round(float(vif_val), 2),
            "Severity": (
                "Severe"   if vif_val > CFG.eda.vif_threshold else
                "Moderate" if vif_val > 5.0 else
                "Low"
            ),
        })

    vif_df = pd.DataFrame(vif_records).sort_values("VIF", ascending=False)
    return vif_df


# ---------------------------------------------------------------------------
# Figure 11: Correlation heatmap
# ---------------------------------------------------------------------------

def _plot_correlation_heatmap(
    corr: pd.DataFrame,
    target_col: str,
    save: bool,
) -> None:
    """
    Full correlation matrix heatmap, annotated with Pearson r values.
    Highlighted column/row for the target (RAINFALL).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                              gridspec_kw={"width_ratios": [3, 1]})

    # --- Left: full correlation heatmap ---
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # upper triangle (keep lower)

    sns.heatmap(
        corr,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="#EEEEEE",
        annot_kws={"size": 8},
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    axes[0].set_title("Pearson Correlation Matrix — All Features + Target",
                      fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # Highlight target column/row with a border
    target_idx = list(corr.columns).index(target_col)
    axes[0].add_patch(plt.Rectangle(
        (target_idx, 0), 1, len(corr),
        fill=False, edgecolor=GREEN, lw=2.5, clip_on=False
    ))

    # --- Right: ranked correlations with target ---
    rain_corr = corr[target_col].drop(target_col).sort_values()
    colors = [RED if v < 0 else GREEN for v in rain_corr.values]
    axes[1].barh(rain_corr.index, rain_corr.values, color=colors, alpha=0.8,
                 edgecolor="none")
    axes[1].axvline(0, color=GRAY, lw=0.8)
    axes[1].set_xlabel("Pearson r with RAINFALL")
    axes[1].set_title("Feature–Rainfall Correlation Ranking")
    for i, (feat, val) in enumerate(rain_corr.items()):
        axes[1].text(
            val + (0.01 if val >= 0 else -0.01),
            i, f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=8, fontweight="bold"
        )
    axes[1].set_xlim(-0.5, 0.6)

    add_figure_title(
        fig, "Feature Correlation Analysis — Lucknow Dataset"
    )
    if save:
        save_figure(fig, "11_correlation_heatmap", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: VIF barplot
# ---------------------------------------------------------------------------

def _plot_vif_barplot(vif_df: pd.DataFrame, save: bool) -> None:
    """
    Horizontal barplot of VIF scores with severity colour coding.
    Reference lines at VIF=5 (moderate) and VIF=10 (severe).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    severity_colors = {"Low": GREEN, "Moderate": ORANGE, "Severe": RED}
    bar_colors = [severity_colors[s] for s in vif_df["Severity"]]

    bars = ax.barh(
        vif_df["Feature"], vif_df["VIF"],
        color=bar_colors, alpha=0.80, edgecolor="none"
    )

    # Threshold lines
    ax.axvline(5,  color=ORANGE, lw=1.2, linestyle="--",
               label="Moderate threshold (VIF = 5)")
    ax.axvline(CFG.eda.vif_threshold, color=RED, lw=1.5, linestyle="--",
               label=f"Severe threshold (VIF = {CFG.eda.vif_threshold:.0f})")

    # Value labels
    for bar, vif_val in zip(bars, vif_df["VIF"]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{vif_val:.1f}",
            va="center", fontsize=9, fontweight="bold"
        )

    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title(
        "Multicollinearity Analysis — Variance Inflation Factors\n"
        "VIF < 5: acceptable | 5–10: moderate | > 10: severe",
        fontweight="bold"
    )
    ax.legend(fontsize=9)

    # Add severity legend patches
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=RED,    alpha=0.8, label="Severe (VIF > 10)"),
        Patch(facecolor=ORANGE, alpha=0.8, label="Moderate (VIF 5–10)"),
        Patch(facecolor=GREEN,  alpha=0.8, label="Low (VIF < 5)"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_els,
              fontsize=8, loc="lower right")
    ax.set_xlim(0, vif_df["VIF"].max() * 1.15)

    if save:
        save_figure(fig, "12_vif_multicollinearity", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: Top-predictor scatter plots
# ---------------------------------------------------------------------------

def _plot_top_predictor_scatters(
    df: pd.DataFrame,
    target_col: str,
    save: bool,
) -> None:
    """
    Scatter plots of the 6 features most correlated with RAINFALL.
    Overlaid with LOWESS smoothing to reveal nonlinear relationships.
    Colour-coded by monsoon / non-monsoon to show regime dependency.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    feature_cols = list(CFG.schema.feature_columns)
    rain_corr = (
        df[feature_cols + [target_col]]
        .corr()[target_col]
        .drop(target_col)
        .abs()
        .sort_values(ascending=False)
    )
    top6 = rain_corr.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    rain = df[target_col]
    # Use log1p(rainfall) for scatter to reduce overplotting by zeros
    log_rain = np.log1p(rain.values)
    monsoon_mask = df["IS_MONSOON"].values == 1

    for i, feat in enumerate(top6):
        ax = axes[i]
        x = df[feat].values

        # Non-monsoon (background)
        ax.scatter(
            x[~monsoon_mask], log_rain[~monsoon_mask],
            s=3, alpha=0.15, color=BLUE, label="Non-monsoon"
        )
        # Monsoon (foreground)
        ax.scatter(
            x[monsoon_mask], log_rain[monsoon_mask],
            s=3, alpha=0.20, color=GREEN, label="Monsoon"
        )

        # LOWESS smoother
        try:
            sorted_idx = np.argsort(x)
            smooth = lowess(
                log_rain[sorted_idx], x[sorted_idx],
                frac=0.15, return_sorted=True
            )
            ax.plot(smooth[:, 0], smooth[:, 1], color=RED, lw=1.8,
                    label="LOWESS", zorder=5)
        except Exception:
            pass

        # Pearson r
        r_val, p_val = stats.pearsonr(x, log_rain)
        ax.set_xlabel(feat)
        ax.set_ylabel("log(1 + Rainfall)")
        ax.set_title(
            f"{feat} vs log(1+Rainfall)\n"
            f"r = {r_val:.3f} (p = {p_val:.2e})",
            fontsize=9
        )
        if i == 0:
            ax.legend(fontsize=7, markerscale=3)

    add_figure_title(
        fig,
        "Feature–Rainfall Scatter Plots (Top 6 Predictors)",
        "Colour: monsoon (green) vs non-monsoon (blue) | Red line: LOWESS smoother",
    )
    if save:
        save_figure(fig, "13_feature_scatter_top6", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 14: Collinear feature group (temperature triad + soil moisture pair)
# ---------------------------------------------------------------------------

def _plot_collinear_group(df: pd.DataFrame, save: bool) -> None:
    """
    Scatter matrix for the two highly collinear feature groups:
      Group 1: TMAX / TMIN / TAVG  (r ~ 0.82–0.96)
      Group 2: SOIL_WET_SURF / SOIL_WET_ROOT  (r = 0.94)

    Also shows the proposed replacement: SOIL_MOISTURE_GRADIENT.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Group 1: Temperature triad ---
    temp_cols = ["TMAX", "TMIN", "TAVG"]
    pairs1 = [("TMAX", "TMIN"), ("TMAX", "TAVG"), ("TMIN", "TAVG")]
    for ax, (c1, c2) in zip(axes[0], pairs1):
        r, p = stats.pearsonr(df[c1], df[c2])
        ax.scatter(df[c1], df[c2], s=2, alpha=0.15, color=ORANGE)
        # Regression line
        m, b, *_ = stats.linregress(df[c1], df[c2])
        x_line = np.array([df[c1].min(), df[c1].max()])
        ax.plot(x_line, m * x_line + b, color=RED, lw=1.5)
        ax.set_xlabel(c1 + " (°C)")
        ax.set_ylabel(c2 + " (°C)")
        ax.set_title(f"{c1} vs {c2}\nr = {r:.3f} ⚠ SEVERE COLLINEARITY")

    # --- Group 2: Soil moisture pair + proposed gradient ---
    sm_cols = ["SOIL_WET_SURF", "SOIL_WET_ROOT"]
    r_sm, _ = stats.pearsonr(df[sm_cols[0]], df[sm_cols[1]])
    axes[1, 0].scatter(df[sm_cols[0]], df[sm_cols[1]], s=2, alpha=0.15, color=BLUE)
    m, b, *_ = stats.linregress(df[sm_cols[0]], df[sm_cols[1]])
    xl = np.array([df[sm_cols[0]].min(), df[sm_cols[0]].max()])
    axes[1, 0].plot(xl, m * xl + b, color=RED, lw=1.5)
    axes[1, 0].set_xlabel("SOIL_WET_SURF")
    axes[1, 0].set_ylabel("SOIL_WET_ROOT")
    axes[1, 0].set_title(
        f"SOIL_WET_SURF vs SOIL_WET_ROOT\nr = {r_sm:.3f} ⚠ SEVERE COLLINEARITY"
    )

    # Proposed replacement: moisture gradient
    df_temp = df.copy()
    df_temp["MOISTURE_GRADIENT"] = df_temp["SOIL_WET_SURF"] - df_temp["SOIL_WET_ROOT"]
    r_grad, _ = stats.pearsonr(df_temp["SOIL_WET_SURF"], df_temp["MOISTURE_GRADIENT"])
    r_rain, _ = stats.pearsonr(df_temp["MOISTURE_GRADIENT"], df_temp["RAINFALL"])
    axes[1, 1].scatter(df_temp["SOIL_WET_SURF"], df_temp["MOISTURE_GRADIENT"],
                        s=2, alpha=0.15, color=GREEN)
    axes[1, 1].set_xlabel("SOIL_WET_SURF")
    axes[1, 1].set_ylabel("MOISTURE_GRADIENT (SURF − ROOT)")
    axes[1, 1].set_title(
        f"Proposed Replacement: MOISTURE_GRADIENT\n"
        f"r with SOIL_WET_SURF = {r_grad:.3f} | r with RAINFALL = {r_rain:.3f}"
    )

    # Moisture gradient vs rainfall
    log_rain = np.log1p(df_temp["RAINFALL"])
    axes[1, 2].scatter(df_temp["MOISTURE_GRADIENT"], log_rain,
                        s=2, alpha=0.15, color=GREEN)
    r_g2, p_g2 = stats.pearsonr(df_temp["MOISTURE_GRADIENT"], log_rain)
    axes[1, 2].set_xlabel("MOISTURE_GRADIENT")
    axes[1, 2].set_ylabel("log(1 + RAINFALL)")
    axes[1, 2].set_title(
        f"MOISTURE_GRADIENT vs log(Rainfall)\nr = {r_g2:.3f} (p = {p_g2:.3e})"
    )

    add_figure_title(
        fig,
        "Multicollinear Feature Groups & Replacement Strategy",
        "Row 1: temperature triad (r=0.82–0.96) | Row 2: soil moisture pair (r=0.94) + gradient replacement",
    )
    if save:
        save_figure(fig, "14_collinear_groups_replacement", subdir="eda")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature recommendation engine
# ---------------------------------------------------------------------------

def _generate_feature_recommendations(
    corr: pd.DataFrame,
    vif_df: pd.DataFrame,
    target_col: str,
) -> List[Dict]:
    """
    Generate structured feature engineering recommendations based on
    correlation analysis and VIF scores.
    """
    vif_threshold = CFG.eda.vif_threshold
    severe_vif = vif_df[vif_df["Severity"] == "Severe"]["Feature"].tolist()
    rain_corr  = corr[target_col].drop(target_col)

    recommendations = []

    # Temperature triad
    if all(c in corr.columns for c in ["TMAX", "TMIN", "TAVG"]):
        r_max = corr.loc["TMAX", "TAVG"]
        r_min = corr.loc["TMIN", "TAVG"]
        recommendations.append({
            "features": ["TMAX", "TMIN", "TAVG"],
            "issue": f"Severe collinearity: r(TMAX,TAVG)={r_max:.2f}, r(TMIN,TAVG)={r_min:.2f}",
            "action_linear_models": "Retain TMIN only (highest rain correlation); remove TMAX, TAVG",
            "action_tree_dl_models": "Retain all three — tree/DL models handle collinearity implicitly",
            "rationale": "TMIN correlates most strongly with rainfall (r=+0.23) among the thermal group",
        })

    # Soil moisture pair
    if all(c in corr.columns for c in ["SOIL_WET_SURF", "SOIL_WET_ROOT"]):
        r_sm = corr.loc["SOIL_WET_SURF", "SOIL_WET_ROOT"]
        recommendations.append({
            "features": ["SOIL_WET_SURF", "SOIL_WET_ROOT"],
            "issue": f"Severe collinearity: r={r_sm:.3f}",
            "action_linear_models": "Replace with SOIL_MOISTURE_GRADIENT = SOIL_WET_SURF − SOIL_WET_ROOT",
            "action_tree_dl_models": "Retain both plus add SOIL_MOISTURE_GRADIENT as engineered feature",
            "rationale": (
                "The gradient captures infiltration dynamics — the difference between "
                "surface and root-zone moisture represents moisture flux direction, "
                "a physically meaningful predictor of near-surface runoff potential"
            ),
        })

    # RH / SOIL_WET_SURF
    if all(c in corr.columns for c in ["RH", "SOIL_WET_SURF"]):
        r_rh_sw = corr.loc["RH", "SOIL_WET_SURF"]
        if abs(r_rh_sw) > 0.85:
            recommendations.append({
                "features": ["RH", "SOIL_WET_SURF"],
                "issue": f"High correlation: r={r_rh_sw:.3f}",
                "action_linear_models": "Monitor VIF; consider retaining RH (atmospheric) over SOIL_WET_SURF",
                "action_tree_dl_models": "Retain both — they measure different hydrological compartments",
                "rationale": "RH is an atmospheric variable; soil moisture is a land-surface variable. "
                             "Their co-movement is physically expected during monsoon but they diverge in dry season.",
            })

    # Best predictors (positive)
    top_pos = rain_corr[rain_corr > 0].sort_values(ascending=False).head(3)
    top_neg = rain_corr[rain_corr < 0].sort_values().head(2)
    recommendations.append({
        "features": list(top_pos.index),
        "issue": "Top positive predictors",
        "action_linear_models": f"Retain {list(top_pos.index)} — statistically and physically justified",
        "action_tree_dl_models": f"Include with lag variants (1, 2, 7 days)",
        "rationale": f"Pearson r values: {dict(top_pos.round(3))}",
    })
    recommendations.append({
        "features": list(top_neg.index),
        "issue": "Top negative predictors",
        "action_linear_models": f"Retain {list(top_neg.index)} — high-pressure → dry",
        "action_tree_dl_models": "Include with lag variants",
        "rationale": f"Pearson r values: {dict(top_neg.round(3))}",
    })

    return recommendations


def _log_recommendations(recs: List[Dict]) -> None:
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING RECOMMENDATIONS")
    logger.info("=" * 60)
    for i, rec in enumerate(recs, 1):
        logger.info(f"\n[{i}] Features: {rec['features']}")
        logger.info(f"    Issue : {rec['issue']}")
        logger.info(f"    Linear: {rec['action_linear_models']}")
        logger.info(f"    DL/ML : {rec['action_tree_dl_models']}")
        logger.info(f"    Why   : {rec['rationale']}")


def _log_vif_results(vif_df: pd.DataFrame) -> None:
    logger.info("\nVIF SCORES:")
    for _, row in vif_df.iterrows():
        symbol = "⚠ " if row["Severity"] != "Low" else "✓ "
        logger.info(
            f"  {symbol}{row['Feature']:20s} VIF = {row['VIF']:6.1f}  [{row['Severity']}]"
        )
