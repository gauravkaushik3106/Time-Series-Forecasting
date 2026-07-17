"""
partial_dependence.py
=====================
Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE)
curves for the XGBoost model.

PDP shows the marginal effect of a feature on the model output after
averaging out all other features.  It answers: "On average, how does
the model's prediction change as CLOUD increases from 0% to 100%?"

ICE curves show the same relationship for individual observations rather
than the average.  They reveal heterogeneity — whether the relationship
is the same for all days or varies by context.

Features analysed
-----------------
- RH (relative humidity): primary moisture driver
- CLOUD (cloud cover %): strongest raw correlation with rainfall (r=0.44)
- SOIL_MOISTURE_GRADIENT: engineered feature replacing collinear soil pair
- RAINFALL_lag1: autoregressive component (ACF lag-1 = 0.49)
- RAINFALL_roll_mean_7: recent wetness regime
- PRESSURE_RH_INTERACTION: engineered interaction capturing monsoon state

Outputs
-------
outputs/figures/explainability/pdp_grid.png
outputs/figures/explainability/pdp_2d_cloud_rh.png
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
    BLUE, RED, GREEN, ORANGE, GRAY, CATEGORICAL_PALETTE,
)

logger = logging.getLogger(__name__)

PDP_FEATURES = [
    "RH",
    "CLOUD",
    "SOIL_MOISTURE_GRADIENT",
    "RAINFALL_lag1",
    "RAINFALL_roll_mean_7",
    "PRESSURE_RH_INTERACTION",
]

N_GRID_POINTS  = 50    # resolution of the PDP grid
N_ICE_SAMPLES  = 100   # number of individual ICE curves to show


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pdp_analysis(
    test_ml:  pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Compute and plot PDPs + ICE curves for the XGBoost regressor.

    Parameters
    ----------
    test_ml : Scaled ML feature set, test split.
    save    : Persist figures to disk.
    """
    apply_style()

    # Load XGBoost regressor
    with open(abs_path("outputs/models/xgboost_model.pkl"), "rb") as fh:
        xgb_data = pickle.load(fh)
    regressor    = xgb_data["regressor"]
    feature_cols = xgb_data["feature_cols"]

    # Wet-day subset (regressor is only meaningful on wet days)
    wet_mask = test_ml["RAIN_OCCURRENCE"] == 1
    X_wet    = test_ml.loc[wet_mask, feature_cols].values.astype(float)

    logger.info(
        f"[PDP] Running on {len(X_wet):,} wet test days | "
        f"{len(feature_cols)} features"
    )

    # Subsample for ICE curves
    rng     = np.random.default_rng(CFG.project.random_seed)
    ice_idx = rng.choice(len(X_wet), size=min(N_ICE_SAMPLES, len(X_wet)), replace=False)
    X_ice   = X_wet[ice_idx]

    # Compute PDP + ICE for each target feature
    pdp_results = {}
    for feat in PDP_FEATURES:
        if feat not in feature_cols:
            logger.warning(f"[PDP] Feature '{feat}' not in model — skipping")
            continue
        grid, pdp, ice = _compute_pdp_ice(
            regressor, X_wet, X_ice, feature_cols, feat
        )
        pdp_results[feat] = (grid, pdp, ice)

    # Plot
    _plot_pdp_ice_grid(pdp_results, save)
    _plot_2d_pdp(regressor, X_wet, feature_cols, save)


# ---------------------------------------------------------------------------
# PDP / ICE computation
# ---------------------------------------------------------------------------

def _compute_pdp_ice(
    regressor,
    X:           np.ndarray,
    X_ice:       np.ndarray,
    feature_cols: List[str],
    target_feat:  str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PDP and ICE curves for one feature.

    Algorithm
    ---------
    For each grid point g in feature range:
      1. Copy all rows of X.
      2. Set the target feature column to g for every row.
      3. Predict and average → one PDP point.
      4. For ICE: keep individual predictions (not averaged).

    Returns
    -------
    grid : (N_GRID_POINTS,) — feature values swept
    pdp  : (N_GRID_POINTS,) — average prediction at each grid point
    ice  : (N_ICE_SAMPLES, N_GRID_POINTS) — individual predictions
    """
    feat_idx = feature_cols.index(target_feat)

    # Grid: 1st–99th percentile to avoid extrapolation artefacts
    lo  = np.percentile(X[:, feat_idx], 1)
    hi  = np.percentile(X[:, feat_idx], 99)
    grid = np.linspace(lo, hi, N_GRID_POINTS)

    pdp     = np.zeros(N_GRID_POINTS)
    ice     = np.zeros((len(X_ice), N_GRID_POINTS))

    for j, g in enumerate(grid):
        # Full PDP: sweep across all X rows
        X_mod        = X.copy()
        X_mod[:, feat_idx] = g
        preds        = regressor.predict(X_mod)
        pdp[j]       = preds.mean()

        # ICE: only on subsample
        X_ice_mod        = X_ice.copy()
        X_ice_mod[:, feat_idx] = g
        ice[:, j]        = regressor.predict(X_ice_mod)

    return grid, pdp, ice


# ---------------------------------------------------------------------------
# Figure E5: PDP + ICE grid
# ---------------------------------------------------------------------------

def _plot_pdp_ice_grid(
    pdp_results: dict,
    save: bool,
) -> None:
    """
    2-row × 3-column grid: one panel per feature.
    Each panel shows ICE curves (light grey), PDP (bold colour), and
    marginal rug plot of the actual feature distribution.
    """
    features = list(pdp_results.keys())
    ncols = 3
    nrows = (len(features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, 5 * nrows),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.30})
    axes = axes.flatten() if nrows > 1 else list(axes)

    for i, feat in enumerate(features):
        ax = axes[i]
        grid, pdp, ice = pdp_results[feat]

        # ICE curves — thin, semi-transparent
        for ice_row in ice:
            ax.plot(grid, ice_row, color=GRAY, lw=0.4, alpha=0.18)

        # PDP — bold
        ax.plot(grid, pdp, color=BLUE, lw=2.2, label="PDP (average)")

        # Confidence band (5th–95th percentile of ICE)
        lo_band = np.percentile(ice, 5, axis=0)
        hi_band = np.percentile(ice, 95, axis=0)
        ax.fill_between(grid, lo_band, hi_band, alpha=0.15, color=BLUE,
                        label="ICE 5th–95th %ile")

        ax.set_xlabel(f"{feat} (scaled)", fontsize=9)
        ax.set_ylabel("Predicted log(1+rainfall)", fontsize=9)
        ax.set_title(f"PDP + ICE: {feat}", fontweight="bold", fontsize=10)
        ax.axhline(pdp.mean(), color=RED, lw=0.8, linestyle=":",
                   label=f"Mean pred = {pdp.mean():.3f}")
        ax.legend(fontsize=7, loc="upper left")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    add_figure_title(
        fig,
        "Partial Dependence Plots + ICE Curves — XGBoost Regressor",
        "Grey = individual ICE curves | Blue = average PDP | Blue band = 5th–95th %ile of ICE",
    )
    if save:
        save_figure(fig, "pdp_grid", subdir="explainability")
    plt.close(fig)
    logger.info("[PDP] PDP+ICE grid saved")


# ---------------------------------------------------------------------------
# Figure E6: 2D PDP (CLOUD × RH interaction)
# ---------------------------------------------------------------------------

def _plot_2d_pdp(
    regressor,
    X:           np.ndarray,
    feature_cols: List[str],
    save: bool,
) -> None:
    """
    2D partial dependence of CLOUD × RH interaction.

    Reveals whether the joint effect of high cloud cover AND high
    humidity is superadditive (moisture convergence regime) or whether
    one factor dominates.
    """
    if "CLOUD" not in feature_cols or "RH" not in feature_cols:
        logger.warning("[PDP] CLOUD or RH not in feature cols — skipping 2D PDP")
        return

    cidx = feature_cols.index("CLOUD")
    ridx = feature_cols.index("RH")
    n_g  = 30

    cloud_grid = np.linspace(np.percentile(X[:, cidx], 1),
                              np.percentile(X[:, cidx], 99), n_g)
    rh_grid    = np.linspace(np.percentile(X[:, ridx], 1),
                              np.percentile(X[:, ridx], 99), n_g)

    Z = np.zeros((n_g, n_g))
    X_mod = X.copy()
    for i, c_val in enumerate(cloud_grid):
        for j, rh_val in enumerate(rh_grid):
            X_mod[:, cidx] = c_val
            X_mod[:, ridx] = rh_val
            Z[i, j] = regressor.predict(X_mod).mean()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.contourf(rh_grid, cloud_grid, Z, levels=20, cmap="YlOrRd")
    ax.contour(rh_grid, cloud_grid, Z, levels=10, colors="white",
               linewidths=0.5, alpha=0.4)
    cbar = fig.colorbar(im, ax=ax, label="Mean predicted log(1+rainfall)")
    ax.set_xlabel("RH (scaled)", fontsize=11)
    ax.set_ylabel("CLOUD (scaled)", fontsize=11)
    ax.set_title(
        "2D Partial Dependence: CLOUD × Relative Humidity\n"
        "High cloud + high RH = maximum predicted rainfall (top-right)",
        fontweight="bold",
    )

    if save:
        save_figure(fig, "pdp_2d_cloud_rh", subdir="explainability")
    plt.close(fig)
    logger.info("[PDP] 2D PDP saved")
