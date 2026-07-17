"""
feature_selector.py
===================
Feature selection pipeline using four complementary lenses:

1. VIF (Variance Inflation Factor)
   Detects multicollinearity in the numeric feature matrix.
   Features with VIF > threshold are candidates for removal in linear models.

2. Correlation filter
   Removes features whose pairwise Pearson |r| exceeds a threshold,
   retaining the one with stronger correlation to the target.

3. XGBoost importance ranking
   Trains a fast XGBoost regressor on log(1+RAINFALL) and ranks features
   by gain-based importance.  Model-agnostic in the sense that importance
   is computed from a tree model independent of the final model family.

4. SHAP pre-screening
   Computes SHAP values for the same XGBoost model and ranks features by
   mean |SHAP|.  SHAP is preferred over raw gain importance because it
   accounts for feature interactions and is not biased toward high-cardinality
   features.

Output: three feature lists —
  - features_all     : all engineered features (full set)
  - features_linear  : VIF-filtered set for SARIMAX
  - features_ml      : SHAP-ranked top-N set for XGBoost / LSTM / GRU
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selection thresholds — all sourced from config or defined here as constants
# ---------------------------------------------------------------------------

VIF_THRESHOLD: float          = CFG.eda.vif_threshold      # 10.0
CORR_FILTER_THRESHOLD: float  = 0.92   # |r| above this triggers pair removal
SHAP_TOP_N: int               = 30     # max features for ML model sets
MIN_IMPORTANCE_FRAC: float    = 0.001  # drop features with XGB gain < 0.1% of total

# Columns that are NEVER candidates for removal regardless of VIF/correlation.
# These are either the target, required structural flags, or leakage guards.
PROTECTED_COLUMNS: set = {
    "RAINFALL",
    "LOG_RAINFALL",
    "RAIN_OCCURRENCE",
    "RAINFALL_WET_ONLY",
    "IS_MONSOON",
    "MONSOON_FLAG",
}

# Columns that encode the same information in different scales — exclude from
# the feature matrix fed to XGBoost (avoid target leakage proxies).
LEAKAGE_COLUMNS: set = {
    "RAINFALL",          # the raw target
    "LOG_RAINFALL",      # monotone transform of target
    "RAIN_OCCURRENCE",   # binary derived from target
    "RAINFALL_WET_ONLY", # masked version of target
}


def run_feature_selection(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "LOG_RAINFALL",
) -> "FeatureSelectionResult":
    """
    Execute the four-stage feature selection pipeline on the training set.

    Parameters
    ----------
    df_train    : Training split with all engineered features.
    feature_cols: All candidate feature column names.
    target_col  : Regression target used for XGBoost + SHAP fitting.

    Returns
    -------
    FeatureSelectionResult with feature lists, scores, and report text.
    """
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)

    # Strip leakage columns and protected columns that are targets
    candidates = [
        c for c in feature_cols
        if c in df_train.columns and c not in LEAKAGE_COLUMNS
    ]

    # Work on complete cases only (drop NaN rows introduced by lags/rolling)
    analysis_df = df_train[candidates + [target_col]].dropna()
    logger.info(
        f"Analysis subset: {len(analysis_df):,} complete rows "
        f"(from {len(df_train):,} training rows after dropping NaN warm-up)"
    )

    # ---- Stage 1: VIF analysis ----
    vif_df = _compute_vif(analysis_df[candidates])

    # ---- Stage 2: Correlation filter ----
    corr_drop = _correlation_filter(
        analysis_df[candidates],
        analysis_df[target_col],
        threshold=CORR_FILTER_THRESHOLD,
    )

    # ---- Stage 3: XGBoost importance ----
    xgb_importance = _xgboost_importance(
        analysis_df[candidates],
        analysis_df[target_col],
    )

    # ---- Stage 4: SHAP pre-screening ----
    shap_importance = _shap_prescreening(
        analysis_df[candidates],
        analysis_df[target_col],
    )

    # ---- Derive recommended feature sets ----
    features_linear = _build_linear_feature_set(
        candidates, vif_df, corr_drop, xgb_importance
    )
    features_ml = _build_ml_feature_set(
        candidates, shap_importance, xgb_importance
    )

    result = FeatureSelectionResult(
        features_all=candidates,
        features_linear=features_linear,
        features_ml=features_ml,
        vif_df=vif_df,
        corr_dropped=corr_drop,
        xgb_importance=xgb_importance,
        shap_importance=shap_importance,
    )

    logger.info(f"Feature sets finalised:")
    logger.info(f"  All features     : {len(candidates)}")
    logger.info(f"  Linear (SARIMAX) : {len(features_linear)}")
    logger.info(f"  ML (XGB/LSTM/GRU): {len(features_ml)}")

    return result


# ---------------------------------------------------------------------------
# Stage 1: VIF
# ---------------------------------------------------------------------------

def _compute_vif(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each numeric feature.

    Drops any constant or near-constant columns before computing VIF
    (variance_inflation_factor crashes on singular matrices).
    """
    df = features_df.copy().select_dtypes(include=[np.number])

    # Remove constant columns (std == 0)
    non_constant = df.columns[df.std() > 1e-8].tolist()
    dropped_const = [c for c in df.columns if c not in non_constant]
    if dropped_const:
        logger.warning(f"Dropped constant columns from VIF: {dropped_const}")
    df = df[non_constant]

    # VIF requires intercept column
    X = df.copy()
    X.insert(0, "_intercept", 1.0)

    records = []
    for i, col in enumerate(df.columns):
        try:
            vif_val = variance_inflation_factor(X.values, i + 1)
        except Exception as e:
            logger.warning(f"VIF computation failed for '{col}': {e}")
            vif_val = np.inf
        records.append({
            "Feature":  col,
            "VIF":      round(float(vif_val), 2),
            "Severity": (
                "Severe"   if vif_val > VIF_THRESHOLD else
                "Moderate" if vif_val > 5.0 else
                "Low"
            ),
        })

    vif_df = pd.DataFrame(records).sort_values("VIF", ascending=False)
    logger.info(
        f"VIF: {(vif_df['Severity']=='Severe').sum()} severe, "
        f"{(vif_df['Severity']=='Moderate').sum()} moderate, "
        f"{(vif_df['Severity']=='Low').sum()} low"
    )
    return vif_df


# ---------------------------------------------------------------------------
# Stage 2: Correlation filter
# ---------------------------------------------------------------------------

def _correlation_filter(
    features_df: pd.DataFrame,
    target: pd.Series,
    threshold: float = CORR_FILTER_THRESHOLD,
) -> List[str]:
    """
    Identify features to drop due to pairwise collinearity.

    Algorithm
    ---------
    For each pair (A, B) with |r(A,B)| > threshold:
      - Keep whichever has higher |r(feature, target)|.
      - Drop the other.
    Applied greedily in descending order of pair correlation magnitude.

    Returns
    -------
    List of column names recommended for removal.
    """
    numeric_df = features_df.select_dtypes(include=[np.number]).dropna()
    corr_matrix = numeric_df.corr().abs()
    target_corr = numeric_df.corrwith(target).abs()

    to_drop: set = set()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Collect all high-correlation pairs sorted by magnitude
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feat_a", "level_1": "feat_b", 0: "corr"})
        .query("corr > @threshold")
        .sort_values("corr", ascending=False)
    )

    for _, row in pairs.iterrows():
        feat_a, feat_b = row["feat_a"], row["feat_b"]
        if feat_a in to_drop or feat_b in to_drop:
            continue
        if feat_a in PROTECTED_COLUMNS or feat_b in PROTECTED_COLUMNS:
            continue
        # Keep the feature with higher target correlation
        r_a = target_corr.get(feat_a, 0.0)
        r_b = target_corr.get(feat_b, 0.0)
        drop = feat_b if r_a >= r_b else feat_a
        to_drop.add(drop)
        logger.debug(
            f"Corr filter: drop '{drop}' (|r(A,B)|={row['corr']:.3f}, "
            f"target-corr: {feat_a}={r_a:.3f}, {feat_b}={r_b:.3f})"
        )

    logger.info(
        f"Correlation filter (threshold={threshold}): "
        f"{len(to_drop)} features flagged for removal"
    )
    return sorted(to_drop)


# ---------------------------------------------------------------------------
# Stage 3: XGBoost importance
# ---------------------------------------------------------------------------

def _xgboost_importance(
    features_df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Train a lightweight XGBoost regressor and return gain-based feature importance.

    XGBoost is fitted on log(1+rainfall) to match the modelling target.
    Hyperparameters are conservative (shallow trees, high regularisation)
    to prevent overfitting from inflating importance of noisy features.
    """
    numeric_df = features_df.select_dtypes(include=[np.number]).dropna()
    common_idx = numeric_df.index.intersection(target.dropna().index)
    X = numeric_df.loc[common_idx]
    y = target.loc[common_idx]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=CFG.project.random_seed,
        n_jobs=-1,
        verbosity=0,
        eval_metric="rmse",
    )
    model.fit(X, y)

    importance_raw = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(importance_raw.values()) or 1.0

    records = []
    for feat in X.columns:
        gain  = importance_raw.get(feat, 0.0)
        cover = model.get_booster().get_score(importance_type="cover").get(feat, 0.0)
        records.append({
            "Feature":     feat,
            "XGB_Gain":    gain,
            "XGB_Gain_Frac": gain / total_gain,
            "XGB_Cover":   cover,
        })

    imp_df = (
        pd.DataFrame(records)
        .sort_values("XGB_Gain", ascending=False)
        .reset_index(drop=True)
    )
    imp_df["XGB_Rank"] = imp_df.index + 1

    logger.info(
        f"XGBoost importance: top feature = '{imp_df.iloc[0]['Feature']}' "
        f"(gain frac = {imp_df.iloc[0]['XGB_Gain_Frac']:.4f})"
    )
    return imp_df


# ---------------------------------------------------------------------------
# Stage 4: SHAP pre-screening
# ---------------------------------------------------------------------------

def _shap_prescreening(
    features_df: pd.DataFrame,
    target: pd.Series,
    sample_size: int = 3000,
) -> pd.DataFrame:
    """
    Compute SHAP values for a lightweight XGBoost model.

    SHAP (SHapley Additive exPlanations) gives a theoretically grounded
    measure of each feature's average marginal contribution to predictions.
    Unlike gain-based importance, SHAP:
      - Accounts for feature interaction effects
      - Is not biased toward continuous or high-cardinality features
      - Sums to the model output for each prediction (additive axiom)

    Parameters
    ----------
    sample_size : SHAP computation is O(N × F); subsample for speed.
    """
    numeric_df = features_df.select_dtypes(include=[np.number]).dropna()
    common_idx = numeric_df.index.intersection(target.dropna().index)
    X = numeric_df.loc[common_idx]
    y = target.loc[common_idx]

    # Subsample for computational efficiency — SHAP scales quadratically
    rng = np.random.default_rng(CFG.project.random_seed)
    if len(X) > sample_size:
        sample_idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample, y_sample = X, y

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CFG.project.random_seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_sample, y_sample)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature":        X_sample.columns.tolist(),
        "SHAP_MeanAbs":   mean_abs_shap,
        "SHAP_MeanAbsFrac": mean_abs_shap / (mean_abs_shap.sum() or 1.0),
    }).sort_values("SHAP_MeanAbs", ascending=False).reset_index(drop=True)

    shap_df["SHAP_Rank"] = shap_df.index + 1

    logger.info(
        f"SHAP pre-screening: top feature = '{shap_df.iloc[0]['Feature']}' "
        f"(mean |SHAP| = {shap_df.iloc[0]['SHAP_MeanAbs']:.4f})"
    )
    return shap_df


# ---------------------------------------------------------------------------
# Build recommended feature sets
# ---------------------------------------------------------------------------

def _build_linear_feature_set(
    candidates: List[str],
    vif_df: pd.DataFrame,
    corr_drop: List[str],
    xgb_importance: pd.DataFrame,
) -> List[str]:
    """
    Build the feature set for linear models (SARIMAX).

    Rules applied in order:
    1. Remove features with VIF > VIF_THRESHOLD (severe multicollinearity).
    2. Remove features flagged by correlation filter.
    3. Remove features with near-zero XGBoost importance (< MIN_IMPORTANCE_FRAC).
    4. Always retain PROTECTED_COLUMNS that are not leakage variables.
    """
    severe_vif = set(
        vif_df.loc[vif_df["Severity"] == "Severe", "Feature"].tolist()
    )
    corr_set   = set(corr_drop)

    low_importance = set(
        xgb_importance.loc[
            xgb_importance["XGB_Gain_Frac"] < MIN_IMPORTANCE_FRAC, "Feature"
        ].tolist()
    )

    to_exclude = (severe_vif | corr_set | low_importance) - PROTECTED_COLUMNS

    linear_features = [
        c for c in candidates
        if c not in to_exclude and c not in LEAKAGE_COLUMNS
    ]

    logger.info(
        f"Linear feature set: removed {len(to_exclude)} "
        f"({len(severe_vif)} severe VIF, {len(corr_set)} corr, "
        f"{len(low_importance)} low importance) → {len(linear_features)} retained"
    )
    return linear_features


def _build_ml_feature_set(
    candidates: List[str],
    shap_importance: pd.DataFrame,
    xgb_importance: pd.DataFrame,
) -> List[str]:
    """
    Build the feature set for ML/DL models (XGBoost, LSTM, GRU).

    Tree-based and neural models handle multicollinearity natively,
    so VIF and correlation filtering are NOT applied here.
    Instead we use SHAP rank as the primary filter, retaining the
    top SHAP_TOP_N features by mean |SHAP| value.
    Any feature with XGB gain < MIN_IMPORTANCE_FRAC is also excluded.
    """
    low_importance = set(
        xgb_importance.loc[
            xgb_importance["XGB_Gain_Frac"] < MIN_IMPORTANCE_FRAC, "Feature"
        ].tolist()
    ) - PROTECTED_COLUMNS

    top_shap = set(
        shap_importance.head(SHAP_TOP_N)["Feature"].tolist()
    )

    # Always include PROTECTED_COLUMNS that are not leakage
    protected_non_leakage = PROTECTED_COLUMNS - LEAKAGE_COLUMNS

    ml_features = [
        c for c in candidates
        if (c in top_shap or c in protected_non_leakage)
        and c not in low_importance
        and c not in LEAKAGE_COLUMNS
    ]

    logger.info(
        f"ML feature set: top-{SHAP_TOP_N} SHAP + protected → {len(ml_features)} features"
    )
    return ml_features


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class FeatureSelectionResult:
    """
    Container for all feature selection outputs.

    Attributes
    ----------
    features_all    : All engineered candidate features.
    features_linear : Recommended set for SARIMAX (VIF + corr filtered).
    features_ml     : Recommended set for XGBoost / LSTM / GRU (SHAP-ranked).
    vif_df          : VIF scores for all features.
    corr_dropped    : Features flagged by correlation filter.
    xgb_importance  : XGBoost gain-based importance table.
    shap_importance : SHAP mean |value| importance table.
    """

    def __init__(
        self,
        features_all: List[str],
        features_linear: List[str],
        features_ml: List[str],
        vif_df: pd.DataFrame,
        corr_dropped: List[str],
        xgb_importance: pd.DataFrame,
        shap_importance: pd.DataFrame,
    ) -> None:
        self.features_all     = features_all
        self.features_linear  = features_linear
        self.features_ml      = features_ml
        self.vif_df           = vif_df
        self.corr_dropped     = corr_dropped
        self.xgb_importance   = xgb_importance
        self.shap_importance  = shap_importance

    def summary(self) -> str:
        lines = [
            "FEATURE SELECTION SUMMARY",
            f"  All candidates   : {len(self.features_all)}",
            f"  Linear (SARIMAX) : {len(self.features_linear)}",
            f"  ML (XGB/LSTM/GRU): {len(self.features_ml)}",
            f"  Corr-dropped     : {len(self.corr_dropped)}",
            f"  Severe VIF       : {(self.vif_df['Severity']=='Severe').sum()}",
        ]
        return "\n".join(lines)
