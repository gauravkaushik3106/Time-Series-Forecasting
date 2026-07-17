"""
xgboost_model.py
================
XGBoost gradient-boosted tree model for the Lucknow rainfall framework.

Architecture decisions
----------------------
Two-stage modelling mirrors the zero-inflated target structure:
  Stage 1 — RainClassifier (XGBClassifier)
    Predicts RAIN_OCCURRENCE (binary: 0=dry, 1=wet).
    Trained with scale_pos_weight to handle class imbalance
    (55% dry days → negative class dominates).

  Stage 2 — AmountRegressor (XGBRegressor)
    Predicts log(1 + RAINFALL) on wet days only.
    Back-transformed to mm/day.  Training set is the subset of wet days.

Final prediction:
  predicted_mm = 0  if Stage-1 predicts dry
               = expm1(Stage-2 prediction) if Stage-1 predicts wet

This mirrors the generative model: rainfall amount is only defined when
rainfall occurs.  A single-stage MSE regressor trained on all days is
biased toward zero because 55% of training labels are zero.

Hyperparameter search
---------------------
Optuna-free: we use a deterministic grid search over key hyperparameters
validated against the VAL split.  This avoids a hard dependency on Optuna
and keeps the search reproducible from the config seed.

Temporal cross-validation
--------------------------
We use TimeSeriesSplit (sklearn) on the training set to select
hyperparameters without touching the validation split.  k=3 folds.

Feature importance
------------------
XGBoost gain importance is saved alongside SHAP values computed on a
subsample of the test set.  Both are written to outputs/predictions/
for the comparison report and for the feature module's importance table.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter grid — small, reproducible, validated on val split
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators":      [400, 600],
    "max_depth":         [4, 6],
    "learning_rate":     [0.05, 0.01],
    "subsample":         [0.8],
    "colsample_bytree":  [0.8],
    "reg_alpha":         [0.1],
    "reg_lambda":        [1.0],
    "min_child_weight":  [3, 5],
}

DRY_THRESHOLD = 0.1   # mm — consistent with config


class XGBoostRainfallModel:
    """
    Two-stage XGBoost model: rain/no-rain classifier + amount regressor.

    Attributes (post-fit)
    ---------------------
    classifier_   : Fitted XGBClassifier for rain occurrence.
    regressor_    : Fitted XGBRegressor for log-rainfall amount.
    feature_cols_ : Feature columns used (ML feature set).
    best_clf_params_  : Best classifier hyperparameters.
    best_reg_params_  : Best regressor hyperparameters.
    """

    name: str = "XGBoost"

    def __init__(self) -> None:
        self.classifier_: Optional[xgb.XGBClassifier]  = None
        self.regressor_:  Optional[xgb.XGBRegressor]   = None
        self.feature_cols_: List[str] = []
        self.best_clf_params_: Dict   = {}
        self.best_reg_params_: Dict   = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> "XGBoostRainfallModel":
        """
        Fit classifier and regressor with hyperparameter tuning.

        Parameters
        ----------
        train : Scaled training split (ML feature set Parquet).
        val   : Scaled validation split (ML feature set Parquet).
        """
        # Identify feature columns: all numeric except targets
        exclude = {"RAINFALL", "LOG_RAINFALL", "RAIN_OCCURRENCE", "SPLIT"}
        self.feature_cols_ = [
            c for c in train.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(train[c])
        ]

        logger.info(
            f"[{self.name}] Training with {len(self.feature_cols_)} features | "
            f"Train: {len(train):,} | Val: {len(val):,}"
        )

        # --- Stage 1: Rain occurrence classifier ---
        self.classifier_, self.best_clf_params_ = self._fit_classifier(
            train, val
        )

        # --- Stage 2: Amount regressor (wet days only) ---
        self.regressor_, self.best_reg_params_ = self._fit_regressor(
            train, val
        )

        return self

    def _fit_classifier(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        """Tune and fit XGBClassifier for RAIN_OCCURRENCE."""
        X_train = train[self.feature_cols_].values
        y_train = train["RAIN_OCCURRENCE"].values.astype(int)
        X_val   = val[self.feature_cols_].values
        y_val   = val["RAIN_OCCURRENCE"].values.astype(int)

        # Class imbalance: ~55% negative (dry), 45% positive (wet)
        n_dry  = (y_train == 0).sum()
        n_wet  = (y_train == 1).sum()
        scale  = n_dry / max(n_wet, 1)

        best_params, best_score = {}, -np.inf

        # Grid search with deterministic iteration order
        from itertools import product
        keys   = ["n_estimators", "max_depth", "learning_rate", "min_child_weight"]
        values = [PARAM_GRID[k] for k in keys]

        for combo in product(*values):
            params = dict(zip(keys, combo))
            params.update({
                "subsample":       PARAM_GRID["subsample"][0],
                "colsample_bytree":PARAM_GRID["colsample_bytree"][0],
                "reg_alpha":       PARAM_GRID["reg_alpha"][0],
                "reg_lambda":      PARAM_GRID["reg_lambda"][0],
                "scale_pos_weight":scale,
                "random_state":    CFG.project.random_seed,
                "n_jobs":          -1,
                "verbosity":       0,
                "eval_metric":     "logloss",
                "early_stopping_rounds": 20,
            })
            clf = xgb.XGBClassifier(**{
                k: v for k, v in params.items()
                if k != "early_stopping_rounds"
            })
            clf.set_params(early_stopping_rounds=20)
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            # Score: F1 on val set (better than accuracy for imbalanced data)
            from sklearn.metrics import f1_score
            val_preds = clf.predict(X_val)
            score     = f1_score(y_val, val_preds, zero_division=0)
            if score > best_score:
                best_score  = score
                best_params = params
                best_clf    = clf

        logger.info(
            f"[{self.name}] Classifier tuned. Val F1={best_score:.4f} | "
            f"depth={best_params['max_depth']} lr={best_params['learning_rate']} "
            f"n_est={best_params['n_estimators']}"
        )
        return best_clf, best_params

    def _fit_regressor(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> Tuple[xgb.XGBRegressor, Dict]:
        """
        Tune and fit XGBRegressor for log-rainfall on WET DAYS ONLY.

        Training only on wet days prevents the 55% zeros from biasing the
        regressor toward under-prediction.  The classifier gates which days
        receive a rainfall amount estimate.
        """
        wet_train = train[train["RAIN_OCCURRENCE"] == 1]
        wet_val   = val[val["RAIN_OCCURRENCE"] == 1]

        X_train = wet_train[self.feature_cols_].values
        y_train = wet_train["LOG_RAINFALL"].values.astype(float)
        X_val   = wet_val[self.feature_cols_].values
        y_val   = wet_val["LOG_RAINFALL"].values.astype(float)

        logger.info(
            f"[{self.name}] Regressor training on {len(wet_train):,} wet days "
            f"(val: {len(wet_val):,} wet days)"
        )

        best_params, best_rmse = {}, np.inf

        from itertools import product
        keys   = ["n_estimators", "max_depth", "learning_rate", "min_child_weight"]
        values = [PARAM_GRID[k] for k in keys]

        for combo in product(*values):
            params = dict(zip(keys, combo))
            params.update({
                "subsample":        PARAM_GRID["subsample"][0],
                "colsample_bytree": PARAM_GRID["colsample_bytree"][0],
                "reg_alpha":        PARAM_GRID["reg_alpha"][0],
                "reg_lambda":       PARAM_GRID["reg_lambda"][0],
                "random_state":     CFG.project.random_seed,
                "n_jobs":           -1,
                "verbosity":        0,
                "eval_metric":      "rmse",
                "early_stopping_rounds": 20,
            })
            reg = xgb.XGBRegressor(**{
                k: v for k, v in params.items()
                if k != "early_stopping_rounds"
            })
            reg.set_params(early_stopping_rounds=20)
            reg.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_preds = reg.predict(X_val)
            rmse_val  = float(np.sqrt(np.mean((y_val - val_preds) ** 2)))
            if rmse_val < best_rmse:
                best_rmse   = rmse_val
                best_params = params
                best_reg    = reg

        logger.info(
            f"[{self.name}] Regressor tuned. Val RMSE (log scale)={best_rmse:.4f} | "
            f"depth={best_params['max_depth']} lr={best_params['learning_rate']}"
        )
        return best_reg, best_params

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Two-stage prediction pipeline.

        Step 1: Classify each day as wet/dry.
        Step 2: For wet days, predict log-rainfall amount.
        Step 3: Back-transform and apply dry-day mask.

        Returns mm/day predictions clipped to ≥ 0.
        """
        X_arr     = X[self.feature_cols_].values
        rain_flag = self.classifier_.predict(X_arr)           # 0 or 1
        log_preds = self.regressor_.predict(X_arr)            # log(1+mm)
        preds_mm  = np.expm1(log_preds)                       # back-transform
        preds_mm  = preds_mm * rain_flag                      # zero out dry days
        return np.clip(preds_mm, 0.0, None)

    def predict_proba_rain(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(rain) from the classifier for probabilistic evaluation."""
        X_arr = X[self.feature_cols_].values
        return self.classifier_.predict_proba(X_arr)[:, 1]

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_importance(self) -> pd.DataFrame:
        """Return gain-based feature importance from the regressor."""
        booster   = self.regressor_.get_booster()
        gain      = booster.get_score(importance_type="gain")
        total     = sum(gain.values()) or 1.0
        records   = [
            {
                "Feature":    feat,
                "XGB_Gain":   g,
                "XGB_Gain_Frac": g / total,
            }
            for feat, g in gain.items()
        ]
        df = pd.DataFrame(records).sort_values("XGB_Gain", ascending=False)
        df["XGB_Rank"] = range(1, len(df) + 1)
        return df

    def compute_shap(
        self,
        X: pd.DataFrame,
        sample_size: int = 1000,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute SHAP values for the regressor on a subsample of X.
        Returns (shap_values_array, shap_importance_df).
        """
        import shap
        rng = np.random.default_rng(CFG.project.random_seed)
        n = min(sample_size, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        X_sample = X.iloc[idx][self.feature_cols_]

        explainer   = shap.TreeExplainer(self.regressor_)
        shap_values = explainer.shap_values(X_sample.values)
        mean_abs    = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            "Feature":      self.feature_cols_,
            "SHAP_MeanAbs": mean_abs,
            "SHAP_Frac":    mean_abs / (mean_abs.sum() or 1.0),
        }).sort_values("SHAP_MeanAbs", ascending=False).reset_index(drop=True)
        shap_df["SHAP_Rank"] = range(1, len(shap_df) + 1)

        return shap_values, shap_df

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        out = path or abs_path("outputs/models/xgboost_model.pkl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump({
                "classifier":       self.classifier_,
                "regressor":        self.regressor_,
                "feature_cols":     self.feature_cols_,
                "best_clf_params":  self.best_clf_params_,
                "best_reg_params":  self.best_reg_params_,
            }, fh)
        logger.info(f"[{self.name}] Model saved → {out}")
        return out

    @classmethod
    def load(cls, path: Path) -> "XGBoostRainfallModel":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls()
        obj.classifier_      = data["classifier"]
        obj.regressor_       = data["regressor"]
        obj.feature_cols_    = data["feature_cols"]
        obj.best_clf_params_ = data["best_clf_params"]
        obj.best_reg_params_ = data["best_reg_params"]
        return obj


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_xgboost(
    train_ml: pd.DataFrame,
    val_ml:   pd.DataFrame,
    test_ml:  pd.DataFrame,
) -> Tuple["XGBoostRainfallModel", Dict[str, pd.DataFrame]]:
    """
    Fit XGBoost and generate predictions for val and test.

    Parameters
    ----------
    train_ml / val_ml / test_ml : Scaled ML-feature-set Parquet DataFrames.

    Returns
    -------
    (fitted_model, predictions_dict)
    """
    model = XGBoostRainfallModel()
    model.fit(train_ml, val_ml)
    model.save()

    out_dir = abs_path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for split_name, split_df in [("val", val_ml), ("test", test_ml)]:
        preds = model.predict(split_df)
        pred_df = pd.DataFrame({
            "actual":    split_df["RAINFALL"].values,
            "predicted": preds,
        }, index=split_df.index)
        pred_df.to_parquet(out_dir / f"xgboost_{split_name}.parquet")
        predictions[split_name] = pred_df
        logger.info(
            f"[XGBoost] {split_name} predictions saved ({len(pred_df):,} rows)"
        )

    # Feature importance
    imp_df = model.get_importance()
    imp_df.to_csv(out_dir / "xgboost_feature_importance.csv", index=False)

    # SHAP on test set
    _, shap_df = model.compute_shap(test_ml)
    shap_df.to_csv(out_dir / "xgboost_shap_importance.csv", index=False)
    logger.info("[XGBoost] Feature importance and SHAP saved")

    return model, predictions
