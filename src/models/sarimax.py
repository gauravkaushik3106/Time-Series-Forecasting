"""
sarimax.py
==========
SARIMAX (Seasonal ARIMA with eXogenous variables) for the Lucknow
rainfall framework.

Design decisions
----------------
Target transformation
  We model log(1 + RAINFALL) rather than raw RAINFALL because:
  - Raw rainfall is severely right-skewed (skewness 6.81); ARIMA assumes
    Gaussian-ish residuals.
  - The log transform reduces skewness to ~1.8 and stabilises variance.
  - All predictions are back-transformed via expm1() before metric computation.

Exogenous variables
  The linear feature set produced by Phase 3 selection (VIF-filtered, no
  severe multicollinearity) is used as the exogenous regressor matrix.
  Lag features (RAINFALL_lag1 etc.) derived from the target are included
  only for the ML models; for SARIMAX the AR terms absorb the autoregressive
  structure, so we restrict to meteorological exogenous variables only
  (CLOUD, WIND, SOLAR_RAD and their lagged versions).

Parameter search
  We use pmdarima.auto_arima with a restricted search grid to avoid the
  combinatorial explosion of full SARIMA search on 6,617 daily observations:
  - Non-seasonal p ∈ {0,1,2}, d ∈ {0,1}, q ∈ {0,1}
  - Seasonal P=1, D=1, Q=1, m=12 (monthly seasonality proxy for speed)
  - Full annual period (m=365) is computationally intractable for daily data;
    monthly captures the dominant Jun-Sep monsoon signal adequately.

Residual saving
  SARIMAX residuals on the training set are saved to outputs/predictions/
  because they will serve as targets for the hybrid SARIMAX+LSTM model
  in Phase 5.  The residuals represent unexplained nonlinear structure
  that the deep learning component is tasked with learning.
"""

from __future__ import annotations

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exogenous feature selection for SARIMAX
# (meteorological only — no rainfall lag features; AR terms handle those)
# ---------------------------------------------------------------------------
SARIMAX_EXOG_COLS: List[str] = [
    "CLOUD",
    "WIND",
    "SOLAR_RAD",
    "CLOUD_lag1",
    "CLOUD_lag3",
    "CLOUD_lag7",
    "MONTH_SIN",
    "MONTH_COS",
    "IS_MONSOON",
]


class SARIMAXModel:
    """
    SARIMAX wrapper with auto-parameter search, forecast generation,
    and residual extraction.

    Attributes (post-fit)
    ---------------------
    model_fit_     : Fitted statsmodels SARIMAX result object.
    order_         : (p, d, q) non-seasonal order.
    seasonal_order_: (P, D, Q, m) seasonal order.
    exog_cols_     : Exogenous column names actually used.
    train_residuals_: In-sample residuals on log scale (for hybrid model).
    """

    name: str = "SARIMAX"

    def __init__(self) -> None:
        self.model_fit_        = None
        self.order_            = None
        self.seasonal_order_   = None
        self.exog_cols_: List[str] = []
        self.train_residuals_: Optional[pd.Series] = None
        self._log_shift: float = 1.0   # for log(x + log_shift)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df_full: pd.DataFrame,
        auto_search: bool = True,
    ) -> "SARIMAXModel":
        """
        Fit SARIMAX on the training split.

        Parameters
        ----------
        df_full     : Full unscaled dataframe with SPLIT column.
        auto_search : If True, use pmdarima auto_arima for order selection.
                      If False, use the default order (1,0,1)(1,1,1,12).
        """
        train_mask = df_full["SPLIT"] == "train"
        df_train   = df_full.loc[train_mask].copy()

        # Log-transform target
        y_train = np.log1p(df_train["RAINFALL"].values.astype(float))

        # Build exogenous matrix — use columns that are present
        self.exog_cols_ = [c for c in SARIMAX_EXOG_COLS if c in df_train.columns]
        X_train = df_train[self.exog_cols_].values.astype(float)

        logger.info(
            f"[{self.name}] Training on {len(y_train):,} observations | "
            f"{len(self.exog_cols_)} exogenous features"
        )

        if auto_search:
            self.order_, self.seasonal_order_ = self._auto_search(y_train, X_train)
        else:
            self.order_          = (1, 0, 1)
            self.seasonal_order_ = (1, 1, 1, 12)

        logger.info(
            f"[{self.name}] Order: {self.order_} × Seasonal: {self.seasonal_order_}"
        )

        # Fit the final model with selected orders
        from statsmodels.tsa.statespace.sarimax import SARIMAX as sm_SARIMAX
        model = sm_SARIMAX(
            endog=y_train,
            exog=X_train,
            order=self.order_,
            seasonal_order=self.seasonal_order_,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit_ = model.fit(
            disp=False,
            maxiter=200,
            method="lbfgs",
        )

        # Extract and save in-sample residuals (log scale)
        self.train_residuals_ = pd.Series(
            self.model_fit_.resid,
            index=df_train.index,
            name="SARIMAX_train_residuals",
        )

        logger.info(
            f"[{self.name}] Fitted. AIC={self.model_fit_.aic:.2f} "
            f"BIC={self.model_fit_.bic:.2f} | "
            f"Residual std={self.train_residuals_.std():.4f}"
        )
        return self

    def _auto_search(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[tuple, tuple]:
        """
        Use pmdarima auto_arima to select p,d,q within a restricted grid.

        Restricted grid prevents combinatorial explosion on daily data:
          - max_p=2, max_q=1 (ACF/PACF showed lag-1 dominance)
          - seasonal m=12 (monthly proxy; m=365 is intractable)
          - stepwise=True for speed
        """
        import pmdarima as pm
        logger.info(f"[{self.name}] Running auto_arima parameter search...")

        # Subsample for speed if training set is large
        max_search_n = 3000
        if len(y) > max_search_n:
            rng = np.random.default_rng(CFG.project.random_seed)
            idx = np.sort(rng.choice(len(y), size=max_search_n, replace=False))
            y_search = y[idx]
            X_search = X[idx] if X.shape[1] > 0 else None
            logger.info(
                f"[{self.name}] Auto-search subsampled to {max_search_n} rows"
            )
        else:
            y_search = y
            X_search = X if X.shape[1] > 0 else None

        try:
            auto_model = pm.auto_arima(
                y_search,
                exogenous=X_search,
                start_p=1, max_p=2,
                start_q=0, max_q=1,
                d=0,                  # log-transform makes series stationary
                start_P=1, max_P=1,
                start_Q=1, max_Q=1,
                D=1,
                m=12,
                seasonal=True,
                stepwise=True,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True,
                n_jobs=1,
            )
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order
            logger.info(
                f"[{self.name}] auto_arima selected: "
                f"order={order}, seasonal_order={seasonal_order}"
            )
        except Exception as e:
            logger.warning(f"[{self.name}] auto_arima failed ({e}); using default orders")
            order, seasonal_order = (1, 0, 1), (1, 1, 1, 12)

        return order, seasonal_order

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        df_full: pd.DataFrame,
        split: str = "test",
    ) -> pd.DataFrame:
        """
        Generate one-step-ahead forecasts for val or test split.

        Uses the apply() method of the fitted result to perform in-sample
        prediction on the target split using the stored model parameters.
        This avoids re-fitting and ensures the same parameters apply.

        Returns DataFrame with columns [actual, predicted] on original scale.
        """
        if self.model_fit_ is None:
            raise RuntimeError("Call .fit() before .predict()")

        train_mask = df_full["SPLIT"] == "train"
        split_mask = df_full["SPLIT"] == split

        # We need the full history to initialise the state
        # Build endog and exog from train+target split
        val_test_mask = (df_full["SPLIT"] == "val") | (df_full["SPLIT"] == "test")

        # For prediction: use apply() to get in-sample fitted values for the
        # new data points, keeping model parameters fixed.
        df_split = df_full.loc[split_mask].copy()
        y_actual_log = np.log1p(df_split["RAINFALL"].values.astype(float))
        X_split = df_split[self.exog_cols_].values.astype(float)

        # Extend the model to the new data window
        try:
            result_applied = self.model_fit_.apply(
                endog=y_actual_log,
                exog=X_split,
                refit=False,
            )
            log_preds = result_applied.fittedvalues
        except Exception as e:
            logger.warning(
                f"[{self.name}] apply() failed ({e}); "
                "falling back to predict() on split indices"
            )
            # Fallback: use the training-fitted values' last state to predict
            n_train = train_mask.sum()
            n_pred  = split_mask.sum()
            fc = self.model_fit_.get_forecast(
                steps=n_pred,
                exog=X_split,
            )
            log_preds = fc.predicted_mean

        # Back-transform: expm1 reverses log1p
        preds_mm = np.expm1(log_preds)
        preds_mm = np.clip(preds_mm, 0.0, None)

        result_df = pd.DataFrame({
            "actual":    df_split["RAINFALL"].values,
            "predicted": preds_mm,
        }, index=df_split.index)

        return result_df

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        out = path or abs_path("outputs/models/sarimax_model.pkl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump({
                "model_fit":       self.model_fit_,
                "order":           self.order_,
                "seasonal_order":  self.seasonal_order_,
                "exog_cols":       self.exog_cols_,
                "train_residuals": self.train_residuals_,
            }, fh)
        logger.info(f"[{self.name}] Model saved → {out}")
        return out

    @classmethod
    def load(cls, path: Path) -> "SARIMAXModel":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls()
        obj.model_fit_        = data["model_fit"]
        obj.order_            = data["order"]
        obj.seasonal_order_   = data["seasonal_order"]
        obj.exog_cols_        = data["exog_cols"]
        obj.train_residuals_  = data["train_residuals"]
        logger.info(f"[{cls.name}] Model loaded from {path}")
        return obj


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_sarimax(
    df_full: pd.DataFrame,
    auto_search: bool = True,
) -> Tuple["SARIMAXModel", Dict[str, pd.DataFrame]]:
    """
    Fit SARIMAX and generate predictions for val and test splits.

    Returns
    -------
    (fitted_model, predictions_dict)
    predictions_dict keys: 'val', 'test'
    """
    model = SARIMAXModel()
    model.fit(df_full, auto_search=auto_search)
    model.save()

    out_dir = abs_path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for split in ["val", "test"]:
        pred_df = model.predict(df_full, split=split)
        pred_df.to_parquet(out_dir / f"sarimax_{split}.parquet")
        predictions[split] = pred_df
        logger.info(
            f"[SARIMAX] {split} predictions saved "
            f"({len(pred_df):,} rows)"
        )

    # Save training residuals for hybrid model
    if model.train_residuals_ is not None:
        model.train_residuals_.to_frame().to_parquet(
            out_dir / "sarimax_train_residuals.parquet"
        )
        logger.info("[SARIMAX] Training residuals saved for hybrid model")

    return model, predictions
