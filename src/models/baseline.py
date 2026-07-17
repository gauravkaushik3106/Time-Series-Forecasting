"""
baseline.py
===========
Naive baseline models for the Lucknow rainfall forecasting framework.

Two baselines are implemented:

1. PersistenceModel
   Forecast(t) = Observed(t−1)
   The simplest possible model: assume tomorrow looks like today.
   Strong lag-1 autocorrelation (r=0.492 from EDA) means this is a
   non-trivial baseline during the monsoon; any useful model must
   substantially beat it.

2. ClimatologyModel
   Forecast(t) = historical mean rainfall for the same day-of-year,
   computed exclusively from the training set.
   Captures the seasonal cycle without any day-to-day information.
   A model that cannot beat climatology has learned nothing beyond
   the mean annual cycle.

Both baselines:
- Operate on the original rainfall scale (mm/day)
- Clip predictions to ≥ 0 (physical floor)
- Expose a .predict(index) interface consistent with the model pipeline
- Save predictions to outputs/predictions/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import abs_path

logger = logging.getLogger(__name__)


class PersistenceModel:
    """
    Lag-1 persistence: predict tomorrow's rainfall = today's observed rainfall.

    At evaluation time, 'today' is the last observed value before the
    prediction date.  For a 1-step-ahead forecast on the test set, this
    means using the test set's own previous-day value — which is always
    a past observation, never future information.
    """

    name: str = "Persistence"

    def fit(self, train: pd.Series) -> "PersistenceModel":
        """No parameters to fit; store training tail for val/test continuity."""
        self._last_train_value = float(train.iloc[-1])
        logger.info(f"[{self.name}] Fitted (no-op). Last train value: {self._last_train_value:.3f} mm")
        return self

    def predict(self, series: pd.Series) -> pd.Series:
        """
        Generate persistence forecasts for the given series.

        For each time t in series, prediction = observed value at t−1.
        The first prediction uses the last value of the preceding context
        (stored during fit or passed as the series's own lag-1).

        Parameters
        ----------
        series : Full observed rainfall series for the evaluation period,
                 in chronological order.  The series itself is used only
                 to derive the lag-1 sequence; no future values are seen.
        """
        # shift(1) produces: prediction[t] = actual[t-1]
        # The NaN at position 0 is filled with the last training value.
        preds = series.shift(1)
        preds.iloc[0] = self._last_train_value
        preds = preds.clip(lower=0.0)
        return preds

    def predict_index(
        self,
        series: pd.Series,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Predict only for the rows corresponding to `index`."""
        full_preds = self.predict(series)
        return full_preds.loc[index]


class ClimatologyModel:
    """
    Day-of-year climatology: predict the historical mean for each calendar day.

    The climatological mean is computed exclusively from the training set.
    Leap-day (DOY 366) is assigned the same value as DOY 365.
    """

    name: str = "Climatology"

    def fit(self, train: pd.Series) -> "ClimatologyModel":
        """
        Compute per-DOY mean from training data.

        Uses a 15-day centred smoothing window over DOY climatology to
        avoid noisy estimates for rare DOYs (e.g., DOY 365 in non-leap years).
        """
        doy_means = train.groupby(train.index.dayofyear).mean()

        # Pad and smooth for DOY stability: circular pad of 15 days, then smooth
        n = len(doy_means)
        padded = pd.concat([
            doy_means.iloc[-15:].set_axis(range(-14, 1)),
            doy_means,
            doy_means.iloc[:15].set_axis(range(n + 1, n + 16)),
        ])
        smoothed = padded.rolling(window=15, center=True, min_periods=1).mean()
        # Re-extract original DOY range
        self._doy_climatology = smoothed.loc[doy_means.index].clip(lower=0.0)

        # Handle DOY 366 (leap day) — assign DOY 365 value
        if 366 not in self._doy_climatology.index:
            self._doy_climatology[366] = self._doy_climatology.get(365, 0.0)

        logger.info(
            f"[{self.name}] Fitted. DOY climatology range: "
            f"{self._doy_climatology.min():.3f}–{self._doy_climatology.max():.3f} mm"
        )
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Return the climatological mean for each date in `index`.

        Parameters
        ----------
        index : DatetimeIndex of dates to predict.
        """
        doys  = index.dayofyear
        preds = np.array([
            self._doy_climatology.get(d, self._doy_climatology.mean())
            for d in doys
        ], dtype=float)
        return pd.Series(preds, index=index, name=self.name).clip(lower=0.0)


# ---------------------------------------------------------------------------
# Runner: fit both baselines on train, predict on val+test, save outputs
# ---------------------------------------------------------------------------

def run_baselines(
    df_full: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Fit and evaluate both baseline models.

    Parameters
    ----------
    df_full : Full unscaled feature dataframe with SPLIT column and RAINFALL.

    Returns
    -------
    dict mapping model name → DataFrame with columns [actual, predicted].
    """
    train_mask = df_full["SPLIT"] == "train"
    val_mask   = df_full["SPLIT"] == "val"
    test_mask  = df_full["SPLIT"] == "test"

    rain_train = df_full.loc[train_mask, "RAINFALL"]
    rain_val   = df_full.loc[val_mask,   "RAINFALL"]
    rain_test  = df_full.loc[test_mask,  "RAINFALL"]

    out_dir = abs_path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for split_name, rain_target, preceding in [
        ("val",  rain_val,  rain_train),
        ("test", rain_test, pd.concat([rain_train, rain_val])),
    ]:
        # Build the continuity series: preceding context + target window
        full_series = pd.concat([preceding, rain_target]).sort_index()

        # --- Persistence ---
        pers = PersistenceModel()
        pers.fit(rain_train)
        pers_preds = pers.predict(full_series).loc[rain_target.index]

        pers_df = pd.DataFrame({
            "actual":    rain_target.values,
            "predicted": pers_preds.values,
        }, index=rain_target.index)
        pers_df.to_parquet(out_dir / f"persistence_{split_name}.parquet")

        # --- Climatology ---
        clim = ClimatologyModel()
        clim.fit(rain_train)
        clim_preds = clim.predict(rain_target.index)

        clim_df = pd.DataFrame({
            "actual":    rain_target.values,
            "predicted": clim_preds.values,
        }, index=rain_target.index)
        clim_df.to_parquet(out_dir / f"climatology_{split_name}.parquet")

        results[f"persistence_{split_name}"] = pers_df
        results[f"climatology_{split_name}"] = clim_df

        logger.info(f"Baselines predicted and saved for {split_name} split")

    return results
