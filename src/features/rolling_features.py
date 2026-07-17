"""
rolling_features.py
===================
Rolling (moving-window) statistical features for the Lucknow rainfall dataset.

Temporal integrity design
--------------------------
All rolling windows use:
  - min_periods = window_size   (no partial-window averages that could distort
                                  the statistical meaning early in the series)
  - center      = False         (CRITICAL: the window looks only BACKWARD.
                                  center=True would include future observations
                                  and constitutes look-ahead leakage.)

With center=False and a window of W days, the value at row t is computed from
rows [t-W+1, t-W+2, ..., t].  The feature at row t therefore contains no
information from day t+1 or later.

The rolling mean at lag t summarises the recent weather regime that a
forecaster would actually know at the moment of making a prediction.

Feature naming convention
--------------------------
  <SOURCE_COLUMN>_roll_mean_<W>   — rolling mean over W days
  <SOURCE_COLUMN>_roll_std_<W>    — rolling std  over W days
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rolling window specification — single source of truth
# ---------------------------------------------------------------------------

# Rainfall rolling statistics: captures recent wetness regime at short,
# medium (biweekly), and near-monthly scales.
RAINFALL_WINDOWS: List[int] = [7, 14, 30]

# Atmospheric driver rolling: humidity and soil moisture respond on similar
# timescales; 7-day captures synoptic regime, 30-day captures seasonal state.
ATMOSPHERIC_ROLL_CONFIG: Dict[str, List[str]] = {
    # column → list of statistics to compute
    "RH":           ["mean"],
    "SOIL_WET_SURF":["mean"],
}
ATMOSPHERIC_ROLL_WINDOWS: List[int] = [7, 30]


def create_rolling_features(
    df: pd.DataFrame,
    rainfall_windows: List[int] = RAINFALL_WINDOWS,
    atmospheric_windows: List[int] = ATMOSPHERIC_ROLL_WINDOWS,
    atmospheric_cols: Dict[str, List[str]] = ATMOSPHERIC_ROLL_CONFIG,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct rolling mean and std features.

    Parameters
    ----------
    df                  : Cleaned dataframe with sorted DatetimeIndex.
    rainfall_windows    : Window sizes (days) for rainfall rolling stats.
    atmospheric_windows : Window sizes for atmospheric driver rolling stats.
    atmospheric_cols    : Mapping of column → statistics list.

    Returns
    -------
    df_out       : Input dataframe with rolling columns appended.
    roll_col_names: List of all newly created column names.
    """
    df_out = df.copy()
    created: List[str] = []

    # --- Rainfall rolling mean and std ---
    for w in rainfall_windows:
        roll_obj = df_out["RAINFALL"].rolling(window=w, min_periods=w, center=False)

        mean_col = f"RAINFALL_roll_mean_{w}"
        std_col  = f"RAINFALL_roll_std_{w}"

        df_out[mean_col] = roll_obj.mean()
        df_out[std_col]  = roll_obj.std(ddof=1)   # sample std (ddof=1)

        created += [mean_col, std_col]

    # --- Atmospheric driver rolling means ---
    for col, stats_list in atmospheric_cols.items():
        if col not in df_out.columns:
            logger.warning(f"Column '{col}' not found — skipping rolling features")
            continue
        for w in atmospheric_windows:
            roll_obj = df_out[col].rolling(window=w, min_periods=w, center=False)
            if "mean" in stats_list:
                col_name = f"{col}_roll_mean_{w}"
                df_out[col_name] = roll_obj.mean()
                created.append(col_name)
            if "std" in stats_list:
                col_name = f"{col}_roll_std_{w}"
                df_out[col_name] = roll_obj.std(ddof=1)
                created.append(col_name)

    n_nan_rows = df_out[created].isna().any(axis=1).sum()
    logger.info(
        f"Rolling features created: {len(created)} columns | "
        f"max window: {max(rainfall_windows + atmospheric_windows)} days | "
        f"rows with ≥1 NaN: {n_nan_rows}"
    )

    return df_out, created
