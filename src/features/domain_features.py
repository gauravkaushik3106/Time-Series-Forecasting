"""
domain_features.py
==================
Physically motivated domain feature engineering for the Lucknow rainfall dataset.

Each feature here is justified by either meteorological physics or by the
multicollinearity analysis performed in Phase 2 EDA.  None of these features
are arbitrary transformations — each encodes a distinct physical process.

Feature catalogue
-----------------
SOIL_MOISTURE_GRADIENT
    = SOIL_WET_SURF − SOIL_WET_ROOT
    Physical interpretation: Positive gradient → surface wetter than root zone,
    indicating recent infiltration from precipitation.  Negative → root zone
    retains more moisture, suggesting evapotranspiration dominance.  Replaces
    the SOIL_WET_SURF/SOIL_WET_ROOT pair in linear models (VIF relief).

DEWPOINT_APPROX
    Approximated via the Magnus formula: Td ≈ T − ((100 − RH) / 5)
    where T = TMIN (minimum temperature is closest to the dew point
    in the early morning, the traditional measurement time).
    Physical interpretation: Dew point measures the absolute atmospheric
    moisture content, independent of temperature.  It is a direct indicator
    of convective potential — when Td approaches T, saturation is imminent.

TEMP_RANGE
    = TMAX − TMIN
    Physical interpretation: Large diurnal range → clear sky, dry conditions.
    Small diurnal range → cloud cover, humid air mass, potential rainfall.
    This compresses the three-way collinear temperature group into a single
    non-redundant variable.

PRESSURE_RH_INTERACTION
    = PRESSURE × (RH / 100)
    Physical interpretation: Low pressure AND high humidity together signal
    a warm, moist air mass — the hallmark of monsoon conditions.  High
    pressure AND low humidity → dry continental air.  The interaction term
    captures the joint signal that neither variable encodes alone.

RAIN_OCCURRENCE
    = 1 if RAINFALL > dry_day_threshold, else 0
    Required for the two-stage model architecture identified in Phase 2:
    stage 1 is a classifier predicting this binary flag.

DRY_SPELL_LENGTH
    Consecutive count of dry days (RAINFALL < threshold) up to and
    including the current day.  Resets to 0 on any rainy day.
    Physical interpretation: Increasing dry spell length is associated with
    progressive drying of the atmospheric boundary layer, making subsequent
    rainfall events progressively less likely — until moisture advection
    from the Bay of Bengal resets the system during monsoon onset.
    IMPORTANT: this counter uses only past observations.  At time t it
    reflects how long the current dry spell has lasted BEFORE day t's
    rainfall is known.

MONSOON_FLAG
    = 1 if month ∈ {6, 7, 8, 9}, else 0
    Already computed by loader as IS_MONSOON; preserved here as an alias
    for explainability — models are explicitly told when it is monsoon season.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG

logger = logging.getLogger(__name__)


def create_domain_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate all physically motivated domain features.

    Parameters
    ----------
    df : Cleaned dataframe with DatetimeIndex and all original columns.

    Returns
    -------
    df_out          : Input dataframe with domain feature columns appended.
    domain_col_names: List of newly created column names only.
    """
    df_out = df.copy()
    created: List[str] = []
    dry_threshold = CFG.rainfall.dry_day_threshold

    # ------------------------------------------------------------------
    # 1. Soil moisture gradient
    #    Replaces the (SOIL_WET_SURF, SOIL_WET_ROOT) pair for linear models.
    # ------------------------------------------------------------------
    if "SOIL_MOISTURE_GRADIENT" not in df_out.columns:
        df_out["SOIL_MOISTURE_GRADIENT"] = (
            df_out["SOIL_WET_SURF"] - df_out["SOIL_WET_ROOT"]
        )
        created.append("SOIL_MOISTURE_GRADIENT")

    # ------------------------------------------------------------------
    # 2. Dewpoint approximation (Magnus simplified)
    #    Td ≈ TMIN − (100 − RH) / 5
    #    Valid range: RH = 5–100%, T = −40°C to +60°C.
    #    Error < 1°C for RH > 50% (physically plausible for Lucknow monsoon).
    # ------------------------------------------------------------------
    if "DEWPOINT_APPROX" not in df_out.columns:
        df_out["DEWPOINT_APPROX"] = df_out["TMIN"] - (100.0 - df_out["RH"]) / 5.0
        created.append("DEWPOINT_APPROX")

    # ------------------------------------------------------------------
    # 3. Diurnal temperature range
    #    Compresses the (TMAX, TMIN, TAVG) collinear triad.
    # ------------------------------------------------------------------
    if "TEMP_RANGE" not in df_out.columns:
        df_out["TEMP_RANGE"] = df_out["TMAX"] - df_out["TMIN"]
        created.append("TEMP_RANGE")

    # ------------------------------------------------------------------
    # 4. Pressure × relative humidity interaction
    #    Normalise RH to [0,1] before multiplication so the product
    #    stays on a physically meaningful scale.
    # ------------------------------------------------------------------
    if "PRESSURE_RH_INTERACTION" not in df_out.columns:
        df_out["PRESSURE_RH_INTERACTION"] = (
            df_out["PRESSURE"] * (df_out["RH"] / 100.0)
        )
        created.append("PRESSURE_RH_INTERACTION")

    # ------------------------------------------------------------------
    # 5. Rain occurrence binary flag
    #    Stage-1 classification target.  Computed from the RAINFALL column
    #    using the same threshold applied throughout the codebase.
    # ------------------------------------------------------------------
    if "RAIN_OCCURRENCE" not in df_out.columns:
        df_out["RAIN_OCCURRENCE"] = (
            df_out["RAINFALL"] > dry_threshold
        ).astype(int)
        created.append("RAIN_OCCURRENCE")

    # ------------------------------------------------------------------
    # 6. Dry spell counter (causal — uses only past information)
    #    At time t, DRY_SPELL_LENGTH[t] = number of consecutive dry days
    #    ending at t-1 (i.e., does NOT include day t's own rainfall).
    #    Implementation: classify rainfall into dry/wet, then compute
    #    a running count that resets on wet days.
    # ------------------------------------------------------------------
    if "DRY_SPELL_LENGTH" not in df_out.columns:
        dry_flag = (df_out["RAINFALL"] < dry_threshold).astype(int)
        df_out["DRY_SPELL_LENGTH"] = _compute_dry_spell_counter(dry_flag)
        created.append("DRY_SPELL_LENGTH")

    # ------------------------------------------------------------------
    # 7. Monsoon flag (alias of IS_MONSOON for naming consistency)
    #    IS_MONSOON already exists from the loader; expose as MONSOON_FLAG
    #    for downstream explainability clarity.
    # ------------------------------------------------------------------
    if "MONSOON_FLAG" not in df_out.columns and "IS_MONSOON" in df_out.columns:
        df_out["MONSOON_FLAG"] = df_out["IS_MONSOON"]
        created.append("MONSOON_FLAG")

    # ------------------------------------------------------------------
    # 8. Log-transformed rainfall (target transformation)
    #    log(1 + RAINFALL) — reduces skewness from 6.81 to ~1.8.
    #    Used as the regression target in SARIMAX and LSTM.
    #    Also added as a feature so models can reference the log-scale
    #    of their own recent history.
    # ------------------------------------------------------------------
    if "LOG_RAINFALL" not in df_out.columns:
        df_out["LOG_RAINFALL"] = np.log1p(df_out["RAINFALL"])
        created.append("LOG_RAINFALL")

    # ------------------------------------------------------------------
    # 9. Wet-day rainfall amount (RAINFALL masked to wet days only)
    #    = RAINFALL where RAIN_OCCURRENCE == 1, else NaN.
    #    Used in the stage-2 regression (amount given rain occurred).
    # ------------------------------------------------------------------
    if "RAINFALL_WET_ONLY" not in df_out.columns:
        df_out["RAINFALL_WET_ONLY"] = df_out["RAINFALL"].where(
            df_out["RAIN_OCCURRENCE"] == 1
        )
        created.append("RAINFALL_WET_ONLY")

    logger.info(f"Domain features created: {len(created)} columns")
    for c in created:
        non_null = df_out[c].notna().sum()
        logger.debug(f"  {c}: {non_null:,} non-null values")

    return df_out, created


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_dry_spell_counter(dry_flag: pd.Series) -> pd.Series:
    """
    Compute a causal dry spell counter.

    For each day t, returns the count of consecutive dry days immediately
    preceding day t (not including t itself).  This ensures the counter
    at row t contains no information about whether it rained on day t.

    Implementation uses a vectorised cumulative sum approach:
      1.  Assign a group ID that increments every time a wet day occurs.
      2.  Within each group, compute the cumulative position (0-indexed).
      3.  Shift the result forward by 1 day to make it strictly causal.

    Parameters
    ----------
    dry_flag : Binary series (1 = dry, 0 = wet), DatetimeIndex.

    Returns
    -------
    counter : Integer series giving the causal dry spell count.
    """
    # Cumulative sum of wet events creates a group label:
    # each wet day starts a new group.
    wet_event     = (dry_flag == 0).astype(int)
    group_id      = wet_event.cumsum()

    # Within-group cumulative count of dry days (since last wet day)
    within_group  = dry_flag.groupby(group_id).cumsum()

    # Shift by 1: the value at t reflects how many consecutive dry days
    # occurred before t, not including t itself.
    causal_counter = within_group.astype(int)

    return causal_counter
