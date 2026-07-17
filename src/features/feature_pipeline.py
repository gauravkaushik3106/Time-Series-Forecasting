"""
feature_pipeline.py
===================
End-to-end feature engineering pipeline for the Lucknow rainfall dataset.

Pipeline stages
---------------
1.  Load processed data (from Phase 2 Parquet output)
2.  Create lag features                 (lag_features.py)
3.  Create rolling statistics           (rolling_features.py)
4.  Create temporal / cyclical features (temporal_features.py)
5.  Create domain features              (domain_features.py)
6.  Drop warm-up rows (NaN from lags / rolling windows)
7.  Run feature selection               (feature_selector.py)
8.  Fit scalers on training split only  (prevents leakage)
9.  Scale features and save split datasets
10. Save feature metadata, importance tables, and engineering report

Scaling philosophy
------------------
StandardScaler is fit ONLY on the training split and applied to val/test.
This mirrors production: at deployment time, the scaler parameters are
frozen from training and applied blindly to new data.
Fitting on val/test (or the full dataset) would leak distributional
information from future observations into the scaler, inflating performance.

Two scalers are maintained separately:
  - feature_scaler  : scales the predictor columns (X)
  - target_scaler   : scales LOG_RAINFALL (y) for neural networks

Outputs
-------
outputs/features/
    full_features.parquet      — all features, unscaled, all splits marked
    train_features.parquet     — training split, scaled, selected features
    val_features.parquet       — validation split, scaled, selected features
    test_features.parquet      — test split, scaled, selected features
    train_features_linear.parquet — linear-model feature set
    val_features_linear.parquet
    test_features_linear.parquet
    scaler_params.json         — mean/std for reproducible inference

outputs/reports/
    03_feature_selection_report.md
    feature_summary_table.csv
    feature_importance_table.csv
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.preprocessing.loader import load_data
from src.features.lag_features import create_lag_features, get_max_lag
from src.features.rolling_features import create_rolling_features, RAINFALL_WINDOWS
from src.features.temporal_features import create_temporal_features
from src.features.domain_features import create_domain_features
from src.features.feature_selector import (
    run_feature_selection,
    FeatureSelectionResult,
    LEAKAGE_COLUMNS,
    PROTECTED_COLUMNS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Columns that are targets or structural — never scaled
# ---------------------------------------------------------------------------

NEVER_SCALE: set = {
    "RAINFALL",
    "LOG_RAINFALL",
    "RAIN_OCCURRENCE",
    "RAINFALL_WET_ONLY",
    "IS_MONSOON",
    "MONSOON_FLAG",
    "SEASON",
    "YEAR",
    "MONTH",
    "QUARTER",
    "DAY_OF_YEAR",
    "WEEK_OF_YEAR",
    "DAY_OF_WEEK",
    "IS_WEEKEND",
    "SEASON_CODE",
    "DRY_SPELL_LENGTH",
}


class FeaturePipeline:
    """
    Orchestrates the complete feature engineering workflow.

    Parameters
    ----------
    raw_path : Override default raw data path from config.
    """

    def __init__(self, raw_path: Optional[Path] = None) -> None:
        self.raw_path = raw_path
        self.df_full_: Optional[pd.DataFrame]      = None
        self.df_engineered_: Optional[pd.DataFrame] = None
        self.selection_result_: Optional[FeatureSelectionResult] = None
        self.feature_scaler_: Optional[StandardScaler] = None
        self.target_scaler_: Optional[StandardScaler]  = None
        self.split_indices_: Dict[str, pd.DatetimeIndex] = {}
        self.all_feature_cols_: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "FeaturePipeline":
        """Execute the complete pipeline. Returns self for chaining."""
        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING PIPELINE — START")
        logger.info("=" * 70)

        # Stage 1: Load data
        self._load()

        # Stage 2–5: Feature creation (on full dataset for temporal integrity)
        self._create_features()

        # Stage 6: Drop warm-up rows
        self._drop_warmup_rows()

        # Stage 7: Re-apply chronological split on reduced index
        self._apply_split()

        # Stage 8: Feature selection (fit on training set only)
        self._select_features()

        # Stage 9: Scaling (fit on training set only)
        self._scale_and_save()

        # Stage 10: Generate reports and tables
        self._generate_outputs()

        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING PIPELINE — COMPLETE")
        logger.info(f"  Engineered features : {len(self.all_feature_cols_)}")
        logger.info(f"  Linear feature set  : {len(self.selection_result_.features_linear)}")
        logger.info(f"  ML feature set      : {len(self.selection_result_.features_ml)}")
        logger.info("=" * 70)
        return self

    # ------------------------------------------------------------------
    # Stage 1: Load processed data
    # ------------------------------------------------------------------

    def _load(self) -> None:
        logger.info("[1/8] Loading data")
        df_full, train, val, test = load_data(raw_path=self.raw_path)

        # Store split boundary dates for re-application after warm-up drop
        self.split_indices_ = {
            "train_start": train.index[0],
            "train_end":   train.index[-1],
            "val_start":   val.index[0],
            "val_end":     val.index[-1],
            "test_start":  test.index[0],
            "test_end":    test.index[-1],
        }
        self.df_full_ = df_full
        logger.info(f"  Loaded {len(df_full):,} rows")

    # ------------------------------------------------------------------
    # Stages 2–5: Feature creation
    # ------------------------------------------------------------------

    def _create_features(self) -> None:
        logger.info("[2/8] Creating features")
        df = self.df_full_.copy()
        all_new: List[str] = []

        # Lag features
        df, lag_cols = create_lag_features(df)
        all_new += lag_cols

        # Rolling features
        df, roll_cols = create_rolling_features(df)
        all_new += roll_cols

        # Temporal / cyclical features
        df, temp_cols = create_temporal_features(df)
        all_new += temp_cols

        # Domain features (physically motivated)
        df, domain_cols = create_domain_features(df)
        all_new += domain_cols

        self.df_engineered_ = df

        # All feature columns = new + original meteorological features
        original_met_features = list(CFG.schema.feature_columns)
        all_candidate_features = original_met_features + all_new

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for c in all_candidate_features:
            if c not in seen and c in df.columns:
                seen.add(c)
                deduped.append(c)

        self.all_feature_cols_ = deduped
        logger.info(f"  Total feature columns: {len(deduped)}")

    # ------------------------------------------------------------------
    # Stage 6: Drop warm-up rows with NaN lag/rolling values
    # ------------------------------------------------------------------

    def _drop_warmup_rows(self) -> None:
        """
        Drop the warm-up period at the head of the dataset where lag and
        rolling features contain NaN (because there is insufficient history).

        The warm-up period = max(lag period, rolling window) = 30 days.
        These rows cannot be used for model training or evaluation because
        their feature vectors are incomplete.

        NaN rows in val/test are also dropped.  In production, the deployed
        model would always have a full history buffer available, so this does
        not represent a real operational limitation.
        """
        max_warmup = max(get_max_lag(), max(RAINFALL_WINDOWS))
        df = self.df_engineered_

        lag_roll_cols = [
            c for c in self.all_feature_cols_
            if "_lag" in c or "_roll_" in c
        ]

        before = len(df)
        df = df.dropna(subset=lag_roll_cols)
        after = len(df)
        self.df_engineered_ = df
        logger.info(
            f"[3/8] Warm-up drop: {before - after} rows removed "
            f"(max warm-up = {max_warmup} days) → {after:,} rows remaining"
        )

    # ------------------------------------------------------------------
    # Stage 7: Re-apply chronological split
    # ------------------------------------------------------------------

    def _apply_split(self) -> None:
        """
        Restore train/val/test splits from the stored boundary dates.
        The warm-up drop may have consumed some early training rows.
        """
        logger.info("[4/8] Re-applying chronological split")
        df = self.df_engineered_
        si = self.split_indices_

        self.train_ = df.loc[si["train_start"]: si["train_end"]].copy()
        self.val_   = df.loc[si["val_start"]:   si["val_end"]].copy()
        self.test_  = df.loc[si["test_start"]:  si["test_end"]].copy()

        logger.info(
            f"  Train: {len(self.train_):,} | "
            f"Val: {len(self.val_):,} | "
            f"Test: {len(self.test_):,}"
        )

        # Sanity: no overlap
        assert self.train_.index[-1] < self.val_.index[0], "Train/val overlap!"
        assert self.val_.index[-1]   < self.test_.index[0], "Val/test overlap!"

    # ------------------------------------------------------------------
    # Stage 8: Feature selection
    # ------------------------------------------------------------------

    def _select_features(self) -> None:
        """Run feature selection exclusively on the training split."""
        logger.info("[5/8] Running feature selection on training split")

        # Exclude columns that are targets, structural, or already-dropped
        candidates = [
            c for c in self.all_feature_cols_
            if c in self.train_.columns
        ]

        self.selection_result_ = run_feature_selection(
            df_train=self.train_,
            feature_cols=candidates,
            target_col="LOG_RAINFALL",
        )
        logger.info(self.selection_result_.summary())

    # ------------------------------------------------------------------
    # Stage 9: Scaling + save
    # ------------------------------------------------------------------

    def _scale_and_save(self) -> None:
        """
        Fit StandardScaler on training features, transform all splits,
        and persist Parquet files and scaler parameters.
        """
        logger.info("[6/8] Scaling and saving feature datasets")
        out_dir = abs_path("outputs/features")
        out_dir.mkdir(parents=True, exist_ok=True)

        sr = self.selection_result_

        # --- Save full unscaled feature dataframe (all rows, split marker) ---
        # Build deduplicated column list — all_feature_cols_ already includes
        # LOG_RAINFALL, RAIN_OCCURRENCE, RAINFALL_WET_ONLY from domain_features;
        # also include RAINFALL explicitly if not already present.
        extra_targets = [
            c for c in ["RAINFALL", "LOG_RAINFALL", "RAIN_OCCURRENCE"]
            if c not in self.all_feature_cols_ and c in self.df_engineered_.columns
        ]
        full_cols_dedup = list(dict.fromkeys(self.all_feature_cols_ + extra_targets))
        df_full_feat = self.df_engineered_[
            [c for c in full_cols_dedup if c in self.df_engineered_.columns]
        ].copy()
        df_full_feat["SPLIT"] = "unknown"
        df_full_feat.loc[self.train_.index, "SPLIT"] = "train"
        df_full_feat.loc[self.val_.index,   "SPLIT"] = "val"
        df_full_feat.loc[self.test_.index,  "SPLIT"] = "test"
        df_full_feat.to_parquet(out_dir / "full_features.parquet", compression="snappy")
        logger.info(f"  Saved full_features.parquet ({len(df_full_feat):,} rows)")

        # --- Scale and save ML feature set (XGB / LSTM / GRU) ---
        self._fit_and_save_scaled_splits(
            feature_cols=sr.features_ml,
            suffix="ml",
            out_dir=out_dir,
        )

        # --- Scale and save linear feature set (SARIMAX) ---
        self._fit_and_save_scaled_splits(
            feature_cols=sr.features_linear,
            suffix="linear",
            out_dir=out_dir,
        )

        logger.info("  Feature datasets saved")

    def _fit_and_save_scaled_splits(
        self,
        feature_cols: List[str],
        suffix: str,
        out_dir: Path,
    ) -> None:
        """
        Fit scaler on training split, transform all splits, save Parquet files.

        The scaler is fit ONLY on training data.  Val/test use the same
        mean and std estimated from training — exactly as in deployment.
        """
        # Columns to scale: numeric, not in NEVER_SCALE
        scalable = [
            c for c in feature_cols
            if c in self.train_.columns
            and pd.api.types.is_numeric_dtype(self.train_[c])
            and c not in NEVER_SCALE
        ]
        non_scalable = [c for c in feature_cols if c not in scalable and c in self.train_.columns]

        target_cols = ["RAINFALL", "LOG_RAINFALL", "RAIN_OCCURRENCE"]

        scaler = StandardScaler()
        scaler.fit(self.train_[scalable])

        scaler_params = {
            "suffix":   suffix,
            "features": scalable,
            "mean":     scaler.mean_.tolist(),
            "std":      scaler.scale_.tolist(),
        }
        params_path = out_dir / f"scaler_params_{suffix}.json"
        with open(params_path, "w") as fh:
            json.dump(scaler_params, fh, indent=2)

        for split_name, split_df in [
            ("train", self.train_),
            ("val",   self.val_),
            ("test",  self.test_),
        ]:
            out_df = split_df[
                [c for c in scalable + non_scalable + target_cols if c in split_df.columns]
            ].copy()

            # Apply scaling
            available_scalable = [c for c in scalable if c in out_df.columns]
            if available_scalable:
                out_df[available_scalable] = scaler.transform(out_df[available_scalable])

            fname = out_dir / f"{split_name}_features_{suffix}.parquet"
            out_df.to_parquet(fname, compression="snappy")
            logger.info(f"  Saved {fname.name} ({len(out_df):,} rows, {len(out_df.columns)} cols)")

        # Store scalers for later inference
        if suffix == "ml":
            self.feature_scaler_ = scaler

    # ------------------------------------------------------------------
    # Stage 10: Reports and tables
    # ------------------------------------------------------------------

    def _generate_outputs(self) -> None:
        logger.info("[7/8] Generating reports and tables")

        sr = self.selection_result_
        reports_dir = abs_path("outputs/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        features_dir = abs_path("outputs/features")

        # --- Feature summary table ---
        summary_rows = []
        for col in self.all_feature_cols_:
            if col not in self.df_engineered_.columns:
                continue
            series = self.df_engineered_[col]
            shap_row = sr.shap_importance.loc[
                sr.shap_importance["Feature"] == col
            ]
            xgb_row = sr.xgb_importance.loc[
                sr.xgb_importance["Feature"] == col
            ]
            vif_row = sr.vif_df.loc[sr.vif_df["Feature"] == col]

            summary_rows.append({
                "Feature":       col,
                "Type":          _infer_feature_type(col),
                "Non_Null":      int(series.notna().sum()),
                "Mean":          round(float(series.mean()), 4) if series.notna().any() else None,
                "Std":           round(float(series.std()), 4)  if series.notna().any() else None,
                "Min":           round(float(series.min()), 4)  if series.notna().any() else None,
                "Max":           round(float(series.max()), 4)  if series.notna().any() else None,
                "VIF":           float(vif_row["VIF"].values[0]) if len(vif_row) else None,
                "VIF_Severity":  vif_row["Severity"].values[0]   if len(vif_row) else "N/A",
                "SHAP_Rank":     int(shap_row["SHAP_Rank"].values[0])    if len(shap_row) else None,
                "SHAP_MeanAbs":  round(float(shap_row["SHAP_MeanAbs"].values[0]), 6) if len(shap_row) else None,
                "XGB_Rank":      int(xgb_row["XGB_Rank"].values[0])      if len(xgb_row) else None,
                "XGB_Gain_Frac": round(float(xgb_row["XGB_Gain_Frac"].values[0]), 6) if len(xgb_row) else None,
                "In_Linear_Set": col in sr.features_linear,
                "In_ML_Set":     col in sr.features_ml,
                "Corr_Dropped":  col in sr.corr_dropped,
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = features_dir / "feature_summary_table.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"  feature_summary_table.csv saved ({len(summary_df)} rows)")

        # --- Feature importance table (SHAP + XGB merged) ---
        imp_df = sr.shap_importance.merge(
            sr.xgb_importance[["Feature", "XGB_Gain", "XGB_Gain_Frac", "XGB_Rank"]],
            on="Feature", how="outer"
        ).sort_values("SHAP_Rank", na_position="last")
        imp_path = features_dir / "feature_importance_table.csv"
        imp_df.to_csv(imp_path, index=False)
        logger.info(f"  feature_importance_table.csv saved")

        # --- Feature selection report ---
        report_text = self._build_selection_report(summary_df, sr)
        report_path = reports_dir / "03_feature_selection_report.md"
        report_path.write_text(report_text, encoding="utf-8")
        logger.info(f"  03_feature_selection_report.md saved")

        logger.info("[8/8] All outputs saved")

    def _build_selection_report(
        self,
        summary_df: pd.DataFrame,
        sr: FeatureSelectionResult,
    ) -> str:
        """Compose the Markdown feature selection report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Lucknow Rainfall Framework — Feature Engineering Report",
            f"Generated: {now}",
            "",
            "---",
            "",
            "## 1. Feature Engineering Summary",
            "",
            f"| Category | Count |",
            f"|---|---|",
        ]

        type_counts = summary_df["Type"].value_counts()
        for ftype, cnt in type_counts.items():
            lines.append(f"| {ftype} | {cnt} |")

        lines += [
            f"| **Total** | **{len(summary_df)}** |",
            "",
            "---",
            "",
            "## 2. Feature Sets",
            "",
            f"| Set | Purpose | Count |",
            f"|---|---|---|",
            f"| All features | Complete engineered set | {len(sr.features_all)} |",
            f"| Linear set (SARIMAX) | VIF + correlation filtered | {len(sr.features_linear)} |",
            f"| ML set (XGB / LSTM / GRU) | SHAP top-{30} ranked | {len(sr.features_ml)} |",
            "",
            "### Linear Feature Set (SARIMAX)",
            "",
            "Features selected after removing severe VIF, high pairwise correlation,",
            "and near-zero importance.  Appropriate for models that assume feature independence.",
            "",
            "```",
        ]
        for i, f in enumerate(sorted(sr.features_linear), 1):
            lines.append(f"  {i:2d}. {f}")
        lines += [
            "```",
            "",
            "### ML Feature Set (XGBoost / LSTM / GRU)",
            "",
            "Top features ranked by SHAP mean |value|.",
            "Tree-based and neural models are multicollinearity-tolerant;",
            "the full set of informative features is retained.",
            "",
            "```",
        ]
        for i, f in enumerate(sr.features_ml, 1):
            lines.append(f"  {i:2d}. {f}")
        lines += [
            "```",
            "",
            "---",
            "",
            "## 3. VIF Analysis",
            "",
            "| Feature | VIF | Severity |",
            "|---|---|---|",
        ]
        for _, row in sr.vif_df.head(20).iterrows():
            tag = "⚠" if row["Severity"] != "Low" else "✓"
            lines.append(f"| {row['Feature']} | {row['VIF']:.1f} | {tag} {row['Severity']} |")

        lines += [
            "",
            "---",
            "",
            "## 4. SHAP Feature Importance (Top 20)",
            "",
            "| Rank | Feature | Mean |SHAP| | % of Total |",
            "|---|---|---|---|",
        ]
        for _, row in sr.shap_importance.head(20).iterrows():
            lines.append(
                f"| {int(row['SHAP_Rank'])} | {row['Feature']} "
                f"| {row['SHAP_MeanAbs']:.5f} "
                f"| {row['SHAP_MeanAbsFrac']*100:.2f}% |"
            )

        lines += [
            "",
            "---",
            "",
            "## 5. Correlation Filter",
            "",
            f"{len(sr.corr_dropped)} feature(s) flagged for removal due to pairwise |r| > 0.92:",
            "",
        ]
        if sr.corr_dropped:
            for f in sr.corr_dropped:
                lines.append(f"- `{f}`")
        else:
            lines.append("_(none beyond threshold)_")

        lines += [
            "",
            "---",
            "",
            "## 6. Feature Engineering Justification",
            "",
            "| Feature Group | Physical Justification |",
            "|---|---|",
            "| RAINFALL_lag1..30 | Autocorrelation: lag-1 r=0.492; 7–30 day memory window |",
            "| RH/PRESSURE/CLOUD_lag1..7 | Synoptic-scale moisture and pressure persistence |",
            "| SOIL_WET_SURF_lag1..7 | Antecedent soil moisture controls infiltration and runoff |",
            "| RAINFALL_roll_mean/std_7..30 | Recent wetness regime; variability as drought signal |",
            "| MONTH_SIN/COS, DOY_SIN/COS | Cyclical seasonal encoding — no discontinuity at year boundary |",
            "| SOIL_MOISTURE_GRADIENT | Replaces collinear SURF/ROOT pair (r=0.94); encodes infiltration direction |",
            "| DEWPOINT_APPROX | Absolute moisture content — independent of temperature |",
            "| TEMP_RANGE | Collapses TMAX/TMIN/TAVG triad; large range = clear dry conditions |",
            "| PRESSURE_RH_INTERACTION | Joint signal: low-P + high-RH = monsoon onset marker |",
            "| DRY_SPELL_LENGTH | Boundary-layer drying trajectory; strongly predictive of rain/no-rain |",
            "| RAIN_OCCURRENCE | Stage-1 classification target (Bernoulli component of ZIP model) |",
            "",
            "---",
            "",
            "## 7. Scaling",
            "",
            "StandardScaler fit on training split only.  Val/test transformed using",
            "training mean and std to prevent distributional leakage.",
            "",
            "Columns excluded from scaling: binary flags, calendar integers,",
            "target variables, and count features (DRY_SPELL_LENGTH, SEASON_CODE).",
            "",
            "---",
            f"*Report generated by `src/features/feature_pipeline.py`*",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_feature_type(col: str) -> str:
    """Infer a human-readable category from a feature column name."""
    if "_lag" in col:
        return "Lag"
    if "_roll_" in col:
        return "Rolling"
    if col in ("MONTH_SIN", "MONTH_COS", "DOY_SIN", "DOY_COS"):
        return "Cyclical"
    if col in ("MONTH", "QUARTER", "DAY_OF_YEAR", "WEEK_OF_YEAR",
               "DAY_OF_WEEK", "IS_WEEKEND", "YEAR_FRAC",
               "SEASON", "SEASON_CODE", "DAYS_FROM_MONSOON_ONSET",
               "DAYS_FROM_MONSOON_WITHDRAWAL"):
        return "Temporal"
    if col in ("SOIL_MOISTURE_GRADIENT", "DEWPOINT_APPROX", "TEMP_RANGE",
               "PRESSURE_RH_INTERACTION", "RAIN_OCCURRENCE",
               "DRY_SPELL_LENGTH", "MONSOON_FLAG", "LOG_RAINFALL",
               "RAINFALL_WET_ONLY"):
        return "Domain"
    if col in ("RAINFALL", "IS_MONSOON", "YEAR"):
        return "Target / Structural"
    return "Original Meteorological"


# ---------------------------------------------------------------------------
# Convenience runner for notebooks / CLI
# ---------------------------------------------------------------------------

def run_feature_pipeline(raw_path: Optional[Path] = None) -> FeaturePipeline:
    """One-call entry point. Returns the completed FeaturePipeline object."""
    pipeline = FeaturePipeline(raw_path=raw_path)
    pipeline.run()
    return pipeline
