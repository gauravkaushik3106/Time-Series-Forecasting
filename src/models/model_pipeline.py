"""
model_pipeline.py
=================
Phase 4 model pipeline orchestrator.

Execution order
---------------
1. Load feature datasets (from Phase 3 Parquet outputs)
2. Fit and predict: Persistence baseline
3. Fit and predict: Climatology baseline
4. Fit and predict: SARIMAX
5. Fit and predict: XGBoost (two-stage)
6. Evaluate all models on val and test splits
7. Generate per-model prediction figures
8. Generate comparison figures (bar chart, seasonal, extremes, Taylor)
9. Write performance report and comparison CSV

No model code is modified here — this module only wires existing modules.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import CFG, abs_path
from src.models.baseline import PersistenceModel, ClimatologyModel
from src.models.sarimax import run_sarimax
from src.models.xgboost_model import run_xgboost
from src.evaluation.metrics import evaluate, build_comparison_table
from src.evaluation.model_comparison import (
    build_all_comparisons,
    generate_performance_report,
)
from src.evaluation.prediction_plots import (
    plot_model_predictions,
    plot_model_comparison,
    plot_seasonal_comparison,
    plot_extreme_event_comparison,
    plot_taylor_diagram,
    plot_all_predictions_overlay,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_pipeline")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_feature_data() -> Tuple[
    pd.DataFrame,  # df_full (unscaled, all splits)
    pd.DataFrame,  # train_ml (scaled)
    pd.DataFrame,  # val_ml
    pd.DataFrame,  # test_ml
]:
    """Load the three feature-set Parquet files produced by Phase 3."""
    features_dir = abs_path("outputs/features")

    df_full  = pd.read_parquet(features_dir / "full_features.parquet")
    train_ml = pd.read_parquet(features_dir / "train_features_ml.parquet")
    val_ml   = pd.read_parquet(features_dir / "val_features_ml.parquet")
    test_ml  = pd.read_parquet(features_dir / "test_features_ml.parquet")

    logger.info(
        f"Loaded feature data: full={len(df_full):,} | "
        f"train={len(train_ml):,} | val={len(val_ml):,} | test={len(test_ml):,}"
    )
    return df_full, train_ml, val_ml, test_ml


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_model_pipeline(
    run_sarimax_search: bool = True,
    save_figures: bool = True,
) -> Dict:
    """
    Execute the complete Phase 4 model pipeline.

    Parameters
    ----------
    run_sarimax_search : Run auto_arima parameter search (slower but optimal).
                         Set False to use default (1,0,1)x(1,1,1,12) orders.
    save_figures       : Write figures to disk.

    Returns
    -------
    dict with keys 'comparison_df', 'pred_dfs', 'models'.
    """
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 4: BASELINES + CLASSICAL + ML MODELS")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("\n[1/9] Loading feature data")
    df_full, train_ml, val_ml, test_ml = _load_feature_data()

    # Unscaled rainfall series for baselines and SARIMAX
    rain_full  = df_full["RAINFALL"]
    train_mask = df_full["SPLIT"] == "train"
    val_mask   = df_full["SPLIT"] == "val"
    test_mask  = df_full["SPLIT"] == "test"

    rain_train = rain_full[train_mask]
    rain_val   = rain_full[val_mask]
    rain_test  = rain_full[test_mask]

    out_pred = abs_path("outputs/predictions")
    out_pred.mkdir(parents=True, exist_ok=True)

    pred_dfs_test: Dict[str, pd.DataFrame] = {}
    pred_dfs_val:  Dict[str, pd.DataFrame] = {}
    models: Dict = {}

    # ------------------------------------------------------------------
    # 2. Persistence baseline
    # ------------------------------------------------------------------
    logger.info("\n[2/9] Persistence baseline")
    t0   = time.time()
    pers = PersistenceModel()
    pers.fit(rain_train)

    for split_name, rain_split, preceding in [
        ("val",  rain_val,  rain_train),
        ("test", rain_test, pd.concat([rain_train, rain_val])),
    ]:
        full_ctx  = pd.concat([preceding, rain_split]).sort_index()
        pers_pred = pers.predict(full_ctx).loc[rain_split.index]
        pred_df   = pd.DataFrame(
            {"actual": rain_split.values, "predicted": pers_pred.values},
            index=rain_split.index,
        )
        pred_df.to_parquet(out_pred / f"persistence_{split_name}.parquet")
        if split_name == "test":
            pred_dfs_test["Persistence"] = pred_df
        else:
            pred_dfs_val["Persistence"] = pred_df

    models["Persistence"] = pers
    logger.info(f"  Persistence done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Climatology baseline
    # ------------------------------------------------------------------
    logger.info("\n[3/9] Climatology baseline")
    t0   = time.time()
    clim = ClimatologyModel()
    clim.fit(rain_train)

    for split_name, rain_split in [("val", rain_val), ("test", rain_test)]:
        clim_pred = clim.predict(rain_split.index)
        pred_df   = pd.DataFrame(
            {"actual": rain_split.values, "predicted": clim_pred.values},
            index=rain_split.index,
        )
        pred_df.to_parquet(out_pred / f"climatology_{split_name}.parquet")
        if split_name == "test":
            pred_dfs_test["Climatology"] = pred_df
        else:
            pred_dfs_val["Climatology"] = pred_df

    models["Climatology"] = clim
    logger.info(f"  Climatology done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. SARIMAX
    # ------------------------------------------------------------------
    logger.info("\n[4/9] SARIMAX")
    t0 = time.time()
    try:
        sarimax_model, sarimax_preds = run_sarimax(
            df_full, auto_search=run_sarimax_search
        )
        models["SARIMAX"] = sarimax_model
        pred_dfs_test["SARIMAX"] = sarimax_preds["test"]
        pred_dfs_val["SARIMAX"]  = sarimax_preds["val"]
        logger.info(f"  SARIMAX done in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.error(f"  SARIMAX failed: {e}. Skipping.")

    # ------------------------------------------------------------------
    # 5. XGBoost
    # ------------------------------------------------------------------
    logger.info("\n[5/9] XGBoost (two-stage)")
    t0 = time.time()
    try:
        xgb_model, xgb_preds = run_xgboost(train_ml, val_ml, test_ml)
        models["XGBoost"] = xgb_model
        pred_dfs_test["XGBoost"] = xgb_preds["test"]
        pred_dfs_val["XGBoost"]  = xgb_preds["val"]
        logger.info(f"  XGBoost done in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.error(f"  XGBoost failed: {e}. Skipping.")

    # ------------------------------------------------------------------
    # 6. Evaluate all models
    # ------------------------------------------------------------------
    logger.info("\n[6/9] Evaluating all models")
    comparison_df_test = build_all_comparisons(pred_dfs_test, split="test")
    comparison_df_val  = build_all_comparisons(pred_dfs_val,  split="val")

    logger.info("\n--- TEST SET RANKING ---")
    for _, row in comparison_df_test.iterrows():
        logger.info(
            f"  #{int(row.get('Rank',0))} {row['Model']:15s}  "
            f"RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}  "
            f"NSE={row['NSE']:.4f}  R2={row['R2']:.4f}"
        )

    # ------------------------------------------------------------------
    # 7. Per-model prediction figures
    # ------------------------------------------------------------------
    if save_figures:
        logger.info("\n[7/9] Per-model prediction figures")
        for model_name, pred_df in pred_dfs_test.items():
            plot_model_predictions(pred_df, model_name, save=True)

    # ------------------------------------------------------------------
    # 8. Comparison figures
    # ------------------------------------------------------------------
    if save_figures:
        logger.info("\n[8/9] Comparison figures")
        plot_model_comparison(comparison_df_test, save=True)
        plot_seasonal_comparison(comparison_df_test, save=True)
        plot_extreme_event_comparison(pred_dfs_test, save=True)
        plot_taylor_diagram(pred_dfs_test, save=True)
        plot_all_predictions_overlay(pred_dfs_test, save=True)

    # ------------------------------------------------------------------
    # 9. Performance report
    # ------------------------------------------------------------------
    logger.info("\n[9/9] Generating performance report")
    generate_performance_report(
        comparison_df=comparison_df_test,
        pred_dfs=pred_dfs_test,
        split="test",
    )

    elapsed = time.time() - t_start
    n_figs  = len(list(abs_path("outputs/figures/models").glob("*.png")))

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total runtime   : {elapsed:.1f}s")
    logger.info(f"  Models trained  : {len(models)}")
    logger.info(f"  Figures saved   : {n_figs}")
    logger.info(f"  Best model      : {comparison_df_test.iloc[0]['Model']} "
                f"(RMSE={comparison_df_test.iloc[0]['RMSE']:.4f} mm/day)")
    logger.info("=" * 70)

    return {
        "comparison_df": comparison_df_test,
        "comparison_df_val": comparison_df_val,
        "pred_dfs_test": pred_dfs_test,
        "pred_dfs_val":  pred_dfs_val,
        "models": models,
    }
