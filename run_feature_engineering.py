"""
run_feature_engineering.py
==========================
Master runner for Phase 3: Feature Engineering.

Execution order
---------------
1.  Load processed data and apply all feature creators
2.  Drop warm-up rows (NaN from lags / rolling windows)
3.  Re-apply chronological split
4.  Run feature selection (VIF + corr + XGBoost + SHAP)
5.  Fit scalers on training split only; transform and save all splits
6.  Generate feature visualisation figures
7.  Write feature summary table, importance table, and selection report

Usage
-----
    # From the project root:
    python run_feature_engineering.py

    # Skip figure saving (fast run):
    python run_feature_engineering.py --no-save
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import abs_path
from src.features.feature_pipeline import run_feature_pipeline
from src.visualization.eda_features import plot_all_feature_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phase3")


def run(data_path: Path | None = None, save_figures: bool = True) -> None:
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("LUCKNOW RAINFALL FRAMEWORK — PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 70)

    # Execute the full feature pipeline (steps 1–7 above)
    pipeline = run_feature_pipeline(raw_path=data_path)

    # Generate feature visualisation figures
    if save_figures:
        logger.info("\n[Visualisation] Generating feature engineering figures")
        plot_all_feature_figures(
            df=pipeline.df_engineered_,
            selection_result=pipeline.selection_result_,
            save=True,
        )

    elapsed = time.time() - t_start

    # Count outputs
    n_figures = len(list(abs_path("outputs/figures/eda").glob("*.png")))
    n_reports = len(list(abs_path("outputs/reports").glob("*.md")))
    n_feature_files = len(list(abs_path("outputs/features").glob("*.parquet")))

    sr = pipeline.selection_result_
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total runtime         : {elapsed:.1f}s")
    logger.info(f"  Total feature columns : {len(pipeline.all_feature_cols_)}")
    logger.info(f"  Linear feature set    : {len(sr.features_linear)} features")
    logger.info(f"  ML feature set        : {len(sr.features_ml)} features")
    logger.info(f"  Figures saved         : {n_figures} PNG files")
    logger.info(f"  Reports saved         : {n_reports} Markdown files")
    logger.info(f"  Feature datasets      : {n_feature_files} Parquet files")
    logger.info("")
    logger.info("Top-5 features by SHAP importance:")
    for _, row in sr.shap_importance.head(5).iterrows():
        logger.info(
            f"    #{int(row['SHAP_Rank'])}: {row['Feature']:<35s} "
            f"mean|SHAP|={row['SHAP_MeanAbs']:.5f}"
        )
    logger.info("")
    logger.info("Output locations:")
    logger.info("  outputs/features/      — scaled Parquet datasets")
    logger.info("  outputs/reports/03_feature_selection_report.md")
    logger.info("  outputs/features/feature_summary_table.csv")
    logger.info("  outputs/features/feature_importance_table.csv")
    logger.info("=" * 70)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lucknow Rainfall Framework — Phase 3: Feature Engineering"
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to raw data file (overrides config)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving figures to disk"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(data_path=args.data, save_figures=not args.no_save)
