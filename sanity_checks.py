"""
sanity_checks.py
================
Pre-flight and post-flight validation for the Lucknow Rainfall Framework.

Checks performed
----------------
1. Python version compatibility
2. Required package versions (with minimum version enforcement)
3. Critical directory structure
4. Raw data file presence and basic integrity
5. Processed / feature output files (if pipeline has been run)
6. Trained model files (pkl, .pt)
7. Prediction outputs (all model Parquet files)
8. Explainability and uncertainty outputs
9. Dashboard assets
10. Config file validity

Usage
-----
    # Full sanity check (before and after running pipeline):
    python sanity_checks.py

    # Check only pre-run requirements (package + data):
    python sanity_checks.py --pre-only

    # Check only post-run outputs (models, predictions, reports):
    python sanity_checks.py --post-only

    # Verbose: show all checks including passed ones:
    python sanity_checks.py --verbose

Exit codes
----------
    0 — all checks passed
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results: List[Tuple[str, str, str]] = []   # (status, category, message)


def check(status: str, category: str, message: str) -> None:
    results.append((status, category, message))


def ok(category: str, message: str) -> None:
    check(PASS, category, message)


def fail(category: str, message: str) -> None:
    check(FAIL, category, message)


def warn(category: str, message: str) -> None:
    check(WARN, category, message)


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------

def check_python() -> None:
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 10:
        ok("Python", f"Python {major}.{minor}.{sys.version_info.micro} (≥3.10 required)")
    elif major == 3 and minor >= 9:
        warn("Python", f"Python {major}.{minor} — supported but 3.10+ recommended")
    else:
        fail("Python", f"Python {major}.{minor} — Python 3.10+ required")


# ---------------------------------------------------------------------------
# 2. Package versions
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    # (import_name, display_name, min_version_tuple, critical)
    ("pandas",       "pandas",      (2, 0),  True),
    ("numpy",        "numpy",       (1, 24), True),
    ("scipy",        "scipy",       (1, 10), True),
    ("matplotlib",   "matplotlib",  (3, 7),  True),
    ("seaborn",      "seaborn",     (0, 12), False),
    ("statsmodels",  "statsmodels", (0, 14), True),
    ("sklearn",      "scikit-learn",(1, 3),  True),
    ("xgboost",      "xgboost",     (2, 0),  True),
    ("shap",         "shap",        (0, 40), True),
    ("torch",        "PyTorch",     (2, 0),  True),
    ("pmdarima",     "pmdarima",    (2, 0),  True),
    ("streamlit",    "streamlit",   (1, 20), True),
    ("plotly",       "plotly",      (5, 0),  True),
    ("pyarrow",      "pyarrow",     (12, 0), True),
    ("yaml",         "PyYAML",      (6, 0),  True),
    ("PIL",          "Pillow",      (9, 0),  False),
]


def check_packages() -> None:
    for import_name, display_name, min_ver, critical in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver_str = getattr(mod, "__version__", "0.0.0")
            ver_parts = tuple(
                int(x) for x in ver_str.split(".")[:2]
                if x.isdigit()
            )
            if ver_parts >= min_ver:
                ok("Packages", f"{display_name} {ver_str} (≥{'.'.join(map(str,min_ver))})")
            else:
                msg = (f"{display_name} {ver_str} — below minimum "
                       f"{'.'.join(map(str,min_ver))}")
                if critical:
                    fail("Packages", msg)
                else:
                    warn("Packages", msg)
        except ImportError:
            msg = f"{display_name} — NOT INSTALLED"
            if critical:
                fail("Packages", msg)
            else:
                warn("Packages", msg)


# ---------------------------------------------------------------------------
# 3. Directory structure
# ---------------------------------------------------------------------------

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/features",
    "config",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/visualization",
    "src/explainability",
    "src/uncertainty",
    "dashboard/components",
    "dashboard/assets",
    "outputs/features",
    "outputs/predictions",
    "outputs/models",
    "outputs/figures/eda",
    "outputs/figures/models",
    "outputs/figures/explainability",
    "outputs/figures/uncertainty",
    "outputs/reports",
    "outputs/explainability",
    "outputs/uncertainty",
]


def check_directories() -> None:
    for d in REQUIRED_DIRS:
        p = _PROJECT_ROOT / d
        if p.exists() and p.is_dir():
            ok("Directories", str(d))
        else:
            fail("Directories", f"Missing directory: {d}")


# ---------------------------------------------------------------------------
# 4. Raw data file
# ---------------------------------------------------------------------------

def check_raw_data() -> None:
    raw = _PROJECT_ROOT / "data" / "raw" / "Lucknow_rainfall_cleaned.xls"
    if not raw.exists():
        fail("Raw Data", f"Raw data file missing: {raw.relative_to(_PROJECT_ROOT)}")
        return

    size_kb = raw.stat().st_size / 1024
    if size_kb < 50:
        warn("Raw Data", f"Raw data file is very small ({size_kb:.0f} KB) — may be truncated")
    else:
        ok("Raw Data", f"Raw data file found ({size_kb:.0f} KB)")

    # Quick content check: try to read first line
    try:
        with open(raw, "r") as fh:
            header = fh.readline().strip()
        expected_cols = ["DATE","RAINFALL","TMAX","TMIN"]
        if all(c in header for c in expected_cols):
            ok("Raw Data", f"CSV header valid: {header[:60]}...")
        else:
            warn("Raw Data", f"Unexpected header: {header[:60]}")
    except Exception as e:
        warn("Raw Data", f"Could not read raw file: {e}")


# ---------------------------------------------------------------------------
# 5. Config file
# ---------------------------------------------------------------------------

def check_config() -> None:
    cfg_path = _PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        fail("Config", "config/config.yaml missing")
        return
    try:
        import yaml
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        required_keys = ["project","paths","schema","split","rainfall","eda","plotting"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            fail("Config", f"config.yaml missing keys: {missing}")
        else:
            ok("Config", "config/config.yaml valid (all required keys present)")
    except Exception as e:
        fail("Config", f"config.yaml parse error: {e}")


# ---------------------------------------------------------------------------
# 6. Runner scripts
# ---------------------------------------------------------------------------

RUNNER_SCRIPTS = [
    "run_preprocessing_eda.py",
    "run_feature_engineering.py",
    "run_models.py",
    "run_deep_learning.py",
    "run_explainability_uncertainty.py",
    "run_all.py",
    "sanity_checks.py",
    "dashboard/app.py",
]


def check_scripts() -> None:
    for script in RUNNER_SCRIPTS:
        p = _PROJECT_ROOT / script
        if p.exists():
            ok("Scripts", script)
        else:
            fail("Scripts", f"Missing: {script}")


# ---------------------------------------------------------------------------
# 7. Processed feature outputs (post-run)
# ---------------------------------------------------------------------------

FEATURE_FILES = [
    "outputs/features/full_features.parquet",
    "outputs/features/train_features_ml.parquet",
    "outputs/features/val_features_ml.parquet",
    "outputs/features/test_features_ml.parquet",
    "outputs/features/train_features_linear.parquet",
    "outputs/features/feature_summary_table.csv",
    "outputs/features/feature_importance_table.csv",
    "outputs/features/scaler_params_ml.json",
    "outputs/features/scaler_params_linear.json",
]


def check_feature_outputs() -> None:
    for rel_path in FEATURE_FILES:
        p = _PROJECT_ROOT / rel_path
        if p.exists():
            size_kb = p.stat().st_size / 1024
            ok("Features", f"{rel_path} ({size_kb:.0f} KB)")
        else:
            warn("Features", f"Missing (run Phase 3): {rel_path}")


# ---------------------------------------------------------------------------
# 8. Model files (post-run)
# ---------------------------------------------------------------------------

MODEL_FILES = [
    ("outputs/models/xgboost_model.pkl",        "XGBoost"),
    ("outputs/models/lstm_model.pt",             "LSTM"),
    ("outputs/models/gru_model.pt",              "GRU"),
    ("outputs/models/hybrid_lstm_residual.pt",   "Hybrid LSTM"),
]

PREDICTION_FILES = [
    "outputs/predictions/persistence_test.parquet",
    "outputs/predictions/climatology_test.parquet",
    "outputs/predictions/sarimax_test.parquet",
    "outputs/predictions/xgboost_test.parquet",
    "outputs/predictions/lstm_test.parquet",
    "outputs/predictions/gru_test.parquet",
    "outputs/predictions/hybrid_test.parquet",
    "outputs/predictions/sarimax_train_residuals.parquet",
    "outputs/predictions/model_comparison_table_test.csv",
    "outputs/predictions/model_comparison_table_phase5.csv",
]


def check_model_outputs() -> None:
    for rel_path, label in MODEL_FILES:
        p = _PROJECT_ROOT / rel_path
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            ok("Models", f"{label} model found ({size_mb:.1f} MB): {rel_path}")
        else:
            warn("Models", f"{label} model missing (run Phase 4/5): {rel_path}")

    for rel_path in PREDICTION_FILES:
        p = _PROJECT_ROOT / rel_path
        if p.exists():
            ok("Predictions", rel_path)
        else:
            warn("Predictions", f"Missing (run Phases 4/5): {rel_path}")


# ---------------------------------------------------------------------------
# 9. Explainability + uncertainty outputs (post-run)
# ---------------------------------------------------------------------------

PHASE6_FILES = [
    "outputs/explainability/shap_summary.csv",
    "outputs/uncertainty/mc_dropout_predictions.parquet",
    "outputs/uncertainty/calibration_metrics.csv",
    "outputs/figures/explainability/shap_beeswarm.png",
    "outputs/figures/explainability/shap_bar_importance.png",
    "outputs/figures/explainability/shap_dependence_grid.png",
    "outputs/figures/explainability/shap_local_waterfall.png",
    "outputs/figures/explainability/pdp_grid.png",
    "outputs/figures/explainability/pdp_2d_cloud_rh.png",
    "outputs/figures/uncertainty/calibration_reliability_diagram.png",
    "outputs/figures/uncertainty/uncertainty_interval_timeseries.png",
]


def check_phase6_outputs() -> None:
    for rel_path in PHASE6_FILES:
        p = _PROJECT_ROOT / rel_path
        if p.exists():
            ok("Phase6", rel_path)
        else:
            warn("Phase6", f"Missing (run Phase 6): {rel_path}")


# ---------------------------------------------------------------------------
# 10. Report files
# ---------------------------------------------------------------------------

REPORT_FILES = [
    "outputs/reports/01_preprocessing_report.md",
    "outputs/reports/02_eda_report.md",
    "outputs/reports/03_feature_selection_report.md",
    "outputs/reports/04_model_performance_report.md",
    "outputs/reports/05_phase5_report.md",
    "outputs/reports/06_phase6_report.md",
]


def check_reports() -> None:
    for rel_path in REPORT_FILES:
        p = _PROJECT_ROOT / rel_path
        if p.exists():
            ok("Reports", rel_path)
        else:
            warn("Reports", f"Missing (run corresponding phase): {rel_path}")


# ---------------------------------------------------------------------------
# 11. Dashboard assets
# ---------------------------------------------------------------------------

def check_dashboard() -> None:
    critical = [
        "dashboard/app.py",
        "dashboard/data_loader.py",
        "dashboard/assets/custom.css",
        "dashboard/components/home.py",
        "dashboard/components/data_explorer.py",
        "dashboard/components/forecast_panel.py",
        "dashboard/components/model_comparison.py",
        "dashboard/components/explainability_panel.py",
        "dashboard/components/uncertainty_panel.py",
        "dashboard/components/research_insights.py",
    ]
    for rel_path in critical:
        p = _PROJECT_ROOT / rel_path
        if p.exists():
            ok("Dashboard", rel_path)
        else:
            fail("Dashboard", f"Missing: {rel_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_checks(pre_only: bool, post_only: bool, verbose: bool) -> int:
    print()
    print("=" * 70)
    print("  LUCKNOW RAINFALL FRAMEWORK — SANITY CHECKS")
    print("=" * 70)

    if not post_only:
        print("\n── Pre-run checks ──────────────────────────────────────────────")
        check_python()
        check_packages()
        check_directories()
        check_raw_data()
        check_config()
        check_scripts()
        check_dashboard()

    if not pre_only:
        print("\n── Post-run checks ─────────────────────────────────────────────")
        check_feature_outputs()
        check_model_outputs()
        check_phase6_outputs()
        check_reports()

    # ── Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)

    current_cat = None
    for status, category, message in results:
        if category != current_cat:
            if not verbose and status == PASS:
                # For non-verbose mode, only print category headers when
                # there are non-passing items in that category
                cat_issues = [r for r in results
                              if r[1] == category and r[0] != PASS]
                if not cat_issues:
                    continue
            print(f"\n  [{category}]")
            current_cat = category
        if verbose or status != PASS:
            print(f"    {status} {message}")

    print()
    print(f"  Summary: {n_pass} passed | {n_warn} warnings | {n_fail} failed")
    print("=" * 70)
    print()

    if n_fail > 0:
        print("  ❌  SANITY CHECK FAILED — resolve the items marked ✗ above.")
        print("      Run with --verbose to see all checks.")
        print()
        return 1
    elif n_warn > 0:
        print("  ⚠   SANITY CHECK PASSED WITH WARNINGS")
        print("      Warnings typically indicate the pipeline has not been run yet.")
        print("      Run: python run_all.py")
        print()
        return 0
    else:
        print("  ✅  ALL CHECKS PASSED")
        print()
        return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lucknow Rainfall Framework — Sanity Check Tool"
    )
    p.add_argument("--pre-only",  action="store_true",
                   help="Check only pre-run requirements")
    p.add_argument("--post-only", action="store_true",
                   help="Check only post-run outputs")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show all checks including passed ones")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = run_checks(
        pre_only=args.pre_only,
        post_only=args.post_only,
        verbose=args.verbose,
    )
    sys.exit(exit_code)
