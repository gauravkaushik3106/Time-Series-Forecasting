# Lucknow Rainfall Framework — Troubleshooting Guide

## Table of Contents

1. [Installation Errors](#1-installation-errors)
2. [Import Errors at Runtime](#2-import-errors-at-runtime)
3. [File Path Issues](#3-file-path-issues)
4. [Data Loading Errors](#4-data-loading-errors)
5. [XGBoost Issues](#5-xgboost-issues)
6. [PyTorch / LSTM / GRU Issues](#6-pytorch--lstm--gru-issues)
7. [SARIMAX / statsmodels Issues](#7-sarimax--statsmodels-issues)
8. [Streamlit Dashboard Issues](#8-streamlit-dashboard-issues)
9. [Memory and Disk Space Issues](#9-memory-and-disk-space-issues)
10. [Platform-Specific Issues](#10-platform-specific-issues)
11. [Output File Issues](#11-output-file-issues)
12. [Diagnostic Commands](#12-diagnostic-commands)

---

## 1. Installation Errors

### `pip install -r requirements.txt` fails with no space left on device

**Symptom:**
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left
```

**Fix:**
```bash
# Clear pip cache first
pip cache purge

# Try again
pip install -r requirements.txt

# If still failing, check disk space
df -h /tmp
df -h ~
```

The PyTorch wheel is ~800 MB. Ensure at least 2 GB free in `/tmp` and the
target environment directory.

---

### `pip install torch` installs but CUDA not detected

**Symptom:**
```python
import torch
torch.cuda.is_available()  # Returns False despite having an NVIDIA GPU
```

**Fix — install the correct CUDA build:**
```bash
# First: check your driver's CUDA version
nvidia-smi | grep "CUDA Version"

# Then install matching build:
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# or
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

---

### statsmodels / scipy fails to install on Windows

**Symptom:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Fix:**
1. Download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install with "Desktop development with C++" workload selected
3. Restart your terminal
4. Re-run `pip install -r requirements.txt`

---

### `pmdarima` install fails

**Symptom:**
```
ERROR: Failed building wheel for pmdarima
```

**Fix:**
```bash
# Ensure Cython is available
pip install cython
pip install pmdarima==2.1.1

# Alternative: install pre-built wheel
pip install pmdarima --no-build-isolation
```

---

### `shap` install fails with NumPy compatibility error

**Symptom:**
```
ImportError: numpy.core.multiarray failed to import
```

**Fix:**
```bash
# Upgrade numpy first, then reinstall shap
pip install "numpy>=2.0" --force-reinstall
pip install shap==0.51.0
```

---

### Virtual environment not activating on Windows (PowerShell)

**Symptom:**
```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

**Fix:**
```powershell
# Run as Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
.venv\Scripts\Activate.ps1
```

---

## 2. Import Errors at Runtime

### `ModuleNotFoundError: No module named 'config'`

**Symptom:** Any script raises `ModuleNotFoundError: No module named 'config'`

**Cause:** Script is not being run from the project root.

**Fix:** Always run scripts from `lucknow_rainfall_framework/`:
```bash
# Wrong:
cd src/models && python model_pipeline.py

# Correct:
cd lucknow_rainfall_framework
python run_models.py
```

All scripts add `_PROJECT_ROOT` to `sys.path` via the pattern at the top
of each file — but this requires `_PROJECT_ROOT` to be computed correctly,
which only happens when running from the project root or using the
top-level runner scripts.

---

### `ModuleNotFoundError: No module named 'src'`

Same cause as above. Always run from the project root.

---

### `ImportError: cannot import name 'RainfallDataLoader' from 'src.preprocessing.loader'`

**Cause:** Stale `__pycache__` from an older version of the file.

**Fix:**
```bash
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
python run_preprocessing_eda.py
```

---

## 3. File Path Issues

### `FileNotFoundError: Raw data file not found: data/raw/Lucknow_rainfall_cleaned.xls`

**Fix:** Place the data file at the correct location:
```bash
# The file must be here relative to the project root:
ls data/raw/Lucknow_rainfall_cleaned.xls

# If it is elsewhere:
cp /path/to/Lucknow_rainfall_cleaned.xls data/raw/
```

The file is named `.xls` but is actually CSV-formatted — this is expected
and handled by the loader automatically.

---

### `FileNotFoundError` for outputs that should exist

**Cause:** A required phase has not been run yet.

**Resolution order:**
```
Phase 2 (preprocessing) must run before Phase 3
Phase 3 (features)      must run before Phase 4
Phase 4 (models)        must run before Phase 5 and 6
Phase 5 (deep learning) must run before Phase 6 (for GRU MC Dropout)
```

Run `python sanity_checks.py` to identify which phases are incomplete.

---

### Paths work on Linux but fail on Windows (backslash issue)

The codebase uses `pathlib.Path` throughout, which handles OS-specific
separators automatically. If you see explicit forward slashes in a
string constant causing issues, open a GitHub issue — it is a bug.

---

## 4. Data Loading Errors

### `pandas.errors.ParserError` when reading the raw file

**Symptom:** The loader fails trying to parse the raw `.xls` file.

**Cause:** File may have been saved with a different line ending or encoding.

**Fix:**
```python
# Test the file manually:
import pandas as pd
df = pd.read_csv("data/raw/Lucknow_rainfall_cleaned.xls")
print(df.head())
print(df.shape)  # Expected: (9497, 12)
```

If this fails, open the file in a text editor and verify:
- First line is: `DATE,RAINFALL,TMAX,TMIN,TAVG,RH,WIND,PRESSURE,CLOUD,SOLAR_RAD,SOIL_WET_SURF,SOIL_WET_ROOT`
- Dates are in `YYYY-MM-DD` format
- Values are comma-separated with no quotes

---

### `pyarrow.lib.ArrowInvalid` when reading Parquet files

**Cause:** A Parquet file was partially written (pipeline crashed mid-run).

**Fix:**
```bash
# Delete corrupted outputs and re-run the relevant phase
rm outputs/features/*.parquet
python run_feature_engineering.py
```

---

## 5. XGBoost Issues

### XGBoost training is very slow

**Cause:** XGBoost uses all available CPU cores by default (`n_jobs=-1`).
On machines with hyperthreading, the effective parallelism may be lower
than expected.

**Diagnostic:**
```python
import xgboost as xgb
print(xgb.__version__)

# Check core usage during training with htop (Linux/Mac) or Task Manager (Windows)
```

**Fix:** Reduce the hyperparameter grid search in `src/models/xgboost_model.py`
by narrowing `PARAM_GRID` to fewer combinations for faster iteration.

---

### `xgboost.core.XGBoostError: [17:32:41] ... Check failed: !info->IsEmpty()`

**Cause:** XGBoost received an empty feature matrix (no wet-day training
samples after filtering).

**Fix:** Verify the training feature set has `RAIN_OCCURRENCE == 1` rows:
```python
import pandas as pd
train = pd.read_parquet("outputs/features/train_features_ml.parquet")
print("Wet days in train:", (train["RAIN_OCCURRENCE"] == 1).sum())
# Expected: ~3000
```

If this is 0, the `RAIN_OCCURRENCE` feature was not created correctly.
Re-run Phase 3: `python run_feature_engineering.py`.

---

### XGBoost early stopping does not trigger

This is expected behaviour when the validation loss continues improving
past the `EARLY_STOP_PAT` threshold. It means the model is still learning —
you may want to increase `MAX_EPOCHS` in `PARAM_GRID` for better results.

---

## 6. PyTorch / LSTM / GRU Issues

### LSTM/GRU training is extremely slow on CPU

**Expected CPU runtime:** ~8 minutes for 20 epochs on the full dataset.

The `run_all.py` script automatically caps epochs at 20 when no GPU
is detected. With a GPU, 100 epochs takes ~2 minutes.

For faster CPU testing:
```bash
# Reduce training data for debugging (edit sequence_generator.py):
# Set BATCH_SIZE = 512 and use a smaller train subset
python run_deep_learning.py  # Will use reduced epoch cap automatically
```

---

### `RuntimeError: CUDA out of memory`

**Fix:**
```bash
# Reduce batch size in src/models/sequence_generator.py:
# Change: BATCH_SIZE = 128
# To:     BATCH_SIZE = 32

# Or clear GPU cache before running:
python -c "import torch; torch.cuda.empty_cache()"
```

---

### `RuntimeError: Expected all tensors to be on the same device`

**Cause:** A tensor was created on CPU but the model is on GPU (or vice versa).

**Fix:** This should not occur with the framework code as-is. If it does after
modifications, ensure you call `.to(self.device)` on all tensors before
passing to the model.

---

### `UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes`

This warning from PyTorch is benign and can be ignored. It does not affect
training or model quality.

---

### GRU/LSTM model produces NaN loss

**Cause:** Exploding gradients, often due to very large input feature values.

**Fix:** Verify that feature scaling was applied correctly:
```python
import pandas as pd
train = pd.read_parquet("outputs/features/train_features_ml.parquet")
print(train.describe())
# All scaled features should have mean ≈ 0 and std ≈ 1
```

If features are not scaled, re-run Phase 3.

The framework applies gradient clipping (`max_norm=1.0`) which should
prevent NaN under normal conditions.

---

### MC Dropout intervals are extremely narrow (ECE ≈ 0.46)

This is a **known, documented limitation** — not a bug. See the full
explanation in the dashboard Uncertainty panel or in
`outputs/reports/06_phase6_report.md`.

Short version: single-layer GRU with 20 training epochs produces
insufficient MC Dropout variance. The architecture code is correct;
the issue is the training budget constraint on CPU.

---

## 7. SARIMAX / statsmodels Issues

### `pmdarima.arima.auto_arima` is very slow

**Cause:** Auto-ARIMA searches over model orders — this is inherently slow
on a 6,617-row daily series.

**Fix (recommended for development):** Use default orders:
```bash
python run_models.py --no-sarimax-search
```

This uses `(1,0,1) × (1,1,1,12)` orders without search.

---

### `LinAlgError: SVD did not converge` in SARIMAX fitting

**Cause:** The SARIMAX model has numerical instability with the given
exogenous variables.

**Fix:**
```python
# In src/models/sarimax.py, reduce SARIMAX_EXOG_COLS to fewer features:
SARIMAX_EXOG_COLS = ["CLOUD", "WIND", "IS_MONSOON"]  # minimal set
```

---

### SARIMAX predictions are all near-zero

**Cause:** The log-transform target (`log1p`) was not back-transformed
before saving. Verify the prediction file:
```python
import pandas as pd
df = pd.read_parquet("outputs/predictions/sarimax_test.parquet")
print(df.describe())
# 'predicted' column should be in mm/day (0–50 range), not log scale (0–4 range)
```

---

### `ConvergenceWarning: Maximum Likelihood optimization failed to converge`

This warning from statsmodels is common with SARIMAX on meteorological data
and does not invalidate the model. The fitted parameters may be slightly
suboptimal but the model is still usable. Increase `maxiter` in
`src/models/sarimax.py` if you want to suppress the warning:
```python
self.model_fit_ = model.fit(disp=False, maxiter=500, method="lbfgs")
```

---

## 8. Streamlit Dashboard Issues

### Dashboard crashes with `ModuleNotFoundError`

**Cause:** Dashboard is being run from the wrong directory.

**Fix — always run from the project root:**
```bash
cd lucknow_rainfall_framework
streamlit run dashboard/app.py
```

**Never run:**
```bash
cd dashboard
streamlit run app.py  # WRONG — path resolution will fail
```

---

### Dashboard shows blank panels with "Figure not found"

**Cause:** Phase 6 (Explainability + Uncertainty) has not been run.

**Fix:**
```bash
python run_explainability_uncertainty.py
```

Then refresh the dashboard.

---

### `st.cache_data` stale after re-running pipeline

Streamlit caches data across sessions. After re-running the pipeline,
clear the cache:

**Option 1:** In the dashboard, click the three-dot menu (top right) →
"Clear cache"

**Option 2:** Restart the Streamlit server:
```bash
# Stop the server (Ctrl+C), then restart:
streamlit run dashboard/app.py
```

---

### Dashboard is slow to load

**Cause:** Parquet files are large (full features = ~1.5 MB).
First load reads from disk; subsequent loads use `@st.cache_data`.

This is expected behaviour. After the first page visit, cached data
renders instantly.

---

### `OSError: [Errno 98] Address already in use` (port 8501)

**Fix — use a different port:**
```bash
streamlit run dashboard/app.py --server.port 8502
```

Or kill the existing process:
```bash
# Linux/Mac:
lsof -i :8501 | awk 'NR>1 {print $2}' | xargs kill -9

# Windows PowerShell:
netstat -ano | findstr :8501
taskkill /PID <PID_FROM_ABOVE> /F
```

---

### `PIL.UnidentifiedImageError` when loading figures

**Cause:** A PNG file was partially written during a failed pipeline run.

**Fix:**
```bash
# Re-run Phase 6 to regenerate all figures:
python run_explainability_uncertainty.py
```

---

### Plotly charts not rendering in dashboard

**Cause:** Outdated Streamlit or Plotly version.

**Fix:**
```bash
pip install --upgrade streamlit plotly
```

---

## 9. Memory and Disk Space Issues

### `MemoryError` during feature engineering

**Cause:** The full feature dataset with all lag/rolling columns can
require ~2 GB of RAM in pandas.

**Fix — check available memory:**
```python
import psutil
print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
```

If available RAM < 4 GB, close other applications before running.

---

### `SARIMAX model pickle is 648 MB`

This is a known issue — the statsmodels SARIMAX fitted result object
stores the full model state including the full Kalman filter history.
The file is automatically removed by `run_all.py` after Phase 5 completes
to free disk space (SARIMAX predictions are already saved as Parquet).

If you need to keep it, ensure at least 3 GB free disk space.

---

### `outputs/` directory grows too large

After a full pipeline run, `outputs/` contains ~700 MB (mostly the
SARIMAX pkl). To reduce size:

```bash
# Remove SARIMAX pkl (predictions are still in outputs/predictions/)
rm outputs/models/sarimax_model.pkl

# Remove EDA figures if not needed
rm -rf outputs/figures/eda/

# Check remaining size
du -sh outputs/
```

---

## 10. Platform-Specific Issues

### macOS: `OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized`

**Fix:**
```python
# Add to the top of any failing script:
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

Or permanently:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Add to ~/.zshrc or ~/.bashrc
```

---

### macOS: matplotlib font warnings on every plot

```
findfont: Font family 'sans-serif' not found.
```

**Fix:**
```bash
# Rebuild matplotlib font cache:
python -c "import matplotlib; matplotlib.font_manager._rebuild()"
```

---

### Windows: `UnicodeDecodeError` when reading files

**Cause:** Windows default encoding is cp1252, not UTF-8.

**Fix:** Add encoding parameter where needed, or set environment variable:
```cmd
set PYTHONIOENCODING=utf-8
```

---

### Windows: Long path errors

**Symptom:**
```
FileNotFoundError: [WinError 3] The system cannot find the path specified
```

**Fix:** Enable long paths in Windows (requires admin):
```powershell
# Run as Administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

## 11. Output File Issues

### Prediction Parquet files are empty

**Cause:** A model failed silently during Phase 4 or 5.

**Diagnostic:**
```python
import pandas as pd
for model in ['persistence','climatology','sarimax','xgboost','lstm','gru','hybrid']:
    try:
        df = pd.read_parquet(f"outputs/predictions/{model}_test.parquet")
        print(f"{model}: {len(df)} rows, cols={list(df.columns)}")
    except Exception as e:
        print(f"{model}: ERROR — {e}")
```

Re-run the failing phase to regenerate.

---

### SHAP summary CSV has wrong features

**Cause:** Phase 3 was re-run after Phase 6, changing the feature set.

**Fix:** Re-run Phase 6 to regenerate SHAP values on the current feature set:
```bash
python run_explainability_uncertainty.py
```

---

## 12. Diagnostic Commands

Run these to collect information for debugging:

```bash
# Full sanity check with verbose output
python sanity_checks.py --verbose

# Check Python and package versions
python -c "
import sys, pandas, numpy, torch, xgboost, shap, streamlit
print('Python:', sys.version[:10])
print('pandas:', pandas.__version__)
print('numpy:', numpy.__version__)
print('torch:', torch.__version__)
print('xgboost:', xgboost.__version__)
print('shap:', shap.__version__)
print('streamlit:', streamlit.__version__)
print('GPU:', torch.cuda.is_available())
"

# Verify all output files exist
python sanity_checks.py --post-only

# Dry run the full pipeline
python run_all.py --dry-run

# Check disk space
df -h .   # Linux/Mac
# dir     # Windows (in project directory)

# Check the raw data file
head -3 data/raw/Lucknow_rainfall_cleaned.xls

# Test loading precomputed predictions
python -c "
import pandas as pd
for m in ['xgboost','sarimax','lstm','gru']:
    df = pd.read_parquet(f'outputs/predictions/{m}_test.parquet')
    print(f'{m}: {len(df)} rows, RMSE={((df.actual-df.predicted)**2).mean()**0.5:.3f}')
"
```

---

## Getting Further Help

If none of the above resolves your issue:

1. Run `python sanity_checks.py --verbose` and note all failures
2. Run `python run_all.py --dry-run` and note any import errors
3. Copy the full error traceback (not just the last line)
4. Note your OS, Python version, and GPU type
