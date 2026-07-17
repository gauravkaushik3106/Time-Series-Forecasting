# Lucknow Rainfall Forecasting Framework — Setup Instructions

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start (All Platforms)](#quick-start)
3. [Linux Setup](#linux-setup)
4. [macOS Setup](#macos-setup)
5. [Windows Setup](#windows-setup)
6. [GPU Support](#gpu-support)
7. [Verify Installation](#verify-installation)
8. [Running the Pipeline](#running-the-pipeline)
9. [Running the Dashboard](#running-the-dashboard)

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| RAM | 8 GB | 16 GB |
| Disk space | 3 GB | 5 GB |
| CPU cores | 4 | 8+ |
| GPU | Not required | CUDA 11.8+ (for DL models) |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Any recent LTS |

---

## Quick Start

For all platforms, the fastest path is:

```bash
# 1. Clone or extract the project
cd lucknow_rainfall_framework

# 2. Create and activate a virtual environment
python -m venv .venv

# Activate (Linux/Mac):
source .venv/bin/activate

# Activate (Windows PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the raw data file
#    Put Lucknow_rainfall_cleaned.xls into:
#    data/raw/Lucknow_rainfall_cleaned.xls

# 5. Run sanity checks
python sanity_checks.py --pre-only

# 6. Run full pipeline
python run_all.py

# 7. Launch dashboard
streamlit run dashboard/app.py
```

---

## Linux Setup

Tested on: Ubuntu 20.04, 22.04, 24.04 / Debian 11+

### Step 1 — System packages

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip git curl
```

If Python 3.12 is not in your package manager:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3.12-dev
```

### Step 2 — Virtual environment

```bash
cd lucknow_rainfall_framework
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support (skip if CPU-only):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Place raw data

```bash
cp /path/to/Lucknow_rainfall_cleaned.xls data/raw/
```

### Step 5 — Verify

```bash
python sanity_checks.py --pre-only --verbose
```

Expected output: all pre-run checks pass.

---

## macOS Setup

Tested on: macOS 12 Monterey, 13 Ventura, 14 Sonoma (Intel + Apple Silicon)

### Step 1 — Install Homebrew and Python

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12
brew install python@3.12

# Add to PATH (add to ~/.zshrc or ~/.bashrc):
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"
```

### Step 2 — Virtual environment

```bash
cd lucknow_rainfall_framework
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3) note:**
> PyTorch ships native ARM binaries. The standard `pip install torch` command
> automatically selects the correct build. No special flags are needed.
> Metal (MPS) acceleration is available for DL training — it will be used
> automatically when `torch.backends.mps.is_available()` returns True.

### Step 4 — Place raw data

```bash
cp ~/Downloads/Lucknow_rainfall_cleaned.xls data/raw/
```

### Step 5 — Verify

```bash
python sanity_checks.py --pre-only --verbose
```

---

## Windows Setup

Tested on: Windows 10 (21H2+) and Windows 11

### Step 1 — Install Python 3.12

1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer — **check "Add Python to PATH"** during installation
3. Verify: open Command Prompt and run:

```cmd
python --version
```

Expected: `Python 3.12.x`

### Step 2 — Virtual environment

Open **Command Prompt** (not PowerShell for simplest compatibility):

```cmd
cd C:\path\to\lucknow_rainfall_framework
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip wheel
```

If using **PowerShell**, you may need to enable script execution first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```cmd
pip install -r requirements.txt
```

> **Windows-specific note — statsmodels / scipy:**
> If you see a C++ runtime error during installation, install the
> [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
> and retry.

> **Windows-specific note — PyTorch:**
> The CPU-only build in requirements.txt installs correctly on Windows.
> For CUDA support on Windows, replace with:
> ```cmd
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4 — Place raw data

```cmd
copy "C:\path\to\Lucknow_rainfall_cleaned.xls" data\raw\
```

### Step 5 — Verify

```cmd
python sanity_checks.py --pre-only --verbose
```

### Windows path separator note

All file paths in the codebase use `pathlib.Path`, which handles
Windows backslash separators automatically. No manual path changes
are needed on Windows.

---

## GPU Support

GPU acceleration significantly reduces LSTM/GRU training time
(~15 min CPU → ~2 min GPU for 100 epochs).

### CUDA (NVIDIA GPU)

```bash
# Check your CUDA version first:
nvidia-smi

# Install matching PyTorch build:
# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### MPS (Apple Silicon)

No extra installation needed. Verify:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Enable full training budget with GPU

When GPU is detected, `run_all.py` automatically uses the full 100-epoch
training budget. On CPU it reduces to 20 epochs to keep runtime feasible.

---

## Verify Installation

Run the full sanity check after placing the data file:

```bash
python sanity_checks.py --pre-only --verbose
```

All items should show ✓. Warnings about post-run outputs are expected
before the pipeline has been run.

---

## Running the Pipeline

### Full pipeline (recommended first run)

```bash
python run_all.py
```

Estimated runtime: ~15 minutes (CPU, no SARIMAX search)

### Faster run (skip SARIMAX auto-search and deep learning)

```bash
python run_all.py --no-sarimax-search --skip-dl
```

Estimated runtime: ~5 minutes

### Individual phases

```bash
python run_preprocessing_eda.py         # Phase 2
python run_feature_engineering.py       # Phase 3
python run_models.py --no-sarimax-search  # Phase 4
python run_deep_learning.py             # Phase 5
python run_explainability_uncertainty.py  # Phase 6
```

### Resume from a specific phase

```bash
python run_all.py --start-phase 4       # Skip Phases 2–3
```

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens automatically at `http://localhost:8501`

### Dashboard does not retrain models

The dashboard is a **read-only presentation layer**. It loads precomputed
artifacts from `outputs/` and renders them interactively. No training,
no SHAP recomputation, and no feature engineering happen when the
dashboard is running.

### Custom port

```bash
streamlit run dashboard/app.py --server.port 8502
```

### Remote server

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.headless true
```

---

## Conda Alternative

If you prefer conda over pip:

```bash
conda env create -f environment.yml
conda activate lucknow-rainfall
python sanity_checks.py --pre-only
python run_all.py
```
