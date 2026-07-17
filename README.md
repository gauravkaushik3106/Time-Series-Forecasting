# 🌧️ AI-Driven Rainfall Forecasting Framework

> **Bachelor Thesis Project (BTP) | IIT Guwahati**

An end-to-end rainfall forecasting framework that combines statistical methods, machine learning, and deep learning to predict daily rainfall using **26 years of meteorological observations (9,400+ records)** from Lucknow, India.

The project was built as a complete machine learning pipeline—from data preprocessing and feature engineering to model comparison, explainability, uncertainty analysis, and deployment through an interactive Streamlit dashboard.

---

# 🚀 Project Highlights

- 📅 Built using **26 years** of meteorological observations (2000–2025)
- 📊 Processed **9,400+ daily weather records**
- 🌳 **XGBoost** achieved the best overall forecasting performance
- 📉 Reduced RMSE by **50%+** compared to baseline forecasting models
- 🧠 Developed domain-driven feature engineering for improved rainfall prediction
- 🔍 Performed model explainability using feature importance and SHAP analysis
- 📈 Built an interactive dashboard for visualization, comparison, and uncertainty analysis
- ⚙️ Developed a fully reproducible pipeline with one-command execution

---

# 📌 Problem Statement

Rainfall forecasting is a challenging time-series problem due to:

- Highly seasonal weather patterns
- Extreme rainfall events
- Large number of dry days
- Strong correlations among meteorological variables
- Non-linear relationships between atmospheric conditions

The objective of this project is to build a robust forecasting framework capable of learning these complex relationships while remaining interpretable and reproducible.

---

# 🏗️ Project Pipeline

```
Raw Weather Data
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ├── Persistence
        ├── Climatology
        ├── SARIMAX
        ├── XGBoost
        ├── LSTM
        ├── GRU
        └── Hybrid Model
        │
        ▼
Model Evaluation
        │
        ▼
Explainability
        │
        ▼
Uncertainty Analysis
        │
        ▼
Interactive Streamlit Dashboard
```

---

# 📊 Models Evaluated

The project compares multiple forecasting paradigms under a common evaluation framework.

| Category | Models |
|-----------|---------|
| Baseline | Persistence, Climatology |
| Statistical | SARIMAX |
| Machine Learning | XGBoost |
| Deep Learning | LSTM, GRU |
| Hybrid | SARIMAX + LSTM |

---

# 🏆 Key Findings

### ✅ XGBoost delivered the best overall performance

- RMSE ≈ **3.74 mm/day**
- NSE ≈ **0.81**

Tree-based learning consistently outperformed deep learning models on this structured meteorological dataset.

---

### ✅ Domain-driven feature engineering improved prediction quality

Engineered lag, rolling-window, and weather-derived features significantly improved forecasting performance.

An engineered **Soil Moisture Gradient** feature emerged as the strongest predictor, demonstrating the value of combining domain knowledge with machine learning.

---

### ✅ Deep learning models were systematically evaluated

LSTM, GRU, and Hybrid forecasting models were implemented and compared under identical evaluation settings to understand their effectiveness for rainfall forecasting.

---

### ✅ Complete Explainability Pipeline

The framework provides:

- Feature importance analysis
- SHAP explainability
- Partial dependence analysis
- Model comparison
- Prediction confidence analysis

making the forecasting process more transparent and interpretable.

---

# 🛠️ Tech Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy
- PyArrow

### Machine Learning

- Scikit-learn
- XGBoost
- Statsmodels

### Deep Learning

- PyTorch

### Explainability

- SHAP

### Visualization

- Plotly
- Matplotlib
- Streamlit

---

# 📂 Project Structure

```text
.
├── config/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard/
│
├── outputs/
│   ├── features/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   └── uncertainty/
│
├── run_all.py
├── sanity_checks.py
├── requirements.txt
└── README.md
```

---

# ⚡ Getting Started

## Clone Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git

cd <repository-name>
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python sanity_checks.py --pre-only
```

---

## Run Complete Pipeline

```bash
python run_all.py
```

The pipeline automatically performs:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Model training
- Model evaluation
- Explainability
- Uncertainty analysis
- Report generation

---

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📷 Dashboard

The Streamlit dashboard provides:

- 📊 Interactive data exploration
- 🌧️ Rainfall forecasting
- 📈 Model comparison
- 🔍 Explainability
- 📉 Prediction confidence analysis
- 💡 Research insights

> *(Add dashboard screenshots here after uploading them.)*

---

# 📈 Future Improvements

- Transformer-based forecasting models
- Real-time weather API integration
- Multi-city forecasting
- Docker containerization
- Cloud deployment
- Improved uncertainty calibration
- Automated model retraining pipeline

---

# 👨‍💻 Author

**Gaurav Singh**

Bachelor of Technology (Civil Engineering)

**Indian Institute of Technology Guwahati**

---

# 📄 License

This repository is intended for academic, educational, and portfolio purposes.
