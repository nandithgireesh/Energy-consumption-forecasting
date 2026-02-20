# ⚡ Energy Consumption Forecasting

> **Claysys AI Hackathon 2026** — Tabular Data Project  
> Predicting future household energy usage using time series analysis and machine learning.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](notebooks/)

---

## 📌 Project Overview

This project builds a robust **Energy Consumption Forecasting** system for a household, using ~4 years of minute-level electricity measurements. The goal is to predict **Global Active Power** consumption over future time horizons using multiple AI/ML approaches including statistical methods, classical ML, and deep learning.

### Problem Statement
Accurate energy forecasting is critical for:
- ⚡ **Grid stability** — utilities need reliable demand forecasts
- 💰 **Cost optimization** — consumers can avoid peak-price usage  
- 🌱 **Sustainability** — better planning reduces carbon footprint

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | [UCI Machine Learning Repository — Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) |
| **File** | `household_power_consumption.txt` |
| **Period** | December 2006 – November 2010 (~4 years) |
| **Granularity** | 1-minute intervals |
| **Records** | 2,075,259 rows |
| **Missing values** | ~1.25% (marked as `?`) |

### Features

| Column | Description | Unit |
|--------|-------------|------|
| `Date` | Date in DD/MM/YYYY format | — |
| `Time` | Time in HH:MM:SS format | — |
| `Global_active_power` | Household global minute-averaged active power | kilowatt |
| `Global_reactive_power` | Household global minute-averaged reactive power | kilowatt |
| `Voltage` | Minute-averaged voltage | volt |
| `Global_intensity` | Household global minute-averaged current intensity | ampere |
| `Sub_metering_1` | Energy sub-metering No. 1 (kitchen) | watt-hour |
| `Sub_metering_2` | Energy sub-metering No. 2 (laundry room) | watt-hour |
| `Sub_metering_3` | Energy sub-metering No. 3 (water heater & AC) | watt-hour |

---

## 🏗️ Project Structure

```
energy-consumption-forecasting/
│
├── data/                          # Dataset files
│   ├── raw/                       # Original dataset
│   └── processed/                 # Cleaned & engineered features
│
├── notebooks/                     # Jupyter notebooks (one per day)
│   ├── Day1_EDA.ipynb             # Exploratory Data Analysis
│   ├── Day2_Preprocessing.ipynb   # Data Cleaning & Feature Engineering
│   ├── Day3_Baseline_Models.ipynb # ARIMA, Holt-Winters
│   ├── Day4_ML_Models.ipynb       # Random Forest, XGBoost
│   ├── Day5_Deep_Learning.ipynb   # LSTM, GRU with PyTorch
│   ├── Day6_Prophet.ipynb         # Facebook Prophet + Ensemble
│   └── Day7_Final_Report.ipynb    # Final evaluation & dashboard
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Data cleaning & feature engineering
│   ├── features.py                # Feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py            # ARIMA, Holt-Winters
│   │   ├── ml_models.py           # Random Forest, XGBoost
│   │   ├── lstm_model.py          # PyTorch LSTM/GRU
│   │   └── prophet_model.py       # Facebook Prophet
│   ├── evaluation.py              # Metrics & visualization
│   └── utils.py                   # Helper functions
│
├── models/                        # Saved trained models
│   └── .gitkeep
│
├── reports/                       # Generated reports & figures
│   └── figures/                   # Plots and charts
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/energy-consumption-forecasting.git
cd energy-consumption-forecasting
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Add the Dataset
Place `household_power_consumption.txt` in the `data/raw/` folder:
```
data/raw/household_power_consumption.txt
```

### Step 5: Launch Jupyter Notebooks
```bash
jupyter lab
```

---

## 📓 Google Colab

Run the full project pipeline on Google Colab (no local setup required):

> **[📎 Open in Google Colab](#)** ← *(link to be added)*

---

## 🔬 Methodology

### 7-Day Development Plan

| Day | Focus | Deliverables |
|-----|-------|-------------|
| **Day 1** | Data Exploration (EDA) | Statistical summary, visualizations, patterns |
| **Day 2** | Preprocessing & Feature Engineering | Clean data, lag features, time features |
| **Day 3** | Baseline Models | ARIMA, Holt-Winters, naive forecasts |
| **Day 4** | Classical ML Models | Random Forest, XGBoost, LightGBM |
| **Day 5** | Deep Learning — LSTM/GRU | Sequence models with PyTorch |
| **Day 6** | Prophet + Ensemble | Meta's Prophet + model stacking |
| **Day 7** | Final Evaluation & Report | Model comparison, dashboard, report |

### Forecasting Approaches
1. **Statistical**: ARIMA, SARIMA, Holt-Winters Exponential Smoothing
2. **Machine Learning**: Random Forest, XGBoost, LightGBM (with lag features)
3. **Deep Learning**: LSTM, GRU (PyTorch)
4. **Prophet**: Facebook's additive decomposition model
5. **Ensemble**: Stacking best models for optimal performance

### Evaluation Metrics
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error  
- **MAPE** — Mean Absolute Percentage Error
- **R²** — Coefficient of Determination

---

## 📈 Results Summary

> *(To be updated daily as models are trained)*

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Naive Baseline | — | — | — | — |
| Holt-Winters | — | — | — | — |
| ARIMA | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| LSTM | — | — | — | — |
| Prophet | — | — | — | — |
| **Ensemble** | **—** | **—** | **—** | **—** |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Statistical Models** | Statsmodels (ARIMA, Holt-Winters) |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Deep Learning** | PyTorch |
| **Time Series** | Prophet (Meta), pmdarima |
| **Notebook** | JupyterLab |

---

## 🗓️ Daily Progress Log

### Day 1 — Feb 19, 2026 — Environment Setup & EDA
- ✅ Set up project structure and GitHub repository
- ✅ Loaded and explored the household power consumption dataset
- ✅ Performed statistical analysis (2,075,259 records, Dec 2006–Nov 2010)
- ✅ Identified missing values (~1.25%) and data patterns
- ✅ Created initial visualizations: time series plots, distribution analysis

### Day 2 — Feb 20, 2026 — Preprocessing & Feature Engineering  
*(To be updated)*

### Day 3 — Feb 21, 2026 — Baseline Statistical Models  
*(To be updated)*

### Day 4 — Feb 22, 2026 — Classical ML Models  
*(To be updated)*

### Day 5 — Feb 23, 2026 — Deep Learning (LSTM/GRU)  
*(To be updated)*

### Day 6 — Feb 24, 2026 — Prophet + Ensemble  
*(To be updated)*

### Day 7 — Feb 25, 2026 — Final Report & Submission  
*(To be updated)*

---

## 👤 Author

**Your Name**  
Claysys AI Hackathon 2026  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/<your-username>)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Submission Links

- **GitHub Repository**: [This Repository](#)
- **Google Colab Notebook**: *(link to be added)*
- **YouTube Demo Video**: *(To be uploaded — unlisted link)*
- **Submission Form**: [https://forms.office.com/r/yjUQQ8fFa9](https://forms.office.com/r/yjUQQ8fFa9)
