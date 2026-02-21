"""
Day 3 — Baseline Statistical Models
=====================================
Claysys AI Hackathon 2026 | Feb 21, 2026

Models:
  1. Naive Seasonal Baseline
  2. Holt-Winters (Triple Exponential Smoothing)
  3. ARIMA (Auto-Regressive Integrated Moving Average)

Goal: Establish performance benchmarks that ML/DL models must beat.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_PROC   = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

print("=" * 60)
print("  DAY 3 - Baseline Statistical Models")
print("  Claysys AI Hackathon 2026")
print("=" * 60)

# ── Load Data ──────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading processed train/test data...")
train_df = pd.read_csv(DATA_PROC / 'train.csv', index_col='Datetime', parse_dates=True)
test_df  = pd.read_csv(DATA_PROC / 'test.csv',  index_col='Datetime', parse_dates=True)

TARGET = 'Global_active_power'
train_series = train_df[TARGET].dropna().asfreq('h', method='ffill')
test_series  = test_df[TARGET].dropna().asfreq('h', method='ffill')

print(f"   Train: {len(train_series):,} hourly records")
print(f"   Test : {len(test_series):,} hourly records")
print(f"   Target mean : {train_series.mean():.3f} kW")
print(f"   Target std  : {train_series.std():.3f} kW")

# ── Metrics Helper ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n   [{model_name}] Results:")
    print(f"   MAE  : {mae:.4f} kW")
    print(f"   RMSE : {rmse:.4f} kW")
    print(f"   MAPE : {mape:.2f} %")
    print(f"   R2   : {r2:.4f}")
    return {'Model': model_name, 'MAE': round(mae,4),
            'RMSE': round(rmse,4), 'MAPE': round(mape,2), 'R2': round(r2,4)}

# ── Save plot helper ───────────────────────────────────────────────────────────
def save_prediction_plot(y_true, y_pred, index, model_name, filename, n=168):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    # Prediction vs Actual
    axes[0].plot(index[:n], y_true[:n], label='Actual',
                 color='steelblue', linewidth=1.5, zorder=5)
    axes[0].plot(index[:n], y_pred[:n], label='Predicted',
                 color='darkorange', linewidth=1.2, linestyle='--')
    axes[0].set_title(f'{model_name} — Actual vs Predicted (First 7 Days of Test)',
                      fontweight='bold')
    axes[0].set_ylabel('Global Active Power (kW)')
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

    # Residuals
    residuals = np.array(y_true[:n]) - np.array(y_pred[:n])
    axes[1].bar(range(len(residuals)), residuals,
                color=['tomato' if r > 0 else 'steelblue' for r in residuals],
                alpha=0.7, width=1)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Residuals (Actual - Predicted)', fontweight='bold')
    axes[1].set_ylabel('Error (kW)')
    axes[1].set_xlabel('Hour Index')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"   Saved: reports/figures/{filename}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: Naive Seasonal Baseline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 1] Naive Seasonal Baseline (24-hour repeat)")
print("─" * 60)
print("  Concept: Predict tomorrow = same as yesterday (same hour)")

# Take the last 24 hours from training as the "season"
season = train_series.values[-24:]
n_test = len(test_series)
n_repeats = int(np.ceil(n_test / 24))
naive_preds = np.tile(season, n_repeats)[:n_test]

metrics_naive = compute_metrics(test_series.values, naive_preds, 'Naive Seasonal')
save_prediction_plot(test_series.values, naive_preds, test_series.index,
                     'Naive Seasonal', 'day3_naive_predictions.png')

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: Holt-Winters Exponential Smoothing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 2] Holt-Winters Triple Exponential Smoothing")
print("─" * 60)
print("  Concept: Weighted average of trend + seasonal + level")
print("  Fitting on last 4,000 hourly records for speed...")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

hw_train = train_series.iloc[-4000:]   # ~167 days

hw_model = ExponentialSmoothing(
    hw_train,
    trend='add',
    seasonal='add',
    seasonal_periods=24,
    initialization_method='estimated',
)
print("  Running optimizer (this may take ~30 seconds)...")
hw_result = hw_model.fit(optimized=True, use_brute=False)

hw_preds = hw_result.forecast(n_test).values
metrics_hw = compute_metrics(test_series.values, hw_preds, 'Holt-Winters')
save_prediction_plot(test_series.values, hw_preds, test_series.index,
                     'Holt-Winters', 'day3_holtwinters_predictions.png')

# Print fitted parameters
print(f"\n   Fitted Parameters:")
print(f"   Alpha (level smoothing)  : {hw_result.params['smoothing_level']:.4f}")
print(f"   Beta  (trend smoothing)  : {hw_result.params['smoothing_trend']:.4f}")
print(f"   Gamma (season smoothing) : {hw_result.params['smoothing_seasonal']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: ARIMA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 3] ARIMA(p,d,q) - Auto-Regressive Integrated Moving Average")
print("─" * 60)
print("  Concept: Model autocorrelations in differenced series")

# Check if pmdarima available, else use manual order
try:
    import pmdarima as pm
    print("  Running auto_arima to find best (p,d,q)...")
    print("  Using last 500 hourly records for AutoARIMA search...")
    arima_train = train_series.iloc[-500:]
    auto_model = pm.auto_arima(
        arima_train,
        seasonal=False,
        max_p=5, max_q=5, max_d=2,
        stepwise=True,
        information_criterion='aic',
        error_action='ignore',
        suppress_warnings=True,
    )
    print(f"  Best ARIMA order: {auto_model.order}")
    arima_order = auto_model.order
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    arima_full = StatsARIMA(train_series.iloc[-1500:], order=arima_order).fit()

except ImportError:
    print("  pmdarima not found — using ARIMA(2,1,2) (a good default)")
    arima_order = (2, 1, 2)
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    arima_full = StatsARIMA(train_series.iloc[-1500:], order=arima_order).fit()

print(f"  Forecasting {n_test} steps ahead...")
arima_forecast = arima_full.forecast(steps=n_test)
arima_preds = np.array(arima_forecast)

# Clip negative predictions (power can't be negative)
arima_preds = np.clip(arima_preds, 0, None)

metrics_arima = compute_metrics(test_series.values, arima_preds, f'ARIMA{arima_order}')
save_prediction_plot(test_series.values, arima_preds, test_series.index,
                     f'ARIMA{arima_order}', 'day3_arima_predictions.png')

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON & SUMMARY PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[COMPARISON] Building model comparison table & charts")
print("─" * 60)

all_metrics = [metrics_naive, metrics_hw, metrics_arima]
df_results = pd.DataFrame(all_metrics).set_index('Model')
df_results = df_results.sort_values('RMSE')

print("\n  === BASELINE MODEL LEADERBOARD ===")
print(df_results.to_string())

# Save results
df_results.to_csv(REPORTS_DIR / 'day3_baseline_results.csv')
print(f"\n  Saved: reports/day3_baseline_results.csv")

# Plot 1: Metric comparison bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
colors = ['#F44336', '#FF9800', '#4CAF50']

for ax, metric, color in zip(axes, metrics_to_plot, colors):
    sorted_df = df_results.sort_values(metric)
    bars = ax.bar(sorted_df.index, sorted_df[metric], color=color, alpha=0.85, edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9, fontweight='bold')
    ax.set_title(f'{metric} (lower = better)', fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)

fig.suptitle('Day 3 — Baseline Model Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day3_model_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day3_model_comparison.png")

# Plot 2: All 3 models vs Actual on same chart (first 7 days)
n = 168
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(test_series.index[:n], test_series.values[:n],
        label='Actual', color='black', linewidth=2, zorder=5)
ax.plot(test_series.index[:n], naive_preds[:n],
        label='Naive Seasonal', color='#9C27B0', linewidth=1, linestyle='--', alpha=0.85)
ax.plot(test_series.index[:n], hw_preds[:n],
        label='Holt-Winters', color='#2196F3', linewidth=1.2, linestyle='--', alpha=0.85)
ax.plot(test_series.index[:n], arima_preds[:n],
        label=f'ARIMA{arima_order}', color='#FF9800', linewidth=1.2, linestyle='-.', alpha=0.85)
ax.set_title('Baseline Models — First 7 Days of Test Period', fontweight='bold', fontsize=12)
ax.set_ylabel('Global Active Power (kW)')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day3_all_baselines.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day3_all_baselines.png")

# Plot 3: ACF/PACF to justify ARIMA order
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
diff_series = train_series.diff().dropna().iloc[-500:]
plot_acf(diff_series, lags=48, ax=axes[0], alpha=0.05, color='steelblue')
axes[0].set_title('ACF — Differenced Series (helps choose q)', fontweight='bold')
plot_pacf(diff_series, lags=48, ax=axes[1], alpha=0.05, color='darkorange', method='ywm')
axes[1].set_title('PACF — Differenced Series (helps choose p)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day3_acf_pacf.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day3_acf_pacf.png")

# ── Final Summary ──────────────────────────────────────────────────────────────
best_model = df_results.index[0]
best_rmse  = df_results['RMSE'].iloc[0]
naive_rmse = df_results.loc[df_results.index.str.contains('Naive'), 'RMSE'].values[0]
improvement = (naive_rmse - best_rmse) / naive_rmse * 100

print("\n" + "=" * 60)
print("  DAY 3 COMPLETE!")
print("=" * 60)
print(f"  Models trained   : 3 (Naive, Holt-Winters, ARIMA)")
print(f"  Best model       : {best_model}")
print(f"  Best RMSE        : {best_rmse:.4f} kW")
print(f"  Improvement over Naive: {improvement:.1f}%")
print(f"  Figures saved    : 5")
print("=" * 60)
print("\nReady for Day 4: Classical ML Models (Random Forest, XGBoost, LightGBM)")
print("These should significantly outperform today's baselines!")
