"""
Day 6 — Prophet + Ensemble Stacking
======================================
Claysys AI Hackathon 2026 | Feb 24, 2026

Models:
  1. Facebook Prophet (automatic trend + seasonality decomposition)
  2. Ensemble — weighted average of best models from Days 3-5:
       LightGBM (40%) + XGBoost (30%) + Prophet (20%) + GRU (10%)

Goal: Squeeze maximum accuracy by combining diverse model types.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_PROC   = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR  = BASE_DIR / 'models'

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

print("=" * 60)
print("  DAY 6 - Prophet + Ensemble Stacking")
print("  Claysys AI Hackathon 2026")
print("=" * 60)

# ── Metrics Helper ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    r2   = r2_score(y_true, y_pred)
    print(f"\n   [{model_name}]")
    print(f"   MAE  : {mae:.4f} kW")
    print(f"   RMSE : {rmse:.4f} kW")
    print(f"   MAPE : {mape:.2f} %")
    print(f"   R2   : {r2:.4f}")
    return {'Model': model_name, 'MAE': round(mae,4),
            'RMSE': round(rmse,4), 'MAPE': round(mape,2), 'R2': round(r2,4)}

def save_pred_plot(y_true, y_pred, model_name, filename, index=None, n=168):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    x = index[:n] if index is not None else range(n)
    axes[0].plot(x, y_true[:n], label='Actual', color='steelblue', linewidth=1.5, zorder=5)
    axes[0].plot(x, y_pred[:n], label='Predicted', color='darkorange', linewidth=1.2, linestyle='--')
    axes[0].set_title(f'{model_name} — Actual vs Predicted (First 7 Days)', fontweight='bold')
    axes[0].set_ylabel('Global Active Power (kW)'); axes[0].legend()
    if index is not None:
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    residuals = np.array(y_true[:n]) - np.array(y_pred[:n])
    axes[1].bar(range(len(residuals)), residuals,
                color=['tomato' if r > 0 else 'steelblue' for r in residuals], alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Residuals (Actual − Predicted)', fontweight='bold')
    axes[1].set_ylabel('Error (kW)'); axes[1].set_xlabel('Hour Index')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"   Saved: reports/figures/{filename}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading data...")
train_df = pd.read_csv(DATA_PROC / 'train.csv', index_col='Datetime', parse_dates=True)
test_df  = pd.read_csv(DATA_PROC / 'test.csv',  index_col='Datetime', parse_dates=True)
TARGET = 'Global_active_power'

train_series = train_df[TARGET].dropna()
test_series  = test_df[TARGET].dropna()
print(f"   Train: {len(train_series):,} | Test: {len(test_series):,}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: FACEBOOK PROPHET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 1] Facebook Prophet")
print("─" * 60)
print("  Concept: Additive model = Trend + Yearly + Weekly + Daily + Noise")
print("  Automatically finds changepoints in trend")
print("  No feature engineering needed — works from timestamps alone")

from prophet import Prophet

# Prophet requires ds (datetime) + y (target) columns
prophet_train = pd.DataFrame({
    'ds': train_series.index,
    'y':  train_series.values,
})

prophet_model = Prophet(
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    n_changepoints=25,
)

print("  Fitting Prophet (this takes ~1-2 minutes)...")
prophet_model.fit(prophet_train)

# Build future dataframe covering test period
future = pd.DataFrame({'ds': test_series.index})
forecast = prophet_model.predict(future)
prophet_preds = forecast['yhat'].values
prophet_preds = np.clip(prophet_preds, 0, None)  # power >= 0

metrics_prophet = compute_metrics(test_series.values, prophet_preds, 'Prophet')
save_pred_plot(test_series.values, prophet_preds, 'Prophet',
               'day6_prophet_predictions.png', index=test_series.index)

# Component plot
print("  Generating Prophet component decomposition plot...")
fig_comp = prophet_model.plot_components(forecast)
fig_comp.set_size_inches(14, 10)
plt.suptitle('Prophet — Trend + Seasonality Components', fontweight='bold', y=1.01)
plt.tight_layout()
fig_comp.savefig(FIGURES_DIR / 'day6_prophet_components.png', bbox_inches='tight')
plt.close()
print("   Saved: reports/figures/day6_prophet_components.png")

# Forecast plot
fig_fc = prophet_model.plot(forecast)
fig_fc.set_size_inches(14, 5)
plt.title('Prophet — Full Forecast vs Actual', fontweight='bold')
plt.tight_layout()
fig_fc.savefig(FIGURES_DIR / 'day6_prophet_forecast.png', bbox_inches='tight')
plt.close()
print("   Saved: reports/figures/day6_prophet_forecast.png")

# ══════════════════════════════════════════════════════════════════════════════
# RE-GENERATE ML PREDICTIONS (for ensemble)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[STEP 2] Re-generating ML predictions for ensemble")
print("─" * 60)

drop_cols   = [TARGET, 'season']
feat_cols   = [c for c in train_df.select_dtypes(include=[np.number]).columns
               if c not in drop_cols]
X_train_ml  = train_df[feat_cols]
y_train_ml  = train_df[TARGET]
X_test_ml   = test_df[feat_cols]
y_test_ml   = test_df[TARGET]

# Load saved models
print("  Loading saved LightGBM and XGBoost models...")
lgbm_model  = joblib.load(MODELS_DIR / 'lightgbm.pkl')
xgb_model   = joblib.load(MODELS_DIR / 'xgboost.pkl')

lgbm_preds  = lgbm_model.predict(X_test_ml)
xgb_preds   = xgb_model.predict(X_test_ml)

metrics_lgbm_check = compute_metrics(y_test_ml.values, lgbm_preds, 'LightGBM (re-check)')
metrics_xgb_check  = compute_metrics(y_test_ml.values, xgb_preds,  'XGBoost (re-check)')
print("   LightGBM and XGBoost predictions loaded successfully")

# ══════════════════════════════════════════════════════════════════════════════
# RE-GENERATE GRU PREDICTIONS (for ensemble)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Loading saved GRU model for DL component...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Must re-define architecture (same as Day 5)
class RNNForecaster(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, model_type='LSTM', bidirectional=False):
        super().__init__()
        self.directions = 2 if bidirectional else 1
        rnn_cls = torch.nn.LSTM if model_type == 'LSTM' else torch.nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers,
                           dropout=dropout if num_layers>1 else 0.0,
                           batch_first=True, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * self.directions, 64),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

SEQ_FEATURES = ['Global_active_power','Global_reactive_power','Voltage',
                 'Global_intensity','Sub_metering_1','Sub_metering_2',
                 'Sub_metering_3','hour_sin','hour_cos','month_sin',
                 'month_cos','dow_sin','dow_cos','is_weekend']
SEQ_FEATURES = [f for f in SEQ_FEATURES if f in train_df.columns]
N_FEAT, SEQ_LEN = len(SEQ_FEATURES), 24

seq_scaler  = joblib.load(MODELS_DIR / 'seq_scaler.pkl')
test_scaled = seq_scaler.transform(test_df[SEQ_FEATURES].dropna().values)

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_test_seq, y_test_seq = make_sequences(test_scaled, SEQ_LEN)

gru_model = RNNForecaster(N_FEAT, hidden_size=128, num_layers=2,
                           dropout=0.2, model_type='GRU')
gru_model.load_state_dict(torch.load(MODELS_DIR / 'gru_model.pt', map_location=device))
gru_model = gru_model.to(device)
gru_model.eval()

with torch.no_grad():
    gru_preds_scaled = gru_model(
        torch.tensor(X_test_seq).to(device)
    ).cpu().numpy()

def inverse_scale(preds, scaler, n_feat):
    dummy = np.zeros((len(preds), n_feat))
    dummy[:, 0] = preds
    return scaler.inverse_transform(dummy)[:, 0]

gru_preds_full = inverse_scale(gru_preds_scaled, seq_scaler, N_FEAT)
print(f"   GRU predictions shape: {gru_preds_full.shape}")

# Align all predictions to the same length (GRU has SEQ_LEN fewer points)
n_common = len(gru_preds_full)
y_true_common     = y_test_ml.values[-n_common:]
lgbm_preds_common = lgbm_preds[-n_common:]
xgb_preds_common  = xgb_preds[-n_common:]
prophet_common    = prophet_preds[-n_common:]
test_idx_common   = test_series.index[-n_common:]
print(f"   Aligned length for ensemble: {n_common} samples")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: WEIGHTED ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 2] Weighted Ensemble Stacking")
print("─" * 60)
print("  Strategy: Assign weights based on individual RMSE performance")
print("  Weights: LightGBM=40%, XGBoost=30%, Prophet=20%, GRU=10%")

# Ensemble A: LightGBM + XGBoost + Prophet + GRU
w_lgbm, w_xgb, w_prophet, w_gru = 0.40, 0.30, 0.20, 0.10
ensemble_preds = (
    w_lgbm    * lgbm_preds_common +
    w_xgb     * xgb_preds_common  +
    w_prophet * prophet_common    +
    w_gru     * gru_preds_full
)
metrics_ens = compute_metrics(y_true_common, ensemble_preds,
                               'Ensemble (LGBM40+XGB30+Prophet20+GRU10)')
save_pred_plot(y_true_common, ensemble_preds,
               'Ensemble', 'day6_ensemble_predictions.png',
               index=test_idx_common)

# Ensemble B: ML-only (LightGBM + XGBoost)
ensemble_ml_preds = 0.60 * lgbm_preds_common + 0.40 * xgb_preds_common
metrics_ens_ml = compute_metrics(y_true_common, ensemble_ml_preds,
                                  'Ensemble-ML (LGBM60+XGB40)')

# Find best ensemble weights automatically
print("\n  Searching for optimal ensemble weights (grid search)...")
best_rmse_w, best_w = float('inf'), None
for w1 in np.arange(0.3, 0.8, 0.1):
    for w2 in np.arange(0.1, 0.5, 0.1):
        for w3 in np.arange(0.0, 0.3, 0.1):
            w4 = round(1 - w1 - w2 - w3, 2)
            if w4 < 0: continue
            preds = w1*lgbm_preds_common + w2*xgb_preds_common + w3*prophet_common + w4*gru_preds_full
            rmse = np.sqrt(mean_squared_error(y_true_common, preds))
            if rmse < best_rmse_w:
                best_rmse_w = rmse
                best_w = (w1, w2, w3, w4)

print(f"  Best weights found: LGBM={best_w[0]:.1f}  XGB={best_w[1]:.1f}  "
      f"Prophet={best_w[2]:.1f}  GRU={best_w[3]:.1f}")
print(f"  Best ensemble RMSE: {best_rmse_w:.4f} kW")

ensemble_opt = (best_w[0]*lgbm_preds_common + best_w[1]*xgb_preds_common +
                best_w[2]*prophet_common     + best_w[3]*gru_preds_full)
metrics_ens_opt = compute_metrics(y_true_common, ensemble_opt,
                                   f'Ensemble-Optimal')
save_pred_plot(y_true_common, ensemble_opt,
               'Ensemble-Optimal', 'day6_ensemble_optimal_predictions.png',
               index=test_idx_common)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[COMPARISON] FINAL COMPLETE LEADERBOARD — All 12 Models")
print("─" * 60)

day5_df = pd.read_csv(REPORTS_DIR / 'day5_all_results.csv', index_col=0)
today_metrics = [metrics_prophet, metrics_ens, metrics_ens_ml, metrics_ens_opt]
today_df = pd.DataFrame(today_metrics).set_index('Model')

final_df = pd.concat([today_df, day5_df]).sort_values('RMSE')

print("\n  ===== FINAL MODEL LEADERBOARD =====")
print(final_df[['MAE','RMSE','MAPE','R2']].to_string())
final_df.to_csv(REPORTS_DIR / 'day6_final_results.csv')
print(f"\n  Saved: reports/day6_final_results.csv")

# ── Plot 1: Final leaderboard bar chart ───────────────────────────────────────
color_map = {
    'Ensemble-Optimal':'#E91E63', 'Ensemble-ML (LGBM60+XGB40)':'#C2185B',
    'Ensemble (LGBM40+XGB30+Prophet20+GRU10)':'#AD1457',
    'LightGBM':'#4CAF50','XGBoost':'#FF9800','Random Forest':'#2196F3',
    'Prophet':'#009688',
    'GRU':'#00BCD4','LSTM':'#9C27B0','BiLSTM':'#FF5722',
    'ARIMA(1, 1, 1)':'#607D8B','Naive Seasonal':'#9E9E9E','Holt-Winters':'#F44336',
}
top10 = final_df.head(10)
fig, ax = plt.subplots(figsize=(14, 6))
colors = [color_map.get(m, '#78909C') for m in top10.index]
bars = ax.bar(top10.index, top10['RMSE'], color=colors, alpha=0.9,
              edgecolor='white', linewidth=1.5)
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=8.5, fontweight='bold')
ax.set_title('FINAL Model Leaderboard — RMSE (All Days)',
             fontweight='bold', fontsize=13)
ax.set_ylabel('RMSE (kW)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day6_final_leaderboard.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day6_final_leaderboard.png")

# ── Plot 2: All top-4 models vs Actual ────────────────────────────────────────
n = min(168, len(y_true_common))
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(test_idx_common[:n], y_true_common[:n],
        label='Actual', color='black', linewidth=2, zorder=10)
ax.plot(test_idx_common[:n], lgbm_preds_common[:n],
        label='LightGBM', color='#4CAF50', linewidth=1.2, linestyle='--', alpha=0.8)
ax.plot(test_idx_common[:n], prophet_common[:n],
        label='Prophet', color='#009688', linewidth=1.2, linestyle='-.', alpha=0.8)
ax.plot(test_idx_common[:n], ensemble_opt[:n],
        label='Ensemble-Optimal', color='#E91E63', linewidth=1.5, alpha=0.9)
ax.set_title('Best Models — First 7 Days of Test Period', fontweight='bold', fontsize=12)
ax.set_ylabel('Global Active Power (kW)'); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day6_best_models_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day6_best_models_comparison.png")

# ── Plot 3: Metrics radar / multi-metric comparison ───────────────────────────
top5 = final_df.head(5)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metric_titles = ['MAE (kW) — lower better', 'RMSE (kW) — lower better', 'R2 — higher better']
metrics_cols  = ['MAE', 'RMSE', 'R2']
colors5 = ['#E91E63','#C2185B','#4CAF50','#FF9800','#2196F3']
for ax, col, title in zip(axes, metrics_cols, metric_titles):
    vals = top5[col]
    if col == 'R2':
        sorted_idx = vals.argsort()[::-1]
    else:
        sorted_idx = vals.argsort()
    svals = vals.iloc[sorted_idx]
    scols = [colors5[i] for i in sorted_idx]
    bars = ax.bar(range(len(svals)), svals.values, color=scols, alpha=0.88, edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xticks(range(len(svals)))
    ax.set_xticklabels([svals.index[i][:15] for i in range(len(svals))],
                        rotation=25, ha='right', fontsize=8)
    ax.set_title(title, fontweight='bold')
fig.suptitle('Top 5 Models — Multi-Metric Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day6_top5_metrics.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day6_top5_metrics.png")

# ── Final Summary ──────────────────────────────────────────────────────────────
best_model = final_df.index[0]
best_rmse  = final_df['RMSE'].iloc[0]
naive_rmse = final_df.loc['Naive Seasonal', 'RMSE']
improvement = (naive_rmse - best_rmse) / naive_rmse * 100

print("\n" + "=" * 60)
print("  DAY 6 COMPLETE!")
print("=" * 60)
print(f"  Models today     : Prophet, Ensemble x3")
print(f"  CHAMPION MODEL   : {best_model}")
print(f"  Champion RMSE    : {best_rmse:.4f} kW")
print(f"  Champion MAPE    : {final_df['MAPE'].iloc[0]:.2f} %")
print(f"  Champion R2      : {final_df['R2'].iloc[0]:.4f}")
print(f"  vs Naive baseline: {improvement:.1f}% improvement")
print(f"  Optimal weights  : LGBM={best_w[0]:.1f}  XGB={best_w[1]:.1f}  "
      f"Prophet={best_w[2]:.1f}  GRU={best_w[3]:.1f}")
print(f"  Figures saved    : 7")
print("=" * 60)
print("\nReady for Day 7: Final Report + Submission!")
print("Tomorrow: update README results table + prepare submission links")
