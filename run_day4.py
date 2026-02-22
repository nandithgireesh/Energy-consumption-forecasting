"""
Day 4 — Classical ML Models
=============================
Claysys AI Hackathon 2026 | Feb 22, 2026

Models:
  1. Random Forest Regressor
  2. XGBoost Regressor
  3. LightGBM Regressor
  4. Side-by-side comparison vs Day 3 baselines

Goal: Leverage 45 engineered features to massively outperform
      statistical baselines (target: beat RMSE < 0.9293 kW)
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_PROC   = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR  = BASE_DIR / 'models'

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

print("=" * 60)
print("  DAY 4 - Classical ML Models")
print("  Claysys AI Hackathon 2026")
print("=" * 60)

# ── Metrics Helper ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"\n   [{model_name}]")
    print(f"   MAE  : {mae:.4f} kW")
    print(f"   RMSE : {rmse:.4f} kW")
    print(f"   MAPE : {mape:.2f} %")
    print(f"   R2   : {r2:.4f}")
    return {'Model': model_name, 'MAE': round(mae,4),
            'RMSE': round(rmse,4), 'MAPE': round(mape,2), 'R2': round(r2,4)}

def save_pred_plot(y_true, y_pred, index, model_name, filename, n=168):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(index[:n], y_true[:n], label='Actual',
                 color='steelblue', linewidth=1.5, zorder=5)
    axes[0].plot(index[:n], y_pred[:n], label='Predicted',
                 color='darkorange', linewidth=1.2, linestyle='--')
    axes[0].set_title(f'{model_name} — Actual vs Predicted (First 7 Days)',
                      fontweight='bold')
    axes[0].set_ylabel('Global Active Power (kW)')
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    residuals = np.array(y_true[:n]) - np.array(y_pred[:n])
    axes[1].bar(range(len(residuals)), residuals,
                color=['tomato' if r > 0 else 'steelblue' for r in residuals],
                alpha=0.7, width=1)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Residuals (Actual − Predicted)', fontweight='bold')
    axes[1].set_ylabel('Error (kW)')
    axes[1].set_xlabel('Hour Index')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"   Saved: reports/figures/{filename}")

# ── Load Data ──────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading processed feature data...")
train_df = pd.read_csv(DATA_PROC / 'train.csv', index_col='Datetime', parse_dates=True)
test_df  = pd.read_csv(DATA_PROC / 'test.csv',  index_col='Datetime', parse_dates=True)

TARGET = 'Global_active_power'

# Drop non-numeric and target from features
drop_cols = [TARGET, 'season']
feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns
                if c not in drop_cols]

X_train = train_df[feature_cols]
y_train = train_df[TARGET]
X_test  = test_df[feature_cols]
y_test  = test_df[TARGET]

# Validation set = last 10% of train (for early stopping in XGB/LGBM)
val_size = int(len(X_train) * 0.10)
X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

print(f"   Train features : {X_train.shape}")
print(f"   Test  features : {X_test.shape}")
print(f"   Feature count  : {len(feature_cols)}")
print(f"   Features used  : {feature_cols[:8]} ...")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 1] Random Forest Regressor")
print("─" * 60)
print("  Concept: Ensemble of 200 decision trees — each trained on")
print("  a random subset of data & features, then averaged.")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    oob_score=True,     # Out-of-bag evaluation (free val set)
)
print("  Training Random Forest (200 trees)...")
rf.fit(X_train, y_train)
print(f"  OOB Score (internal validation): {rf.oob_score_:.4f}")

rf_preds = rf.predict(X_test)
metrics_rf = compute_metrics(y_test.values, rf_preds, 'Random Forest')
save_pred_plot(y_test.values, rf_preds, test_df.index,
               'Random Forest', 'day4_rf_predictions.png')

# Feature importance
fi = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
top_fi = fi.head(20)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_fi)))[::-1]
ax.barh(top_fi['feature'][::-1], top_fi['importance'][::-1], color=colors, alpha=0.9)
ax.set_title('Random Forest — Top 20 Feature Importances', fontweight='bold')
ax.set_xlabel('Importance (Gini)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day4_rf_feature_importance.png', bbox_inches='tight')
plt.close()
print("   Saved: reports/figures/day4_rf_feature_importance.png")
print(f"\n   Top 5 important features:")
for _, row in fi.head(5).iterrows():
    print(f"   {row['feature']:40s}  {row['importance']:.4f}")

# Save model
joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')
print("   Model saved: models/random_forest.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 2] XGBoost Regressor")
print("─" * 60)
print("  Concept: Gradient boosting — sequentially builds trees,")
print("  each correcting errors of the previous one.")

try:
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='rmse',
        early_stopping_rounds=50,
    )

    print("  Training XGBoost (up to 1000 rounds, early stopping at 50)...")
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = xgb.best_iteration
    print(f"  Best iteration: {best_iter}")

    xgb_preds = xgb.predict(X_test)
    metrics_xgb = compute_metrics(y_test.values, xgb_preds, 'XGBoost')
    save_pred_plot(y_test.values, xgb_preds, test_df.index,
                   'XGBoost', 'day4_xgb_predictions.png')
    joblib.dump(xgb, MODELS_DIR / 'xgboost.pkl')
    print("   Model saved: models/xgboost.pkl")

    # XGB Feature importance
    xgb_fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n   Top 5 XGBoost features:")
    for _, row in xgb_fi.head(5).iterrows():
        print(f"   {row['feature']:40s}  {row['importance']:.4f}")

except ImportError:
    print("  XGBoost not available — skipping")
    metrics_xgb = {'Model':'XGBoost','MAE':None,'RMSE':None,'MAPE':None,'R2':None}
    xgb_preds = np.zeros(len(y_test))

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: LightGBM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 3] LightGBM Regressor")
print("─" * 60)
print("  Concept: Gradient boosting with leaf-wise tree growth")
print("  (faster than XGBoost, often similar accuracy)")

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("  Training LightGBM (up to 1000 rounds, early stopping at 50)...")
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
    )
    print(f"  Best iteration: {lgbm.best_iteration_}")

    lgbm_preds = lgbm.predict(X_test)
    metrics_lgbm = compute_metrics(y_test.values, lgbm_preds, 'LightGBM')
    save_pred_plot(y_test.values, lgbm_preds, test_df.index,
                   'LightGBM', 'day4_lgbm_predictions.png')
    joblib.dump(lgbm, MODELS_DIR / 'lightgbm.pkl')
    print("   Model saved: models/lightgbm.pkl")

    # LGBM Feature importance
    lgbm_fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n   Top 5 LightGBM features:")
    for _, row in lgbm_fi.head(5).iterrows():
        print(f"   {row['feature']:40s}  {row['importance']:.4f}")

except ImportError:
    print("  LightGBM not available — skipping")
    metrics_lgbm = {'Model':'LightGBM','MAE':None,'RMSE':None,'MAPE':None,'R2':None}
    lgbm_preds = np.zeros(len(y_test))

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS & LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[COMPARISON] Building full leaderboard vs Day 3 baselines")
print("─" * 60)

# Load Day 3 results
day3_df = pd.read_csv(REPORTS_DIR / 'day3_baseline_results.csv', index_col=0)

# Today's results
day4_metrics = [metrics_rf, metrics_xgb, metrics_lgbm]
day4_df = pd.DataFrame([m for m in day4_metrics if m['RMSE'] is not None]).set_index('Model')

# Combined leaderboard
all_df = pd.concat([day4_df, day3_df]).sort_values('RMSE')

print("\n  === FULL LEADERBOARD (Day 3 + Day 4) ===")
print(all_df[['MAE','RMSE','MAPE','R2']].to_string())

all_df.to_csv(REPORTS_DIR / 'day4_all_results.csv')
print(f"\n  Saved: reports/day4_all_results.csv")

# Plot: All models RMSE comparison
fig, ax = plt.subplots(figsize=(12, 5))
colors_map = {
    'Random Forest' : '#2196F3',
    'XGBoost'       : '#FF9800',
    'LightGBM'      : '#4CAF50',
    'ARIMA(1, 1, 1)': '#9C27B0',
    'Naive Seasonal': '#607D8B',
    'Holt-Winters'  : '#F44336',
}
bar_colors = [colors_map.get(m, '#78909C') for m in all_df.index]
bars = ax.bar(all_df.index, all_df['RMSE'], color=bar_colors, alpha=0.88, edgecolor='white', linewidth=1.2)
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9, fontweight='bold')
ax.axhline(all_df.loc[all_df.index.str.contains('Naive'), 'RMSE'].values[0],
           color='gray', linestyle='--', linewidth=1.2, label='Naive baseline')
ax.set_title('Day 4 — All Models RMSE Comparison\n(ML Models vs Statistical Baselines)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('RMSE (kW)')
ax.legend()
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day4_full_leaderboard.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day4_full_leaderboard.png")

# Plot: All 3 ML models vs Actual (first week)
n = 168
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(test_df.index[:n], y_test.values[:n],
        label='Actual', color='black', linewidth=2, zorder=5)
ax.plot(test_df.index[:n], rf_preds[:n],
        label='Random Forest', color='#2196F3', linewidth=1.2, linestyle='--', alpha=0.85)
if metrics_xgb['RMSE']:
    ax.plot(test_df.index[:n], xgb_preds[:n],
            label='XGBoost', color='#FF9800', linewidth=1.2, linestyle='--', alpha=0.85)
if metrics_lgbm['RMSE']:
    ax.plot(test_df.index[:n], lgbm_preds[:n],
            label='LightGBM', color='#4CAF50', linewidth=1.2, linestyle='-.', alpha=0.85)
ax.set_title('ML Models — First 7 Days of Test Period', fontweight='bold', fontsize=12)
ax.set_ylabel('Global Active Power (kW)')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day4_ml_comparison_week.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day4_ml_comparison_week.png")

# Plot: Combined Feature Importance (RF + XGB + LGBM)
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (fi_df, name, color) in zip(axes, [
        (fi,      'Random Forest', '#2196F3'),
        (xgb_fi,  'XGBoost',       '#FF9800'),
        (lgbm_fi, 'LightGBM',      '#4CAF50'),
    ]):
        top = fi_df.head(15)
        ax.barh(top['feature'][::-1], top['importance'][::-1],
                color=color, alpha=0.85)
        ax.set_title(f'{name}\nTop 15 Features', fontweight='bold')
        ax.set_xlabel('Importance')
    fig.suptitle('Feature Importance Comparison Across ML Models',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'day4_feature_importance_all.png', bbox_inches='tight')
    plt.close()
    print("  Saved: reports/figures/day4_feature_importance_all.png")
except Exception:
    pass

# ── Final Summary ──────────────────────────────────────────────────────────────
best_today = day4_df.sort_values('RMSE').index[0]
best_rmse  = day4_df.sort_values('RMSE')['RMSE'].iloc[0]
naive_rmse = 0.9408  # Day 3 Naive baseline
improvement = (naive_rmse - best_rmse) / naive_rmse * 100

print("\n" + "=" * 60)
print("  DAY 4 COMPLETE!")
print("=" * 60)
print(f"  Models trained   : 3 (RF, XGBoost, LightGBM)")
print(f"  Best ML model    : {best_today}")
print(f"  Best RMSE        : {best_rmse:.4f} kW")
print(f"  Improvement vs Naive baseline : {improvement:.1f}%")
print(f"  Improvement vs ARIMA baseline : {((0.9293-best_rmse)/0.9293*100):.1f}%")
print(f"  Figures saved    : 6")
print("=" * 60)
print("\nReady for Day 5: Deep Learning — LSTM & GRU (PyTorch)")
print(f"Target to beat: RMSE < {best_rmse:.4f} kW")
