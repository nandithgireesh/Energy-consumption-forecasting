"""
Day 2 — Data Preprocessing & Feature Engineering Pipeline
==========================================================
Claysys AI Hackathon 2026 | Feb 20, 2026
Runs the complete preprocessing pipeline and saves all artifacts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_RAW    = BASE_DIR / 'data' / 'raw' / 'household_power_consumption.txt'
DATA_PROC   = BASE_DIR / 'data' / 'processed'
MODELS_DIR  = BASE_DIR / 'models'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'

for d in [DATA_PROC, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  DAY 2 — Preprocessing & Feature Engineering")
print("  Claysys AI Hackathon 2026")
print("=" * 60)

# ── STEP 1: Load Raw Data ──────────────────────────────────────────────────────
print("\n📂 STEP 1: Loading raw dataset...")
df = pd.read_csv(
    DATA_RAW, sep=';', na_values='?',
    low_memory=False,
    dtype={
        'Global_active_power'   : 'float32',
        'Global_reactive_power' : 'float32',
        'Voltage'               : 'float32',
        'Global_intensity'      : 'float32',
        'Sub_metering_1'        : 'float32',
        'Sub_metering_2'        : 'float32',
        'Sub_metering_3'        : 'float32',
    }
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.drop(columns=['Date', 'Time'], inplace=True)
df.set_index('Datetime', inplace=True)
df.sort_index(inplace=True)
print(f"   ✅ Loaded {len(df):,} records | {df.index.min()} → {df.index.max()}")
print(f"   Missing values: {df.isna().sum().sum():,} ({df.isna().sum().sum()/df.size*100:.2f}%)")

# ── STEP 2: Handle Missing Values ──────────────────────────────────────────────
print("\n🧹 STEP 2: Handling missing values (linear interpolation)...")
df_clean = df.interpolate(method='linear', limit_direction='both')
remaining_na = df_clean.isna().sum().sum()
print(f"   ✅ Missing after interpolation: {remaining_na}")

# ── STEP 3: Resample to Hourly ─────────────────────────────────────────────────
print("\n⏱️  STEP 3: Resampling 1-minute → Hourly averages...")
df_hourly = df_clean.resample('h').mean(numeric_only=True).dropna()
print(f"   ✅ Hourly records: {len(df_hourly):,}  |  Shape: {df_hourly.shape}")

# ── STEP 4: Outlier Removal ────────────────────────────────────────────────────
print("\n🔍 STEP 4: Removing extreme outliers (Z-score > 4)...")
target_col = 'Global_active_power'
z_scores = (df_hourly[target_col] - df_hourly[target_col].mean()) / df_hourly[target_col].std()
n_before = len(df_hourly)
df_hourly = df_hourly[z_scores.abs() <= 4.0]
print(f"   ✅ Removed {n_before - len(df_hourly)} outliers | Remaining: {len(df_hourly):,}")

# ── STEP 5: Domain-Derived Features ───────────────────────────────────────────
print("\n⚙️  STEP 5: Adding domain-derived features...")
df_feat = df_hourly.copy()
df_feat['apparent_power']       = np.sqrt(df_feat['Global_active_power']**2 + df_feat['Global_reactive_power']**2)
df_feat['power_factor']         = df_feat['Global_active_power'] / (df_feat['apparent_power'] + 1e-8)
df_feat['sub_metering_remainder'] = (
    df_feat['Global_active_power'] * 1000 / 60
    - df_feat['Sub_metering_1']
    - df_feat['Sub_metering_2']
    - df_feat['Sub_metering_3']
)
print("   ✅ apparent_power, power_factor, sub_metering_remainder added")

# ── STEP 6: Calendar + Cyclical Time Features ──────────────────────────────────
print("\n📅 STEP 6: Adding time & cyclical features...")
idx = df_feat.index
df_feat['hour']       = idx.hour
df_feat['day']        = idx.day
df_feat['dayofweek']  = idx.dayofweek
df_feat['month']      = idx.month
df_feat['quarter']    = idx.quarter
df_feat['year']       = idx.year
df_feat['dayofyear']  = idx.dayofyear
df_feat['is_weekend'] = (idx.dayofweek >= 5).astype(int)
df_feat['season']     = df_feat['month'].map({
    12:'winter',1:'winter',2:'winter',
    3:'spring',4:'spring',5:'spring',
    6:'summer',7:'summer',8:'summer',
    9:'autumn',10:'autumn',11:'autumn'
})
# Cyclical encoding
df_feat['hour_sin']  = np.sin(2*np.pi*df_feat['hour']  / 24)
df_feat['hour_cos']  = np.cos(2*np.pi*df_feat['hour']  / 24)
df_feat['month_sin'] = np.sin(2*np.pi*df_feat['month'] / 12)
df_feat['month_cos'] = np.cos(2*np.pi*df_feat['month'] / 12)
df_feat['dow_sin']   = np.sin(2*np.pi*df_feat['dayofweek'] / 7)
df_feat['dow_cos']   = np.cos(2*np.pi*df_feat['dayofweek'] / 7)
print("   ✅ 13 time features added (calendar + cyclical)")

# ── STEP 7: Lag Features ───────────────────────────────────────────────────────
print("\n⏪ STEP 7: Adding lag features...")
lags = [1, 2, 3, 6, 12, 24, 48, 168]
for lag in lags:
    df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
print(f"   ✅ {len(lags)} lag features added: {lags}")

# ── STEP 8: Rolling Statistics ─────────────────────────────────────────────────
print("\n📊 STEP 8: Adding rolling mean & std features...")
windows = [3, 6, 12, 24, 48, 168]
for w in windows:
    df_feat[f'{target_col}_rollmean_{w}'] = df_feat[target_col].shift(1).rolling(w).mean()
    df_feat[f'{target_col}_rollstd_{w}']  = df_feat[target_col].shift(1).rolling(w).std()
print(f"   ✅ {len(windows)*2} rolling features added (mean + std for windows: {windows})")

# Drop NaN rows created by lags/rolling
df_feat = df_feat.dropna()
print(f"\n   Final shape after dropping NaN rows: {df_feat.shape}")
print(f"   Total features: {df_feat.shape[1]}")

# ── STEP 9: Train / Test Split ─────────────────────────────────────────────────
print("\n✂️  STEP 9: Train / Test split (hold out last 3 months)...")
split_date = df_feat.index.max() - pd.DateOffset(months=3)
train_df = df_feat[df_feat.index <= split_date]
test_df  = df_feat[df_feat.index >  split_date]
print(f"   Train: {train_df.index.min()} → {train_df.index.max()} | {len(train_df):,} records")
print(f"   Test : {test_df.index.min()}  → {test_df.index.max()}  | {len(test_df):,} records")

# ── STEP 10: Normalize (fit on train only) ─────────────────────────────────────
print("\n📐 STEP 10: Normalizing features (MinMaxScaler fit on train)...")
scale_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns
              if c not in ['is_weekend','hour','dayofweek','month','quarter','year','dayofyear']]
scaler = MinMaxScaler()
train_scaled = train_df.copy()
test_scaled  = test_df.copy()
train_scaled[scale_cols] = scaler.fit_transform(train_df[scale_cols])
test_scaled[scale_cols]  = scaler.transform(test_df[scale_cols])
joblib.dump(scaler, MODELS_DIR / 'minmax_scaler.pkl')
print(f"   ✅ Scaler saved: models/minmax_scaler.pkl")

# ── STEP 11: Save Processed Files ─────────────────────────────────────────────
print("\n💾 STEP 11: Saving processed datasets...")
df_feat.to_csv(DATA_PROC / 'features_hourly.csv')
train_df.to_csv(DATA_PROC / 'train.csv')
test_df.to_csv(DATA_PROC / 'test.csv')
train_scaled.to_csv(DATA_PROC / 'train_scaled.csv')
test_scaled.to_csv(DATA_PROC / 'test_scaled.csv')
print("   ✅ Saved:")
print("      → data/processed/features_hourly.csv")
print("      → data/processed/train.csv")
print("      → data/processed/test.csv")
print("      → data/processed/train_scaled.csv")
print("      → data/processed/test_scaled.csv")

# ── STEP 12: Visualizations ────────────────────────────────────────────────────
print("\n📈 STEP 12: Generating Day 2 visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

# Plot 1: Outlier removal boxplot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(df_clean.resample('h').mean()[target_col].dropna(), bins=80,
             color='tomato', edgecolor='white', alpha=0.85)
axes[0].set_title('Before Outlier Removal\n(Hourly Avg Distribution)', fontweight='bold')
axes[0].set_xlabel('Global Active Power (kW)')
axes[1].hist(df_hourly[target_col], bins=80, color='steelblue', edgecolor='white', alpha=0.85)
axes[1].set_title('After Outlier Removal (Z>4)\n(Hourly Avg Distribution)', fontweight='bold')
axes[1].set_xlabel('Global Active Power (kW)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day2_outlier_removal.png', bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/day2_outlier_removal.png")

# Plot 2: Train/test split
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_df.index, train_df[target_col], label='Train', color='steelblue', linewidth=0.6, alpha=0.9)
ax.plot(test_df.index,  test_df[target_col],  label='Test (hold-out)', color='tomato', linewidth=0.8)
ax.axvline(test_df.index.min(), color='black', linestyle='--', linewidth=1.5, label='Split point')
ax.set_title('Train / Test Split — Global Active Power', fontweight='bold')
ax.set_ylabel('kW'); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day2_train_test_split.png', bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/day2_train_test_split.png")

# Plot 3: Feature correlation with target (top 20)
numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
corr_target = df_feat[numeric_cols].corr()[target_col].drop(target_col)
corr_sorted = corr_target.abs().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#F44336' if corr_target[f] > 0 else '#2196F3' for f in corr_sorted.index]
ax.barh(corr_sorted.index[::-1], corr_sorted.values[::-1], color=colors[::-1], alpha=0.85)
ax.set_title('Top 20 Features — Correlation with Global Active Power\n(Red = positive, Blue = negative)',
             fontweight='bold')
ax.set_xlabel('|Pearson Correlation|')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day2_feature_correlation.png', bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/day2_feature_correlation.png")

# Plot 4: Rolling mean feature preview
fig, ax = plt.subplots(figsize=(14, 4))
sample = df_feat.iloc[-1000:]
ax.plot(sample.index, sample[target_col], label='Actual', color='steelblue', linewidth=0.8, alpha=0.7)
ax.plot(sample.index, sample[f'{target_col}_rollmean_24'],
        label='24h Rolling Mean', color='darkorange', linewidth=1.5)
ax.plot(sample.index, sample[f'{target_col}_rollmean_168'],
        label='7-day Rolling Mean', color='green', linewidth=1.5, linestyle='--')
ax.set_title('Rolling Mean Features vs Actual (Last 1000 Hours)', fontweight='bold')
ax.set_ylabel('kW'); ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day2_rolling_features.png', bbox_inches='tight')
plt.close()
print("   ✅ Saved: reports/figures/day2_rolling_features.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ DAY 2 COMPLETE!")
print("=" * 60)
print(f"  Raw records       : 2,075,259 (1-minute)")
print(f"  After resampling  : {len(df_hourly):,} (hourly)")
print(f"  After outliers    : {len(df_hourly):,}")
print(f"  Train rows        : {len(train_df):,}")
print(f"  Test  rows        : {len(test_df):,}")
print(f"  Total features    : {df_feat.shape[1]}")
print(f"  Lag features      : {len(lags)}")
print(f"  Rolling features  : {len(windows)*2}")
print(f"  Time features     : 13")
print(f"  Figures saved     : 4")
print("=" * 60)
print("\n🚀 Ready for Day 3: Baseline Statistical Models (ARIMA, Holt-Winters)")
