"""
Day 5 — Deep Learning: LSTM & GRU with PyTorch
=================================================
Claysys AI Hackathon 2026 | Feb 23, 2026

Models:
  1. LSTM  (Long Short-Term Memory)
  2. GRU   (Gated Recurrent Unit)
  3. Bidirectional LSTM (Bonus)

Approach: Multivariate sliding-window sequences
  - Look-back  : 24 hours
  - Features   : 7 raw electrical measurements + cyclical time features
  - Target     : Global_active_power (t+1)

Goal: Capture temporal dependencies — try to beat LightGBM RMSE=0.0077
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_PROC   = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR  = BASE_DIR / 'models'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

print("=" * 60)
print("  DAY 5 - Deep Learning: LSTM & GRU")
print("  Claysys AI Hackathon 2026")
print("=" * 60)
print(f"\n  Device : {device}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load & Prepare Sequence Data
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading & preparing sequence data...")

train_df = pd.read_csv(DATA_PROC / 'train.csv', index_col='Datetime', parse_dates=True)
test_df  = pd.read_csv(DATA_PROC / 'test.csv',  index_col='Datetime', parse_dates=True)

TARGET = 'Global_active_power'

# Select multivariate features for LSTM input
# Use electrical readings + cyclical time (avoid lag/rolling to keep it pure DL)
SEQ_FEATURES = [
    'Global_active_power',     # target — always col 0
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos',
    'is_weekend',
]
SEQ_FEATURES = [f for f in SEQ_FEATURES if f in train_df.columns]

train_data = train_df[SEQ_FEATURES].dropna().values
test_data  = test_df[SEQ_FEATURES].dropna().values

# Scale to [0, 1]
seq_scaler = MinMaxScaler()
train_scaled = seq_scaler.fit_transform(train_data)
test_scaled  = seq_scaler.transform(test_data)
joblib.dump(seq_scaler, MODELS_DIR / 'seq_scaler.pkl')

SEQ_LEN  = 24    # 24-hour look-back window
N_FEAT   = len(SEQ_FEATURES)

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])          # shape: (seq_len, n_features)
        y.append(data[i + seq_len, 0])            # col 0 = target (t+1)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train_seq, y_train_seq = make_sequences(train_scaled, SEQ_LEN)
X_test_seq,  y_test_seq  = make_sequences(test_scaled,  SEQ_LEN)

# Validation split (last 10% of sequences)
val_sz = int(len(X_train_seq) * 0.10)
X_tr, X_val = X_train_seq[:-val_sz], X_train_seq[-val_sz:]
y_tr, y_val = y_train_seq[:-val_sz], y_train_seq[-val_sz:]

print(f"   Features in sequence : {N_FEAT}  {SEQ_FEATURES}")
print(f"   Sequence length      : {SEQ_LEN} hours")
print(f"   X_train shape        : {X_tr.shape}")
print(f"   X_val   shape        : {X_val.shape}")
print(f"   X_test  shape        : {X_test_seq.shape}")

# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

BATCH_SIZE = 128
train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr),    batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TimeSeriesDataset(X_val, y_val),  batch_size=BATCH_SIZE)
test_loader  = DataLoader(TimeSeriesDataset(X_test_seq, y_test_seq), batch_size=BATCH_SIZE)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
class RNNForecaster(nn.Module):
    """Flexible RNN model supporting LSTM, GRU, and Bidirectional variants."""
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, model_type='LSTM', bidirectional=False):
        super().__init__()
        self.model_type    = model_type
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if model_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]           # take last timestep
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)

# ── Training Function ──────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, model_name,
                lr=1e-3, epochs=60, patience=10):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_ctr  = 0
    train_losses, val_losses = [], []
    best_weights = None

    print(f"\n  Training {model_name}...")
    print(f"  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>10} | {'LR':>10}")
    print(f"  {'-'*5} | {'-'*11} | {'-'*10} | {'-'*10}")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        t_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                v_loss += criterion(pred, yb).item() * len(xb)
        v_loss /= len(val_loader.dataset)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5} | {t_loss:>11.6f} | {v_loss:>10.6f} | {current_lr:>10.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_ctr  = 0
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping at epoch {epoch} (val loss no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_weights)
    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses

# ── Inference Function ─────────────────────────────────────────────────────────
def get_predictions(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)

# ── Inverse-scale target only ──────────────────────────────────────────────────
def inverse_scale(preds_scaled, scaler, n_features):
    dummy = np.zeros((len(preds_scaled), n_features))
    dummy[:, 0] = preds_scaled
    return scaler.inverse_transform(dummy)[:, 0]

# ── Training history plot ──────────────────────────────────────────────────────
def plot_history(train_losses, val_losses, model_name, filename):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
    ax.plot(val_losses,   label='Val Loss',   color='tomato',    linewidth=1.5)
    ax.set_title(f'{model_name} — Training History (MSE Loss)', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"   Saved: reports/figures/{filename}")

def save_pred_plot(y_true, y_pred, model_name, filename, n=168):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(y_true[:n], label='Actual',    color='steelblue', linewidth=1.5, zorder=5)
    axes[0].plot(y_pred[:n], label='Predicted', color='darkorange', linewidth=1.2, linestyle='--')
    axes[0].set_title(f'{model_name} — Actual vs Predicted (First 7 Days)', fontweight='bold')
    axes[0].set_ylabel('Global Active Power (kW)'); axes[0].legend()
    residuals = y_true[:n] - y_pred[:n]
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
# MODEL 1: LSTM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 1] LSTM — Long Short-Term Memory")
print("─" * 60)
print("  Architecture : 2-layer LSTM → Dropout → FC(128→64→1)")
print(f"  Hidden size  : 128   |  Layers: 2   |  Dropout: 0.2")
print(f"  Features in  : {N_FEAT}   |  Seq length: {SEQ_LEN} hours")

lstm_model = RNNForecaster(N_FEAT, hidden_size=128, num_layers=2,
                            dropout=0.2, model_type='LSTM')
n_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"  Trainable params: {n_params:,}")

lstm_model, lstm_train_loss, lstm_val_loss = train_model(
    lstm_model, train_loader, val_loader, 'LSTM',
    lr=1e-3, epochs=60, patience=10
)

lstm_preds_scaled = get_predictions(lstm_model, test_loader)
lstm_preds = inverse_scale(lstm_preds_scaled, seq_scaler, N_FEAT)
y_test_actual = inverse_scale(y_test_seq, seq_scaler, N_FEAT)

metrics_lstm = compute_metrics(y_test_actual, lstm_preds, 'LSTM')
plot_history(lstm_train_loss, lstm_val_loss, 'LSTM', 'day5_lstm_history.png')
save_pred_plot(y_test_actual, lstm_preds, 'LSTM', 'day5_lstm_predictions.png')

torch.save(lstm_model.state_dict(), MODELS_DIR / 'lstm_model.pt')
print("   Model saved: models/lstm_model.pt")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: GRU
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 2] GRU — Gated Recurrent Unit")
print("─" * 60)
print("  Architecture : 2-layer GRU → Dropout → FC(128→64→1)")
print("  (Fewer parameters than LSTM — often trains faster)")

gru_model = RNNForecaster(N_FEAT, hidden_size=128, num_layers=2,
                           dropout=0.2, model_type='GRU')
n_params = sum(p.numel() for p in gru_model.parameters() if p.requires_grad)
print(f"  Trainable params: {n_params:,}")

gru_model, gru_train_loss, gru_val_loss = train_model(
    gru_model, train_loader, val_loader, 'GRU',
    lr=1e-3, epochs=60, patience=10
)

gru_preds_scaled = get_predictions(gru_model, test_loader)
gru_preds = inverse_scale(gru_preds_scaled, seq_scaler, N_FEAT)

metrics_gru = compute_metrics(y_test_actual, gru_preds, 'GRU')
plot_history(gru_train_loss, gru_val_loss, 'GRU', 'day5_gru_history.png')
save_pred_plot(y_test_actual, gru_preds, 'GRU', 'day5_gru_predictions.png')

torch.save(gru_model.state_dict(), MODELS_DIR / 'gru_model.pt')
print("   Model saved: models/gru_model.pt")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: Bidirectional LSTM (Bonus)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[MODEL 3] Bidirectional LSTM (Bonus)")
print("─" * 60)
print("  Reads sequences forward AND backward — richer context")

bilstm_model = RNNForecaster(N_FEAT, hidden_size=64, num_layers=2,
                              dropout=0.2, model_type='LSTM', bidirectional=True)
n_params = sum(p.numel() for p in bilstm_model.parameters() if p.requires_grad)
print(f"  Trainable params: {n_params:,}")

bilstm_model, bilstm_train_loss, bilstm_val_loss = train_model(
    bilstm_model, train_loader, val_loader, 'BiLSTM',
    lr=1e-3, epochs=60, patience=10
)

bilstm_preds_scaled = get_predictions(bilstm_model, test_loader)
bilstm_preds = inverse_scale(bilstm_preds_scaled, seq_scaler, N_FEAT)

metrics_bilstm = compute_metrics(y_test_actual, bilstm_preds, 'BiLSTM')
plot_history(bilstm_train_loss, bilstm_val_loss, 'BiLSTM', 'day5_bilstm_history.png')
save_pred_plot(y_test_actual, bilstm_preds, 'BiLSTM', 'day5_bilstm_predictions.png')

torch.save(bilstm_model.state_dict(), MODELS_DIR / 'bilstm_model.pt')
print("   Model saved: models/bilstm_model.pt")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON & LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[COMPARISON] Full leaderboard — Days 3 + 4 + 5")
print("─" * 60)

day4_df = pd.read_csv(REPORTS_DIR / 'day4_all_results.csv', index_col=0)
dl_metrics = [metrics_lstm, metrics_gru, metrics_bilstm]
dl_df = pd.DataFrame(dl_metrics).set_index('Model')

all_df = pd.concat([dl_df, day4_df]).sort_values('RMSE')

print("\n  === FULL LEADERBOARD (Day 3 + 4 + 5) ===")
print(all_df[['MAE','RMSE','MAPE','R2']].to_string())
all_df.to_csv(REPORTS_DIR / 'day5_all_results.csv')
print(f"\n  Saved: reports/day5_all_results.csv")

# Plot 1: Training curves comparison (all 3 DL models)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, tl, vl, color) in zip(axes, [
    ('LSTM',   lstm_train_loss,   lstm_val_loss,   '#9C27B0'),
    ('GRU',    gru_train_loss,    gru_val_loss,    '#00BCD4'),
    ('BiLSTM', bilstm_train_loss, bilstm_val_loss, '#FF9800'),
]):
    ax.plot(tl, label='Train', color=color, linewidth=1.5)
    ax.plot(vl, label='Val',   color=color, linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_title(f'{name} Training', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_yscale('log'); ax.legend(fontsize=8)
fig.suptitle('Day 5 — LSTM / GRU / BiLSTM Training Curves', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day5_training_curves.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day5_training_curves.png")

# Plot 2: All DL models vs Actual (first week)
n = min(168, len(y_test_actual))
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test_actual[:n], label='Actual',  color='black', linewidth=2, zorder=5)
ax.plot(lstm_preds[:n],    label='LSTM',    color='#9C27B0', linewidth=1.2, linestyle='--')
ax.plot(gru_preds[:n],     label='GRU',     color='#00BCD4', linewidth=1.2, linestyle='--')
ax.plot(bilstm_preds[:n],  label='BiLSTM',  color='#FF9800', linewidth=1.2, linestyle='-.')
ax.set_title('LSTM / GRU / BiLSTM — First 7 Days of Test', fontweight='bold', fontsize=12)
ax.set_ylabel('Global Active Power (kW)'); ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day5_dl_comparison_week.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day5_dl_comparison_week.png")

# Plot 3: Full leaderboard RMSE
fig, ax = plt.subplots(figsize=(13, 5))
color_map = {
    'LightGBM':'#4CAF50','XGBoost':'#FF9800','Random Forest':'#2196F3',
    'LSTM':'#9C27B0','GRU':'#00BCD4','BiLSTM':'#FF5722',
    'ARIMA(1, 1, 1)':'#607D8B','Naive Seasonal':'#9E9E9E','Holt-Winters':'#F44336',
}
bar_colors = [color_map.get(m,'#78909C') for m in all_df.index]
bars = ax.bar(all_df.index, all_df['RMSE'], color=bar_colors, alpha=0.88,
              edgecolor='white', linewidth=1.2)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8, fontweight='bold')
ax.set_title('Full Model Leaderboard — RMSE (Days 3 + 4 + 5)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('RMSE (kW)')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'day5_full_leaderboard.png', bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/day5_full_leaderboard.png")

# ── Summary ────────────────────────────────────────────────────────────────────
best_today = dl_df.sort_values('RMSE').index[0]
best_rmse  = dl_df.sort_values('RMSE')['RMSE'].iloc[0]
lgbm_rmse  = 0.0077

print("\n" + "=" * 60)
print("  DAY 5 COMPLETE!")
print("=" * 60)
print(f"  DL models trained  : LSTM, GRU, BiLSTM")
print(f"  Best DL model      : {best_today}")
print(f"  Best DL RMSE       : {best_rmse:.4f} kW")
print(f"  LightGBM RMSE (D4) : {lgbm_rmse:.4f} kW")
if best_rmse < lgbm_rmse:
    print(f"  DL beat ML! Improvement: {(lgbm_rmse-best_rmse)/lgbm_rmse*100:.1f}% better")
else:
    print(f"  ML still leads — DL is {(best_rmse-lgbm_rmse)/lgbm_rmse*100:.1f}% above LightGBM")
    print(f"  (Classic result: tree models often beat LSTM on tabular data)")
print(f"  Figures saved      : 8")
print("=" * 60)
print("\nReady for Day 6: Prophet + Ensemble Stacking!")
print("The ensemble will combine the best of all worlds.")
