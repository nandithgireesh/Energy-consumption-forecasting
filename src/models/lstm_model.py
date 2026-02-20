"""
lstm_model.py
-------------
LSTM and GRU deep learning models for energy consumption forecasting using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).parent.parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────────
# 1. PyTorch Model Definitions
# ──────────────────────────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """Stacked LSTM network for univariate/multivariate time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])   # Take last time step
        return out.squeeze(-1)


class GRUNet(nn.Module):
    """Stacked GRU network — lighter than LSTM, often similar accuracy."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Training Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DeepLearningForecaster:
    """
    High-level wrapper for training LSTM or GRU regression models.

    Usage
    -----
    model = DeepLearningForecaster(model_type='LSTM', input_size=1, seq_length=24)
    model.fit(X_train, y_train, X_val, y_val, epochs=30)
    predictions = model.predict(X_test)
    """

    def __init__(self, model_type: str = 'LSTM', input_size: int = 1,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, seq_length: int = 24,
                 batch_size: int = 64, learning_rate: float = 1e-3):

        self.model_type   = model_type
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.dropout      = dropout
        self.seq_length   = seq_length
        self.batch_size   = batch_size
        self.learning_rate = learning_rate
        self.device       = DEVICE
        self.net          = None
        self.train_losses = []
        self.val_losses   = []

        print(f"🖥️  Using device: {DEVICE}")

    def _build_model(self):
        cls = LSTMNet if self.model_type == 'LSTM' else GRUNet
        self.net = cls(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        print(f"🔧 Built {self.model_type} model with {sum(p.numel() for p in self.net.parameters()):,} parameters")

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 50, patience: int = 10):
        """
        Train the model.

        Parameters
        ----------
        X_train : np.ndarray  (n_samples, seq_length, input_size)
        y_train : np.ndarray  (n_samples,)
        X_val, y_val : validation set (optional)
        epochs : int
        patience : int  Early stopping patience
        """
        self._build_model()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.MSELoss()

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        best_val_loss = np.inf
        patience_counter = 0

        print(f"\n🚀 Training {self.model_type} — {epochs} epochs, batch_size={self.batch_size}")
        for epoch in range(1, epochs + 1):
            # Training
            self.net.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.net(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            epoch_loss /= len(X_train)
            self.train_losses.append(epoch_loss)

            # Validation
            if X_val is not None and y_val is not None:
                self.net.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    val_pred = self.net(X_v)
                    val_loss = criterion(val_pred, y_v).item()
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)

                if epoch % 5 == 0 or epoch == 1:
                    print(f"   Epoch {epoch:3d}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.net.state_dict(), MODELS_DIR / f'best_{self.model_type.lower()}.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"⏹️  Early stopping at epoch {epoch} (val loss not improving)")
                        break
            else:
                if epoch % 10 == 0:
                    print(f"   Epoch {epoch:3d}/{epochs} | Train Loss: {epoch_loss:.4f}")

        print(f"✅ Training complete! Best Val Loss: {best_val_loss:.4f}")
        # Load best weights
        best_path = MODELS_DIR / f'best_{self.model_type.lower()}.pt'
        if best_path.exists():
            self.net.load_state_dict(torch.load(best_path, map_location=self.device))
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model not trained yet.")
        self.net.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(self.device)
            preds = self.net(X_t)
        return preds.cpu().numpy()

    def plot_training_history(self, save: bool = True):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.train_losses, label='Train Loss', color='steelblue')
        if self.val_losses:
            ax.plot(self.val_losses, label='Val Loss', color='darkorange')
        ax.set_title(f'{self.model_type} Training History', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(MODELS_DIR.parent / 'reports' / 'figures' / f'{self.model_type.lower()}_training.png',
                        bbox_inches='tight')
        plt.show()
        return fig
