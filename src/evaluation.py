"""
evaluation.py
-------------
Metrics, scoring, and visualization utilities for forecast evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoids division by zero)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = 'Model') -> dict:
    """
    Compute all evaluation metrics.

    Returns
    -------
    dict with keys: model, MAE, RMSE, MAPE, R2
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics = {
        'Model': model_name,
        'MAE':   round(mae, 4),
        'RMSE':  round(rmse, 4),
        'MAPE':  round(mape_val, 2),
        'R²':    round(r2, 4),
    }
    print(f"\n📊 {model_name} — Evaluation Metrics")
    print(f"   MAE  : {metrics['MAE']:.4f} kW")
    print(f"   RMSE : {metrics['RMSE']:.4f} kW")
    print(f"   MAPE : {metrics['MAPE']:.2f} %")
    print(f"   R²   : {metrics['R²']:.4f}")
    return metrics


def compare_models(results: list) -> pd.DataFrame:
    """
    Create a comparison table from a list of metric dicts.

    Parameters
    ----------
    results : list of dict (output from compute_metrics)

    Returns
    -------
    pd.DataFrame sorted by RMSE ascending.
    """
    df = pd.DataFrame(results).set_index('Model')
    df = df.sort_values('RMSE')
    print("\n🏆 Model Comparison Table:")
    print(df.to_string())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def set_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })


def plot_time_series(series: pd.Series, title: str = 'Time Series',
                     ylabel: str = 'Global Active Power (kW)',
                     save: bool = True, filename: str = 'time_series.png'):
    """Plot a single time series."""
    set_style()
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(series.index, series.values, linewidth=0.7, color='steelblue', alpha=0.85)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.tight_layout()
    if save:
        path = FIGURES_DIR / filename
        fig.savefig(path, bbox_inches='tight')
        print(f"💾 Saved: {path}")
    plt.show()
    return fig


def plot_predictions(y_true, y_pred, index=None, model_name: str = 'Model',
                     save: bool = True, filename: str = None):
    """Plot actual vs. predicted values."""
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Time plot
    if index is not None:
        axes[0].plot(index, y_true, label='Actual', color='steelblue', linewidth=1.2)
        axes[0].plot(index, y_pred, label='Predicted', color='darkorange', linewidth=1.2, linestyle='--')
    else:
        axes[0].plot(y_true, label='Actual', color='steelblue', linewidth=1.2)
        axes[0].plot(y_pred, label='Predicted', color='darkorange', linewidth=1.2, linestyle='--')
    axes[0].set_title(f'{model_name} — Actual vs. Predicted', fontweight='bold')
    axes[0].set_ylabel('Global Active Power (kW)')
    axes[0].legend()

    # Residual plot
    residuals = np.array(y_true) - np.array(y_pred)
    axes[1].bar(range(len(residuals)), residuals, color='tomato', alpha=0.6, width=1)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Residuals', fontweight='bold')
    axes[1].set_ylabel('Error (kW)')
    axes[1].set_xlabel('Sample Index')

    fig.tight_layout()
    fname = filename or f"{model_name.lower().replace(' ', '_')}_predictions.png"
    if save:
        path = FIGURES_DIR / fname
        fig.savefig(path, bbox_inches='tight')
        print(f"💾 Saved: {path}")
    plt.show()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = 'RMSE',
                           save: bool = True):
    """Bar chart comparing models on a given metric."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(comparison_df)))
    bars = ax.barh(comparison_df.index, comparison_df[metric], color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.set_xlabel(metric)
    ax.set_title(f'Model Comparison — {metric}', fontweight='bold')
    ax.invert_yaxis()
    fig.tight_layout()
    if save:
        path = FIGURES_DIR / f'model_comparison_{metric.lower()}.png'
        fig.savefig(path, bbox_inches='tight')
        print(f"💾 Saved: {path}")
    plt.show()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    """Correlation heatmap of all numeric features."""
    set_style()
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', mask=mask, cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    fig.tight_layout()
    if save:
        path = FIGURES_DIR / 'correlation_heatmap.png'
        fig.savefig(path, bbox_inches='tight')
        print(f"💾 Saved: {path}")
    plt.show()
    return fig


def plot_seasonal_decomposition(series: pd.Series, period: int = 24,
                                 save: bool = True):
    """Plot trend, seasonality, and residuals."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    set_style()
    result = seasonal_decompose(series.dropna(), model='additive', period=period, extrapolate_trend='freq')
    fig = result.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle('Seasonal Decomposition (Additive)', y=1.02, fontweight='bold')
    fig.tight_layout()
    if save:
        path = FIGURES_DIR / 'seasonal_decomposition.png'
        fig.savefig(path, bbox_inches='tight')
        print(f"💾 Saved: {path}")
    plt.show()
    return fig
