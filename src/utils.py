"""
utils.py
--------
General utility helpers for the Energy Consumption Forecasting project.
"""

import time
import functools
import numpy as np
import pandas as pd
from pathlib import Path


def timer(func):
    """Decorator that prints how long a function takes to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def memory_usage(df: pd.DataFrame) -> str:
    """Return the memory usage of a DataFrame as a human-readable string."""
    mem = df.memory_usage(deep=True).sum()
    if mem < 1e6:
        return f"{mem/1e3:.1f} KB"
    elif mem < 1e9:
        return f"{mem/1e6:.1f} MB"
    else:
        return f"{mem/1e9:.2f} GB"


def ensure_dirs(*paths):
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def print_section(title: str, width: int = 60):
    """Print a formatted section header."""
    print('\n' + '=' * width)
    print(f"  {title}")
    print('=' * width)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"🎲 Random seeds set to {seed}")


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     index: pd.DatetimeIndex, model_name: str,
                     output_dir: str = '../reports'):
    """Save predictions alongside actuals to a CSV file."""
    ensure_dirs(output_dir)
    df = pd.DataFrame({
        'datetime': index[:len(y_true)],
        'actual': y_true,
        'predicted': y_pred,
        'error': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
    })
    filename = Path(output_dir) / f'{model_name.lower().replace(" ", "_")}_predictions.csv'
    df.to_csv(filename, index=False)
    print(f"💾 Predictions saved: {filename}")
    return df
