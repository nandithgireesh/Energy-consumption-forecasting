"""
preprocessing.py
----------------
Data cleaning and feature engineering for the power consumption dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# 1. Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in the power consumption dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        'interpolate' (linear interpolation, recommended for time series),
        'ffill' (forward fill), 'drop' (drop rows with NaN)

    Returns
    -------
    pd.DataFrame
    """
    missing_before = df.isna().sum().sum()
    print(f"🧹 Missing values before cleaning: {missing_before:,}")

    if strategy == 'interpolate':
        df = df.interpolate(method='linear', limit_direction='both')
    elif strategy == 'ffill':
        df = df.ffill().bfill()
    elif strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    missing_after = df.isna().sum().sum()
    print(f"   Missing values after  cleaning: {missing_after:,}")
    return df


def remove_outliers(df: pd.DataFrame, column: str, z_thresh: float = 4.0) -> pd.DataFrame:
    """
    Remove extreme outliers using Z-score thresholding.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column name to check for outliers.
    z_thresh : float
        Z-score threshold (default 4.0 — keeps 99.99% of normal data).

    Returns
    -------
    pd.DataFrame
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    n_outliers = (z_scores > z_thresh).sum()
    print(f"🔍 Outliers found in '{column}': {n_outliers} (z > {z_thresh})")
    df = df[z_scores <= z_thresh]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and cyclical time-based features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    idx = df.index

    # Calendar features
    df['hour']        = idx.hour
    df['day']         = idx.day
    df['dayofweek']   = idx.dayofweek          # 0=Monday, 6=Sunday
    df['month']       = idx.month
    df['quarter']     = idx.quarter
    df['year']        = idx.year
    df['dayofyear']   = idx.dayofyear
    df['weekofyear']  = idx.isocalendar().week.astype(int)
    df['is_weekend']  = (idx.dayofweek >= 5).astype(int)

    # Season encoding
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3:  'spring', 4: 'spring', 5: 'spring',
        6:  'summer', 7: 'summer', 8: 'summer',
        9:  'autumn', 10: 'autumn', 11: 'autumn',
    })

    # Cyclical encoding (for hour and month — avoids discontinuity at wrap-around)
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']   = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['dayofweek'] / 7)

    print(f"✅ Time features added: {[c for c in df.columns if c not in ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']]}")
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = 'Global_active_power',
                      lags: list = None) -> pd.DataFrame:
    """
    Add lagged versions of the target column.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    lags : list of int
        List of lag periods (in time steps).

    Returns
    -------
    pd.DataFrame
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]   # 1h … 1 week ago (assumes hourly)

    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    print(f"✅ Lag features added: lags = {lags}")
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = 'Global_active_power',
                          windows: list = None) -> pd.DataFrame:
    """
    Add rolling mean and std features.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    windows : list of int
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
    """
    if windows is None:
        windows = [3, 6, 12, 24, 48, 168]       # 3h to 1 week

    df = df.copy()
    for w in windows:
        df[f'{target_col}_rollmean_{w}'] = df[target_col].shift(1).rolling(window=w).mean()
        df[f'{target_col}_rollstd_{w}']  = df[target_col].shift(1).rolling(window=w).std()

    print(f"✅ Rolling features added: windows = {windows}")
    return df


def add_energy_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features derived from the power domain knowledge.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Apparent power (kVA) = sqrt(P² + Q²)
    if 'Global_active_power' in df.columns and 'Global_reactive_power' in df.columns:
        df['apparent_power'] = np.sqrt(
            df['Global_active_power']**2 + df['Global_reactive_power']**2
        )
        # Power factor = P / S
        df['power_factor'] = df['Global_active_power'] / (df['apparent_power'] + 1e-8)

    # Energy not sub-metered (as a new feature)
    if all(c in df.columns for c in ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']):
        df['sub_metering_remainder'] = (
            df['Global_active_power'] * 1000 / 60
            - df['Sub_metering_1']
            - df['Sub_metering_2']
            - df['Sub_metering_3']
        )

    print("✅ Domain-derived features added: apparent_power, power_factor, sub_metering_remainder")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize(train: pd.DataFrame, test: pd.DataFrame, columns: list, method: str = 'minmax'):
    """
    Fit a scaler on train set, transform both train and test.

    Parameters
    ----------
    train, test : pd.DataFrame
    columns : list of str
    method : 'minmax' or 'standard'

    Returns
    -------
    tuple: (train_scaled, test_scaled, scaler)
    """
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    train_scaled = train.copy()
    test_scaled  = test.copy()
    train_scaled[columns] = scaler.fit_transform(train[columns])
    test_scaled[columns]  = scaler.transform(test[columns])
    print(f"✅ Data normalized using {method} scaler.")
    return train_scaled, test_scaled, scaler


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sequence Builder (for LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray, seq_length: int = 24, horizon: int = 1):
    """
    Create sliding window sequences for LSTM/GRU models.

    Parameters
    ----------
    data : np.ndarray  (shape: [n_samples, n_features])
    seq_length : int   Look-back window length
    horizon : int      Forecast horizon

    Returns
    -------
    X : np.ndarray  (n_sequences, seq_length, n_features)
    y : np.ndarray  (n_sequences,)  — target is first column
    """
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length + horizon - 1, 0])   # 0 = target column
    return np.array(X), np.array(y)
