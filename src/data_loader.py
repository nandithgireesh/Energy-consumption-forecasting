"""
data_loader.py
--------------
Utilities for loading and initial inspection of the Household Power Consumption dataset.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_raw_data(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load the raw household power consumption dataset.

    Parameters
    ----------
    filepath : str
        Path to 'household_power_consumption.txt'
    verbose : bool
        If True, print summary statistics about the loaded data.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with datetime index.
    """
    print(f"📂 Loading dataset from: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=';',
        na_values='?',
        low_memory=False,
        dtype={
            'Global_active_power': 'float32',
            'Global_reactive_power': 'float32',
            'Voltage': 'float32',
            'Global_intensity': 'float32',
            'Sub_metering_1': 'float32',
            'Sub_metering_2': 'float32',
            'Sub_metering_3': 'float32',
        }
    )

    # Combine Date and Time into a single datetime index
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    if verbose:
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Shape          : {df.shape}")
        print(f"   Date range     : {df.index.min()} → {df.index.max()}")
        print(f"   Missing values : {df.isna().sum().sum():,} ({df.isna().sum().sum()/df.size*100:.2f}%)")
        print(f"   Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(f"\n   Columns        : {df.columns.tolist()}")
        print(f"\n   Data Types:\n{df.dtypes}")

    return df


def resample_data(df: pd.DataFrame, freq: str = 'h') -> pd.DataFrame:
    """
    Resample minute-level data to a lower frequency (hourly, daily, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Raw minute-level dataframe with datetime index.
    freq : str
        Resampling frequency. Options: 'h' (hourly), 'D' (daily), 'W' (weekly), 'ME' (monthly).

    Returns
    -------
    pd.DataFrame
        Resampled dataframe using mean aggregation.
    """
    freq_label = {
        'h': 'Hourly', 'D': 'Daily', 'W': 'Weekly', 'ME': 'Monthly'
    }.get(freq, freq)

    print(f"⏱️  Resampling to {freq_label} frequency...")
    resampled = df.resample(freq).mean(numeric_only=True)
    print(f"   New shape: {resampled.shape}")
    return resampled


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a detailed summary of the dataset including stats on missing values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isna().sum(),
        'null_pct': (df.isna().sum() / len(df) * 100).round(2),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True),
    })
    return summary


def split_train_test(df: pd.DataFrame, test_months: int = 3):
    """
    Split dataframe into train and test sets by time.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime index.
    test_months : int
        Number of months to hold out for testing.

    Returns
    -------
    tuple: (train_df, test_df)
    """
    split_date = df.index.max() - pd.DateOffset(months=test_months)
    train = df[df.index <= split_date]
    test = df[df.index > split_date]
    print(f"✂️  Train: {train.index.min()} → {train.index.max()} ({len(train):,} records)")
    print(f"   Test : {test.index.min()} → {test.index.max()} ({len(test):,} records)")
    return train, test


if __name__ == '__main__':
    # Quick test
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'household_power_consumption.txt'
    if data_path.exists():
        df = load_raw_data(str(data_path))
        print(get_data_summary(df))
    else:
        print(f"Dataset not found at {data_path}")
