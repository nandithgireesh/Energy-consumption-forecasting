"""
baseline.py
-----------
Statistical baseline models: Naive, Holt-Winters, ARIMA/SARIMA
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# 1. Naive Baseline
# ──────────────────────────────────────────────────────────────────────────────

class NaiveForecaster:
    """
    Naive / Seasonal Naive forecaster.
    Predicts the value from 'seasonality' steps ago.
    """
    def __init__(self, seasonality: int = 24):
        self.seasonality = seasonality

    def fit(self, train: pd.Series):
        self.train = train
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """Repeat last full season 'n_steps' times."""
        season = self.train.values[-self.seasonality:]
        n_repeats = int(np.ceil(n_steps / self.seasonality))
        forecast = np.tile(season, n_repeats)[:n_steps]
        return forecast


# ──────────────────────────────────────────────────────────────────────────────
# 2. Holt-Winters (Exponential Smoothing)
# ──────────────────────────────────────────────────────────────────────────────

class HoltWintersModel:
    """
    Triple Exponential Smoothing (Holt-Winters) for seasonal time series.
    """
    def __init__(self, seasonal: str = 'add', seasonal_periods: int = 24):
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.result = None

    def fit(self, series: pd.Series):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        print(f"🔧 Fitting Holt-Winters (seasonal={self.seasonal}, periods={self.seasonal_periods})...")
        self.model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method='estimated',
        )
        self.result = self.model.fit(optimized=True, use_brute=False)
        print("✅ Holt-Winters fitted!")
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        forecast = self.result.forecast(n_steps)
        return forecast.values


# ──────────────────────────────────────────────────────────────────────────────
# 3. ARIMA / SARIMA
# ──────────────────────────────────────────────────────────────────────────────

class ARIMAModel:
    """
    Auto-ARIMA wrapper using pmdarima for automatic order selection.
    Falls back to manual order if pmdarima not available.
    """
    def __init__(self, seasonal: bool = True, m: int = 24,
                 max_p: int = 3, max_q: int = 3, max_P: int = 1, max_Q: int = 1):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.model = None

    def fit(self, series: pd.Series):
        try:
            import pmdarima as pm
            print(f"🔧 Running Auto-ARIMA (seasonal={self.seasonal}, m={self.m})...")
            self.model = pm.auto_arima(
                series,
                seasonal=self.seasonal,
                m=self.m,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                stepwise=True,
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True,
                n_jobs=-1,
            )
            print(f"✅ Auto-ARIMA order: {self.model.order}, seasonal: {self.model.seasonal_order}")
        except ImportError:
            from statsmodels.tsa.arima.model import ARIMA
            print("⚠️  pmdarima not found, using default ARIMA(1,1,1)")
            arima = ARIMA(series, order=(1, 1, 1))
            self.model = arima.fit()
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        try:
            forecast = self.model.predict(n_periods=n_steps)
        except AttributeError:
            forecast = self.model.forecast(steps=n_steps)
        return np.array(forecast)

    def update(self, new_obs):
        """Update ARIMA model with new observations (online learning)."""
        try:
            self.model.update(new_obs)
        except Exception:
            pass
