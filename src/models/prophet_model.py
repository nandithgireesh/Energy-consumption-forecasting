"""
prophet_model.py
----------------
Facebook Prophet model for time series forecasting with seasonal regressors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = Path(__file__).parent.parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class ProphetForecaster:
    """
    Wrapper around Meta's Prophet model for energy consumption forecasting.

    Prophet works best with daily/hourly granularity and handles:
    - Multiple seasonalities (daily, weekly, yearly)
    - Holidays and special events
    - Missing data
    - Changepoints (structural breaks)
    """

    def __init__(self, seasonality_mode: str = 'additive',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 interval_width: float = 0.95):
        self.params = dict(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            interval_width=interval_width,
        )
        self.model = None
        self.forecast_df = None

    def fit(self, series: pd.Series):
        """
        Fit Prophet to a time series.

        Parameters
        ----------
        series : pd.Series with DatetimeIndex
        """
        try:
            from prophet import Prophet
        except ImportError:
            print("❌ prophet not installed. Run: pip install prophet")
            return self

        # Prophet requires a DataFrame with columns 'ds' and 'y'
        df_prophet = pd.DataFrame({
            'ds': series.index,
            'y': series.values,
        }).reset_index(drop=True)

        print(f"🔧 Fitting Prophet ({self.params['seasonality_mode']} seasonality)...")
        self.model = Prophet(**self.params)
        self.model.fit(df_prophet)
        print("✅ Prophet fitted!")
        return self

    def predict(self, n_steps: int, freq: str = 'h') -> pd.DataFrame:
        """
        Generate future forecast.

        Parameters
        ----------
        n_steps : int
            Number of time periods to forecast.
        freq : str
            Frequency string ('h', 'D', etc.)

        Returns
        -------
        pd.DataFrame
            Prophet forecast dataframe with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        future = self.model.make_future_dataframe(periods=n_steps, freq=freq)
        self.forecast_df = self.model.predict(future)
        return self.forecast_df

    def get_predictions_array(self, n_steps: int, freq: str = 'h') -> np.ndarray:
        """Return just the yhat values for the forecast horizon."""
        forecast = self.predict(n_steps, freq=freq)
        return forecast['yhat'].values[-n_steps:]

    def plot_forecast(self, save: bool = True):
        """Plot Prophet's built-in forecast visualization."""
        if self.model is None or self.forecast_df is None:
            raise RuntimeError("Fit and predict first.")
        fig = self.model.plot(self.forecast_df, figsize=(14, 5))
        fig.suptitle('Prophet Forecast — Global Active Power', fontweight='bold')
        if save:
            path = FIGURES_DIR / 'prophet_forecast.png'
            fig.savefig(path, bbox_inches='tight')
            print(f"💾 Saved: {path}")
        plt.show()
        return fig

    def plot_components(self, save: bool = True):
        """Plot trend and seasonal components."""
        if self.model is None or self.forecast_df is None:
            raise RuntimeError("Fit and predict first.")
        fig = self.model.plot_components(self.forecast_df)
        fig.suptitle('Prophet Components', fontweight='bold')
        if save:
            path = FIGURES_DIR / 'prophet_components.png'
            fig.savefig(path, bbox_inches='tight')
            print(f"💾 Saved: {path}")
        plt.show()
        return fig
