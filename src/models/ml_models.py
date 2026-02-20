"""
ml_models.py
------------
Tree-based and ensemble ML models for energy forecasting.
Uses lag/rolling features to convert time series into supervised learning.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

MODELS_DIR = Path(__file__).parent.parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Random Forest
# ──────────────────────────────────────────────────────────────────────────────

class RandomForestForecaster:
    """
    Random Forest Regressor for time-series forecasting with lag features.
    """
    def __init__(self, n_estimators: int = 200, max_depth: int = None,
                 n_jobs: int = -1, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            oob_score=True,
        )
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = X_train.columns.tolist()
        print(f"🔧 Training Random Forest with {X_train.shape[1]} features, {len(X_train):,} samples...")
        self.model.fit(X_train, y_train)
        print(f"✅ Random Forest fitted! OOB Score: {self.model.oob_score_:.4f}")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def feature_importance(self) -> pd.DataFrame:
        fi = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        return fi

    def save(self, filename: str = 'random_forest.pkl'):
        path = MODELS_DIR / filename
        joblib.dump(self.model, path)
        print(f"💾 Model saved: {path}")

    @classmethod
    def load(cls, filename: str = 'random_forest.pkl'):
        path = MODELS_DIR / filename
        obj = cls()
        obj.model = joblib.load(path)
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# 2. XGBoost
# ──────────────────────────────────────────────────────────────────────────────

class XGBoostForecaster:
    """
    XGBoost Regressor for energy consumption forecasting.
    """
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05,
                 max_depth: int = 6, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
            self._available = True
        except ImportError:
            print("⚠️  XGBoost not installed. pip install xgboost")
            self._available = False
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if not self._available:
            return self
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        eval_set = [(X_val, y_val)] if X_val is not None else None
        print(f"🔧 Training XGBoost — {X_train.shape[1]} features, {len(X_train):,} samples...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        print("✅ XGBoost fitted!")
        return self

    def predict(self, X_test) -> np.ndarray:
        if not self._available:
            return np.zeros(len(X_test))
        return self.model.predict(X_test)

    def feature_importance(self) -> pd.DataFrame:
        if not self._available or self.feature_names is None:
            return pd.DataFrame()
        fi = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        return fi

    def save(self, filename: str = 'xgboost.pkl'):
        path = MODELS_DIR / filename
        joblib.dump(self.model, path)
        print(f"💾 Model saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. LightGBM
# ──────────────────────────────────────────────────────────────────────────────

class LightGBMForecaster:
    """
    LightGBM Regressor — fast tree-based model.
    """
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05,
                 num_leaves: int = 63, random_state: int = 42):
        try:
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            self._available = True
        except ImportError:
            print("⚠️  LightGBM not installed. pip install lightgbm")
            self._available = False
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if not self._available:
            return self
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        print(f"🔧 Training LightGBM — {X_train.shape[1]} features, {len(X_train):,} samples...")
        callbacks = []
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            callbacks=callbacks,
        )
        print("✅ LightGBM fitted!")
        return self

    def predict(self, X_test) -> np.ndarray:
        if not self._available:
            return np.zeros(len(X_test))
        return self.model.predict(X_test)

    def save(self, filename: str = 'lightgbm.pkl'):
        path = MODELS_DIR / filename
        joblib.dump(self.model, path)
        print(f"💾 Model saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ensemble (Simple Average / Weighted Average)
# ──────────────────────────────────────────────────────────────────────────────

class EnsembleForecaster:
    """
    Simple weighted-average ensemble of multiple forecasters.
    """
    def __init__(self, models: list, weights: list = None):
        """
        Parameters
        ----------
        models : list of fitted forecaster objects (must have .predict())
        weights : list of floats (must sum to 1). If None, equal weights.
        """
        self.models = models
        n = len(models)
        self.weights = weights if weights else [1.0 / n] * n
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1."

    def predict(self, X_test) -> np.ndarray:
        preds = [w * m.predict(X_test) for w, m in zip(self.weights, self.models)]
        return np.sum(preds, axis=0)
