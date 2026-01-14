"""
Chapter 2: Model Implementations

Trains multiple forecasting models:
1. Exponential Smoothing (baseline)
2. ARIMA/SARIMA
3. Prophet (if available)
4. XGBoost (if available)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result from model forecast"""
    unique_id: str
    split_id: int
    model_name: str
    forecast: np.ndarray
    actual: np.ndarray
    train_time: float
    forecast_time: float

    @property
    def valid_mask(self) -> np.ndarray:
        """Mask of valid (non-NaN) predictions"""
        return np.isfinite(self.forecast) & np.isfinite(self.actual)

    @property
    def valid_count(self) -> int:
        """Number of valid predictions"""
        return self.valid_mask.sum()


class ForecastModel(ABC):
    """Base class for forecasting models"""

    @abstractmethod
    def fit(self, y: np.ndarray, **kwargs):
        """Fit model to training data"""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecast for given horizon"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Model name"""
        pass


class ExponentialSmoothingModel(ForecastModel):
    """Simple Exponential Smoothing baseline"""

    def __init__(self):
        self.train_data = None

    def fit(self, y: np.ndarray, **kwargs):
        """Fit exponential smoothing to data"""
        self.train_data = y.copy()

    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecast using exponential smoothing"""
        if self.train_data is None or len(self.train_data) == 0:
            return np.full(horizon, np.nan)

        # Simple exponential smoothing with trend
        alpha = 0.3

        # Initial level and trend
        level = self.train_data[-1]

        # Estimate trend from last 10 observations
        if len(self.train_data) >= 10:
            recent = self.train_data[-10:]
            trend = np.mean(np.diff(recent))
        else:
            trend = 0

        # Generate forecast
        forecast = []
        for t in range(1, horizon + 1):
            pred = level + trend * t
            forecast.append(max(pred, 0))  # No negative generation

        return np.array(forecast)

    def get_name(self) -> str:
        return "exponential_smoothing"


class ARIMAModel(ForecastModel):
    """ARIMA/SARIMA model wrapper"""

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.train_data = None

    def fit(self, y: np.ndarray, **kwargs):
        """Fit ARIMA to data"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            self.train_data = y.copy()

            # Fit ARIMA
            self.model = ARIMA(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            self.model = self.model.fit()

            logger.debug(f"ARIMA{self.order} fitted successfully")

        except ImportError:
            logger.warning("statsmodels not available, using exponential smoothing fallback")
            self.train_data = y.copy()
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}, using fallback")
            self.train_data = y.copy()

    def predict(self, horizon: int) -> np.ndarray:
        """Generate ARIMA forecast"""
        if self.model is None or self.train_data is None:
            # Fallback: use trend
            if len(self.train_data) > 0:
                trend = np.mean(np.diff(self.train_data[-10:]))
                return np.array([self.train_data[-1] + trend * (i + 1) for i in range(horizon)])
            else:
                return np.full(horizon, np.nan)

        try:
            forecast = self.model.get_forecast(steps=horizon)
            preds = forecast.predicted_mean
            # Handle both Series and ndarray
            if hasattr(preds, 'values'):
                preds = preds.values
            return np.maximum(preds, 0)  # No negative values
        except Exception as e:
            logger.warning(f"ARIMA prediction failed: {e}")
            return np.full(horizon, np.nan)

    def get_name(self) -> str:
        return f"arima{self.order}"


class ProphetModel(ForecastModel):
    """Facebook Prophet model wrapper"""

    def __init__(self, yearly_seasonality=False, weekly_seasonality=True):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model = None
        self.train_dates = None
        self.train_data = None

    def fit(self, y: np.ndarray, dates: Optional[np.ndarray] = None, **kwargs):
        """Fit Prophet to data"""
        try:
            from prophet import Prophet

            self.train_data = y.copy()
            self.train_dates = dates

            if dates is None:
                # Create default dates
                dates = pd.date_range(end=pd.Timestamp.now(), periods=len(y), freq='H')

            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': dates,
                'y': y
            })

            # Fit Prophet
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=False,
                interval_width=0.95
            )
            self.model.fit(df)

            logger.debug("Prophet model fitted successfully")

        except ImportError:
            logger.warning("Prophet not available, using exponential smoothing fallback")
            self.train_data = y.copy()
        except Exception as e:
            logger.warning(f"Prophet fitting failed: {e}, using fallback")
            self.train_data = y.copy()

    def predict(self, horizon: int) -> np.ndarray:
        """Generate Prophet forecast"""
        if self.model is None or self.train_data is None:
            # Fallback
            if len(self.train_data) > 0:
                trend = np.mean(np.diff(self.train_data[-10:]))
                return np.array([self.train_data[-1] + trend * (i + 1) for i in range(horizon)])
            else:
                return np.full(horizon, np.nan)

        try:
            # Create future dataframe
            if self.train_dates is None:
                last_date = pd.Timestamp.now()
            else:
                last_date = self.train_dates[-1]

            future = pd.DataFrame({
                'ds': pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='H')
            })

            forecast = self.model.predict(future)
            preds = forecast['yhat'].values
            return np.maximum(preds, 0)  # No negative values

        except Exception as e:
            logger.warning(f"Prophet prediction failed: {e}")
            return np.full(horizon, np.nan)

    def get_name(self) -> str:
        return "prophet"


class XGBoostModel(ForecastModel):
    """XGBoost model for time series forecasting"""

    def __init__(self, n_lags: int = 24, max_depth: int = 5, learning_rate: float = 0.1):
        self.n_lags = n_lags
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.train_data = None
        self.scaler_min = None
        self.scaler_max = None

    def _create_lagged_features(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for XGBoost"""
        X = []
        Y = []

        for i in range(self.n_lags, len(y)):
            X.append(y[i - self.n_lags:i])
            Y.append(y[i])

        return np.array(X), np.array(Y)

    def fit(self, y: np.ndarray, **kwargs):
        """Fit XGBoost to data"""
        try:
            import xgboost as xgb

            self.train_data = y.copy()

            # Normalize data
            self.scaler_min = np.min(y)
            self.scaler_max = np.max(y)

            if self.scaler_max > self.scaler_min:
                y_norm = (y - self.scaler_min) / (self.scaler_max - self.scaler_min)
            else:
                y_norm = y

            # Create features
            X, Y = self._create_lagged_features(y_norm)

            if len(X) == 0:
                logger.warning("Not enough data for XGBoost")
                self.model = None
                return

            # Fit XGBoost
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                verbosity=0
            )
            self.model.fit(X, Y, verbose=False)

            logger.debug("XGBoost model fitted successfully")

        except ImportError:
            logger.warning("XGBoost not available")
            self.train_data = y.copy()
        except Exception as e:
            logger.warning(f"XGBoost fitting failed: {e}")
            self.train_data = y.copy()

    def predict(self, horizon: int) -> np.ndarray:
        """Generate XGBoost forecast"""
        if self.model is None or self.train_data is None or len(self.train_data) < self.n_lags:
            # Fallback
            if len(self.train_data) > 0:
                trend = np.mean(np.diff(self.train_data[-10:]))
                return np.array([self.train_data[-1] + trend * (i + 1) for i in range(horizon)])
            else:
                return np.full(horizon, np.nan)

        try:
            # Normalize last n_lags values
            y_last = self.train_data[-self.n_lags:].copy()

            if self.scaler_max > self.scaler_min:
                y_last_norm = (y_last - self.scaler_min) / (self.scaler_max - self.scaler_min)
            else:
                y_last_norm = y_last

            # Generate forecast
            forecast = []
            current_lags = y_last_norm.copy()

            for _ in range(horizon):
                pred_norm = self.model.predict(current_lags.reshape(1, -1))[0]

                # Denormalize
                if self.scaler_max > self.scaler_min:
                    pred = pred_norm * (self.scaler_max - self.scaler_min) + self.scaler_min
                else:
                    pred = pred_norm

                forecast.append(max(pred, 0))

                # Update lags
                current_lags = np.append(current_lags[1:], pred_norm)

            return np.array(forecast)

        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            return np.full(horizon, np.nan)

    def get_name(self) -> str:
        return "xgboost"


class ModelFactory:
    """Factory for creating model instances"""

    _models = {
        "exponential_smoothing": ExponentialSmoothingModel,
        "arima": ARIMAModel,
        "prophet": ProphetModel,
        "xgboost": XGBoostModel,
    }

    @classmethod
    def create(cls, model_name: str, **kwargs) -> ForecastModel:
        """Create model by name"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")

        return cls._models[model_name](**kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available models"""
        return list(cls._models.keys())
        return list(cls._models.keys())
