"""
Chapter 2: Experiment Runner Module

Implements the complete backtesting and model selection framework:
- Backtesting strategies (rolling, expanding)
- Model implementations (exponential smoothing, ARIMA, Prophet, XGBoost)
- Evaluation metrics (RMSE, MAE, MAPE, MASE)
- Model comparison and selection
- Training orchestration
"""

from .backtesting import (BacktestingStrategy, BacktestSplit,
                          ExpandingWindowBacktest, RollingWindowBacktest,
                          validate_backtesting_splits)
from .evaluation import (ForecastMetrics, aggregate_metrics,
                         compute_series_metrics)
from .models import (ARIMAModel, ExponentialSmoothingModel, ForecastModel,
                     ForecastResult, ModelFactory, ProphetModel, XGBoostModel)
from .training import ModelSelector, TrainingPipeline

__all__ = [
    # Backtesting
    "BacktestSplit",
    "RollingWindowBacktest",
    "ExpandingWindowBacktest",
    "BacktestingStrategy",
    "validate_backtesting_splits",
    # Models
    "ForecastModel",
    "ExponentialSmoothingModel",
    "ARIMAModel",
    "ProphetModel",
    "XGBoostModel",
    "ModelFactory",
    "ForecastResult",
    # Evaluation
    "ForecastMetrics",
    "compute_series_metrics",
    "aggregate_metrics",
    # Training
    "TrainingPipeline",
    "ModelSelector",
]
