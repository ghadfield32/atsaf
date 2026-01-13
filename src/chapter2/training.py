"""
Chapter 2: Model Training and Forecasting Pipeline

Orchestrates training and evaluation across backtesting splits.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtesting import BacktestingStrategy, BacktestSplit
from .evaluation import compute_series_metrics
from .models import ForecastResult, ModelFactory

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Trains models across all backtesting splits"""
    
    def __init__(
        self,
        models: List[str] = None,
        backtesting_strategy: str = "rolling"
    ):
        """
        Initialize training pipeline
        
        Args:
            models: List of model names to train
            backtesting_strategy: "rolling" or "expanding"
        """
        if models is None:
            models = ["exponential_smoothing", "arima"]
        
        self.models = models
        self.backtesting_strategy = backtesting_strategy
        self.results = []
    
    def run(
        self,
        data: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Run complete training pipeline
        
        Args:
            data: DataFrame with columns [unique_id, ds, y]
            output_dir: Directory to save results
            
        Returns:
            DataFrame with all forecast results
        """
        logger.info(f"Starting training pipeline with {len(self.models)} models")
        logger.info(f"Backtesting strategy: {self.backtesting_strategy}")
        
        # Generate backtesting splits
        backtest = BacktestingStrategy(
            strategy=self.backtesting_strategy,
            min_train_size=100,
            test_size=24,
            n_splits=10 if self.backtesting_strategy == "expanding" else 5
        )
        splits_by_series = backtest.generate_splits(data)
        
        total_splits = sum(len(s) for s in splits_by_series.values())
        logger.info(f"Generated {total_splits} backtesting splits")
        
        # Train models on each split
        for series_idx, (unique_id, splits) in enumerate(splits_by_series.items()):
            logger.info(f"Processing series {series_idx + 1}/{len(splits_by_series)}: {unique_id}")
            
            for split in splits:
                # Get train/test data
                train_df, test_df = backtest.get_split_data(data, unique_id, split)
                
                y_train = train_df["y"].values
                y_test = test_df["y"].values
                
                # Train each model
                for model_name in self.models:
                    try:
                        result = self._train_and_forecast(
                            model_name=model_name,
                            unique_id=unique_id,
                            split_id=split.split_id,
                            y_train=y_train,
                            y_test=y_test,
                            test_dates=test_df["ds"].values
                        )
                        self.results.append(result)
                    
                    except Exception as e:
                        logger.warning(
                            f"Failed to train {model_name} on {unique_id} split {split.split_id}: {e}"
                        )
                        continue
        
        # Convert results to DataFrame
        results_df = self._create_results_dataframe()
        
        # Save results if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = output_dir / "training_results.parquet"
            results_df.to_parquet(results_path, index=False)
            logger.info(f"Saved results to {results_path}")
        
        return results_df
    
    def _train_and_forecast(
        self,
        model_name: str,
        unique_id: str,
        split_id: int,
        y_train: np.ndarray,
        y_test: np.ndarray,
        test_dates: np.ndarray
    ) -> Dict:
        """Train model and generate forecast"""
        # Create model
        model = ModelFactory.create(model_name)
        
        # Train
        start_time = time.time()
        model.fit(y_train, dates=test_dates)
        train_time = time.time() - start_time
        
        # Forecast
        start_time = time.time()
        forecast = model.predict(horizon=len(y_test))
        forecast_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_series_metrics(
            y_true=y_test,
            y_pred=forecast,
            y_train=y_train
        )
        
        return {
            "unique_id": unique_id,
            "split_id": split_id,
            "model_name": model_name,
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "mase": metrics.get("mase"),
            "valid_count": metrics.get("valid_count", 0),
            "train_time": train_time,
            "forecast_time": forecast_time,
            "forecast": forecast,
            "actual": y_test
        }
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        # Extract forecast arrays to separate structure
        forecast_data = []
        metric_data = []
        
        for result in self.results:
            forecast = result["forecast"]
            actual = result["actual"]
            
            # Store forecast-level data
            for i, (pred, true) in enumerate(zip(forecast, actual)):
                forecast_data.append({
                    "unique_id": result["unique_id"],
                    "split_id": result["split_id"],
                    "model_name": result["model_name"],
                    "step": i,
                    "forecast": pred,
                    "actual": true
                })
            
            # Store split-level metrics
            metric_data.append({
                "unique_id": result["unique_id"],
                "split_id": result["split_id"],
                "model_name": result["model_name"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
                "mase": result["mase"],
                "valid_count": result["valid_count"],
                "train_time": result["train_time"],
                "forecast_time": result["forecast_time"]
            })
        
        return pd.DataFrame(metric_data)


class ModelSelector:
    """Select best model based on performance"""
    
    def __init__(self, primary_metric: str = "rmse", min_valid: int = 20):
        """
        Initialize model selector
        
        Args:
            primary_metric: Metric for ranking ("rmse", "mae", "mape", "mase")
            min_valid: Minimum valid predictions required
        """
        self.primary_metric = primary_metric
        self.min_valid = min_valid
    
    def select_best_model(
        self,
        results: pd.DataFrame
    ) -> Dict:
        """
        Select best model across all series and splits
        
        Args:
            results: Training results DataFrame
            
        Returns:
            Dictionary with best model info
        """
        # Filter valid results
        valid_results = results[results["valid_count"] >= self.min_valid].copy()
        
        if valid_results.empty:
            logger.error("No valid results found")
            return {}
        
        # Aggregate by model
        agg_dict = {
            "valid_count": "sum",
            "rmse": "mean",
            "mae": "mean",
            "mape": "mean",
            "mase": "mean"
        }
        
        model_performance = valid_results.groupby("model_name").agg(agg_dict)
        model_performance = model_performance.reset_index()
        
        # Sort by primary metric mean (lower is better)
        primary_metric_col = self.primary_metric
        model_performance = model_performance.sort_values(
            primary_metric_col,
            ascending=True
        )
        
        best_model = model_performance.iloc[0]
        
        return {
            "model_name": best_model["model_name"],
            "rmse_mean": float(best_model["rmse"]),
            "mae_mean": float(best_model["mae"]),
            "mape_mean": float(best_model["mape"]),
            "mase_mean": float(best_model["mase"]),
            "primary_metric": self.primary_metric,
            "primary_metric_mean": float(best_model[primary_metric_col]),
            "total_splits": int(best_model["valid_count"]),
            "ranking": model_performance.reset_index(drop=True)
        }
    
    def generate_leaderboard(
        self,
        results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate model leaderboard
        
        Args:
            results: Training results DataFrame
            
        Returns:
            Leaderboard DataFrame
        """
        # Filter valid results
        valid_results = results[results["valid_count"] >= self.min_valid].copy()
        
        # Aggregate by model
        leaderboard = valid_results.groupby("model_name").agg({
            "rmse": ["mean", "std"],
            "mae": ["mean", "std"],
            "mape": ["mean", "std"],
            "mase": ["mean", "std"],
            "valid_count": "sum",
            "train_time": ["mean", "max"],
            "forecast_time": ["mean", "max"]
        }).round(3)
        
        # Flatten column names
        leaderboard.columns = [
            "_".join(col).strip() for col in leaderboard.columns.values
        ]
        
        # Add rank
        leaderboard["rmse_rank"] = leaderboard["rmse_mean"].rank()
        leaderboard["mae_rank"] = leaderboard["mae_mean"].rank()
        leaderboard["mape_rank"] = leaderboard["mape_mean"].rank()
        leaderboard["mase_rank"] = leaderboard["mase_mean"].rank()
        
        # Average rank
        rank_cols = ["rmse_rank", "mae_rank", "mape_rank", "mase_rank"]
        leaderboard["avg_rank"] = leaderboard[rank_cols].mean(axis=1)
        
        return leaderboard.sort_values("avg_rank")
        return leaderboard.sort_values("avg_rank")
