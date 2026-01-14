"""
Phase 2: Chapter 2 Experiment Runner

Complete orchestration of:
1. Data loading from Phase 1 artifacts
2. Backtesting split generation
3. Model training and evaluation
4. Model comparison and selection
5. Leaderboard generation

Usage:
    python scripts/chapter2_runner.py \
        --input artifacts/baseline/clean.parquet \
        --output artifacts/chapter2 \
        --models exponential_smoothing arima prophet xgboost \
        --strategy rolling
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Chapter 2 modules
from src.chapter2.backtesting import (BacktestingStrategy,
                                      validate_backtesting_splits)
from src.chapter2.training import ModelSelector, TrainingPipeline


class Chapter2Runner:
    """Orchestrates Chapter 2 experiment runner"""

    def __init__(self, output_dir: Path):
        """Initialize with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {}
        self.results = {}

    def run(
        self,
        input_path: str,
        models: list = None,
        backtesting_strategy: str = "rolling",
        baseline_metrics_path: str = None
    ) -> Dict[str, Any]:
        """
        Run complete Chapter 2 pipeline

        Args:
            input_path: Path to clean.parquet from Phase 1
            models: List of models to train
            backtesting_strategy: "rolling" or "expanding"
            baseline_metrics_path: Path to baseline metrics

        Returns:
            Dictionary with status and results
        """
        if models is None:
            models = ["exponential_smoothing", "arima"]

        logger.info("=" * 80)
        logger.info("PHASE 2: Chapter 2 Experiment Runner")
        logger.info("=" * 80)

        try:
            # Step 1: Load data
            logger.info("\n[1/5] Loading data from Phase 1...")
            data = self._load_data(input_path)

            # Step 2: Generate backtesting splits
            logger.info("\n[2/5] Generating backtesting splits...")
            splits_info = self._generate_splits(data, backtesting_strategy)

            # Step 3: Train models
            logger.info("\n[3/5] Training models...")
            training_results = self._train_models(data, models, backtesting_strategy)

            # Step 4: Compare models
            logger.info("\n[4/5] Comparing models against baseline...")
            leaderboard, best_model = self._compare_models(
                training_results, baseline_metrics_path
            )

            # Step 5: Generate outputs
            logger.info("\n[5/5] Generating outputs...")
            self._save_results(
                training_results, leaderboard, best_model, splits_info
            )

            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2 COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"\nBest Model: {best_model['model_name']}")
            logger.info(f"  RMSE: {best_model['rmse_mean']:.2f}")
            logger.info(f"  MAE: {best_model['mae_mean']:.2f}")
            logger.info(f"  MAPE: {best_model['mape_mean']:.2f}%")
            logger.info(f"\nLeaderboard saved to: {self.output_dir / 'leaderboard.json'}")
            logger.info(f"Best model info saved to: {self.output_dir / 'best_model_info.json'}")

            return {
                "status": "SUCCESS",
                "output_dir": str(self.output_dir),
                "best_model": best_model["model_name"],
                "models_trained": len(models),
                "total_splits": splits_info["total_splits"],
                "artifacts": {
                    "leaderboard": str(self.output_dir / "leaderboard.json"),
                    "best_model_info": str(self.output_dir / "best_model_info.json"),
                    "training_results": str(self.output_dir / "training_results.parquet"),
                    "selection_report": str(self.output_dir / "selection_report.json")
                }
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.exception(e)
            return {
                "status": "FAILED",
                "error": str(e),
                "output_dir": str(self.output_dir)
            }

    def _load_data(self, input_path: str) -> pd.DataFrame:
        """Load and validate data"""
        logger.info(f"Loading data from {input_path}...")

        data = pd.read_parquet(input_path)

        # Validate columns (allow either 'y' or 'value')
        required_cols = ["unique_id", "ds"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Rename value to y if needed
        if "value" in data.columns and "y" not in data.columns:
            data = data.rename(columns={"value": "y"})

        # Convert ds to datetime if needed
        if data["ds"].dtype != "datetime64[ns]":
            data["ds"] = pd.to_datetime(data["ds"])

        logger.info(f"  Loaded {len(data)} observations")
        logger.info(f"  Series: {data['unique_id'].nunique()}")
        logger.info(f"  Date range: {data['ds'].min()} to {data['ds'].max()}")

        return data

    def _generate_splits(
        self,
        data: pd.DataFrame,
        strategy: str
    ) -> Dict[str, Any]:
        """Generate backtesting splits"""
        backtest = BacktestingStrategy(
            strategy=strategy,
            min_train_size=100,
            test_size=24,
            n_splits=10 if strategy == "expanding" else 5
        )

        splits_by_series = backtest.generate_splits(data)

        # Validate splits
        validation = validate_backtesting_splits(data, splits_by_series)

        total_splits = sum(len(s) for s in splits_by_series.values())
        valid_splits = sum(1 for v in validation.values() if v)

        logger.info(f"  Generated {total_splits} splits ({strategy})")
        logger.info(f"  Valid: {valid_splits}/{len(validation)}")

        # Serialize splits info
        splits_info = backtest.serialize_splits(splits_by_series)

        return {
            "strategy": strategy,
            "total_splits": total_splits,
            "series_count": len(splits_by_series),
            "validation": validation,
            "splits_info": splits_info
        }

    def _train_models(
        self,
        data: pd.DataFrame,
        models: list,
        strategy: str
    ) -> pd.DataFrame:
        """Train models"""
        logger.info(f"Training {len(models)} models...")

        pipeline = TrainingPipeline(
            models=models,
            backtesting_strategy=strategy
        )

        results = pipeline.run(data, output_dir=self.output_dir)

        logger.info(f"  Completed training")
        logger.info(f"  Results shape: {results.shape}")
        logger.info(f"  Models: {results['model_name'].unique().tolist()}")

        return results

    def _compare_models(
        self,
        results: pd.DataFrame,
        baseline_path: str = None
    ) -> tuple:
        """Compare models and select best"""
        logger.info("Comparing models...")

        selector = ModelSelector(primary_metric="rmse", min_valid=20)

        # Select best model
        best_model = selector.select_best_model(results)

        # Generate leaderboard
        leaderboard = selector.generate_leaderboard(results)

        logger.info(f"  Best model: {best_model['model_name']}")
        logger.info(f"  RMSE: {best_model['rmse_mean']:.2f}")
        logger.info(f"  MAE: {best_model['mae_mean']:.2f}")

        # Compare against baseline if provided
        if baseline_path:
            baseline_metrics = pd.read_parquet(baseline_path)
            baseline_rmse = baseline_metrics["rmse"].mean()
            improvement = (baseline_rmse - best_model['rmse_mean']) / baseline_rmse * 100

            logger.info(f"  vs Baseline RMSE: {improvement:+.1f}%")
            best_model['baseline_comparison'] = {
                'baseline_rmse': float(baseline_rmse),
                'improvement_pct': float(improvement)
            }

        return leaderboard, best_model

    def _save_results(
        self,
        results: pd.DataFrame,
        leaderboard: pd.DataFrame,
        best_model: Dict,
        splits_info: Dict
    ):
        """Save all results"""
        logger.info("Saving results...")

        # Save training results
        results_path = self.output_dir / "training_results.parquet"
        results.to_parquet(results_path, index=False)
        logger.info(f"  Saved: training_results.parquet ({len(results)} rows)")

        # Save leaderboard
        leaderboard_json = self.output_dir / "leaderboard.json"
        leaderboard.to_json(leaderboard_json, orient="index", indent=2, default_handler=str)
        logger.info(f"  Saved: leaderboard.json")

        # Save best model info
        best_model_path = self.output_dir / "best_model_info.json"
        with open(best_model_path, "w") as f:
            json.dump(best_model, f, indent=2, default=str)
        logger.info(f"  Saved: best_model_info.json")

        # Save selection report
        report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "chapter2",
            "best_model": best_model,
            "backtesting": splits_info,
            "results_summary": {
                "total_results": len(results),
                "models": results["model_name"].unique().tolist(),
                "series": results["unique_id"].unique().tolist(),
                "avg_results_per_model": len(results) / results["model_name"].nunique()
            }
        }

        report_path = self.output_dir / "selection_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"  Saved: selection_report.json")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Phase 2: Chapter 2 Experiment Runner"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/baseline/clean.parquet",
        help="Input data path (Phase 1 output)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/chapter2",
        help="Output directory"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["exponential_smoothing", "arima"],
        help="Models to train"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="rolling",
        choices=["rolling", "expanding"],
        help="Backtesting strategy"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="artifacts/baseline/metrics.parquet",
        help="Baseline metrics path"
    )

    args = parser.parse_args()

    runner = Chapter2Runner(args.output)
    result = runner.run(
        input_path=args.input,
        models=args.models,
        backtesting_strategy=args.strategy,
        baseline_metrics_path=args.baseline if os.path.exists(args.baseline) else None
    )

    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    exit(main())
    exit(main())
