"""
Chapter 2: Time Series Backtesting Framework

Implements multiple backtesting strategies for fair model comparison:
1. Rolling Window: Train on fixed window, slide forward
2. Expanding Window: Train on growing window

Both strategies ensure no information leakage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestSplit:
    """Represents a single train/test split"""
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray
    
    def __post_init__(self):
        """Validate no leakage"""
        if self.train_end >= self.test_start:
            raise ValueError(
                f"Train/test leakage: train_end ({self.train_end}) >= "
                f"test_start ({self.test_start})"
            )
        
    @property
    def train_size(self) -> int:
        """Number of training observations"""
        return len(self.train_indices)
    
    @property
    def test_size(self) -> int:
        """Number of test observations"""
        return len(self.test_indices)
    
    @property
    def info(self) -> Dict:
        """Serialize split info"""
        return {
            "split_id": self.split_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_size": self.train_size,
            "test_size": self.test_size
        }


class RollingWindowBacktest:
    """Rolling window backtesting strategy"""
    
    def __init__(
        self,
        min_train_size: int = 100,
        test_size: int = 24,
        step_size: int = 24
    ):
        """
        Initialize rolling window backtesting
        
        Args:
            min_train_size: Minimum training observations
            test_size: Number of test observations (forecast horizon)
            step_size: How many steps to slide forward
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def generate_splits(
        self,
        data: pd.DataFrame,
        unique_id: str
    ) -> List[BacktestSplit]:
        """
        Generate rolling window splits for a single series
        
        Args:
            data: DataFrame with columns [unique_id, ds, y]
            unique_id: Series identifier
            
        Returns:
            List of BacktestSplit objects
        """
        series = data[data["unique_id"] == unique_id].sort_values("ds").reset_index(drop=True)
        n = len(series)
        
        if n < self.min_train_size + self.test_size:
            raise ValueError(
                f"Series {unique_id} too short: {n} < "
                f"{self.min_train_size + self.test_size}"
            )
        
        splits = []
        split_id = 0
        
        for train_end_idx in range(
            self.min_train_size,
            n - self.test_size + 1,
            self.step_size
        ):
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.test_size, n)
            
            if test_end_idx - test_start_idx < self.test_size:
                # Skip incomplete test sets
                continue
            
            train_indices = np.arange(train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            split = BacktestSplit(
                split_id=split_id,
                train_start=series.iloc[0]["ds"],
                train_end=series.iloc[train_end_idx - 1]["ds"],
                test_start=series.iloc[test_start_idx]["ds"],
                test_end=series.iloc[test_end_idx - 1]["ds"],
                train_indices=train_indices,
                test_indices=test_indices
            )
            
            splits.append(split)
            split_id += 1
        
        logger.info(f"Generated {len(splits)} rolling splits for {unique_id}")
        return splits


class ExpandingWindowBacktest:
    """Expanding window backtesting strategy"""
    
    def __init__(
        self,
        min_train_size: int = 100,
        test_size: int = 24,
        n_splits: int = 10
    ):
        """
        Initialize expanding window backtesting
        
        Args:
            min_train_size: Starting training size
            test_size: Number of test observations (forecast horizon)
            n_splits: Number of expanding windows
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.n_splits = n_splits
    
    def generate_splits(
        self,
        data: pd.DataFrame,
        unique_id: str
    ) -> List[BacktestSplit]:
        """
        Generate expanding window splits for a single series
        
        Args:
            data: DataFrame with columns [unique_id, ds, y]
            unique_id: Series identifier
            
        Returns:
            List of BacktestSplit objects
        """
        series = data[data["unique_id"] == unique_id].sort_values("ds").reset_index(drop=True)
        n = len(series)
        
        if n < self.min_train_size + self.test_size * self.n_splits:
            raise ValueError(
                f"Series {unique_id} too short: {n} < "
                f"{self.min_train_size + self.test_size * self.n_splits}"
            )
        
        splits = []
        
        for i in range(self.n_splits):
            test_start_idx = self.min_train_size + (i * self.test_size)
            test_end_idx = test_start_idx + self.test_size
            
            if test_end_idx > n:
                # Not enough data
                break
            
            train_indices = np.arange(test_start_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            split = BacktestSplit(
                split_id=i,
                train_start=series.iloc[0]["ds"],
                train_end=series.iloc[test_start_idx - 1]["ds"],
                test_start=series.iloc[test_start_idx]["ds"],
                test_end=series.iloc[test_end_idx - 1]["ds"],
                train_indices=train_indices,
                test_indices=test_indices
            )
            
            splits.append(split)
        
        logger.info(f"Generated {len(splits)} expanding splits for {unique_id}")
        return splits


class BacktestingStrategy:
    """Unified interface for backtesting strategies"""
    
    def __init__(
        self,
        strategy: str = "rolling",
        min_train_size: int = 100,
        test_size: int = 24,
        step_size: int = 24,
        n_splits: int = 10,
        **kwargs
    ):
        """
        Initialize backtesting strategy
        
        Args:
            strategy: "rolling" or "expanding"
            min_train_size: Minimum training observations
            test_size: Test size (horizon)
            step_size: Step size for rolling window
            n_splits: Number of splits for expanding window
            **kwargs: Additional parameters (ignored)
        """
        self.strategy_name = strategy
        
        if strategy == "rolling":
            self.strategy = RollingWindowBacktest(
                min_train_size=min_train_size,
                test_size=test_size,
                step_size=step_size
            )
        elif strategy == "expanding":
            self.strategy = ExpandingWindowBacktest(
                min_train_size=min_train_size,
                test_size=test_size,
                n_splits=n_splits
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def generate_splits(
        self,
        data: pd.DataFrame
    ) -> Dict[str, List[BacktestSplit]]:
        """
        Generate splits for all series
        
        Args:
            data: DataFrame with columns [unique_id, ds, y]
            
        Returns:
            Dictionary mapping unique_id to list of BacktestSplit
        """
        logger.info(f"Generating {self.strategy_name} backtesting splits...")
        
        splits_by_series = {}
        
        for unique_id in data["unique_id"].unique():
            try:
                splits = self.strategy.generate_splits(data, unique_id)
                splits_by_series[unique_id] = splits
            except Exception as e:
                logger.warning(f"Failed to generate splits for {unique_id}: {e}")
                continue
        
        return splits_by_series
    
    def get_split_data(
        self,
        data: pd.DataFrame,
        unique_id: str,
        split: BacktestSplit
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test data for a specific split
        
        Args:
            data: Full dataset
            unique_id: Series identifier
            split: BacktestSplit object
            
        Returns:
            (train_df, test_df) tuple
        """
        series = data[data["unique_id"] == unique_id].sort_values("ds").reset_index(drop=True)
        
        train_df = series.iloc[split.train_indices].copy()
        test_df = series.iloc[split.test_indices].copy()
        
        return train_df, test_df
    
    def serialize_splits(
        self,
        splits_by_series: Dict[str, List[BacktestSplit]]
    ) -> Dict:
        """
        Serialize splits for saving to JSON
        
        Args:
            splits_by_series: Dictionary of splits
            
        Returns:
            Serializable dictionary
        """
        serialized = {}
        
        for unique_id, splits in splits_by_series.items():
            serialized[unique_id] = {
                "strategy": self.strategy_name,
                "n_splits": len(splits),
                "splits": [split.info for split in splits]
            }
        
        return serialized


def validate_backtesting_splits(
    data: pd.DataFrame,
    splits_by_series: Dict[str, List[BacktestSplit]]
) -> Dict[str, bool]:
    """
    Validate backtesting splits for no leakage
    
    Args:
        data: Full dataset
        splits_by_series: Dictionary of splits
        
    Returns:
        Dictionary mapping unique_id to validation result
    """
    logger.info("Validating backtesting splits...")
    
    validation_results = {}
    
    for unique_id, splits in splits_by_series.items():
        is_valid = True
        
        for split in splits:
            # Check 1: No temporal leakage
            if split.train_end >= split.test_start:
                logger.error(f"{unique_id} split {split.split_id}: temporal leakage")
                is_valid = False
            
            # Check 2: No duplicate indices
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)
            if train_set & test_set:
                logger.error(f"{unique_id} split {split.split_id}: overlapping indices")
                is_valid = False
            
            # Check 3: Minimum train size
            if len(split.train_indices) < 50:
                logger.error(f"{unique_id} split {split.split_id}: train size too small")
                is_valid = False
            
            # Check 4: Test size consistency
            if len(split.test_indices) != 24:
                logger.error(f"{unique_id} split {split.split_id}: test size != 24")
                is_valid = False
        
        validation_results[unique_id] = is_valid
        
        if is_valid:
            logger.info(f"{unique_id}: {len(splits)} splits VALID")
        else:
            logger.error(f"{unique_id}: INVALID splits")
    
    return validation_results
    return validation_results
