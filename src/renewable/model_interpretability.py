# src/renewable/model_interpretability.py
"""
Model interpretability: SHAP, Partial Dependence Plots, and Feature Importance.

This module provides functions to generate comprehensive interpretability reports
for LightGBM-based forecasting models, including:
- Feature importance extraction
- SHAP value computation and visualization
- Partial dependence plots
- Local prediction explanations (waterfall plots)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports - gracefully handle if not installed
try:
    import matplotlib  # noqa: E402
    matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
    import matplotlib.pyplot as plt  # noqa: E402
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed - visualization features will be unavailable")

try:
    import shap  # noqa: E402
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed - SHAP plots will be unavailable")

try:
    from sklearn.inspection import PartialDependenceDisplay  # noqa: E402
    PDP_AVAILABLE = True
except ImportError:
    PDP_AVAILABLE = False
    logger.warning("sklearn not installed - PDP plots will be unavailable")


@dataclass
class InterpretabilityReport:
    """Container for all interpretability artifacts."""

    series_id: str
    feature_importance: pd.DataFrame
    shap_summary_path: Optional[Path] = None
    shap_dependence_paths: dict[str, Path] = field(default_factory=dict)
    pdp_path: Optional[Path] = None
    waterfall_path: Optional[Path] = None
    top_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_feature_importance(forecaster: Any) -> pd.DataFrame:
    """
    Extract feature importance from a fitted skforecast forecaster.

    Args:
        forecaster: Fitted skforecast ForecasterRecursive or similar

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance desc
    """
    try:
        importance = forecaster.get_feature_importances()
        if isinstance(importance, pd.DataFrame):
            # Ensure standard column names
            if 'feature' not in importance.columns:
                importance = importance.reset_index()
                importance.columns = ['feature', 'importance']
            return importance.sort_values('importance', ascending=False).reset_index(drop=True)
        else:
            logger.warning("get_feature_importances() did not return a DataFrame")
            return pd.DataFrame(columns=['feature', 'importance'])
    except Exception as e:
        logger.error(f"Failed to extract feature importance: {e}")
        return pd.DataFrame(columns=['feature', 'importance'])


def compute_shap_values(
    estimator: Any,
    X_train: pd.DataFrame,
    sample_frac: float = 0.5,
    max_samples: int = 1000,
    seed: int = 42,
) -> tuple[Optional[Any], Optional[np.ndarray], pd.DataFrame]:
    """
    Compute SHAP values for a tree-based model.

    Args:
        estimator: Fitted tree-based estimator (LGBMRegressor, XGBRegressor, etc.)
        X_train: Training features DataFrame
        sample_frac: Fraction of data to sample for SHAP computation
        max_samples: Maximum number of samples to use
        seed: Random seed for reproducibility

    Returns:
        Tuple of (explainer, shap_values, X_sample) or (None, None, X_train) on error
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available - skipping SHAP computation")
        return None, None, X_train

    try:
        explainer = shap.TreeExplainer(estimator)

        # Sample for performance
        n_samples = min(int(len(X_train) * sample_frac), max_samples)
        rng = np.random.default_rng(seed=seed)
        sample_idx = rng.choice(len(X_train), size=n_samples, replace=False)
        X_sample = X_train.iloc[sample_idx].copy()

        shap_values = explainer.shap_values(X_sample)

        logger.info(f"Computed SHAP values for {len(X_sample)} samples")
        return explainer, shap_values, X_sample

    except Exception as e:
        logger.error(f"Failed to compute SHAP values: {e}")
        return None, None, X_train


def generate_shap_summary_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 15,
    title: Optional[str] = None,
) -> bool:
    """
    Generate SHAP summary plot showing feature importance and impact direction.

    Args:
        shap_values: SHAP values array from TreeExplainer
        X_sample: Feature DataFrame corresponding to SHAP values
        output_path: Path to save the PNG file
        max_display: Maximum number of features to display
        title: Optional title for the plot

    Returns:
        True if plot was generated successfully, False otherwise
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE or shap_values is None:
        return False

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=max_display,
            show=False,
            plot_size=None,
        )

        if title:
            plt.title(title, fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Generated SHAP summary plot: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate SHAP summary plot: {e}")
        plt.close()
        return False


def generate_shap_bar_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 15,
    title: Optional[str] = None,
) -> bool:
    """
    Generate SHAP bar plot showing mean absolute SHAP values.

    Args:
        shap_values: SHAP values array
        X_sample: Feature DataFrame
        output_path: Path to save the PNG file
        max_display: Maximum features to display
        title: Optional title

    Returns:
        True if successful, False otherwise
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE or shap_values is None:
        return False

    try:
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )

        if title:
            plt.title(title, fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Generated SHAP bar plot: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate SHAP bar plot: {e}")
        plt.close()
        return False


def generate_shap_dependence_plot(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    output_path: Path,
    interaction_feature: Optional[str] = "auto",
) -> bool:
    """
    Generate SHAP dependence plot for a single feature.

    Shows how the feature value affects the prediction, with optional
    interaction coloring.

    Args:
        shap_values: SHAP values array
        X_sample: Feature DataFrame
        feature: Feature name to plot
        output_path: Path to save the PNG file
        interaction_feature: Feature for interaction coloring ("auto" or feature name)

    Returns:
        True if successful, False otherwise
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE or shap_values is None:
        return False

    if feature not in X_sample.columns:
        logger.warning(f"Feature '{feature}' not in X_sample columns")
        return False

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feature,
            shap_values,
            X_sample,
            ax=ax,
            show=False,
            interaction_index=interaction_feature,
        )

        ax.set_title(f"SHAP Dependence: {feature}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Generated SHAP dependence plot for {feature}: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate SHAP dependence plot for {feature}: {e}")
        plt.close()
        return False


def generate_shap_waterfall(
    explainer: Any,
    X_predict: pd.DataFrame,
    prediction_idx: int,
    output_path: Path,
    title: Optional[str] = None,
) -> bool:
    """
    Generate SHAP waterfall plot for a single prediction (local interpretability).

    Shows how each feature contributed to moving the prediction away from
    the expected value.

    Args:
        explainer: SHAP TreeExplainer
        X_predict: Features for prediction(s)
        prediction_idx: Index of the prediction to explain
        output_path: Path to save the PNG file
        title: Optional title

    Returns:
        True if successful, False otherwise
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE or explainer is None:
        return False

    try:
        shap_values_single = explainer(X_predict)

        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values_single[prediction_idx], show=False)

        if title:
            plt.title(title, fontsize=10)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Generated SHAP waterfall plot: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate SHAP waterfall plot: {e}")
        plt.close()
        return False


def generate_partial_dependence_plot(
    estimator: Any,
    X_train: pd.DataFrame,
    features: list[str],
    output_path: Path,
    kind: str = "both",
    title: Optional[str] = None,
) -> bool:
    """
    Generate partial dependence plot using sklearn.

    Shows the marginal effect of features on the predicted outcome,
    optionally with individual conditional expectation (ICE) curves.

    Args:
        estimator: Fitted sklearn-compatible estimator
        X_train: Training features DataFrame
        features: List of feature names to plot (1-3 features recommended)
        output_path: Path to save the PNG file
        kind: "average", "individual", or "both" (PDP + ICE)
        title: Optional title

    Returns:
        True if successful, False otherwise
    """
    if not PDP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        logger.warning("PDP or matplotlib not available - skipping partial dependence plot")
        return False

    # Filter to features that exist in X_train
    valid_features = [f for f in features if f in X_train.columns]
    if not valid_features:
        logger.warning(f"No valid features found in X_train for PDP")
        return False

    try:
        fig, ax = plt.subplots(figsize=(4 * len(valid_features), 4))

        PartialDependenceDisplay.from_estimator(
            estimator=estimator,
            X=X_train,
            features=valid_features,
            kind=kind,
            ax=ax,
            n_jobs=-1,
        )

        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Generated partial dependence plot: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate partial dependence plot: {e}")
        plt.close()
        return False


def _select_representative_sample(X_train: pd.DataFrame, series_id: str) -> int:
    """
    Select a representative sample for the SHAP waterfall plot.

    For solar series (_SUN): Select a daytime sample with positive lag_1
    For wind series (_WND): Select a sample near the median of lag_1

    This ensures the waterfall shows meaningful feature contributions,
    not just "everything is zero" for nighttime solar.

    Args:
        X_train: Training features DataFrame
        series_id: Series identifier (e.g., "CALI_SUN", "ERCO_WND")

    Returns:
        Index of the selected sample
    """
    if len(X_train) == 0:
        return 0

    # Check if lag_1 exists (main feature for both solar and wind)
    if "lag_1" not in X_train.columns:
        logger.warning(f"[{series_id}] lag_1 not in X_train, using middle sample")
        return len(X_train) // 2

    lag_1 = X_train["lag_1"]

    is_solar = series_id.endswith("_SUN")

    if is_solar:
        # For solar: pick a daytime sample where lag_1 > 0 (generation was happening)
        # Also prefer samples where direct_radiation > 0 if available
        daytime_mask = lag_1 > 0

        if "direct_radiation" in X_train.columns:
            daytime_mask = daytime_mask & (X_train["direct_radiation"] > 0)

        daytime_indices = X_train.index[daytime_mask].tolist()

        if daytime_indices:
            # Pick one near the median generation among daytime samples
            daytime_lags = lag_1.loc[daytime_indices]
            median_val = daytime_lags.median()
            closest_idx = (daytime_lags - median_val).abs().idxmin()
            sample_pos = X_train.index.get_loc(closest_idx)
            logger.info(f"[{series_id}] Selected daytime sample at idx {sample_pos} (lag_1={lag_1.iloc[sample_pos]:.1f})")
            return sample_pos
        else:
            logger.warning(f"[{series_id}] No daytime samples found, using middle")
            return len(X_train) // 2

    else:
        # For wind: pick a sample near the median lag_1 (moderate wind)
        median_val = lag_1.median()
        closest_idx = (lag_1 - median_val).abs().idxmin()
        sample_pos = X_train.index.get_loc(closest_idx)
        logger.info(f"[{series_id}] Selected median-wind sample at idx {sample_pos} (lag_1={lag_1.iloc[sample_pos]:.1f})")
        return sample_pos


def generate_full_interpretability_report(
    forecaster: Any,
    X_train: pd.DataFrame,
    series_id: str,
    output_dir: Path,
    top_n_features: int = 5,
    shap_sample_frac: float = 0.5,
    shap_max_samples: int = 1000,
) -> InterpretabilityReport:
    """
    Generate complete interpretability report with all artifacts.

    Creates:
    - feature_importance.csv
    - shap_summary.png
    - shap_bar.png
    - shap_dependence_{feature}.png for top features
    - partial_dependence.png

    Args:
        forecaster: Fitted skforecast ForecasterRecursive
        X_train: Training features DataFrame
        series_id: Unique identifier for the series (e.g., "CALI_WND")
        output_dir: Directory to save artifacts
        top_n_features: Number of top features for dependence plots
        shap_sample_frac: Fraction of data for SHAP computation
        shap_max_samples: Maximum samples for SHAP

    Returns:
        InterpretabilityReport with paths to all generated artifacts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = InterpretabilityReport(
        series_id=series_id,
        feature_importance=pd.DataFrame(columns=['feature', 'importance']),
        metadata={"output_dir": str(output_dir)},
    )

    # 1. Feature importance
    logger.info(f"[{series_id}] Computing feature importance...")
    importance = compute_feature_importance(forecaster)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    report.feature_importance = importance
    report.top_features = importance.head(top_n_features)["feature"].tolist()

    # Get the internal estimator for SHAP/PDP
    # Use 'estimator' (new API) with fallback to 'regressor' (deprecated)
    try:
        if hasattr(forecaster, "estimator"):
            estimator = forecaster.estimator
        else:
            estimator = forecaster.regressor
    except AttributeError:
        logger.warning(f"[{series_id}] Could not access forecaster estimator")
        return report

    # 2. SHAP values
    logger.info(f"[{series_id}] Computing SHAP values...")
    explainer, shap_values, X_sample = compute_shap_values(
        estimator,
        X_train,
        sample_frac=shap_sample_frac,
        max_samples=shap_max_samples,
    )

    # 3. SHAP summary plot
    if shap_values is not None:
        shap_summary_path = output_dir / "shap_summary.png"
        if generate_shap_summary_plot(
            shap_values, X_sample, shap_summary_path,
            title=f"SHAP Summary: {series_id}"
        ):
            report.shap_summary_path = shap_summary_path

        # SHAP bar plot
        shap_bar_path = output_dir / "shap_bar.png"
        generate_shap_bar_plot(
            shap_values, X_sample, shap_bar_path,
            title=f"Mean |SHAP|: {series_id}"
        )

        # 4. SHAP dependence plots for top features
        logger.info(f"[{series_id}] Generating SHAP dependence plots...")
        for feat in report.top_features:
            if feat in X_sample.columns:
                dep_path = output_dir / f"shap_dependence_{feat}.png"
                if generate_shap_dependence_plot(shap_values, X_sample, feat, dep_path):
                    report.shap_dependence_paths[feat] = dep_path

    # 5. Partial dependence plots
    logger.info(f"[{series_id}] Generating partial dependence plot...")
    pdp_features = [f for f in report.top_features[:3] if f in X_train.columns]
    if pdp_features:
        pdp_path = output_dir / "partial_dependence.png"
        if generate_partial_dependence_plot(
            estimator, X_train, pdp_features, pdp_path,
            title=f"Partial Dependence: {series_id}"
        ):
            report.pdp_path = pdp_path

    # 6. Waterfall for a sample prediction (select representative sample)
    if explainer is not None and len(X_train) > 0:
        waterfall_path = output_dir / "shap_waterfall_sample.png"

        # Smart sample selection - pick a representative sample, not random/middle
        sample_idx = _select_representative_sample(X_train, series_id)

        if generate_shap_waterfall(
            explainer,
            X_train.iloc[[sample_idx]],
            0,
            waterfall_path,
            title=f"Prediction Explanation: {series_id}"
        ):
            report.waterfall_path = waterfall_path

    report.metadata["n_features"] = len(importance)
    report.metadata["n_shap_samples"] = len(X_sample) if shap_values is not None else 0

    logger.info(f"[{series_id}] Interpretability report complete: {output_dir}")
    return report


if __name__ == "__main__":
    # Quick test with synthetic data
    logging.basicConfig(level=logging.INFO)

    try:
        from lightgbm import LGBMRegressor
        from skforecast.recursive import ForecasterRecursive

        # Create synthetic time series
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        y = pd.Series(
            100 + 20 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.randn(n) * 5,
            index=dates,
            name="y"
        )
        exog = pd.DataFrame({
            "temperature": 20 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.randn(n),
            "wind_speed": 5 + 3 * np.random.randn(n).clip(-2, 2),
        }, index=dates)

        # Fit forecaster
        forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(random_state=42, verbose=-1, n_estimators=50),
            lags=24,
        )
        forecaster.fit(y=y, exog=exog)

        # Get training matrices
        X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)

        # Generate report
        output_dir = Path("data/renewable/interpretability/TEST_SERIES")
        report = generate_full_interpretability_report(
            forecaster=forecaster,
            X_train=X_train,
            series_id="TEST_SERIES",
            output_dir=output_dir,
        )

        print(f"\nInterpretability Report for {report.series_id}:")
        print(f"  Top features: {report.top_features}")
        print(f"  SHAP summary: {report.shap_summary_path}")
        print(f"  SHAP dependence plots: {list(report.shap_dependence_paths.keys())}")
        print(f"  PDP path: {report.pdp_path}")
        print(f"\nFeature Importance:")
        print(report.feature_importance.head(10))

    except ImportError as e:
        print(f"Test requires lightgbm and skforecast: {e}")
