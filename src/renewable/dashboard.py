# file: src/renewable/dashboard.py
"""Streamlit dashboard for renewable energy forecasting.

Provides:
- Forecast visualization with prediction intervals
- Drift monitoring and alerts
- Coverage analysis (nominal vs empirical)
- Weather features by region

Run with:
    streamlit run src/renewable/dashboard.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.renewable.db import (
    connect,
    get_drift_alerts,
    get_recent_forecasts,
    init_renewable_db,
)
from src.renewable.regions import FUEL_TYPES, REGIONS

# Page config
st.set_page_config(
    page_title="Renewable Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)


def main():
    """Main dashboard application."""
    st.title("⚡ Renewable Energy Forecast Dashboard")
    st.markdown("Next-24h wind/solar generation forecasts with drift monitoring")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        db_path = st.text_input(
            "Database Path",
            value="data/renewable/renewable.db",
        )

        # Initialize database if it doesn't exist
        if not Path(db_path).exists():
            init_renewable_db(db_path)
            st.info("Database initialized")

        st.divider()

        # Region filter
        all_regions = list(REGIONS.keys())
        selected_regions = st.multiselect(
            "Regions",
            options=all_regions,
            default=["CALI", "ERCO", "MISO"],
        )

        # Fuel type filter
        fuel_type = st.selectbox(
            "Fuel Type",
            options=["WND", "SUN", "Both"],
            index=0,
        )

        st.divider()

        # Actions
        show_debug = st.checkbox("Show Debug", value=False)
        if st.button("🔄 Refresh Data", width="stretch"):
            st.rerun()

        if st.button("📊 Run Pipeline", width="stretch"):
            run_pipeline_from_dashboard(db_path, selected_regions, fuel_type)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecasts",
        "⚠️ Drift Monitor",
        "📊 Coverage",
        "🌤️ Weather",
        "🔍 Interpretability",
    ])

    with tab1:
        render_forecasts_tab(db_path, selected_regions, fuel_type, show_debug=show_debug)

    with tab2:
        render_drift_tab(db_path)

    with tab3:
        render_coverage_tab(db_path)

    with tab4:
        render_weather_tab(db_path, selected_regions)

    with tab5:
        render_interpretability_tab(selected_regions, fuel_type)


def render_forecasts_tab(db_path: str, regions: list, fuel_type: str, *, show_debug: bool = False):
    """Render forecast visualization with prediction intervals."""
    st.subheader("Generation Forecasts")

    forecasts_df = pd.DataFrame()
    data_source = "none"
    derived_columns: list[str] = []

    # Try to load from parquet file first (pipeline output)
    parquet_path = Path("data/renewable/forecasts.parquet")
    if parquet_path.exists():
        try:
            forecasts_df = pd.read_parquet(parquet_path)
            data_source = f"parquet:{parquet_path}"
            # Add region/fuel_type columns if missing
            if "unique_id" in forecasts_df.columns:
                parts = forecasts_df["unique_id"].astype(str).str.split("_", n=1, expand=True)
                if "region" not in forecasts_df.columns:
                    forecasts_df["region"] = parts[0]
                    derived_columns.append("region")
                if "fuel_type" not in forecasts_df.columns:
                    forecasts_df["fuel_type"] = parts[1] if parts.shape[1] > 1 else pd.NA
                    derived_columns.append("fuel_type")
            st.success(f"Loaded {len(forecasts_df)} forecasts from pipeline")

            # Calculate and display data freshness
            if not forecasts_df.empty and "ds" in forecasts_df.columns:
                earliest_forecast_ts = forecasts_df["ds"].min()
                now_utc = pd.Timestamp.now(tz="UTC").floor("h")

                # Forecasts start from last_data + 1h, so last_data = earliest_forecast - 1h
                last_data_ts = earliest_forecast_ts - pd.Timedelta(hours=1)

                # Ensure both timestamps are timezone-aware for comparison
                if not hasattr(last_data_ts, 'tz') or last_data_ts.tz is None:
                    last_data_ts = pd.Timestamp(last_data_ts, tz="UTC")

                data_age_hours = (now_utc - last_data_ts).total_seconds() / 3600

                # Show warning if data is > 6 hours old
                if data_age_hours > 6:
                    st.warning(
                        f"⚠️ Forecasts are based on **{data_age_hours:.1f} hour old** data "
                        f"(last EIA data: {last_data_ts.strftime('%b %d %H:%M')} UTC). "
                        f"Click 'Refresh Forecasts' button in sidebar to update."
                    )
                else:
                    st.info(
                        f"✅ Forecasts from {last_data_ts.strftime('%b %d %H:%M')} UTC data "
                        f"({data_age_hours:.1f}h old)"
                    )

        except Exception as e:
            st.warning(f"Could not load parquet: {e}")

    # Fall back to database
    if forecasts_df.empty:
        try:
            forecasts_df = get_recent_forecasts(db_path, hours=72)
            data_source = f"db:{db_path}"
        except Exception as e:
            st.warning(f"Could not load from database: {e}")

    if forecasts_df.empty:
        # Show demo data
        st.info("No forecasts found. Showing demo data.")
        forecasts_df = generate_demo_forecasts(regions, fuel_type)
        data_source = "demo"

    if show_debug:
        with st.expander("Debug: Forecast Data", expanded=False):
            st.markdown("**Source**")
            st.code(data_source)
            st.markdown("**Columns**")
            st.code(", ".join(forecasts_df.columns.tolist()))

            st.markdown("**Counts (pre-filter)**")
            st.write({"rows": int(len(forecasts_df))})

            if derived_columns:
                st.markdown("**Derived Columns**")
                st.write(derived_columns)

            if "unique_id" in forecasts_df.columns:
                st.markdown("**unique_id sample**")
                st.write(forecasts_df["unique_id"].dropna().astype(str).head(10).tolist())

            if "fuel_type" in forecasts_df.columns:
                st.markdown("**fuel_type counts**")
                st.dataframe(forecasts_df["fuel_type"].value_counts(dropna=False).to_frame())

                unknown = sorted(
                    {str(v) for v in forecasts_df["fuel_type"].dropna().unique()}
                    - set(FUEL_TYPES.keys())
                )
                if unknown:
                    st.warning(f"Unknown fuel_type values: {unknown}")

            if "region" in forecasts_df.columns:
                st.markdown("**region counts**")
                st.dataframe(forecasts_df["region"].value_counts(dropna=False).to_frame())

    # Filter by selections
    if fuel_type != "Both":
        forecasts_df = forecasts_df[forecasts_df["fuel_type"] == fuel_type]

    if regions:
        forecasts_df = forecasts_df[forecasts_df["region"].isin(regions)]

    if show_debug:
        with st.expander("Debug: Filter Result", expanded=False):
            st.markdown("**Applied Filters**")
            st.write({"fuel_type": fuel_type, "regions": regions})
            st.markdown("**Counts (post-filter)**")
            st.write({"rows": int(len(forecasts_df))})
            if "unique_id" in forecasts_df.columns:
                st.markdown("**unique_id after filter**")
                st.write(sorted(forecasts_df["unique_id"].dropna().astype(str).unique().tolist()))

    if forecasts_df.empty:
        st.warning("No data matching filters")
        return

    # Series selector
    series_options = forecasts_df["unique_id"].unique().tolist()
    selected_series = st.selectbox(
        "Select Series",
        options=series_options,
        index=0 if series_options else None,
        key="forecast_series_select",
    )

    if selected_series:
        series_data = forecasts_df[forecasts_df["unique_id"] == selected_series].copy()
        series_data = series_data.sort_values("ds")

        # Convert to local timezone for display
        region_code = series_data["unique_id"].iloc[0].split("_")[0]
        region_info = REGIONS.get(region_code)
        timezone_name = region_info.timezone if region_info else "UTC"

        # Create forecast plot with intervals
        fig = create_forecast_plot(series_data, selected_series, timezone_name)
        st.plotly_chart(fig, width="stretch")

        # Show data table
        with st.expander("View Data"):
            st.dataframe(
                series_data[["ds", "yhat", "yhat_lo_80", "yhat_hi_80", "yhat_lo_95", "yhat_hi_95"]],
                width="stretch",
            )


def create_forecast_plot(df: pd.DataFrame, title: str, timezone_name: str = "UTC") -> go.Figure:
    """Create Plotly figure with forecast and prediction intervals.

    Args:
        df: Forecast dataframe with ds (timestamp), yhat, and interval columns
        title: Series name for chart title
        timezone_name: IANA timezone name for display (e.g., "America/Chicago")
    """
    fig = go.Figure()

    # Convert timestamps to local timezone for display
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Convert UTC to local timezone
    if timezone_name != "UTC":
        df["ds"] = df["ds"].dt.tz_localize("UTC").dt.tz_convert(timezone_name)

    # Get timezone abbreviation for display (e.g., "CST", "PST")
    if timezone_name != "UTC" and len(df) > 0:
        tz_abbr = df["ds"].iloc[0].strftime("%Z")
    else:
        tz_abbr = "UTC"

    # 95% interval (outer, lighter)
    if "yhat_lo_95" in df.columns and "yhat_hi_95" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["ds"], df["ds"][::-1]]),
            y=pd.concat([df["yhat_hi_95"], df["yhat_lo_95"][::-1]]),
            fill="toself",
            fillcolor="rgba(68, 138, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Interval",
            hoverinfo="skip",
        ))

    # 80% interval (inner, darker)
    if "yhat_lo_80" in df.columns and "yhat_hi_80" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["ds"], df["ds"][::-1]]),
            y=pd.concat([df["yhat_hi_80"], df["yhat_lo_80"][::-1]]),
            fill="toself",
            fillcolor="rgba(68, 138, 255, 0.4)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Interval",
            hoverinfo="skip",
        ))

    # Point forecast
    fig.add_trace(go.Scatter(
        x=df["ds"],
        y=df["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="#1f77b4", width=2),
    ))

    # Actuals if available
    if "y" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="markers",
            name="Actual",
            marker=dict(color="#2ca02c", size=6),
        ))

    fig.update_layout(
        title=f"Forecast: {title}",
        xaxis_title=f"Time ({tz_abbr})",
        yaxis_title="Generation (MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
    )

    return fig


def render_drift_tab(db_path: str):
    """Render drift monitoring and alerts."""
    st.subheader("Drift Detection")

    col1, col2, col3 = st.columns(3)

    # Try to load alerts
    try:
        alerts_df = get_drift_alerts(db_path, hours=48)
    except Exception:
        alerts_df = pd.DataFrame()

    # Summary metrics
    with col1:
        critical = len(alerts_df[alerts_df["severity"] == "critical"]) if not alerts_df.empty else 0
        st.metric(
            "Critical Alerts",
            critical,
            delta=None,
            delta_color="inverse" if critical > 0 else "off",
        )

    with col2:
        warning = len(alerts_df[alerts_df["severity"] == "warning"]) if not alerts_df.empty else 0
        st.metric("Warnings", warning)

    with col3:
        stable = len(alerts_df[alerts_df["alert_type"] == "drift_check"]) if not alerts_df.empty else 0
        st.metric("Stable Checks", stable)

    st.divider()

    if alerts_df.empty:
        st.info("No drift alerts in the last 48 hours. System is stable.")

        # Show demo drift status
        st.markdown("### Demo Drift Status")
        demo_drift = pd.DataFrame({
            "Series": ["CALI_WND", "ERCO_WND", "MISO_WND", "CALI_SUN", "ERCO_SUN"],
            "Current RMSE": [125.3, 98.7, 156.2, 45.1, 67.8],
            "Threshold": [150.0, 120.0, 180.0, 60.0, 80.0],
            "Status": ["✅ Stable", "✅ Stable", "✅ Stable", "✅ Stable", "✅ Stable"],
        })
        st.dataframe(demo_drift, width="stretch")
    else:
        # Show alerts table
        st.dataframe(
            alerts_df[["alert_at", "unique_id", "severity", "current_rmse", "threshold_rmse", "message"]],
            width="stretch",
        )

        # Drift timeline
        if len(alerts_df) > 1:
            alerts_df["alert_at"] = pd.to_datetime(alerts_df["alert_at"])
            fig = px.scatter(
                alerts_df,
                x="alert_at",
                y="current_rmse",
                color="severity",
                size="current_rmse",
                hover_data=["unique_id", "message"],
                title="Drift Timeline",
            )
            fig.add_hline(
                y=alerts_df["threshold_rmse"].mean(),
                line_dash="dash",
                annotation_text="Avg Threshold",
            )
            st.plotly_chart(fig, width="stretch")


def render_coverage_tab(db_path: str):
    """Render coverage analysis comparing nominal vs empirical."""
    st.subheader("Prediction Interval Coverage")

    st.markdown("""
    **Coverage** measures how often actual values fall within prediction intervals.
    - **Nominal**: The expected coverage (80% or 95%)
    - **Empirical**: The actual observed coverage
    - **Gap**: Difference indicates calibration quality
    """)

    # Demo coverage data
    coverage_data = pd.DataFrame({
        "Series": ["CALI_WND", "ERCO_WND", "MISO_WND", "SWPP_WND", "CALI_SUN", "ERCO_SUN"],
        "Nominal 80%": [80, 80, 80, 80, 80, 80],
        "Empirical 80%": [78.5, 82.1, 76.3, 79.8, 81.2, 77.9],
        "Nominal 95%": [95, 95, 95, 95, 95, 95],
        "Empirical 95%": [93.2, 96.1, 91.5, 94.8, 95.7, 92.3],
    })

    coverage_data["Gap 80%"] = coverage_data["Empirical 80%"] - coverage_data["Nominal 80%"]
    coverage_data["Gap 95%"] = coverage_data["Empirical 95%"] - coverage_data["Nominal 95%"]

    # Summary
    col1, col2 = st.columns(2)

    with col1:
        avg_80 = coverage_data["Empirical 80%"].mean()
        st.metric("Avg 80% Coverage", f"{avg_80:.1f}%", f"{avg_80 - 80:.1f}%")

    with col2:
        avg_95 = coverage_data["Empirical 95%"].mean()
        st.metric("Avg 95% Coverage", f"{avg_95:.1f}%", f"{avg_95 - 95:.1f}%")

    st.divider()

    # Coverage comparison chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="80% Empirical",
        x=coverage_data["Series"],
        y=coverage_data["Empirical 80%"],
        marker_color="rgba(68, 138, 255, 0.7)",
    ))

    fig.add_trace(go.Bar(
        name="95% Empirical",
        x=coverage_data["Series"],
        y=coverage_data["Empirical 95%"],
        marker_color="rgba(68, 138, 255, 0.4)",
    ))

    # Nominal lines
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% Nominal")
    fig.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="95% Nominal")

    fig.update_layout(
        title="Coverage by Series",
        xaxis_title="Series",
        yaxis_title="Coverage (%)",
        barmode="group",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

    # Detailed table
    with st.expander("View Coverage Data"):
        st.dataframe(coverage_data, width="stretch")


def render_weather_tab(db_path: str, regions: list):
    """Render weather features visualization."""
    st.subheader("Weather Features")

    weather_df = pd.DataFrame()

    # Prefer real pipeline output; no demo fallback.
    parquet_path = Path("data/renewable/weather.parquet")
    if parquet_path.exists():
        try:
            weather_df = pd.read_parquet(parquet_path)
            st.success(f"Loaded {len(weather_df)} weather rows from pipeline")
        except Exception as exc:
            st.warning(f"Could not load weather parquet: {exc}")

    if weather_df.empty and Path(db_path).exists():
        try:
            with connect(db_path) as con:
                weather_df = pd.read_sql_query(
                    "SELECT * FROM weather_features ORDER BY ds ASC",
                    con,
                )
            if not weather_df.empty:
                st.success(f"Loaded {len(weather_df)} weather rows from database")
        except Exception as exc:
            st.warning(f"Could not load weather data from database: {exc}")

    if weather_df.empty:
        st.warning("No weather data available. Run the pipeline to populate weather features.")
        return

    weather_df["ds"] = pd.to_datetime(weather_df["ds"], errors="coerce")
    if regions:
        weather_df = weather_df[weather_df["region"].isin(regions)]
    if weather_df.empty:
        st.warning("No weather data matching selected regions.")
        return

    # Variable selector
    weather_vars = [
        col for col in ["wind_speed_10m", "wind_speed_100m", "direct_radiation", "cloud_cover"]
        if col in weather_df.columns
    ]
    if not weather_vars:
        st.warning("Weather data missing expected variables.")
        return
    selected_var = st.selectbox("Weather Variable", options=weather_vars)

    # Plot
    fig = px.line(
        weather_df,
        x="ds",
        y=selected_var,
        color="region",
        title=f"{selected_var} by Region",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

    # Summary stats
    st.markdown("### Current Conditions")

    cols = st.columns(len(regions[:4]))
    for i, region in enumerate(regions[:4]):
        if i < len(cols):
            with cols[i]:
                region_data = weather_df[weather_df["region"] == region].iloc[-1] if len(weather_df[weather_df["region"] == region]) > 0 else {}
                st.metric(
                    region,
                    f"{region_data.get('wind_speed_10m', 0):.1f} m/s",
                    help="Wind speed at 10m",
                )


def render_interpretability_tab(regions: list, fuel_type: str):
    """Render model interpretability visualizations (SHAP, feature importance, PDP)."""
    st.subheader("Model Interpretability")

    # Model Leaderboard Section
    st.markdown("### 🏆 Model Leaderboard (Cross-Validation)")

    # Model descriptions for education
    MODEL_INFO = {
        "AutoARIMA": {
            "type": "Statistical",
            "description": "Auto-tuned ARIMA with automatic p,d,q selection. Good for univariate series with trend/seasonality.",
            "strengths": "Robust, well-understood, good prediction intervals",
        },
        "MSTL_ARIMA": {
            "type": "Statistical",
            "description": "Multiple Seasonal-Trend decomposition + ARIMA. Handles daily (24h) and weekly (168h) seasonality.",
            "strengths": "Best for multi-seasonal patterns like energy data",
        },
        "AutoETS": {
            "type": "Statistical",
            "description": "Exponential smoothing with automatic error/trend/season selection.",
            "strengths": "Simple, fast, works well for smooth series",
        },
        "AutoTheta": {
            "type": "Statistical",
            "description": "Theta method with automatic decomposition. Robust to outliers.",
            "strengths": "Competition winner (M3), handles level shifts",
        },
        "CES": {
            "type": "Statistical",
            "description": "Complex Exponential Smoothing. Captures complex seasonal patterns.",
            "strengths": "Good for complex seasonality",
        },
        "SeasonalNaive": {
            "type": "Baseline",
            "description": "Uses value from same hour last week. Baseline benchmark.",
            "strengths": "Simple benchmark - if beaten, models add value",
        },
    }

    run_log_path = Path("data/renewable/run_log.json")
    if run_log_path.exists():
        try:
            import json
            run_log = json.loads(run_log_path.read_text())
            pipeline_results = run_log.get("pipeline_results", {})
            leaderboard_data = pipeline_results.get("leaderboard", [])

            if leaderboard_data:
                leaderboard_df = pd.DataFrame(leaderboard_data)
                best_model = pipeline_results.get("best_model", "")
                best_rmse = pipeline_results.get("best_rmse", 0)

                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Model", best_model)
                with col2:
                    st.metric("Best RMSE", f"{best_rmse:.3f}")
                with col3:
                    st.metric("Models Evaluated", len(leaderboard_data))
                with col4:
                    # Calculate improvement over baseline
                    baseline_rmse = leaderboard_df[leaderboard_df["model"] == "SeasonalNaive"]["rmse"].values
                    if len(baseline_rmse) > 0 and best_rmse > 0:
                        improvement = ((baseline_rmse[0] - best_rmse) / baseline_rmse[0]) * 100
                        st.metric("vs Baseline", f"{improvement:+.1f}%", help="Improvement over SeasonalNaive")
                    else:
                        st.metric("vs Baseline", "N/A")

                # Selection rationale
                st.markdown("#### Why This Model?")
                st.info(f"""
                **{best_model}** was selected because it has the **lowest RMSE** on cross-validation.

                - **RMSE (Root Mean Square Error)**: Penalizes large errors more heavily. Best for energy forecasting where big misses are costly.
                - **Selection method**: Time-series CV with {run_log.get('config', {}).get('cv_windows', 2)} windows, step size {run_log.get('config', {}).get('cv_step_size', 168)}h
                - **Horizon**: {run_log.get('config', {}).get('horizon', 24)}h ahead forecasts
                """)

                # Model description for winner
                if best_model in MODEL_INFO:
                    info = MODEL_INFO[best_model]
                    st.success(f"**{info['type']} Model**: {info['description']}")

                # Full leaderboard with visualization
                st.markdown("#### All Models Ranked by RMSE")

                display_cols = [c for c in ["model", "rmse", "mae", "mape", "coverage_80", "coverage_95"]
                               if c in leaderboard_df.columns]

                # Create visualization
                if "rmse" in leaderboard_df.columns:
                    fig = px.bar(
                        leaderboard_df.sort_values("rmse"),
                        x="model",
                        y="rmse",
                        title="Model Comparison (Lower RMSE = Better)",
                        color="rmse",
                        color_continuous_scale="RdYlGn_r",
                    )
                    fig.add_hline(y=best_rmse, line_dash="dash", line_color="green",
                                  annotation_text=f"Best: {best_rmse:.3f}")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, width="stretch")

                # Format numeric columns for table
                styled_df = leaderboard_df[display_cols].copy()
                for col in ["rmse", "mae", "mape"]:
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                for col in ["coverage_80", "coverage_95"]:
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

                st.dataframe(styled_df, width="stretch", hide_index=True)

                # Coverage analysis
                if "coverage_80" in leaderboard_df.columns:
                    st.markdown("#### Prediction Interval Coverage")
                    st.markdown("""
                    **Coverage** measures if prediction intervals are well-calibrated:
                    - **80% interval** should contain ~80% of actual values
                    - **95% interval** should contain ~95% of actual values
                    - **Under-coverage** (<target) = intervals too narrow, overconfident
                    - **Over-coverage** (>target) = intervals too wide, conservative
                    """)

                    coverage_df = leaderboard_df[["model", "coverage_80", "coverage_95"]].copy()
                    coverage_df["coverage_80_status"] = coverage_df["coverage_80"].apply(
                        lambda x: "Under" if x < 75 else ("Over" if x > 85 else "Good") if pd.notna(x) else "N/A"
                    )
                    coverage_df["coverage_95_status"] = coverage_df["coverage_95"].apply(
                        lambda x: "Under" if x < 90 else ("Over" if x > 99 else "Good") if pd.notna(x) else "N/A"
                    )

                # Model descriptions expander
                with st.expander("Model Descriptions"):
                    for model_name, info in MODEL_INFO.items():
                        st.markdown(f"**{model_name}** ({info['type']})")
                        st.markdown(f"- {info['description']}")
                        st.markdown(f"- *Strengths*: {info['strengths']}")
                        st.markdown("---")

                # CV configuration expander
                config = run_log.get("config", {})
                with st.expander("CV Configuration"):
                    st.write({
                        "cv_windows": config.get("cv_windows"),
                        "cv_step_size": config.get("cv_step_size"),
                        "horizon": config.get("horizon"),
                        "regions": config.get("regions"),
                        "fuel_types": config.get("fuel_types"),
                        "run_at": run_log.get("run_at_utc", "N/A"),
                    })
            else:
                st.info("Leaderboard not available. Run the pipeline with the latest code to generate.")
        except Exception as e:
            st.warning(f"Could not load leaderboard: {e}")
    else:
        st.info("No run log found. Run the pipeline to generate model comparison.")

    st.divider()

    st.markdown("### 🔍 Per-Series Interpretability")
    st.markdown("""
    **LightGBM** models are trained alongside statistical models (MSTL/ARIMA) to provide
    interpretability insights. The statistical models generate the primary forecasts,
    while LightGBM helps understand feature importance and relationships.
    """)

    interp_dir = Path("data/renewable/interpretability")

    if not interp_dir.exists():
        st.info("No interpretability data available. Run the pipeline to generate SHAP and PDP plots.")
        return

    # Get available series
    series_dirs = sorted([d.name for d in interp_dir.iterdir() if d.is_dir()])

    if not series_dirs:
        st.warning("Interpretability directory exists but contains no series data.")
        return

    # Filter by selected regions and fuel type
    filtered_series = []
    for series_id in series_dirs:
        parts = series_id.split("_")
        if len(parts) == 2:
            region, ft = parts
            if regions and region not in regions:
                continue
            if fuel_type != "Both" and ft != fuel_type:
                continue
            filtered_series.append(series_id)

    if not filtered_series:
        st.warning("No interpretability data for selected filters.")
        return

    # Series selector
    selected_series = st.selectbox(
        "Select Series",
        options=filtered_series,
        index=0,
        key="interpretability_series_select",
    )

    if not selected_series:
        return

    series_dir = interp_dir / selected_series

    # Layout: Feature Importance + SHAP Summary side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Feature Importance")
        importance_path = series_dir / "feature_importance.csv"
        if importance_path.exists():
            try:
                importance_df = pd.read_csv(importance_path)
                # Show top 15 features
                top_features = importance_df.head(15)

                # Create bar chart
                fig = px.bar(
                    top_features,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top Features: {selected_series}",
                    labels={"importance": "Importance", "feature": "Feature"},
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
                st.plotly_chart(fig, width="stretch")

                with st.expander("Full Feature List"):
                    st.dataframe(importance_df, width="stretch")
            except Exception as e:
                st.error(f"Error loading feature importance: {e}")
        else:
            st.info("Feature importance not available.")

    with col2:
        st.markdown("### SHAP Summary")
        shap_summary_path = series_dir / "shap_summary.png"
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), width="stretch")
        else:
            # Try bar plot as fallback
            shap_bar_path = series_dir / "shap_bar.png"
            if shap_bar_path.exists():
                st.image(str(shap_bar_path), width="stretch")
            else:
                st.info("SHAP summary not available.")

    st.divider()

    # SHAP Dependence Plots
    st.markdown("### SHAP Dependence Plots")
    st.markdown("Shows how individual feature values affect predictions.")

    shap_dep_files = list(series_dir.glob("shap_dependence_*.png"))
    if shap_dep_files:
        # Create columns for dependence plots
        n_cols = min(3, len(shap_dep_files))
        cols = st.columns(n_cols)

        for i, dep_file in enumerate(shap_dep_files[:6]):  # Limit to 6 plots
            feature_name = dep_file.stem.replace("shap_dependence_", "")
            with cols[i % n_cols]:
                st.markdown(f"**{feature_name}**")
                st.image(str(dep_file), width="stretch")
    else:
        st.info("SHAP dependence plots not available.")

    st.divider()

    # Partial Dependence Plot
    st.markdown("### Partial Dependence Plot")
    st.markdown("Shows the average effect of features on predictions (marginal effect).")

    pdp_path = series_dir / "partial_dependence.png"
    if pdp_path.exists():
        st.image(str(pdp_path), width="stretch")
    else:
        st.info("Partial dependence plot not available.")

    # Waterfall plot for sample prediction
    waterfall_path = series_dir / "shap_waterfall_sample.png"
    if waterfall_path.exists():
        st.markdown("### Sample Prediction Explanation")
        st.markdown("SHAP waterfall showing how features contributed to a single prediction.")
        st.image(str(waterfall_path), width="stretch")


def generate_demo_forecasts(regions: list, fuel_type: str) -> pd.DataFrame:
    """Generate demo forecast data for display."""
    data = []
    base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    fuel_types = [fuel_type] if fuel_type != "Both" else ["WND", "SUN"]

    for region in regions[:3]:
        for ft in fuel_types:
            unique_id = f"{region}_{ft}"
            base_value = 500 if ft == "WND" else 300

            for h in range(24):
                ds = base_time + timedelta(hours=h)

                # Add daily pattern
                if ft == "SUN":
                    hour_factor = max(0, np.sin((ds.hour - 6) * np.pi / 12)) if 6 < ds.hour < 18 else 0
                    yhat = base_value * hour_factor + np.random.normal(0, 20)
                else:
                    yhat = base_value + np.sin(ds.hour * np.pi / 12) * 100 + np.random.normal(0, 30)

                yhat = max(0, yhat)

                data.append({
                    "unique_id": unique_id,
                    "region": region,
                    "fuel_type": ft,
                    "ds": ds,
                    "yhat": yhat,
                    "yhat_lo_80": yhat * 0.85,
                    "yhat_hi_80": yhat * 1.15,
                    "yhat_lo_95": yhat * 0.75,
                    "yhat_hi_95": yhat * 1.25,
                })

    return pd.DataFrame(data)


def run_pipeline_from_dashboard(db_path: str, regions: list, fuel_type: str):
    """Run the forecasting pipeline from the dashboard."""
    with st.spinner("Refreshing forecasts... (may take 2-3 minutes)"):
        try:
            from src.renewable.jobs import run_hourly

            # Run the hourly pipeline job
            run_hourly.main()

            st.success("Pipeline completed! Forecasts have been updated with latest EIA data.")
            st.info("Reloading page to show new forecasts...")

            # Wait a moment then reload
            import time
            time.sleep(2)
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
