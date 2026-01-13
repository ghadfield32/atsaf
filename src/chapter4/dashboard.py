"""
Chapter 4: Streamlit Monitoring Dashboard

Run with:
    streamlit run src/chapter4/dashboard.py

Displays:
- Pipeline health status
- Data freshness metrics
- Rolling performance charts
- Drift detection status
- Active alerts
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Import monitoring modules
from src.chapter4.health import full_health_check, HealthReport
from src.chapter4.drift import rolling_performance, detect_data_drift, DriftReport
from src.chapter4.alerts import check_alerts, AlertConfig, format_alert_summary, AlertSeverity

# Page config
st.set_page_config(
    page_title="EIA Forecasting Monitor",
    page_icon="📊",
    layout="wide",
)


def load_health() -> HealthReport:
    """Load current health status"""
    return full_health_check()


def load_performance() -> pd.DataFrame:
    """Load rolling performance metrics"""
    return rolling_performance()


def load_drift() -> DriftReport:
    """Load drift detection results"""
    # Load reference and current data if available
    ref_path = Path("data/clean.parquet")
    if not ref_path.exists():
        return None

    df = pd.read_parquet(ref_path)
    if len(df) < 100:
        return None

    # Use older data as reference, recent as current
    n = len(df)
    ref_df = df.iloc[:int(n * 0.8)]
    cur_df = df.iloc[int(n * 0.8):]

    return detect_data_drift(ref_df, cur_df, value_column="value")


def render_health_metrics(health: HealthReport):
    """Render health status metrics"""
    st.subheader("Pipeline Health")

    # Health indicator
    if health.is_healthy:
        st.success("✅ Pipeline Healthy")
    else:
        st.error("❌ Pipeline Unhealthy")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        freshness_color = "normal" if health.freshness_hours < 6 else "inverse"
        st.metric(
            "Data Freshness",
            f"{health.freshness_hours:.1f}h",
            delta=None,
        )

    with col2:
        st.metric(
            "Missing Hours",
            health.missing_hour_count,
        )

    with col3:
        st.metric(
            "Forecast Age",
            f"{health.forecast_age_hours:.1f}h",
        )

    with col4:
        st.metric(
            "Data Rows",
            f"{health.data_rows:,}",
        )

    # Timestamps
    st.caption(f"Last ingest: {health.last_ingest_time or 'Never'}")
    st.caption(f"Last forecast: {health.last_forecast_time or 'Never'}")


def render_performance_chart(perf_df: pd.DataFrame):
    """Render rolling performance chart"""
    st.subheader("Rolling Performance")

    if perf_df.empty:
        st.info("No performance data available yet.")
        return

    # Performance metrics chart
    chart_data = perf_df.set_index("timestamp")[["rmse", "mape"]]
    st.line_chart(chart_data)

    # Latest metrics
    latest = perf_df.iloc[-1] if len(perf_df) > 0 else None
    if latest is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest RMSE", f"{latest['rmse']:.2f}")
        with col2:
            st.metric("Latest MAPE", f"{latest['mape']:.1%}")
        with col3:
            st.metric("Coverage", f"{latest['coverage']:.1%}")


def render_drift_status(drift: DriftReport):
    """Render drift detection status"""
    st.subheader("Drift Detection")

    if drift is None:
        st.info("Insufficient data for drift detection.")
        return

    # Drift indicator
    if drift.drift_detected:
        st.warning(f"⚠️ Drift Detected (score: {drift.drift_score:.2f})")
    else:
        st.success(f"✅ No Drift (score: {drift.drift_score:.2f})")

    # Details
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reference Size", f"{drift.reference_size:,}")
    with col2:
        st.metric("Current Size", f"{drift.current_size:,}")

    if drift.drifted_columns:
        st.write("Drifted columns:", drift.drifted_columns)


def render_alerts(health: HealthReport, drift: DriftReport):
    """Render active alerts"""
    st.subheader("Active Alerts")

    alerts = check_alerts(health, drift)

    if not alerts:
        st.success("No active alerts")
        return

    # Group by severity
    critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
    warning = [a for a in alerts if a.severity == AlertSeverity.WARNING]

    if critical:
        for alert in critical:
            st.error(f"🔴 **CRITICAL**: {alert.message}")

    if warning:
        for alert in warning:
            st.warning(f"🟡 **WARNING**: {alert.message}")


def render_data_preview():
    """Show preview of latest data"""
    st.subheader("Data Preview")

    clean_path = Path("data/clean.parquet")
    if not clean_path.exists():
        st.info("No data available.")
        return

    df = pd.read_parquet(clean_path)

    # Show latest rows
    st.write(f"Showing last 10 of {len(df):,} rows")
    st.dataframe(df.tail(10))


def main():
    """Main dashboard entry point"""
    st.title("📊 EIA Forecasting Monitor")
    st.markdown("Real-time monitoring of the EIA natural gas forecasting pipeline.")

    # Sidebar controls
    st.sidebar.header("Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)

    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 60 seconds")
        # Note: In production, use st.experimental_rerun() with time.sleep()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Links**")
    st.sidebar.markdown("- [MLflow UI](http://localhost:5000)")
    st.sidebar.markdown("- [Airflow UI](http://localhost:8080)")

    # Load data
    with st.spinner("Loading pipeline status..."):
        health = load_health()
        perf_df = load_performance()
        drift = load_drift()

    # Main content
    render_health_metrics(health)

    st.markdown("---")

    # Two-column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        render_performance_chart(perf_df)

    with col2:
        render_drift_status(drift)

    st.markdown("---")

    render_alerts(health, drift)

    st.markdown("---")

    # Expandable sections
    with st.expander("📋 Data Preview"):
        render_data_preview()

    with st.expander("⚙️ Alert Configuration"):
        config = AlertConfig()
        st.write("Current thresholds:")
        st.json({
            "freshness_warning_hours": config.freshness_warning_hours,
            "freshness_critical_hours": config.freshness_critical_hours,
            "drift_warning_score": config.drift_warning_score,
            "drift_critical_score": config.drift_critical_score,
        })

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == "__main__":
    main()
