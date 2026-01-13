# file: src/chapter4/dashboard_app.py
import pandas as pd
import streamlit as st
from src.chapter4.db import connect

st.set_page_config(page_title="EIA Forecast Monitoring", layout="wide")
st.title("EIA Forecast Monitoring")

db_path = st.sidebar.text_input("DB Path", "artifacts/monitoring/monitoring.sqlite")

con = connect(db_path)

runs = pd.read_sql_query("SELECT * FROM pipeline_runs ORDER BY ts_utc DESC LIMIT 100", con)
alerts = pd.read_sql_query("SELECT * FROM alerts ORDER BY alert_ts_utc DESC LIMIT 200", con)
scores = pd.read_sql_query("SELECT * FROM forecast_scores ORDER BY scored_ts_utc DESC LIMIT 2000", con)

con.close()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Recent Pipeline Runs")
    st.dataframe(runs, use_container_width=True)

with c2:
    st.subheader("Alerts")
    st.dataframe(alerts, use_container_width=True)

st.subheader("Accuracy (Scored Forecasts)")
if not scores.empty:
    scores["scored_ts_utc"] = pd.to_datetime(scores["scored_ts_utc"], utc=True)
    st.line_chart(
        scores.sort_values("scored_ts_utc").set_index("scored_ts_utc")[["mape", "rmse"]],
        use_container_width=True
    )
    st.dataframe(scores.head(200), use_container_width=True)
else:
    st.info("No scored forecasts yet. Run the scoring job after actuals arrive.")
