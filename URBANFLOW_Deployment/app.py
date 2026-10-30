
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="URBANFLOW",
    page_icon="🚕",
    layout="wide"
)

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "urbanflow_dashboard_test_v2.parquet"
)

@st.cache_data
def load_data():
    data = pd.read_parquet(DATA_PATH)
    data["time_bin"] = pd.to_datetime(data["time_bin"])
    data["date"] = data["time_bin"].dt.date
    return data

st.title("URBANFLOW")
st.subheader("NYC Taxi Demand Forecasting")

st.caption(
    "Historical 15-minute demand forecasting | "
    "Frozen LightGBM V2 | November-December 2025"
)

if not DATA_PATH.exists():
    st.error("Dashboard dataset not found.")
    st.stop()

df = load_data()

overall_mae = np.mean(
    np.abs(df["actual"] - df["predicted"])
)

baseline_mae = np.mean(
    np.abs(df["actual"] - df["baseline"])
)

improvement = (
    (baseline_mae - overall_mae)
    / baseline_mae * 100
)

col1, col2, col3 = st.columns(3)

col1.metric("LightGBM MAE", f"{overall_mae:.4f}")
col2.metric("Baseline MAE", f"{baseline_mae:.4f}")
col3.metric("MAE Improvement", f"{improvement:.2f}%")

st.divider()

zones = sorted(df["zone_id"].unique())

zone = st.sidebar.selectbox(
    "Select NYC Taxi Zone",
    zones,
    index=zones.index(237) if 237 in zones else 0
)

min_date = df["date"].min()
max_date = df["date"].max()

selected_date = st.sidebar.date_input(
    "Select Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

zone_df = df[df["zone_id"] == zone]

daily_df = zone_df[
    zone_df["date"] == selected_date
].sort_values("time_bin")

if daily_df.empty:
    st.warning("No data found for this selection.")
    st.stop()

st.header(f"Taxi Zone {zone}")

zone_mae = np.mean(
    np.abs(zone_df["actual"] - zone_df["predicted"])
)

zone_baseline_mae = np.mean(
    np.abs(zone_df["actual"] - zone_df["baseline"])
)

a, b, c = st.columns(3)

a.metric("Zone Model MAE", f"{zone_mae:.3f}")
b.metric("Zone Baseline MAE", f"{zone_baseline_mae:.3f}")
c.metric(
    "Selected Day Pickups",
    f"{daily_df['actual'].sum():,.0f}"
)

st.subheader("Actual vs Predicted Demand")

chart_data = daily_df.melt(
    id_vars=["time_bin"],
    value_vars=["actual", "predicted", "baseline"],
    var_name="Series",
    value_name="Pickup Count"
)

fig = px.line(
    chart_data,
    x="time_bin",
    y="Pickup Count",
    color="Series",
    title="15-Minute Taxi Pickup Demand"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Errors")

daily_df = daily_df.copy()

daily_df["absolute_error"] = np.abs(
    daily_df["actual"] - daily_df["predicted"]
)

fig_error = px.bar(
    daily_df,
    x="time_bin",
    y="absolute_error",
    title="LightGBM Absolute Error by Interval"
)

st.plotly_chart(fig_error, use_container_width=True)

st.subheader("Forecasting Data")

st.dataframe(
    daily_df[
        ["time_bin", "actual", "predicted",
         "baseline", "absolute_error"]
    ],
    use_container_width=True
)

st.download_button(
    "Download Selected Data",
    data=daily_df.to_csv(index=False),
    file_name=f"urbanflow_zone_{zone}.csv",
    mime="text/csv"
)

st.caption(
    "Historical test-set predictions, not live taxi-demand forecasts."
)
