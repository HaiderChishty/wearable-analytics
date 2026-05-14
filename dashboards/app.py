# dashboards/app.py

# ======================================================
# Wearable Analytics Dashboard
# ======================================================
#
# Run with:
#
# streamlit run dashboards/app.py
#
# ======================================================

import os
import sys

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# Allow imports from src/
# ======================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ======================================================
# Pipeline Imports
# ======================================================

from src.data_generation import generate_wearable_data
from src.preprocessing import preprocess_data
from src.features import generate_features
from src.scoring import generate_scores

# ======================================================
# Streamlit Page Config
# ======================================================

st.set_page_config(
    page_title="Wearable Analytics Dashboard",
    layout="wide"
)

st.title("📊 Wearable Analytics Dashboard")

st.markdown(
    """
    Simulated wearable analytics pipeline inspired by WHOOP and Oura.

    This dashboard demonstrates:
    - physiological signal generation
    - preprocessing
    - feature engineering
    - wearable-style scoring
    """
)

# ======================================================
# Sidebar Controls
# ======================================================

st.sidebar.header("Dashboard Settings")

num_days = st.sidebar.slider(
    "Number of Days",
    min_value=3,
    max_value=14,
    value=7
)

# ======================================================
# Generate Pipeline Data
# ======================================================

@st.cache_data
def run_pipeline(days):

    raw_df = generate_wearable_data(days=days)

    processed_df = preprocess_data(raw_df)

    features_df = generate_features(processed_df)

    scores_df = generate_scores(features_df)

    return processed_df, scores_df


processed_df, scores_df = run_pipeline(num_days)

# ======================================================
# Daily Scores Overview
# ======================================================

st.header("🏆 Daily Scores")

latest_scores = scores_df.iloc[-1]

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------
# Recovery
# ------------------------------------------------------

with col1:

    st.metric(
        label="Recovery Score",
        value=f"{latest_scores['recovery_score']:.1f}"
    )

# ------------------------------------------------------
# Strain
# ------------------------------------------------------

with col2:

    st.metric(
        label="Strain Score",
        value=f"{latest_scores['strain_score']:.1f}"
    )

# ------------------------------------------------------
# Sleep
# ------------------------------------------------------

with col3:

    st.metric(
        label="Sleep Score",
        value=f"{latest_scores['sleep_score']:.1f}"
    )

# ======================================================
# Recommendations
# ======================================================

st.header("🧠 Recommendations")

recovery = latest_scores["recovery_score"]
strain = latest_scores["strain_score"]
sleep = latest_scores["sleep_score"]

recommendations = []

# ------------------------------------------------------
# Recovery Logic
# ------------------------------------------------------

if recovery < 40:
    recommendations.append(
        "🔴 Low recovery — consider rest or light activity."
    )

elif recovery < 70:
    recommendations.append(
        "🟡 Moderate recovery — train carefully."
    )

else:
    recommendations.append(
        "🟢 High recovery — ready for intense activity."
    )

# ------------------------------------------------------
# Sleep Logic
# ------------------------------------------------------

if sleep < 60:
    recommendations.append(
        "😴 Sleep quality was suboptimal — prioritize sleep tonight."
    )

# ------------------------------------------------------
# Strain Logic
# ------------------------------------------------------

if strain > 75:
    recommendations.append(
        "🔥 High cardiovascular strain detected."
    )

for rec in recommendations:
    st.write(rec)

# ======================================================
# Daily Score Trends
# ======================================================

st.header("📈 Daily Score Trends")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    scores_df["date"],
    scores_df["recovery_score"],
    marker="o",
    linewidth=2,
    label="Recovery"
)

ax.plot(
    scores_df["date"],
    scores_df["strain_score"],
    marker="o",
    linewidth=2,
    label="Strain"
)

ax.plot(
    scores_df["date"],
    scores_df["sleep_score"],
    marker="o",
    linewidth=2,
    label="Sleep"
)

ax.set_ylabel("Score")
ax.set_xlabel("Date")
ax.set_title("Daily Wearable Scores")

ax.grid(alpha=0.3)

ax.legend()

st.pyplot(fig)

# ======================================================
# Heart Rate Time Series
# ======================================================

st.header("❤️ Heart Rate")

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(
    processed_df["timestamp"],
    processed_df["hr"],
    alpha=0.3,
    label="Raw HR"
)

ax.plot(
    processed_df["timestamp"],
    processed_df["hr_smooth"],
    linewidth=2,
    label="Smoothed HR"
)

ax.set_title("Heart Rate Time Series")
ax.set_ylabel("BPM")
ax.set_xlabel("Timestamp")

ax.grid(alpha=0.3)

ax.legend()

st.pyplot(fig)

# ======================================================
# HRV Time Series
# ======================================================

st.header("🫀 Heart Rate Variability")

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(
    processed_df["timestamp"],
    processed_df["hrv"],
    alpha=0.3,
    label="Raw HRV"
)

ax.plot(
    processed_df["timestamp"],
    processed_df["hrv_smooth"],
    linewidth=2,
    label="Smoothed HRV"
)

ax.set_title("HRV Time Series")
ax.set_ylabel("HRV (ms)")
ax.set_xlabel("Timestamp")

ax.grid(alpha=0.3)

ax.legend()

st.pyplot(fig)

# ======================================================
# Daily Physiological Features
# ======================================================

st.header("📋 Daily Feature Summary")

summary_cols = [
    "date",
    "resting_hr",
    "nightly_hrv",
    "sleep_duration_hours",
    "sleep_efficiency",
    "daily_activity_load",
    "recovery_score",
    "strain_score",
    "sleep_score"
]

st.dataframe(
    scores_df[summary_cols].round(2),
    use_container_width=True
)

# ======================================================
# Footer
# ======================================================

st.markdown("---")

st.caption(
    """
    Wearable Analytics Portfolio Project
    Inspired by WHOOP / Oura-style physiological analytics.
    """
)