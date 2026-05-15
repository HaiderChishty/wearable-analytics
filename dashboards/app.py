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
    page_title="Welcome back, Haider",
    layout="wide"
)

st.markdown(
    """
    <style>

    /* Make chart containers invisible / black */
    div[data-testid="stPlotlyChart"],
    div[data-testid="stImage"],
    div[data-testid="stPyplot"] {
        background-color: black;
        border: none;
        box-shadow: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h2 style='
        margin-top: -60px;
        margin-bottom: 10px;
        font-size: 28px;
        font-weight: 600;
    '>
        Welcome back, Haider
    </h2>
    """,
    unsafe_allow_html=True
)



# ======================================================
# Helper Functions 
# ======================================================

def draw_gauge(ax, value, label, color):
    """
    value: 0–100
    """

    # Smaller donut
    ax.pie(
        [value, 100 - value],
        startangle=90,
        radius=0.25,  # smaller circle
        colors=[color, "#2A2A2A"],
        wedgeprops={
            "width": 0.18,
            "edgecolor": "none"
        },
    )

    # Smaller center number
    ax.text(
        0,
        0,
        f"{value:.0f}",
        ha="center",
        va="center",
        fontsize=12,      # reduced
        fontweight="bold",
        color="white"
    )

    # Smaller label
    ax.set_title(
        label,
        pad=2,
        fontsize=11,
        color="white"
    )

    # Move chart upward
    ax.set_position([0.15, 0.25, 0.7, 0.7])

    # Remove background
    ax.set_facecolor("black")

    # Remove whitespace
    ax.axis("equal")

# ======================================================
# Dashboard Settings
# ======================================================

num_days = 5

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
# Day Navigation
# ======================================================

if "selected_day_idx" not in st.session_state:
    st.session_state.selected_day_idx = len(scores_df) - 1

nav_col1, nav_col2, nav_col3, spacer = st.columns([0.5, 0.9, 0.5, 8])

with nav_col1:

    if st.button("<"):

        st.session_state.selected_day_idx = max(
            0,
            st.session_state.selected_day_idx - 1
        )

selected_scores = scores_df.iloc[
    st.session_state.selected_day_idx
]

with nav_col2:

    st.markdown(
        f"""
        <div>
            <h3 style='
                margin: 0;
                margin-top: -5px;
                font-size: 25px;
            '>
                {selected_scores['date']}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col3:

    if st.button(">"):

        st.session_state.selected_day_idx = min(
            len(scores_df) - 1,
            st.session_state.selected_day_idx + 1
        )

selected_scores = scores_df.iloc[
    st.session_state.selected_day_idx
]

col1, col2, col3 = st.columns(3)

st.markdown(
    """
    <h3 style='
        font-size: 28px;
        margin-bottom: 0px;
        margin-top: 0px;
    '>
        Daily Recovery Overview
    </h3>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

fig1, ax1 = plt.subplots(figsize=(2, 2), facecolor="black")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig2, ax2 = plt.subplots(figsize=(2, 2), facecolor="black")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig3, ax3 = plt.subplots(figsize=(2, 2), facecolor="black")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

draw_gauge(ax1, selected_scores["recovery_score"], "Recovery", "#4CAF50")
draw_gauge(ax2, selected_scores["strain_score"], "Strain", "#FF5722")
draw_gauge(ax3, selected_scores["sleep_score"], "Sleep", "#3F51B5")

with col1:
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.pyplot(fig2, use_container_width=True)

with col3:
    st.pyplot(fig3, use_container_width=True)

# ======================================================
# Recommendations
# ======================================================

st.header("🧠 Recommendations")

recovery = selected_scores["recovery_score"]
strain = selected_scores["strain_score"]
sleep = selected_scores["sleep_score"]

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