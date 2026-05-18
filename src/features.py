# features.py

import os
import numpy as np
import pandas as pd

from src.data_generation import generate_wearable_data
from src.preprocessing import preprocess_data


# =========================================================
# Helper: Add date column
# =========================================================

def add_date_column(df):

    df = df.copy()

    df["date"] = df["timestamp"].dt.date

    return df


# =========================================================
# Resting Heart Rate
# =========================================================
#
# Approximation:
# Lowest sustained HR during sleep
#
# Real wearables often estimate RHR during:
# - deep sleep
# - low-motion periods
# - overnight windows
#
# Here:
# Use 5th percentile sleeping HR
# =========================================================

def calculate_resting_hr(df):

    sleep_df = df[df["sleep_state"] > 0]

    resting_hr = (
        sleep_df
        .groupby("date")["hr_smooth"]
        .quantile(0.05)
        .rename("resting_hr")
    )

    return resting_hr


# =========================================================
# Nightly HRV
# =========================================================
#
# Average HRV during sleep
#
# Higher HRV generally reflects:
# - better recovery
# - lower stress
# - stronger parasympathetic activity
# =========================================================

def calculate_nightly_hrv(df):

    sleep_df = df[df["sleep_state"] > 0]

    nightly_hrv = (
        sleep_df
        .groupby("date")["hrv_smooth"]
        .mean()
        .rename("nightly_hrv")
    )

    return nightly_hrv


# =========================================================
# Sleep Duration
# =========================================================
#
# Total minutes asleep
# Converted to hours
# =========================================================

def calculate_sleep_duration(df):

    sleep_minutes = (
        df[df["sleep_state"] > 0]
        .groupby("date")
        .size()
    )

    sleep_duration = (
        sleep_minutes / 60.0
    ).rename("sleep_duration_hours")

    return sleep_duration


# =========================================================
# Sleep Efficiency
# =========================================================
#
# Simple approximation:
#
# sleep time / time in bed
#
# Since synthetic data currently assumes fixed
# sleep opportunity window (11 PM - 7 AM),
# efficiency estimates fragmentation.
# =========================================================

def calculate_sleep_efficiency(df):

    sleep_window_minutes = 8 * 60

    sleep_minutes = (
        df[df["sleep_state"] > 0]
        .groupby("date")
        .size()
    )

    sleep_efficiency = (
        100 * sleep_minutes / sleep_window_minutes
    ).rename("sleep_efficiency")

    return sleep_efficiency


# =========================================================
# Sleep Stage Percentages
# =========================================================
#
# 1 = Light
# 2 = Deep
# 3 = REM
# =========================================================

def calculate_sleep_stage_percentages(df):

    sleep_df = df[df["sleep_state"] > 0]

    total_sleep = sleep_df.groupby("date").size()

    rem_pct = (
        sleep_df[sleep_df["sleep_state"] == 3]
        .groupby("date")
        .size()
        / total_sleep
        * 100
    ).rename("rem_sleep_pct")

    deep_pct = (
        sleep_df[sleep_df["sleep_state"] == 2]
        .groupby("date")
        .size()
        / total_sleep
        * 100
    ).rename("deep_sleep_pct")

    light_pct = (
        sleep_df[sleep_df["sleep_state"] == 1]
        .groupby("date")
        .size()
        / total_sleep
        * 100
    ).rename("light_sleep_pct")

    return rem_pct, deep_pct, light_pct


# =========================================================
# Activity Features
# =========================================================

def calculate_activity_features(df):

    daily_activity = (
        df.groupby("date")["activity_smooth"]
        .sum()
        .rename("daily_activity_load")
    )

    active_minutes = (
        df[df["activity_smooth"] > 0.5]
        .groupby("date")
        .size()
        .rename("active_minutes")
    )

    return daily_activity, active_minutes


# =========================================================
# Heart Rate Zone Features
# =========================================================
#
# Simple wearable-style HR zones
#
# Zone 1: < 90 bpm
# Zone 2: 90–110
# Zone 3: 110–130
# Zone 4: 130–150
# Zone 5: > 150
#
# Later:
# Can personalize using HRmax/reserve
# =========================================================

def calculate_hr_zone_features(df):

    hr = df["hr_smooth"]

    zones = {
        "zone1_minutes": (hr < 90),
        "zone2_minutes": ((hr >= 90) & (hr < 110)),
        "zone3_minutes": ((hr >= 110) & (hr < 130)),
        "zone4_minutes": ((hr >= 130) & (hr < 150)),
        "zone5_minutes": (hr >= 150),
    }

    features = {}

    for feature_name, condition in zones.items():

        features[feature_name] = (
            df[condition]
            .groupby("date")
            .size()
        )

    zone_df = pd.DataFrame(features).fillna(0)

    return zone_df

# =========================================================
# Daily Heart Rate Summary
# =========================================================

def calculate_daily_hr_summary(df):

    summary = df.groupby("date")["hr_smooth"].agg([
        ("avg_hr", "mean"),
        ("max_hr", "max"),
        ("min_hr", "min")
    ])

    return summary


# =========================================================
# Full Feature Pipeline
# =========================================================

def generate_features(df):

    df = add_date_column(df)

    # Recovery-related
    resting_hr = calculate_resting_hr(df)
    nightly_hrv = calculate_nightly_hrv(df)

    # Sleep-related
    sleep_duration = calculate_sleep_duration(df)
    sleep_efficiency = calculate_sleep_efficiency(df)

    rem_pct, deep_pct, light_pct = (
        calculate_sleep_stage_percentages(df)
    )

    # Activity-related
    daily_activity, active_minutes = (
        calculate_activity_features(df)
    )

    # HR zones
    hr_zones = calculate_hr_zone_features(df)

    # HR summary
    hr_summary = calculate_daily_hr_summary(df)

    # Combine all features
    features = pd.concat([
        resting_hr,
        nightly_hrv,
        sleep_duration,
        sleep_efficiency,
        rem_pct,
        deep_pct,
        light_pct,
        daily_activity,
        active_minutes,
        hr_zones,
        hr_summary
    ], axis=1)

    features = features.reset_index()

    return features


# =========================================================
# Save Features
# =========================================================

def save_features(df, filename="daily_features.csv"):

    base_dir = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(
        base_dir,
        "..",
        "data",
        "processed",
        filename
    )

    output_path = os.path.normpath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # Generate synthetic data
    raw_df = generate_wearable_data(days=3)

    # Preprocess
    processed_df = preprocess_data(raw_df)

    # Feature engineering
    features_df = generate_features(processed_df)

    print(features_df.head())

    # Save
    save_features(features_df)