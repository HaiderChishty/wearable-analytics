# scoring.py

import os
import numpy as np
import pandas as pd

from src.data_generation import generate_wearable_data
from src.preprocessing import preprocess_data
from src.features import generate_features


# =========================================================
# Helper: Normalize values to wearable-style scores
# =========================================================


def normalize_score(value, min_val, max_val, reverse=False):

    score = 100 * (value - min_val) / (max_val - min_val)

    score = np.clip(score, 0, 100)

    if reverse:
        score = 100 - score

    return score


# =========================================================
# Sleep Score
# =========================================================
#
# Components:
#
# - Sleep duration
# - Sleep efficiency
# - REM sleep percentage
#
# Interpretation:
#
# - Longer sleep improves recovery
# - Higher efficiency suggests consolidated sleep
# - REM contributes to cognitive recovery
#
# Output:
# 0–100
# =========================================================

def calculate_sleep_score(features_df):

    df = features_df.copy()

    # -----------------------------------------------------
    # Component scores
    # -----------------------------------------------------

    duration_score = normalize_score(
        df["sleep_duration_hours"],
        min_val=4,
        max_val=9
    )

    efficiency_score = normalize_score(
        df["sleep_efficiency"],
        min_val=70,
        max_val=100
    )

    rem_score = normalize_score(
        df["rem_sleep_pct"],
        min_val=10,
        max_val=30
    )

    # -----------------------------------------------------
    # Weighted combination
    # -----------------------------------------------------

    sleep_score = (
        0.5 * duration_score
        + 0.3 * efficiency_score
        + 0.2 * rem_score
    )

    df["sleep_score"] = sleep_score.round(1)

    return df


# =========================================================
# Recovery Score
# =========================================================
#
# Components:
#
# - High HRV → positive
# - Low resting HR → positive
# - Good sleep → positive
#
# Formula:
#
# recovery =
#     0.4 * hrv_score
#   + 0.3 * sleep_score
#   + 0.3 * resting_hr_score
#
# Output:
# 0–100
# =========================================================

def calculate_recovery_score(features_df):

    df = features_df.copy()

    # -----------------------------------------------------
    # HRV score
    # -----------------------------------------------------

    hrv_score = normalize_score(
        df["nightly_hrv"],
        min_val=20,
        max_val=80
    )

    # -----------------------------------------------------
    # Resting HR score
    #
    # Lower resting HR is generally better.
    # reverse=True flips scoring.
    # -----------------------------------------------------

    resting_hr_score = normalize_score(
        df["resting_hr"],
        min_val=45,
        max_val=90,
        reverse=True
    )

    # -----------------------------------------------------
    # Sleep score
    # -----------------------------------------------------

    if "sleep_score" not in df.columns:
        df = calculate_sleep_score(df)

    # -----------------------------------------------------
    # Final recovery score
    # -----------------------------------------------------

    recovery_score = (
        0.4 * hrv_score
        + 0.3 * df["sleep_score"]
        + 0.3 * resting_hr_score
    )

    df["recovery_score"] = recovery_score.round(1)

    return df


# =========================================================
# Strain Score
# =========================================================
#
# Based on time spent in elevated HR zones.
#
# Higher HR zones contribute more strain.
#
# Simple weighted load model:
#
# Zone 1 -> low strain
# Zone 5 -> very high strain
#
# Output:
# 0–100
# =========================================================

def calculate_strain_score(features_df):

    df = features_df.copy()

    # -----------------------------------------------------
    # Weighted HR zone load
    # -----------------------------------------------------

    weighted_load = (
        1.0 * df["zone1_minutes"]
        + 2.0 * df["zone2_minutes"]
        + 3.5 * df["zone3_minutes"]
        + 5.0 * df["zone4_minutes"]
        + 7.0 * df["zone5_minutes"]
    )

    # -----------------------------------------------------
    # Normalize to wearable-style 0–100 scale
    # -----------------------------------------------------

    strain_score = normalize_score(
        weighted_load,
        min_val=200,
        max_val=2500
    )

    df["strain_score"] = strain_score.round(1)

    return df


# =========================================================
# Full Scoring Pipeline
# =========================================================

def generate_scores(features_df):

    df = features_df.copy()

    df = calculate_sleep_score(df)

    df = calculate_recovery_score(df)

    df = calculate_strain_score(df)

    return df


# =========================================================
# Save Scores
# =========================================================

def save_scores(df, filename="daily_scores.csv"):

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

    print(f"Saved scores to: {output_path}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Generate synthetic data
    # -----------------------------------------------------

    raw_df = generate_wearable_data(days=5)

    # -----------------------------------------------------
    # Preprocess signals
    # -----------------------------------------------------

    processed_df = preprocess_data(raw_df)

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    features_df = generate_features(processed_df)

    # -----------------------------------------------------
    # Generate scores
    # -----------------------------------------------------

    scores_df = generate_scores(features_df)

    print(scores_df[[
        "date",
        "sleep_score",
        "recovery_score",
        "strain_score"
    ]])

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_scores(scores_df)