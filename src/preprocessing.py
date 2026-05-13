#preprocessing.py

import numpy as np
import pandas as pd
import os

from src.data_generation import generate_wearable_data


# =========================================================
# Add timestamps
# =========================================================

def add_timestamps(df, start_date="2024-01-01"):
    
    df = df.copy()

    df["timestamp"] = pd.date_range(
        start=start_date,
        periods=len(df),
        freq="min"
    )

    return df


# =========================================================
# Simulate missing data
# =========================================================

def inject_missing_values(df, missing_fraction=0.01, seed=42):

    np.random.seed(seed)

    df = df.copy()

    n_missing = int(len(df) * missing_fraction)

    missing_idx = np.random.choice(
        len(df),
        n_missing,
        replace=False
    )

    signal_cols = ["hr", "hrv", "activity"]

    for col in signal_cols:
        df.loc[missing_idx, col] = np.nan

    return df


# =========================================================
# Fill missing data
# =========================================================

def interpolate_signals(df):

    df = df.copy()

    signal_cols = ["hr", "hrv", "activity"]

    for col in signal_cols:
        df[col] = df[col].interpolate(method="linear")

    return df


# =========================================================
# Smooth noisy physiological signals
# =========================================================

def smooth_signals(df, window=20):

    df = df.copy()

    df["hr_smooth"] = (
        df["hr"]
        .rolling(window=window, center=True)
        .mean()
    )

    df["hrv_smooth"] = (
        df["hrv"]
        .rolling(window=window, center=True)
        .mean()
    )

    df["activity_smooth"] = (
        df["activity"]
        .rolling(window=window, center=True)
        .mean()
    )

    return df


# =========================================================
# Full preprocessing pipeline
# =========================================================

def preprocess_data(df):

    df = add_timestamps(df)

    df = inject_missing_values(df)

    df = interpolate_signals(df)

    df = smooth_signals(df)

    return df


# =========================================================
# Save processed data
# =========================================================

def save_processed_data(df, filename="wearable_data_processed.csv"):

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

    print(f"Saved processed data to: {output_path}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    df = generate_wearable_data(days=3)

    df_processed = preprocess_data(df)

    print(df_processed.head())

    save_processed_data(df_processed)