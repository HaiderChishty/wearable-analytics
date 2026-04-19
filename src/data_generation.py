import numpy as np
import pandas as pd
import os


def generate_wearable_data(days=3, freq_min=1, seed=42):
    np.random.seed(seed)

    minutes = days * 24 * 60
    t = np.arange(minutes)

    # -------------------------
    # Circadian rhythm (HR baseline)
    # -------------------------

    phase_shift = 6 * 60  # minutes

    circadian = 60 + 15 * np.sin(2 * np.pi * (t - phase_shift) / (24 * 60))
    circadian += 3 * np.sin(2 * np.pi * (t - phase_shift) / (12 * 60))
    circadian += np.random.randn(minutes) * 0.5

    # -------------------------
    # Activity signal (bursty)
    # -------------------------
    activity = np.random.rand(minutes)

    exercise_events = []

    for d in range(days):
        start = d * 24 * 60 + np.random.randint(8 * 60, 20 * 60)
        duration = np.random.randint(20, 60)

        activity[start:start + duration] += 2.5
        exercise_events.append((start, start + duration))

    activity = np.clip(activity, 0, None)

    # =========================================================
    # HEART RATE (UPDATED: physiologically realistic dynamics)
    # =========================================================

    hr = np.zeros(minutes)

    # 1. circadian baseline
    hr += circadian

    # 2. helper: physiological lag (activity response smoothing)
    def smooth_signal(x, alpha=0.97):
        y = np.zeros_like(x)
        for i in range(1, len(x)):
            y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
        return y

    activity_lag = smooth_signal(activity, alpha=0.97)

    # slow metabolic response (ramp-up behavior)
    hr += 24 * activity_lag

    # fast sympathetic spike (onset response)
    hr += 10 * activity

    # 3. recovery dynamics (post-exercise elevation)
    recovery_effect = np.zeros(minutes)

    for start, end in exercise_events:
        for i in range(end, min(end + 240, minutes)):
            decay = np.exp(-(i - end) / 60)
            recovery_effect[i] += 15 * decay

    hr += recovery_effect

    # 4. sleep modulation (physiological lowering)
    sleep_state = np.zeros(minutes)

    for d in range(days):
        sleep_start = d * 24 * 60 + 23 * 60
        sleep_end = d * 24 * 60 + 7 * 60

        if sleep_end < sleep_start:
            sleep_state[sleep_start:] = 1
            sleep_state[:sleep_end] = 1

            deep_start = d * 24 * 60 + 1 * 60
            sleep_state[deep_start:deep_start + 180] = 2

            rem_start = d * 24 * 60 + 4 * 60
            sleep_state[rem_start:rem_start + 90] = 3

    hr -= 6 * (sleep_state > 0)
    hr -= 2 * (sleep_state == 2)
    hr -= 1 * (sleep_state == 3)

    # 5. noise (keep small)
    hr += np.random.randn(minutes) * 2.0

    # 6. physiological bounds (not overly aggressive clipping)
    hr = np.clip(hr, 20, 190)

    # -------------------------
    # HRV (inverse coupling, slightly smoother)
    # -------------------------
    hrv = 85 - 0.55 * hr + np.random.randn(minutes) * 4
    hrv = np.clip(hrv, 15, 120)

    # -------------------------
    # Assemble dataframe
    # -------------------------
    df = pd.DataFrame({
        "minute": t,
        "hr": hr,
        "hrv": hrv,
        "activity": activity,
        "sleep_state": sleep_state
    })

    return df


def save_data(df, filename="wearable_data.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(
        base_dir,
        "..",
        "data",
        "raw",
        filename
    )

    output_path = os.path.normpath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved data to: {output_path}")


if __name__ == "__main__":
    df = generate_wearable_data()

    print(df.head())

    save_data(df)