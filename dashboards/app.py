# ======================================================
# Wearable Analytics Dashboard
# ======================================================
#
# Run with:
#   streamlit run dashboards/app.py
#
# ======================================================

import os
import sys
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ======================================================
# Allow imports from src/
# ======================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_generation import generate_wearable_data
from src.preprocessing import preprocess_data
from src.features import generate_features
from src.scoring import generate_scores

# ======================================================
# Page config
# ======================================================

st.set_page_config(
    page_title="Wearable Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# Global CSS (structural only — no card classes)
# ======================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #0f1117; }

.block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ======================================================
# Color constants
# ======================================================

DARK_BG     = "#1a1d27"
CARD_BORDER = "#2a2d3e"
GRID_COLOR  = "#2a2d3e"
TEXT_MID    = "#9ca3af"
TEXT_DIM    = "#6b7280"


# ======================================================
# Pipeline (cached for full dataset)
# ======================================================

@st.cache_data
def run_pipeline(seed=42, total_days=14):
    raw_df       = generate_wearable_data(days=total_days, seed=seed)
    processed_df = preprocess_data(raw_df)
    features_df  = generate_features(processed_df)
    scores_df    = generate_scores(features_df)
    return processed_df, scores_df

TOTAL_DAYS = 14
processed_df, scores_df = run_pipeline(total_days=TOTAL_DAYS)

all_dates = sorted(scores_df["date"].unique())


# ======================================================
# SVG ring rendered as base64 <img>
# Streamlit strips raw <svg> in markdown; <img src="data:..."> is safe.
# ======================================================

def score_ring_img(score, label, color, size=160):
    r    = 54
    cx   = cy = size // 2
    circ = 2 * np.pi * r
    filled = max(0.0, min(1.0, score / 100)) * circ
    gap    = circ - filled
    score_int = int(round(score))

    svg = (
        f'<svg width="{size}" height="{size + 24}" '
        f'viewBox="0 0 {size} {size + 24}" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="#22253a" stroke-width="9"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="{color}" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{filled:.2f} {gap:.2f}" '
        f'transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" '
        f'font-family="DM Mono, monospace" font-size="22" font-weight="500" '
        f'fill="#ffffff">{score_int}</text>'
        f'<text x="{cx}" y="{size + 18}" text-anchor="middle" '
        f'font-family="DM Sans, sans-serif" font-size="11" '
        f'letter-spacing="2" fill="#6b7280">{label.upper()}</text>'
        f'</svg>'
    )

    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" />'


# ======================================================
# Matplotlib dark style
# ======================================================

def apply_dark(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_MID, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MID)
    ax.yaxis.label.set_color(TEXT_MID)
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD_BORDER)
    ax.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


# ======================================================
# HTML helpers (all inline styles — no CSS classes)
# ======================================================

CARD_OPEN = (
    '<div style="background:#1a1d27; border:1px solid #2a2d3e; '
    'border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:0;">'
)
CARD_CLOSE = "</div>"

def card_title(text):
    return (
        '<div style="font-size:10px; font-weight:600; letter-spacing:0.1em; '
        f'text-transform:uppercase; color:#4b5563; margin-bottom:0.7rem;">'
        f'{text}</div>'
    )

def stat_row(label, value):
    return (
        '<div style="display:flex; justify-content:space-between; '
        'align-items:center; padding:6px 0; border-bottom:1px solid #1f2233;">'
        f'<span style="font-size:13px; color:#6b7280;">{label}</span>'
        f'<span style="font-family:DM Mono,monospace; font-size:13px; '
        f'color:#e8eaf0;">{value}</span>'
        '</div>'
    )

def rec_pill(text, kind="green"):
    palette = {
        "green":  ("#0d2318", "#34d399", "#065f46"),
        "yellow": ("#1c1a07", "#fbbf24", "#78350f"),
        "red":    ("#200d0d", "#f87171", "#7f1d1d"),
        "blue":   ("#0c1829", "#60a5fa", "#1e3a5f"),
    }
    bg, fg, border = palette.get(kind, palette["green"])
    return (
        f'<div style="background:{bg}; color:{fg}; border:1px solid {border}; '
        f'border-radius:999px; padding:6px 14px; font-size:13px; font-weight:500; '
        f'margin:4px 0; display:inline-block;">{text}</div><br>'
    )


# ======================================================
# HEADER
# ======================================================

st.markdown(
    '<div style="font-size:28px; font-weight:300; color:#e8eaf0; '
    'letter-spacing:-0.02em; margin-bottom:4px;">'
    'Welcome back, <span style="font-weight:600; color:#fff;">Haider</span>'
    '</div>'
    '<div style="font-size:15px; color:#CCCCCC; margin-bottom:1.5rem;">'
    'Your physiological snapshot</div>',
    unsafe_allow_html=True
)


# ======================================================
# DAY SELECTOR  (Prev / Date display / Next)
# ======================================================

if "day_idx" not in st.session_state:
    st.session_state.day_idx = len(all_dates) - 1   # most recent day

nav1, nav2, nav3, _ = st.columns([1, 2, 1, 12])

with nav1:
    if st.button("◀  Prev", use_container_width=True):
        st.session_state.day_idx = max(0, st.session_state.day_idx - 1)

with nav3:
    if st.button("Next  ▶", use_container_width=True):
        st.session_state.day_idx = min(
            len(all_dates) - 1, st.session_state.day_idx + 1
        )

with nav2:
    selected_date = all_dates[st.session_state.day_idx]
    st.markdown(
        '<div style="text-align:center; font-family:DM Mono,monospace; '
        f'font-size:15px; color:#e8eaf0; padding:6px 0;">'
        f'{str(selected_date)}</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── Filter to selected day ───────────────────────────

day_processed = processed_df[
    processed_df["timestamp"].dt.date == selected_date
].copy()

day_scores = scores_df[scores_df["date"] == selected_date]

if day_scores.empty:
    st.warning("No score data available for this day.")
    st.stop()

today = day_scores.iloc[0]


# ======================================================
# SCORE RINGS
# ======================================================

ring_cols = st.columns([1, 1, 1, 4])
ring_data = [
    (today["recovery_score"], "Recovery", "#34d399"),
    (today["strain_score"],   "Strain",   "#fb923c"),
    (today["sleep_score"],    "Sleep",    "#818cf8"),
]

for col, (score, label, color) in zip(ring_cols[:3], ring_data):
    with col:
        st.markdown(
            f'<div style="display:flex; justify-content:center;">'
            f'{score_ring_img(score, label, color)}</div>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<hr style="border-color:#1f2233; margin:0.5rem 0 1.2rem 0;">',
    unsafe_allow_html=True
)


# ======================================================
# ROW 1 — Heart Rate  |  HRV
# ======================================================

col1, col2 = st.columns(2)

# ── Heart Rate ───────────────────────────────────────

with col1:
    st.markdown(CARD_OPEN + card_title("Heart Rate"), unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 2.6))
    apply_dark(ax, fig)
    ax.plot(day_processed["timestamp"], day_processed["hr"],
            color="#374151", linewidth=0.8, alpha=0.5, label="Raw")
    ax.plot(day_processed["timestamp"], day_processed["hr_smooth"],
            color="#f87171", linewidth=2, label="Smoothed")
    ax.set_ylabel("bpm", fontsize=9, color=TEXT_MID)
    # ax.legend(fontsize=8, facecolor=DARK_BG, edgecolor=CARD_BORDER,
    #           labelcolor=TEXT_MID)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    rhr    = today["resting_hr"]
    avg_hr = day_processed["hr"].mean()
    max_hr = day_processed["hr"].max()
    st.markdown(
        stat_row("Resting HR", f"{rhr:.0f} bpm") +
        stat_row("Daily average", f"{avg_hr:.0f} bpm") +
        stat_row("Daily max", f"{max_hr:.0f} bpm"),
        unsafe_allow_html=True
    )
    st.markdown(CARD_CLOSE, unsafe_allow_html=True)


# ── HRV ─────────────────────────────────────────────

with col2:
    st.markdown(CARD_OPEN + card_title("Heart Rate Variability"), unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 2.6))
    apply_dark(ax, fig)
    ax.plot(day_processed["timestamp"], day_processed["hrv"],
            color="#374151", linewidth=0.8, alpha=0.5, label="Raw")
    ax.plot(day_processed["timestamp"], day_processed["hrv_smooth"],
            color="#818cf8", linewidth=2, label="Smoothed")
    ax.set_ylabel("ms", fontsize=9, color=TEXT_MID)
    # ax.legend(fontsize=8, facecolor=DARK_BG, edgecolor=CARD_BORDER,
    #           labelcolor=TEXT_MID)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    nhrv    = today["nightly_hrv"]
    avg_hrv = day_processed["hrv"].mean()
    max_hrv = day_processed["hrv"].max()
    st.markdown(
        stat_row("Nightly HRV", f"{nhrv:.1f} ms") +
        stat_row("Daily average", f"{avg_hrv:.1f} ms") +
        stat_row("Daily max", f"{max_hrv:.1f} ms"),
        unsafe_allow_html=True
    )
    st.markdown(CARD_CLOSE, unsafe_allow_html=True)


st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ======================================================
# ROW 2 — Sleep Staging  |  HR Zone Distribution
# ======================================================

col3, col4 = st.columns(2)

# ── Sleep Staging ────────────────────────────────────

with col3:
    st.markdown(CARD_OPEN + card_title("Sleep Staging"), unsafe_allow_html=True)

    sleep_window = day_processed[day_processed["sleep_state"] > 0]

    if len(sleep_window) > 10:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        apply_dark(ax, fig)

        for stage, color in [(1, "#3b82f6"), (2, "#1d4ed8"), (3, "#818cf8")]:
            mask  = sleep_window["sleep_state"] == stage
            times = sleep_window.loc[mask, "timestamp"]
            for t in times:
                ax.axvspan(t, t + pd.Timedelta(minutes=1),
                           color=color, alpha=0.85, linewidth=0)

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Light", "Deep", "REM"], fontsize=9, color=TEXT_MID)
        ax.set_ylim(0.5, 3.5)
        patches = [
            mpatches.Patch(color="#3b82f6", label="Light"),
            mpatches.Patch(color="#1d4ed8", label="Deep"),
            mpatches.Patch(color="#818cf8", label="REM"),
        ]
        ax.legend(handles=patches, fontsize=8, facecolor=DARK_BG,
                  edgecolor=CARD_BORDER, labelcolor=TEXT_MID)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.markdown(
            '<p style="color:#4b5563; font-size:13px; padding:1rem 0;">'
            'No sleep data for this day.</p>',
            unsafe_allow_html=True
        )

    dur  = today["sleep_duration_hours"]
    eff  = today["sleep_efficiency"]
    rem  = today["rem_sleep_pct"]
    deep = today["deep_sleep_pct"]
    st.markdown(
        stat_row("Duration", f"{dur:.1f} hrs") +
        stat_row("Efficiency", f"{eff:.0f}%") +
        stat_row("REM / Deep", f"{rem:.0f}% / {deep:.0f}%"),
        unsafe_allow_html=True
    )
    st.markdown(CARD_CLOSE, unsafe_allow_html=True)


# ── HR Zone Distribution ─────────────────────────────

with col4:
    st.markdown(CARD_OPEN + card_title("HR Zone Distribution"), unsafe_allow_html=True)

    zone_cols   = ["zone1_minutes", "zone2_minutes", "zone3_minutes",
                   "zone4_minutes", "zone5_minutes"]
    zone_labels = ["Z1  <90 bpm", "Z2  90–110", "Z3  110–130",
                   "Z4  130–150", "Z5  >150 bpm"]
    zone_colors = ["#1e3a5f", "#2563eb", "#d97706", "#ea580c", "#dc2626"]

    zone_minutes = [float(today.get(z, 0)) for z in zone_cols]
    total_min    = sum(zone_minutes)

    fig, ax = plt.subplots(figsize=(6, 2.6))
    apply_dark(ax, fig)

    y_pos = list(range(len(zone_labels)))
    bars  = ax.barh(y_pos, zone_minutes, color=zone_colors, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(zone_labels, fontsize=9, color=TEXT_MID)
    ax.set_xlabel("minutes", fontsize=9, color=TEXT_MID)
    ax.invert_yaxis()
    for bar, mins in zip(bars, zone_minutes):
        if mins > 2:
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{mins:.0f}", va="center", fontsize=8, color=TEXT_MID)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    high_zones    = sum(today.get(z, 0) for z in zone_cols[2:])
    peak_zone_idx = int(np.argmax(zone_minutes))
    st.markdown(
        stat_row("Most time in", zone_labels[peak_zone_idx]) +
        stat_row("High-intensity (Z3–Z5)", f"{high_zones:.0f} min") +
        stat_row("Total HR zone minutes", f"{total_min:.0f} min"),
        unsafe_allow_html=True
    )
    st.markdown(CARD_CLOSE, unsafe_allow_html=True)


st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ======================================================
# ROW 3 — Activity Signal  |  Readiness Insights
# ======================================================

col5, col6 = st.columns([3, 2])

# ── Activity ─────────────────────────────────────────

with col5:
    st.markdown(CARD_OPEN + card_title("Activity Signal"), unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    apply_dark(ax, fig)
    ax.fill_between(day_processed["timestamp"],
                    day_processed["activity_smooth"],
                    color="#fb923c", alpha=0.2, linewidth=0)
    ax.plot(day_processed["timestamp"], day_processed["activity_smooth"],
            color="#fb923c", linewidth=1.8)
    ax.set_ylabel("a.u.", fontsize=9, color=TEXT_MID)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    active_min = today.get("active_minutes", 0)
    load       = today.get("daily_activity_load", 0)
    st.markdown(
        stat_row("Active minutes", f"{active_min:.0f} min") +
        stat_row("Activity load", f"{load:.1f}"),
        unsafe_allow_html=True
    )
    st.markdown(CARD_CLOSE, unsafe_allow_html=True)


# ── Readiness Insights ───────────────────────────────

with col6:
    st.markdown(CARD_OPEN + card_title("Readiness Insights"), unsafe_allow_html=True)

    recovery = today["recovery_score"]
    strain   = today["strain_score"]
    sleep    = today["sleep_score"]

    pills = []

    if recovery >= 70:
        pills.append(("green",  "High recovery — ready for intensity."))
    elif recovery >= 45:
        pills.append(("yellow", "Moderate recovery — train smart."))
    else:
        pills.append(("red",    "Low recovery — prioritize rest."))

    if sleep >= 80:
        pills.append(("green", "Excellent sleep quality."))
    elif sleep < 55:
        pills.append(("blue",  "Poor sleep — aim for 8 hrs tonight."))

    if strain > 75:
        pills.append(("yellow", "High strain — monitor fatigue."))
    elif strain < 30:
        pills.append(("blue",   "Low strain — room for more activity."))

    prior = scores_df[scores_df["date"] < selected_date]
    if len(prior) >= 2:
        delta = today["nightly_hrv"] - prior["nightly_hrv"].mean()
        if delta > 3:
            pills.append(("green", f"HRV up +{delta:.1f} ms vs average."))
        elif delta < -3:
            pills.append(("red",   f"HRV down {delta:.1f} ms vs average."))

    st.markdown(
        "".join(rec_pill(text, kind) for kind, text in pills),
        unsafe_allow_html=True
    )
    st.markdown(CARD_CLOSE, unsafe_allow_html=True)


st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ======================================================
# ROW 4 — Score History strip (all days, selected highlighted)
# ======================================================

st.markdown(CARD_OPEN + card_title("Score History — All Days"), unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(14, 2.4))
apply_dark(ax, fig)

dates_str = scores_df["date"].astype(str)

ax.plot(dates_str, scores_df["recovery_score"],
        marker="o", markersize=5, linewidth=2,
        color="#34d399", label="Recovery")
ax.plot(dates_str, scores_df["sleep_score"],
        marker="s", markersize=5, linewidth=2,
        color="#818cf8", label="Sleep")
ax.plot(dates_str, scores_df["strain_score"],
        marker="^", markersize=5, linewidth=2,
        color="#fb923c", label="Strain", linestyle="--")

ax.axvline(str(selected_date), color="#ffffff",
           linewidth=1.2, linestyle=":", alpha=0.5)

ax.set_ylim(0, 105)
ax.set_ylabel("Score", fontsize=9, color=TEXT_MID)
ax.legend(fontsize=9, facecolor=DARK_BG, edgecolor=CARD_BORDER,
          labelcolor=TEXT_MID, loc="lower right")
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout(pad=0.4)
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown(CARD_CLOSE, unsafe_allow_html=True)


# ======================================================
# Footer
# ======================================================

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; font-size:11px; color:#374151; '
    'letter-spacing:0.05em;">'
    'WEARABLE ANALYTICS &nbsp;·&nbsp; SIMULATED PHYSIOLOGICAL DATA'
    '&nbsp;·&nbsp; INSPIRED BY WHOOP / OURA</div>',
    unsafe_allow_html=True
)