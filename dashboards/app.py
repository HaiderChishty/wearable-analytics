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
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import streamlit.components.v1 as components

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
# Global CSS
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

[data-testid="column"] { padding: 0 0.4rem; }

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
# Pipeline — anchor end date to today
# ======================================================

TOTAL_DAYS = 5

@st.cache_data
def run_pipeline(seed=48, total_days=TOTAL_DAYS):
    raw_df       = generate_wearable_data(days=total_days, seed=seed)
    processed_df = preprocess_data(raw_df)

    today_ts = pd.Timestamp(datetime.date.today())
    last_ts  = processed_df["timestamp"].max()
    shift    = (today_ts + pd.Timedelta(hours=23, minutes=59)) - last_ts
    processed_df["timestamp"] = processed_df["timestamp"] + shift

    features_df = generate_features(processed_df)
    scores_df   = generate_scores(features_df)
    return processed_df, scores_df

processed_df, scores_df = run_pipeline()
all_dates = sorted(scores_df["date"].unique())


# ======================================================
# Animated score rings (JS-driven, rendered via components)
# ======================================================

def score_rings_component(rec_score, str_score, slp_score, animate=False):
    """
    Renders three score rings side-by-side in a sandboxed iframe using
    streamlit.components.v1.html().

    When animate=True, JavaScript drives:
      - stroke-dasharray count-up from 0 → target (easeOutQuart, 900 ms)
      - integer text count-up from 0 → target score in sync

    When animate=False the rings are drawn in their final state instantly.
    """
    rings = [
        {"score": rec_score, "label": "RECOVERY", "color": "#34d399"},
        {"score": str_score, "label": "STRAIN",   "color": "#fb923c"},
        {"score": slp_score, "label": "SLEEP",    "color": "#818cf8"},
    ]

    size      = 118
    r         = int(size * 0.34)
    cx = cy   = size // 2
    circ      = 2 * math.pi * r
    stroke_w  = 7

    # Build one <svg> per ring with stable IDs
    svgs = []
    for i, ring in enumerate(rings):
        filled    = max(0.0, min(1.0, ring["score"] / 100)) * circ
        gap       = circ - filled
        score_int = int(round(ring["score"]))
        label_y   = size + 17

        svg = f"""
<svg id="ring_svg_{i}" width="{size}" height="{size + 22}"
     viewBox="0 0 {size} {size + 22}"
     xmlns="http://www.w3.org/2000/svg"
     style="display:block;">
  <!-- background track -->
  <circle cx="{cx}" cy="{cy}" r="{r}"
          fill="none" stroke="#22253a" stroke-width="{stroke_w}"/>
  <!-- filled arc — JS will animate stroke-dasharray -->
  <circle id="arc_{i}" cx="{cx}" cy="{cy}" r="{r}"
          fill="none"
          stroke="{ring['color']}" stroke-width="{stroke_w}"
          stroke-linecap="round"
          stroke-dasharray="{'0 ' + str(round(circ, 2)) if animate else str(round(filled, 2)) + ' ' + str(round(gap, 2))}"
          transform="rotate(-90 {cx} {cy})"/>
  <!-- score number — JS will update textContent -->
  <text id="num_{i}" x="{cx}" y="{cy + 6}"
        text-anchor="middle"
        font-family="DM Mono, monospace"
        font-size="16" font-weight="500"
        fill="#ffffff">{'0' if animate else score_int}</text>
  <!-- label -->
  <text x="{cx}" y="{label_y}"
        text-anchor="middle"
        font-family="DM Sans, sans-serif"
        font-size="9" letter-spacing="1.5"
        fill="#6b7280">{ring['label']}</text>
</svg>"""
        svgs.append(svg)

    # JS targets: list of [filled, gap, score_int] per ring
    targets_js = "[" + ",".join(
        f"[{max(0.0, min(1.0, r['score']/100))*circ:.3f},"
        f"{circ - max(0.0, min(1.0, r['score']/100))*circ:.3f},"
        f"{int(round(r['score']))}]"
        for r in rings
    ) + "]"

    anim_js = ""
    if animate:
        anim_js = f"""
<script>
(function() {{
  const DURATION = 900;   // ms
  const targets  = {targets_js};
  const circ     = {circ:.3f};

  // easeOutQuart for a snappy deceleration
  function ease(t) {{
    return 1 - Math.pow(1 - t, 4);
  }}

  const arcs = targets.map((_, i) => document.getElementById('arc_' + i));
  const nums = targets.map((_, i) => document.getElementById('num_' + i));

  const start = performance.now();

  function frame(now) {{
    const raw = Math.min((now - start) / DURATION, 1.0);
    const t   = ease(raw);

    targets.forEach(([filled, gap, score], i) => {{
      // arc
      const f = filled * t;
      const g = circ - f;
      arcs[i].setAttribute('stroke-dasharray', f.toFixed(3) + ' ' + g.toFixed(3));
      // integer count-up
      nums[i].textContent = Math.round(score * t);
    }});

    if (raw < 1.0) requestAnimationFrame(frame);
  }}

  requestAnimationFrame(frame);
}})();
</script>"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    background: transparent;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 14px;
    padding: 4px 0;
  }}
</style>
</head>
<body>
{''.join(svgs)}
{anim_js}
</body>
</html>"""

    # height = svg height + a little breathing room
    components.html(html, height=size + 36, scrolling=False)

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
# Shared time-axis formatter
# ======================================================

import matplotlib.dates as mdates

def format_time_axis(ax, interval=2):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.tick_params(axis="x", rotation=0)
    fig = ax.get_figure()
    fig.canvas.draw()
    labels = []
    for dt in ax.get_xticks():
        try:
            labels.append(mdates.num2date(dt).strftime("%I:%M %p").lstrip("0"))
        except Exception:
            labels.append("")
    ax.set_xticklabels(labels, ha="center")


# ======================================================
# HTML helpers
# ======================================================

def stat_row(label, value):
    return (
        '<div style="display:flex; justify-content:space-between; '
        'align-items:center; padding:7px 0; border-bottom:1px solid #1f2233;">'
        f'<span style="font-size:15px; color:#9ca3af;">{label}</span>'
        f'<span style="font-family:DM Mono,monospace; font-size:15px; '
        f'font-weight:500; color:#f3f4f6;">{value}</span>'
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
        f'border-radius:999px; padding:5px 13px; font-size:13px; font-weight:500; '
        f'margin:3px 0; display:inline-block;">{text}</div>'
    )


# ======================================================
# DAY INDEX — resolved once, buttons rerun immediately
# ======================================================

if "day_idx" not in st.session_state:
    st.session_state.day_idx = len(all_dates) - 1

st.session_state.day_idx = min(st.session_state.day_idx, len(all_dates) - 1)

if "prev_day_idx" not in st.session_state:
    st.session_state.prev_day_idx = st.session_state.day_idx

day_changed = (st.session_state.day_idx != st.session_state.prev_day_idx)
st.session_state.prev_day_idx = st.session_state.day_idx

selected_date = all_dates[st.session_state.day_idx]

day_processed = processed_df[
    processed_df["timestamp"].dt.date == selected_date
].copy()

day_scores = scores_df[scores_df["date"] == selected_date]
if day_scores.empty:
    st.warning("No score data for this day.")
    st.stop()

today     = day_scores.iloc[0]
day_start = pd.Timestamp(selected_date)
day_end   = day_start + pd.Timedelta(days=1)


# ======================================================
# Readiness pills
# ======================================================

recovery = today["recovery_score"]
strain   = today["strain_score"]
sleep_s  = today["sleep_score"]

pills = []
if recovery >= 70:
    pills.append(("green",  "High recovery — ready for intensity."))
elif recovery >= 45:
    pills.append(("yellow", "Moderate recovery — train smart."))
else:
    pills.append(("red",    "Low recovery — prioritize rest."))

if sleep_s >= 80:
    pills.append(("green", "Excellent sleep quality."))
elif sleep_s < 55:
    pills.append(("blue",  "Poor sleep — aim for 8 hrs tonight."))

if strain > 75:
    pills.append(("yellow", "High strain — monitor fatigue."))
elif strain < 30:
    pills.append(("blue",   "Low strain — room for more activity."))

prior = scores_df[scores_df["date"] < selected_date]
if len(prior) >= 2:
    delta = today["nightly_hrv"] - prior["nightly_hrv"].mean()
    if delta > 3:
        pills.append(("green", f"HRV up +{delta:.1f} ms vs avg."))
    elif delta < -3:
        pills.append(("red",   f"HRV down {delta:.1f} ms vs avg."))


# ======================================================
# HEADER ROW
# ======================================================

hdr_l, hdr_c, hdr_r = st.columns([2.6, 3.2, 2.2])

with hdr_l:
    st.markdown(
        '<div style="padding-top:0.45rem;">'
        '<div style="font-size:28px; font-weight:300; color:#e8eaf0; '
        'letter-spacing:-0.03em; line-height:1.1;">'
        'Welcome back, <span style="font-weight:600; color:#fff;">Haider</span>'
        '</div>'
        '<div style="font-size:14px; color:#6b7280; margin-top:5px;">'
        'Your physiological snapshot</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("◀", key="btn_prev", use_container_width=True):
            if st.session_state.day_idx > 0:
                st.session_state.day_idx -= 1
                st.rerun()
    with nav3:
        if st.button("▶", key="btn_next", use_container_width=True):
            if st.session_state.day_idx < len(all_dates) - 1:
                st.session_state.day_idx += 1
                st.rerun()
    with nav2:
        st.markdown(
            f'<div style="text-align:center; font-family:DM Mono,monospace; '
            f'font-size:13px; background:#151925; border:1px solid #232838; '
            f'border-radius:10px; padding:8px 10px; color:#e8eaf0;">'
            f'{selected_date.strftime("%A, %B %d")}</div>',
            unsafe_allow_html=True
        )

with hdr_c:
    # JS-driven rings: animate on day change, static on first load
    score_rings_component(
        rec_score=today["recovery_score"],
        str_score=today["strain_score"],
        slp_score=today["sleep_score"],
        animate=day_changed,
    )

with hdr_r:
    st.markdown(
        '<div style="padding-top:1rem; padding-left:0.5rem;">'
        '<div style="font-size:11px; letter-spacing:0.12em; font-weight:600; '
        'color:#6b7280; margin-bottom:10px;">READINESS INSIGHTS</div>'
        '<div style="display:flex; flex-direction:column; '
        'align-items:flex-start; gap:7px;">'
        + "".join(rec_pill(text, kind) for kind, text in pills[:4])
        + '</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
st.markdown('<hr style="border-color:#1f2233; margin:0 0 1rem 0;">', unsafe_allow_html=True)


# ======================================================
# ROW 1 — Heart Rate  |  HRV
# ======================================================

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    apply_dark(ax, fig)
    ax.plot(day_processed["timestamp"], day_processed["hr"],
            color="#374151", linewidth=0.8, alpha=0.5)
    ax.plot(day_processed["timestamp"], day_processed["hr_smooth"],
            color="#f87171", linewidth=2)
    ax.set_title("Heart Rate", fontsize=13, color="#e8eaf0",
                 fontweight="500", loc="left", pad=8)
    ax.set_ylabel("bpm", fontsize=10, color=TEXT_MID)
    ax.set_xlim(day_start, day_end)
    format_time_axis(ax, interval=4)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    rhr    = today["resting_hr"]
    avg_hr = day_processed["hr"].mean()
    max_hr = day_processed["hr"].max()
    st.markdown(
        stat_row("Resting HR",    f"{rhr:.0f} bpm") +
        stat_row("Daily average", f"{avg_hr:.0f} bpm") +
        stat_row("Daily max",     f"{max_hr:.0f} bpm"),
        unsafe_allow_html=True
    )

with col2:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    apply_dark(ax, fig)
    ax.plot(day_processed["timestamp"], day_processed["hrv"],
            color="#374151", linewidth=0.8, alpha=0.5)
    ax.plot(day_processed["timestamp"], day_processed["hrv_smooth"],
            color="#818cf8", linewidth=2)
    ax.set_title("Heart Rate Variability", fontsize=13, color="#e8eaf0",
                 fontweight="500", loc="left", pad=8)
    ax.set_ylabel("ms", fontsize=10, color=TEXT_MID)
    ax.set_xlim(day_start, day_end)
    format_time_axis(ax, interval=4)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    nhrv    = today["nightly_hrv"]
    avg_hrv = day_processed["hrv"].mean()
    max_hrv = day_processed["hrv"].max()
    st.markdown(
        stat_row("Nightly HRV",   f"{nhrv:.1f} ms") +
        stat_row("Daily average", f"{avg_hrv:.1f} ms") +
        stat_row("Daily max",     f"{max_hrv:.1f} ms"),
        unsafe_allow_html=True
    )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ======================================================
# ROW 2 — Sleep Staging  |  HR Zone Distribution
# ======================================================

col3, col4 = st.columns(2)

with col3:
    night_start = pd.Timestamp(selected_date) - pd.Timedelta(hours=4)
    night_end   = pd.Timestamp(selected_date) + pd.Timedelta(hours=10)

    sleep_window = processed_df[
        (processed_df["timestamp"] >= night_start) &
        (processed_df["timestamp"] <= night_end) &
        (processed_df["sleep_state"] > 0)
    ]

    if len(sleep_window) > 10:
        fig, ax = plt.subplots(figsize=(6, 2.35))
        apply_dark(ax, fig)

        for stage, color in [
            (1, "#38bdf8"),
            (2, "#2563eb"),
            (3, "#a600bf"),
        ]:
            mask  = sleep_window["sleep_state"] == stage
            times = sleep_window.loc[mask, "timestamp"]
            for t in times:
                ax.axvspan(t, t + pd.Timedelta(minutes=1),
                           color=color, alpha=0.85, linewidth=0)

        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Sleep Staging", fontsize=13, color="#e8eaf0",
                     fontweight="500", loc="left", pad=8)

        patches = [
            mpatches.Patch(color="#38bdf8", label="Light"),
            mpatches.Patch(color="#2563eb", label="Deep"),
            mpatches.Patch(color="#a600bf", label="REM"),
        ]
        ax.legend(handles=patches, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), fontsize=9,
                  facecolor=DARK_BG, edgecolor=CARD_BORDER,
                  labelcolor=TEXT_MID, frameon=False)

        start = sleep_window["timestamp"].min()
        end   = sleep_window["timestamp"].max()
        ax.set_xlim(start, end)

        ticks = pd.date_range(start=start.floor("h"), end=end.ceil("h"), freq="1h")
        ticks = sorted(set(
            [t for t in ticks if start <= t <= end] + [start.floor("h")]
        ))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [t.strftime("%I:%M %p").lstrip("0") for t in ticks],
            color=TEXT_MID, fontsize=9
        )
        ax.tick_params(axis="x", rotation=0)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.markdown(
            '<p style="color:#9ca3af; font-size:14px; padding:1rem 0;">'
            'No sleep data for this night.</p>',
            unsafe_allow_html=True
        )

    dur  = today["sleep_duration_hours"]
    eff  = today["sleep_efficiency"]
    rem  = today["rem_sleep_pct"]
    deep = today["deep_sleep_pct"]
    st.markdown(
        stat_row("Duration",   f"{dur:.1f} hrs") +
        stat_row("REM / Deep", f"{rem:.0f}% / {deep:.0f}%"),
        unsafe_allow_html=True
    )

with col4:
    zone_cols   = ["zone1_minutes", "zone2_minutes", "zone3_minutes",
                   "zone4_minutes", "zone5_minutes"]
    zone_labels = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"]
    zone_colors = ["#3b82f6", "#34d399", "#fbbf24", "#fb923c", "#f87171"]

    zone_minutes = [
        float(today[z]) if z in today.index and pd.notna(today[z]) else 0.0
        for z in zone_cols
    ]
    total_min = sum(zone_minutes)

    fig, ax = plt.subplots(figsize=(6, 2.75))
    apply_dark(ax, fig)

    bar_h       = 0.38
    y_positions = list(range(len(zone_cols)))

    for i, (mins, label, color) in enumerate(
            zip(zone_minutes, zone_labels, zone_colors)):
        pct = (mins / total_min * 100) if total_min > 0 else 0
        ax.barh(i, pct, height=bar_h, color=color, alpha=0.88)
        ax.text(pct + 0.8, i, f"{pct:.1f}%",
                va="center", ha="left", fontsize=9,
                color="#e8eaf0", fontfamily="monospace")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(zone_labels, fontsize=9, color=TEXT_MID)
    ax.set_xlabel("% of day", fontsize=9, color=TEXT_MID)
    ax.set_xlim(0, 110)
    ax.set_title("HR Zone Distribution", fontsize=13, color="#e8eaf0",
                 fontweight="500", loc="left", pad=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    high_zones = sum(
        float(today[z]) if z in today.index and pd.notna(today[z]) else 0.0
        for z in zone_cols[2:]
    )
    peak_zone_idx = int(np.argmax(zone_minutes))
    st.markdown(
        stat_row("Most time in",           zone_labels[peak_zone_idx]) +
        stat_row("High-intensity (Z3-Z5)", f"{high_zones:.0f} min"),
        unsafe_allow_html=True
    )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ======================================================
# ROW 3 — Activity Signal  |  Score History
# ======================================================

col5, col6 = st.columns(2)

with col5:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    apply_dark(ax, fig)
    ax.fill_between(day_processed["timestamp"],
                    day_processed["activity_smooth"],
                    color="#fb923c", alpha=0.2, linewidth=0)
    ax.plot(day_processed["timestamp"], day_processed["activity_smooth"],
            color="#fb923c", linewidth=1.8)
    ax.set_title("Activity Signal", fontsize=13, color="#e8eaf0",
                 fontweight="500", loc="left", pad=8)
    ax.set_ylabel("a.u.", fontsize=10, color=TEXT_MID)
    ax.set_xlim(day_start, day_end)
    format_time_axis(ax, interval=4)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    active_min = today.get("active_minutes", 0)
    st.markdown(
        stat_row("Active minutes", f"{active_min:.0f} min"),
        unsafe_allow_html=True
    )

with col6:
    history_df = scores_df.tail(5).copy()
    dates_str  = [d.strftime("%m/%d") for d in history_df["date"]]
    x          = np.arange(len(dates_str))
    width      = 0.25

    fig, ax = plt.subplots(figsize=(6, 3.3))
    apply_dark(ax, fig)

    # Bar order matches ring order: Recovery → Strain → Sleep
    bars_rec    = ax.bar(x - width, history_df["recovery_score"],
                         width, color="#34d399", label="Recovery", alpha=0.85)
    bars_strain = ax.bar(x,         history_df["strain_score"],
                         width, color="#fb923c", label="Strain",   alpha=0.85)
    bars_sleep  = ax.bar(x + width, history_df["sleep_score"],
                         width, color="#818cf8", label="Sleep",    alpha=0.85)

    sel_str = pd.Timestamp(selected_date).strftime("%m/%d")
    if sel_str in dates_str:
        sel_idx = dates_str.index(sel_str)
        for bg in [bars_rec, bars_strain, bars_sleep]:
            bg[sel_idx].set_edgecolor("#ffffff")
            bg[sel_idx].set_linewidth(1.5)

    ax.set_title("Score History — Last 5 Days", fontsize=13, color="#e8eaf0",
                 fontweight="500", loc="left", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dates_str, fontsize=10, color=TEXT_MID,
                       rotation=0, ha="center")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score", fontsize=10, color=TEXT_MID)
    ax.legend(fontsize=9, facecolor=DARK_BG, edgecolor=CARD_BORDER,
              labelcolor=TEXT_MID, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
    plt.tight_layout(pad=0.4, rect=[0, 0.12, 1, 1])
    st.pyplot(fig, use_container_width=True)
    plt.close()


