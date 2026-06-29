"""
Dronomy live dashboard.

Launch while vo_run.py is running (or to review a completed run):
    uv run streamlit run code/dashboard.py
"""
from __future__ import annotations
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "frames_15hz"
DATA_DIR   = ROOT / "data"


def find_run_dir() -> Path:
    """Return the active results folder (latest.txt pointer → newest run_* → legacy nb3v2)."""
    latest_txt = ROOT / "results" / "latest.txt"
    if latest_txt.exists():
        p = Path(latest_txt.read_text().strip())
        if p.exists():
            return p
    runs = sorted((ROOT / "results").glob("run_*/"))
    if runs:
        return runs[-1]
    return ROOT / "results" / "run_20260623_093123"   # legacy fallback


RUN_DIR = find_run_dir()
VO_CSV  = RUN_DIR / "dl_vo_log_full.csv"

# Nominal flight GPS (used to convert VO pixels → approx lat/lon for the map)
NOM_LAT      = 43.521955
NOM_LON      = -5.624290
SCALE_M_PX = 0.7763   # m/px from notebook Cell 8 (manual-anchor fit)
FPS_SOURCE = 15       # extraction rate


def count_frames() -> int:
    """Count extracted frames on disk; fall back to VO CSV max if dir missing."""
    if FRAMES_DIR.exists():
        n = len(list(FRAMES_DIR.glob("*.jpg")))
        if n > 0:
            return n
    return 3429   # full-video fallback

REFRESH_S = 3

# ── Data loading ──────────────────────────────────────────────────────────────

def load_vo_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "frame_idx":    int(row["frame_idx"]),
                "cum_x":        float(row["cum_x"]),
                "cum_y":        float(row["cum_y"]),
                "cum_yaw":      float(row["cum_yaw"]),
                "dx":           float(row["dx"]),
                "dy":           float(row["dy"]),
                "inlier_ratio": float(row["inlier_ratio"]),
                "good_matches": int(row["good_matches"]),
                "inliers":      int(row["inliers"]),
                "mean_score":   float(row["mean_score"]),
            })
    return rows or None


def vo_to_gps(rows: list[dict]) -> tuple[list[float], list[float]]:
    lats, lons = [], []
    cos_lat = math.cos(math.radians(NOM_LAT))
    for r in rows:
        east_m  =  r["cum_x"] * SCALE_M_PX
        north_m = -r["cum_y"] * SCALE_M_PX   # image y-axis points down
        lats.append(NOM_LAT + north_m / 111320)
        lons.append(NOM_LON + east_m  / (111320 * cos_lat))
    return lats, lons


@st.cache_data(show_spinner=False)
def load_sat_tile() -> tuple[np.ndarray | None, str]:
    # Prefer google > esri, higher zoom first (same logic as sat_anchors / fuse)
    def priority(p: Path) -> tuple[int, int]:
        n = p.stem
        prov = 0 if "google" in n else 1
        zoom = int(n.split("_z")[-1].split(".")[0]) if "_z" in n else 0
        return (prov, -zoom)
    candidates = sorted(DATA_DIR.glob("satellite_*.png"), key=priority)
    for p in candidates:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if img is not None:
            return img, p.name
    return None, "not found"


@st.cache_data(show_spinner=False)
def list_frame_paths() -> list[Path]:
    return sorted(FRAMES_DIR.glob("*.jpg"))


def load_frame_rgb(idx: int) -> np.ndarray | None:
    paths = list_frame_paths()
    if not paths or idx >= len(paths):
        return None
    return cv2.cvtColor(cv2.imread(str(paths[idx])), cv2.COLOR_BGR2RGB)


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dronomy Dashboard", layout="wide")
st.title("Dronomy VO — Live Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_rate = st.slider("Refresh interval (s)", 1, 30, REFRESH_S)
    show_table_rows = st.slider("Table rows to show", 10, 200, 50)
    st.markdown("---")
    st.markdown(f"**VO CSV:** `{VO_CSV.relative_to(ROOT)}`")
    st.markdown(f"**Frames dir:** `{FRAMES_DIR.relative_to(ROOT)}`")
    st.markdown(f"**Scale:** {SCALE_M_PX} m/px")
    if st.button("Force refresh"):
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
rows = load_vo_csv(VO_CSV)

if rows is None:
    st.warning(f"VO log not found at `{VO_CSV}`. Start `vo_run.py` to begin.")
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    st.stop()

n_done       = rows[-1]["frame_idx"]
total_frames = count_frames()
pct          = n_done / total_frames * 100
elapsed      = None   # not available from CSV alone

# ── Metrics row ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

is_done = n_done >= total_frames
status  = "Complete" if is_done else "Running"
m1.metric("Status", status)
m2.metric("Frames processed", f"{n_done} / {total_frames}")
m3.metric("Progress", f"{pct:.1f}%")

valid = [r for r in rows[1:] if r["good_matches"] > 0]
mean_ir    = np.mean([r["inlier_ratio"] for r in valid]) if valid else 0.0
mean_score = np.mean([r["mean_score"]   for r in valid]) if valid else 0.0
m4.metric("Mean inlier ratio", f"{mean_ir:.3f}")
m5.metric("Mean LG score",     f"{mean_score:.3f}")

st.progress(min(pct / 100, 1.0))

st.markdown("---")

# ── Current frame + satellite tile ────────────────────────────────────────────
col_frame, col_sat = st.columns(2)

with col_frame:
    st.subheader(f"Drone frame #{n_done}")
    frame_img = load_frame_rgb(n_done)
    if frame_img is not None:
        st.image(frame_img, use_container_width=True)
    else:
        st.info("Frame image not available.")

with col_sat:
    sat_img, sat_name = load_sat_tile()
    st.subheader(f"Satellite tile — {sat_name}")
    if sat_img is not None:
        st.image(sat_img, use_container_width=True)
    else:
        st.info("No satellite tile found in `data/`.")

st.markdown("---")

# ── Trajectory map ────────────────────────────────────────────────────────────
# Load fused result if fuse.py has been run (most recent fuse_* folder)
def load_fused_track() -> list[dict] | None:
    runs = sorted(RUN_DIR.glob("fuse_*/fused_track.csv"))
    if not runs:
        return None
    rows_f = []
    with open(runs[-1], newline="") as fh:
        for row in csv.DictReader(fh):
            rows_f.append({"frame_idx": int(row["frame_idx"]),
                           "est_lat":   float(row["est_lat"]),
                           "est_lon":   float(row["est_lon"])})
    return rows_f or None

fused = load_fused_track()

if fused:
    map_title   = "Trajectory map — fused (drift-corrected with anchors)"
    map_caption = f"Showing anchor-calibrated result from `{RUN_DIR.name}/fuse_*/fused_track.csv`. Scale and orientation fitted from manual anchors."
    lats = [r["est_lat"] for r in fused]
    lons = [r["est_lon"] for r in fused]
    times_s = [r["frame_idx"] / FPS_SOURCE for r in fused]
    hover  = [f"Frame {r['frame_idx']}  t={r['frame_idx']/FPS_SOURCE:.1f}s" for r in fused]
else:
    map_title   = "Approximate trajectory map using VO"
    map_caption = f"Based on past results we use **{SCALE_M_PX} m/px** to convert VO pixel displacements to GPS — this gets adjusted after satellite anchoring."
    lats, lons = vo_to_gps(rows)
    times_s    = [r["frame_idx"] / FPS_SOURCE for r in rows]
    hover      = [f"Frame {r['frame_idx']}  t={t:.1f}s<br>"
                  f"IR={r['inlier_ratio']:.3f}  score={r['mean_score']:.3f}"
                  for r, t in zip(rows, times_s)]

st.subheader(map_title)
st.caption(map_caption)

fig_map = go.Figure()

fig_map.add_trace(go.Scattermapbox(
    lat=lats, lon=lons,
    mode="lines+markers",
    marker=dict(
        size=4,
        color=times_s,
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(title="Time (s)", thickness=12),
    ),
    line=dict(width=2, color="cyan"),
    text=hover,
    hoverinfo="text",
    name="Fused path" if fused else "VO path",
))

fig_map.add_trace(go.Scattermapbox(
    lat=[lats[0]], lon=[lons[0]],
    mode="markers",
    marker=dict(size=14, color="lime"),
    name="Start",
))
fig_map.add_trace(go.Scattermapbox(
    lat=[lats[-1]], lon=[lons[-1]],
    mode="markers",
    marker=dict(size=14, color="red"),
    name="End" if fused else "Current / End",
))

# Auto-fit zoom to trajectory extent
_lat_span = max(lats) - min(lats)
_lon_span = max(lons) - min(lons)
_span     = max(_lat_span, _lon_span, 1e-6)
_zoom     = max(12, min(17, round(13.5 - math.log2(_span * 111))))
_center   = dict(lat=sum(lats) / len(lats), lon=sum(lons) / len(lons))

fig_map.update_layout(
    mapbox=dict(style="open-street-map", center=_center, zoom=_zoom),
    margin=dict(l=0, r=0, t=0, b=0),
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ── Coordinates table ─────────────────────────────────────────────────────────
st.subheader(f"Coordinates — last {show_table_rows} frames")

table_rows = rows[-show_table_rows:]
lats_t, lons_t = vo_to_gps(table_rows)

table_data = {
    "frame_idx":    [r["frame_idx"]    for r in table_rows],
    "time_s":       [f"{r['frame_idx']/FPS_SOURCE:.1f}" for r in table_rows],
    "est_lat":      [f"{lat:.6f}"      for lat in lats_t],
    "est_lon":      [f"{lon:.6f}"      for lon in lons_t],
    "cum_x (px)":   [f"{r['cum_x']:.1f}"  for r in table_rows],
    "cum_y (px)":   [f"{r['cum_y']:.1f}"  for r in table_rows],
    "cum_yaw (°)":  [f"{r['cum_yaw']:.1f}" for r in table_rows],
    "inlier_ratio": [f"{r['inlier_ratio']:.3f}" for r in table_rows],
    "matches":      [r["good_matches"] for r in table_rows],
    "inliers":      [r["inliers"]      for r in table_rows],
    "LG score":     [f"{r['mean_score']:.3f}" for r in table_rows],
}
st.dataframe(table_data, use_container_width=True, height=350)

# ── Footer + auto-refresh ─────────────────────────────────────────────────────
st.markdown(
    f"<small>Last updated: {time.strftime('%H:%M:%S')} &nbsp;|&nbsp; "
    f"CSV: `{VO_CSV.name}`</small>",
    unsafe_allow_html=True,
)

if auto_refresh and not is_done:
    time.sleep(refresh_rate)
    st.rerun()
