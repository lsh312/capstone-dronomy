"""
GPS ground-truth benchmark for the drone geolocalization pipeline.

Scores estimated positions (satellite anchors, or any per-frame trajectory)
against DJI on-board GPS recorded in the video's .SRT subtitle track.

The harness is matcher-agnostic: it consumes a CSV of estimates
(`frame_idx, est_lat, est_lon`) plus the DJI .SRT, and reports per-fix
position error in metres. The same harness therefore scores LightGlue,
SIFT, SURF and MatchAnything runs on equal footing.

Alignment: extracted frames are named `frame_<ms>ms.jpg`, where <ms> is the
video timestamp (cv2 CAP_PROP_POS_MSEC). DJI .SRT cue times are on the same
video clock, so truth is aligned to each frame by timestamp with no offset
(an explicit offset is available via --time-offset-ms if clocks ever differ).

CLI:
    python gps_benchmark.py \
        --srt ../data/flight.SRT \
        --anchors ../experiments/deep_learning/results/deep-learning-8inliers+tuning/dl_anchors_log_20260623_115208.csv \
        --frames ../data/frames_15hz \
        --out ../experiments/comparison/results/benchmark

Library:
    from gps_benchmark import parse_dji_srt, load_frame_timestamps, benchmark_estimates
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Nominal flight coordinates (from the video filename). Used only to
# disambiguate lat/lon ordering in the older DJI GPS(...) SRT format.
NOMINAL_LAT = 43.521955
NOMINAL_LON = -5.624290

# An estimate is counted as a correct fix if it lands within this distance
# of ground truth. A few metres of error is expected (per the mentor brief).
DEFAULT_GOOD_M = 15.0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# DJI .SRT parsing
# --------------------------------------------------------------------------- #
class GpsFix(NamedTuple):
    t_ms: int       # video timestamp (ms) from the SRT cue start
    lat: float
    lon: float
    alt: float      # relative/absolute altitude (m); NaN if absent
    yaw: float = float("nan")   # gimbal yaw / heading (deg, 0-360 N-ref); NaN if absent


_CUE_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
# Modern DJI format: [latitude: 43.52] [longitude: -5.62] [rel_alt: 1.3 abs_alt: 100]
_LAT_RE = re.compile(r"\[?\s*latitude\s*[:=]\s*(-?\d+\.\d+)", re.IGNORECASE)
_LON_RE = re.compile(r"\[?\s*longitude\s*[:=]\s*(-?\d+\.\d+)", re.IGNORECASE)
_ALT_RE = re.compile(r"(?:rel_alt|abs_alt|altitude)\s*[:=]\s*(-?\d+\.?\d*)", re.IGNORECASE)
# Heading/orientation truth: [gb_yaw: 170.3 gb_pitch: -90.0 gb_roll: 0.0]
_YAW_RE = re.compile(r"gb_yaw\s*[:=]\s*(-?\d+\.?\d*)", re.IGNORECASE)
# Older DJI format: GPS(-5.624290,43.521955,18)  -> (lon, lat, sats) usually
_GPS_RE = re.compile(r"GPS\s*\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.?\d*)\s*\)")


def _cue_start_ms(line: str) -> int | None:
    m = _CUE_RE.search(line)
    if not m:
        return None
    h, mn, s, ms = (int(m.group(i)) for i in range(1, 5))
    return ((h * 60 + mn) * 60 + s) * 1000 + ms


def _assign_lat_lon(a: float, b: float,
                    nominal_lat: float, nominal_lon: float) -> tuple[float, float]:
    """Disambiguate a GPS(a,b,...) pair into (lat, lon).

    DJI's older format is GPS(lon, lat, sats), but order varies by firmware.
    Pick the assignment whose point is closest to the nominal coordinate.
    """
    as_lon_lat = haversine_m(b, a, nominal_lat, nominal_lon)   # a=lon, b=lat
    as_lat_lon = haversine_m(a, b, nominal_lat, nominal_lon)   # a=lat, b=lon
    return (b, a) if as_lon_lat <= as_lat_lon else (a, b)


def parse_dji_srt(path: str | Path,
                  nominal_lat: float = NOMINAL_LAT,
                  nominal_lon: float = NOMINAL_LON) -> list[GpsFix]:
    """Parse a DJI .SRT subtitle track into time-ordered GPS fixes.

    Handles both the modern `[latitude:][longitude:]` layout and the older
    `GPS(lon,lat,sats)` layout. Each subtitle cue contributes one fix, stamped
    with the cue's start time on the video clock.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Split on blank lines into subtitle blocks.
    blocks = re.split(r"\n\s*\n", text)
    fixes: list[GpsFix] = []

    for block in blocks:
        if not block.strip():
            continue
        t_ms = None
        for line in block.splitlines():
            t = _cue_start_ms(line)
            if t is not None:
                t_ms = t
                break
        if t_ms is None:
            continue

        lat = lon = None
        lat_m, lon_m = _LAT_RE.search(block), _LON_RE.search(block)
        if lat_m and lon_m:
            lat, lon = float(lat_m.group(1)), float(lon_m.group(1))
        else:
            gps_m = _GPS_RE.search(block)
            if gps_m:
                a, b = float(gps_m.group(1)), float(gps_m.group(2))
                lat, lon = _assign_lat_lon(a, b, nominal_lat, nominal_lon)
        if lat is None or lon is None:
            continue

        alt_m = _ALT_RE.search(block)
        alt = float(alt_m.group(1)) if alt_m else float("nan")
        yaw_m = _YAW_RE.search(block)
        yaw = float(yaw_m.group(1)) if yaw_m else float("nan")
        fixes.append(GpsFix(t_ms, lat, lon, alt, yaw))

    fixes.sort(key=lambda f: f.t_ms)
    return fixes


# --------------------------------------------------------------------------- #
# Frame / truth alignment
# --------------------------------------------------------------------------- #
def load_frame_timestamps(frames_dir: str | Path,
                          n_needed: int | None = None,
                          fps_hint: float = 15.0) -> np.ndarray:
    """Return per-frame video timestamps (ms), indexed by frame_idx.

    frame_idx in the pipeline logs is the position in the sorted frame list,
    matching the notebook's `sorted(FRAMES_DIR.glob('*.jpg'))`.

    Frame extraction is regular (15 Hz), so when fewer frame files are present
    than `n_needed` (e.g. only the first 50 frames are committed to the repo
    while logs reference frame_idx up to ~3429), the cadence is fit from the
    available frames and extrapolated linearly. Falls back to `fps_hint` if no
    frames are on disk at all.
    """
    paths = sorted(Path(frames_dir).glob("*.jpg"))
    ts = np.array([int(p.stem.split("_")[1].replace("ms", "")) for p in paths],
                  dtype=np.int64)

    if n_needed is None or n_needed <= len(ts):
        return ts

    if len(ts) >= 2:
        # Median per-frame step from observed cadence (robust to the source
        # video's 33/34 ms POS_MSEC jitter).
        step = float(np.median(np.diff(ts)))
        t0 = float(ts[0])
    else:
        step = 1000.0 / fps_hint
        t0 = float(ts[0]) if len(ts) else 0.0
    full = (t0 + step * np.arange(n_needed)).round().astype(np.int64)
    # Preserve the exact measured timestamps where we have real files.
    full[: len(ts)] = ts
    return full


def interp_truth(fixes: list[GpsFix], t_ms: float) -> tuple[float, float] | None:
    """Linearly interpolate ground-truth lat/lon at a video timestamp.

    Returns None if t_ms falls outside the SRT time span (no extrapolation).
    """
    if not fixes:
        return None
    times = np.array([f.t_ms for f in fixes], dtype=np.float64)
    if t_ms < times[0] or t_ms > times[-1]:
        return None
    lats = np.array([f.lat for f in fixes], dtype=np.float64)
    lons = np.array([f.lon for f in fixes], dtype=np.float64)
    return float(np.interp(t_ms, times, lats)), float(np.interp(t_ms, times, lons))


def interp_truth_yaw(fixes: list[GpsFix], t_ms: float) -> float | None:
    """Circularly interpolate the ground-truth heading (deg) at a timestamp.

    Headings wrap at 360 deg, so interpolation is done on the unit circle
    (interpolate sin/cos, then atan2) rather than linearly. Returns a value in
    [0, 360), or None if t_ms is outside the SRT span or no yaw truth exists.
    """
    if not fixes:
        return None
    times = np.array([f.t_ms for f in fixes], dtype=np.float64)
    if t_ms < times[0] or t_ms > times[-1]:
        return None
    yaws = np.array([f.yaw for f in fixes], dtype=np.float64)
    if not np.isfinite(yaws).any():
        return None
    rad = np.radians(yaws)
    s = float(np.interp(t_ms, times, np.sin(rad)))
    c = float(np.interp(t_ms, times, np.cos(rad)))
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def heading_error_deg(est_deg: float, truth_deg: float) -> float:
    """Smallest absolute angular difference between two headings, in [0, 180]."""
    return abs((est_deg - truth_deg + 180.0) % 360.0 - 180.0)


def best_yaw_offset_deg(signed_errors_deg: np.ndarray) -> float:
    """Circular mean of signed (est-truth) errors — the best constant offset.

    If subtracting this offset collapses the heading error, the estimator is
    informative but mis-referenced (a fixable convention/frame issue) rather
    than genuinely random.
    """
    rad = np.radians(signed_errors_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


# --------------------------------------------------------------------------- #
# Estimate loading + scoring
# --------------------------------------------------------------------------- #
class Scored(NamedTuple):
    frame_idx: int
    t_ms: int
    est_lat: float
    est_lon: float
    true_lat: float
    true_lon: float
    error_m: float
    is_good: bool


def load_estimates(csv_path: str | Path,
                   lat_col: str = "est_lat",
                   lon_col: str = "est_lon") -> list[tuple[int, float, float]]:
    """Load (frame_idx, lat, lon) rows from a pipeline CSV log."""
    out: list[tuple[int, float, float]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return out
        # Tolerate alternative column names from other matchers / trajectories.
        lat_key = lat_col if lat_col in reader.fieldnames else _first_match(
            reader.fieldnames, ("est_lat", "lat", "latitude"))
        lon_key = lon_col if lon_col in reader.fieldnames else _first_match(
            reader.fieldnames, ("est_lon", "lon", "longitude"))
        idx_key = _first_match(reader.fieldnames, ("frame_idx", "frame", "idx"))
        if not (lat_key and lon_key and idx_key):
            raise ValueError(
                f"{csv_path}: need frame_idx/lat/lon columns, got {reader.fieldnames}")
        for row in reader:
            out.append((int(float(row[idx_key])),
                        float(row[lat_key]), float(row[lon_key])))
    return out


def _first_match(fields, candidates):
    lower = {f.lower(): f for f in fields}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def benchmark_estimates(estimates: list[tuple[int, float, float]],
                        fixes: list[GpsFix],
                        frame_ts: np.ndarray,
                        good_m: float = DEFAULT_GOOD_M,
                        time_offset_ms: int = 0) -> tuple[list[Scored], dict]:
    """Score each estimate against interpolated ground truth.

    Returns (per-estimate scored rows, summary dict).
    """
    scored: list[Scored] = []
    for frame_idx, est_lat, est_lon in estimates:
        if frame_idx < 0 or frame_idx >= len(frame_ts):
            continue
        t_ms = int(frame_ts[frame_idx]) + time_offset_ms
        truth = interp_truth(fixes, t_ms)
        if truth is None:
            continue
        true_lat, true_lon = truth
        err = haversine_m(est_lat, est_lon, true_lat, true_lon)
        scored.append(Scored(frame_idx, t_ms, est_lat, est_lon,
                             true_lat, true_lon, err, err <= good_m))

    errs = np.array([s.error_m for s in scored], dtype=np.float64)
    summary = {
        "n_estimates": len(estimates),
        "n_scored": len(scored),
        "n_unmatched": len(estimates) - len(scored),
        "good_threshold_m": good_m,
    }
    if errs.size:
        summary.update({
            "mean_error_m": float(np.mean(errs)),
            "median_error_m": float(np.median(errs)),
            "p90_error_m": float(np.percentile(errs, 90)),
            "max_error_m": float(np.max(errs)),
            "min_error_m": float(np.min(errs)),
            "rmse_m": float(np.sqrt(np.mean(errs ** 2))),
            "n_good": int(np.sum(errs <= good_m)),
            "pct_good": float(100.0 * np.mean(errs <= good_m)),
        })
    return scored, summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_scored_csv(scored: list[Scored], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_idx", "t_ms", "est_lat", "est_lon",
                    "true_lat", "true_lon", "error_m", "is_good"])
        for s in scored:
            w.writerow([s.frame_idx, s.t_ms, f"{s.est_lat:.7f}", f"{s.est_lon:.7f}",
                        f"{s.true_lat:.7f}", f"{s.true_lon:.7f}",
                        f"{s.error_m:.2f}", int(s.is_good)])


def plot_report(scored: list[Scored], fixes: list[GpsFix], summary: dict,
                out_path: str | Path, title: str = "GPS benchmark") -> None:
    """Three-panel report: error vs time, error CDF, spatial overlay."""
    import matplotlib.pyplot as plt

    if not scored:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errs = np.array([s.error_m for s in scored])
    t_s = np.array([s.t_ms for s in scored]) / 1000.0
    good_m = summary.get("good_threshold_m", DEFAULT_GOOD_M)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Error over time
    ax = axes[0]
    ax.scatter(t_s, errs, c=np.where(errs <= good_m, "tab:green", "tab:red"),
               s=20, alpha=0.8)
    ax.axhline(good_m, ls="--", c="gray", lw=1, label=f"{good_m:.0f} m threshold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
    ax.set_title("Error over time"); ax.legend(); ax.grid(alpha=0.3)

    # 2. Error CDF
    ax = axes[1]
    se = np.sort(errs)
    cdf = np.arange(1, se.size + 1) / se.size * 100
    ax.plot(se, cdf, lw=2)
    ax.axvline(good_m, ls="--", c="gray", lw=1)
    ax.axvline(summary.get("median_error_m", np.median(errs)), ls=":", c="tab:blue",
               lw=1, label=f"median {np.median(errs):.1f} m")
    ax.set_xlabel("position error (m)"); ax.set_ylabel("% of estimates ≤ error")
    ax.set_title("Error CDF"); ax.legend(); ax.grid(alpha=0.3)

    # 3. Spatial overlay: truth track + estimates joined to their truth point
    ax = axes[2]
    if fixes:
        ax.plot([f.lon for f in fixes], [f.lat for f in fixes],
                "-", c="0.6", lw=1, label="GPS truth track")
    for s in scored:
        c = "tab:green" if s.is_good else "tab:red"
        ax.plot([s.est_lon, s.true_lon], [s.est_lat, s.true_lat], "-", c=c, lw=0.6, alpha=0.6)
        ax.scatter(s.est_lon, s.est_lat, c=c, s=18, zorder=3)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("Estimates vs truth"); ax.legend(); ax.grid(alpha=0.3)
    ax.ticklabel_format(useOffset=False, style="plain")

    fig.suptitle(f"{title}  —  median {summary.get('median_error_m', float('nan')):.1f} m, "
                 f"{summary.get('pct_good', 0):.0f}% within {good_m:.0f} m "
                 f"(n={summary.get('n_scored', 0)})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: dict, label: str = "") -> None:
    head = f"GPS benchmark{f' — {label}' if label else ''}"
    print(head)
    print("=" * len(head))
    print(f"  estimates scored : {summary.get('n_scored', 0)} / {summary.get('n_estimates', 0)}"
          f"  ({summary.get('n_unmatched', 0)} outside SRT time span)")
    if "median_error_m" not in summary:
        print("  (no estimates could be scored)")
        return
    g = summary["good_threshold_m"]
    print(f"  median error     : {summary['median_error_m']:.1f} m")
    print(f"  mean error       : {summary['mean_error_m']:.1f} m")
    print(f"  RMSE             : {summary['rmse_m']:.1f} m")
    print(f"  p90 error        : {summary['p90_error_m']:.1f} m")
    print(f"  min / max        : {summary['min_error_m']:.1f} m / {summary['max_error_m']:.1f} m")
    print(f"  within {g:.0f} m      : {summary['n_good']} ({summary['pct_good']:.0f}%)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark estimated positions vs DJI .SRT GPS truth")
    ap.add_argument("--srt", required=True, help="DJI .SRT subtitle file")
    ap.add_argument("--anchors", "--estimates", dest="estimates", required=True,
                    help="CSV of estimates with frame_idx, est_lat, est_lon")
    ap.add_argument("--frames", default="../data/frames_15hz",
                    help="frames dir (for frame_idx -> timestamp mapping)")
    ap.add_argument("--out", default="../experiments/comparison/results/benchmark", help="output dir")
    ap.add_argument("--good-m", type=float, default=DEFAULT_GOOD_M,
                    help="distance (m) under which a fix counts as correct")
    ap.add_argument("--time-offset-ms", type=int, default=0,
                    help="add this offset to frame timestamps before truth lookup")
    ap.add_argument("--label", default="", help="label for plot title / output names")
    args = ap.parse_args()

    fixes = parse_dji_srt(args.srt)
    if not fixes:
        raise SystemExit(f"No GPS fixes parsed from {args.srt} — check the SRT format.")
    span_s = (fixes[-1].t_ms - fixes[0].t_ms) / 1000.0
    print(f"Parsed {len(fixes)} GPS fixes from SRT  "
          f"({span_s:.1f} s span, {fixes[0].t_ms/1000:.1f}–{fixes[-1].t_ms/1000:.1f} s)")

    estimates = load_estimates(args.estimates)
    max_idx = max((e[0] for e in estimates), default=0)
    frame_ts = load_frame_timestamps(args.frames, n_needed=max_idx + 1)
    n_real = len(sorted(Path(args.frames).glob("*.jpg")))
    note = "" if n_real >= max_idx + 1 else f"  ({n_real} real + extrapolated to {max_idx + 1})"
    print(f"Frame timestamps: {len(frame_ts)}{note}")
    print(f"Loaded {len(estimates)} estimates from {args.estimates}\n")

    scored, summary = benchmark_estimates(estimates, fixes, frame_ts,
                                          good_m=args.good_m,
                                          time_offset_ms=args.time_offset_ms)
    label = args.label or Path(args.estimates).stem
    print_summary(summary, label)

    out_dir = Path(args.out)
    csv_out = out_dir / f"benchmark_{label}.csv"
    png_out = out_dir / f"benchmark_{label}.png"
    write_scored_csv(scored, csv_out)
    plot_report(scored, fixes, summary, png_out, title=label)
    print(f"\nWrote {csv_out}")
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    main()
