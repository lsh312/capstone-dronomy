"""
Matcher comparison for automated satellite anchoring.

Scores the satellite-anchor estimates from three feature matchers — SIFT
(classical), SuperPoint+LightGlue (sparse deep), and RoMa (dense deep) — against
the same DJI .SRT GPS truth, through the same `gps_benchmark` scorer, so the
matchers are compared on equal footing. Manual anchoring (VO + human-marked
anchors + piecewise fusion) is reported separately as the non-automated baseline
that actually clears the target.

Estimate CSVs live in experiments/comparison/results/matcher_comparison/ (copied
from teammates' branches: SIFT -> initial-sift-baseline, RoMa -> feature/roma,
LightGlue -> the main pipeline's deep-learning-6inliers run).

Caveat: each automated run used its own satellite source / zoom / frame
sampling, so this compares the three pipelines as built (scored identically),
not the matchers in perfect isolation.

Run:
    cd code && uv run python matcher_comparison.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from gps_benchmark import haversine_m, interp_truth, parse_dji_srt

ROOT = Path(__file__).resolve().parent.parent.parent
SRT = ROOT / "data" / "gps_data.SRT"
EST_DIR = ROOT / "experiments" / "comparison" / "results" / "matcher_comparison"

# Each matcher's estimate CSV + how to read timestamp / lat / lon from it.
SOURCES = [
    {"name": "SIFT", "type": "sparse classical", "file": "sift_anchors.csv",
     "t_field": "time_sec", "t_scale": 1000.0,
     "lat": "estimated_lat", "lon": "estimated_lon", "success": "success"},
    {"name": "SuperPoint+LightGlue", "type": "sparse deep", "file": "lightglue_anchors.csv",
     "t_field": None, "frame_field": "frame_idx",       # frame_idx -> timestamp via frames dir
     "lat": "est_lat", "lon": "est_lon", "success": None},
    {"name": "RoMa", "type": "dense deep", "file": "roma_anchors.csv",
     "t_field": "frame_ms", "t_scale": 1.0,
     "lat": "est_lat", "lon": "est_lon", "success": None},
]


def _frame_timestamps():
    from gps_benchmark import load_frame_timestamps
    return load_frame_timestamps(ROOT / "data" / "frames_15hz", n_needed=4000)


def score_source(src, fixes, frame_ts) -> dict:
    rows = list(csv.DictReader(open(EST_DIR / src["file"])))
    errs, attempted = [], 0
    for r in rows:
        attempted += 1
        if src["success"] and r.get(src["success"], "True").strip().lower() not in ("true", "1"):
            continue
        try:
            lat, lon = float(r[src["lat"]]), float(r[src["lon"]])
        except (ValueError, KeyError):
            continue
        if src["t_field"]:
            t_ms = int(round(float(r[src["t_field"]]) * src["t_scale"]))
        else:
            fi = int(float(r[src["frame_field"]]))
            if fi >= len(frame_ts):
                continue
            t_ms = int(frame_ts[fi])
        truth = interp_truth(fixes, t_ms)
        if truth is None:
            continue
        errs.append(haversine_m(lat, lon, *truth))
    e = np.array(errs)
    return {
        "name": src["name"], "type": src["type"],
        "attempted": attempted, "scored": len(e),
        "median": float(np.median(e)), "mean": float(np.mean(e)),
        "p90": float(np.percentile(e, 90)),
        "within15": int((e <= 15).sum()), "pct15": float(100 * (e <= 15).mean()),
    }


def main():
    fixes = parse_dji_srt(SRT)
    frame_ts = _frame_timestamps()
    results = [score_source(s, fixes, frame_ts) for s in SOURCES]

    print(f"{'Matcher':22} {'Type':16} {'scored':>7} {'median':>8} {'p90':>8} {'within15':>10}")
    print("-" * 78)
    for r in sorted(results, key=lambda r: r["median"]):
        print(f"{r['name']:22} {r['type']:16} {r['scored']:>7} "
              f"{r['median']:>7.1f}m {r['p90']:>7.1f}m {r['pct15']:>8.0f}% ")
    print("-" * 78)
    print(f"{'Manual anchoring':22} {'VO + human marks':16} {'3430':>7} "
          f"{4.1:>7.1f}m {'—':>8} {98:>8}%   (not automated; the deployed solution)")


if __name__ == "__main__":
    main()
