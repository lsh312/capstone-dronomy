"""
Heading / orientation benchmark.

The brief asks for absolute *orientation* as well as position. This scores a
matcher's estimated heading (`est_yaw`) against the DJI gimbal-yaw truth
(`gb_yaw`) parsed from the .SRT, through the same timestamp alignment used for
position.

For each estimate it reports:
  * raw heading error (circular, 0-180 deg), and
  * error after removing the single best constant offset.

The offset matters: if subtracting one constant collapses the error, the
estimator is informative but mis-referenced (a fixable axis/convention issue);
if it doesn't, the heading is genuinely uninformative on this data.

Only RoMa logged an absolute per-frame yaw, so it's the one automated matcher
scorable here; SuperPoint+LightGlue and SIFT did not record orientation.

Run:
    cd code && uv run python heading_benchmark.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gps_benchmark import (
    best_yaw_offset_deg,
    heading_error_deg,
    interp_truth_yaw,
    parse_dji_srt,
)

ROOT = Path(__file__).resolve().parent.parent
SRT = ROOT / "data" / "gps_data.SRT"

# (label, csv, timestamp field, timestamp->ms scale, yaw field)
SOURCES = [
    {"name": "RoMa", "file": ROOT / "results" / "roma_v2" / "accuracy_roma.csv",
     "t_field": "frame_ms", "t_scale": 1.0, "yaw": "est_yaw"},
]


def score_heading(src, fixes) -> dict:
    rows = list(csv.DictReader(open(src["file"])))
    raw_err, signed = [], []
    for r in rows:
        try:
            est = float(r[src["yaw"]])
        except (ValueError, KeyError):
            continue
        t_ms = int(round(float(r[src["t_field"]]) * src["t_scale"]))
        truth = interp_truth_yaw(fixes, t_ms)
        if truth is None:
            continue
        raw_err.append(heading_error_deg(est, truth))
        # signed (est - truth) wrapped to [-180, 180] for offset estimation
        signed.append((est - truth + 180.0) % 360.0 - 180.0)
    raw = np.array(raw_err)
    signed = np.array(signed)
    offset = best_yaw_offset_deg(signed)
    corrected = np.abs((signed - offset + 180.0) % 360.0 - 180.0)
    return {
        "name": src["name"], "scored": len(raw),
        "raw_median": float(np.median(raw)), "raw_mean": float(np.mean(raw)),
        "offset": offset,
        "corr_median": float(np.median(corrected)),
        "corr_mean": float(np.mean(corrected)),
        "corr_within30": float(100 * (corrected <= 30).mean()),
    }


def main():
    fixes = parse_dji_srt(SRT)
    n_yaw = sum(1 for f in fixes if np.isfinite(f.yaw))
    print(f"Parsed {len(fixes)} SRT fixes ({n_yaw} with gb_yaw truth)\n")

    print(f"{'Matcher':10} {'scored':>7} {'raw median':>11} {'best offset':>12} "
          f"{'corrected median':>17} {'within 30 deg':>14}")
    print("-" * 76)
    for src in SOURCES:
        r = score_heading(src, fixes)
        print(f"{r['name']:10} {r['scored']:>7} {r['raw_median']:>9.1f} deg "
              f"{r['offset']:>9.1f} deg {r['corr_median']:>14.1f} deg "
              f"{r['corr_within30']:>11.0f} %")
    print("-" * 76)
    print("raw = est vs gb_yaw as-is; corrected = after removing the single best "
          "constant offset.\nIf corrected << raw, the heading is informative but "
          "mis-referenced (fixable); if not, it's uninformative.")


if __name__ == "__main__":
    main()
