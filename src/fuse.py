"""
Fuse the VO trajectory with manual (and optionally automated) anchors
to produce a fully geolocalized flight path.

Reads:
    results/run_<ts>/dl_vo_log_full.csv   (from vo_run.py)
    data/manual_anchors.json              ({frame_idx: [lat, lon]} human marks)
    data/satellite_*.png + .bbox.json     (optional, for trajectory overlay)
    data/gps_data.SRT                     (optional, for GPS ground-truth benchmark)

Also accepts automated anchors from sat_anchors.py to augment manual ones.

Writes:
    results/run_<ts>/fused_track_<ts>.csv
    results/run_<ts>/fused_trajectory_<ts>.png
    results/run_<ts>/benchmark_fused_<ts>.csv/.png   (if SRT present)

Usage:
    uv run python src/fuse.py
    uv run python src/fuse.py --anchors data/manual_anchors.json
    uv run python src/fuse.py --sat-anchors results/run_<ts>/sat_anchors_<ts>.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import manual_anchors as ma
import gps_benchmark as gpsb

NOM_LAT = 43.521955
NOM_LON = -5.624290


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_vo_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "frame_idx": int(row["frame_idx"]),
                "cum_x":     float(row["cum_x"]),
                "cum_y":     float(row["cum_y"]),
                "cum_yaw":   float(row["cum_yaw"]),
            })
    return rows


def load_manual_anchors_json(path: Path) -> dict[int, tuple[float, float]]:
    raw = json.loads(path.read_text())
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_sat_anchors_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "frame_idx": int(row["frame_idx"]),
                "est_lat":   float(row["est_lat"]),
                "est_lon":   float(row["est_lon"]),
                "vo_cum_x":  float(row["vo_cum_x"]),
                "vo_cum_y":  float(row["vo_cum_y"]),
                "source":    "satellite",
            })
    return rows


def find_tile(data_dir: Path) -> tuple[np.ndarray | None, dict | None]:
    """Return (rgb_image, bbox_dict) for the best available satellite tile."""
    def priority(p: Path) -> tuple[int, int]:
        n = p.stem
        return (0 if "google" in n else 1, -int(n.split("_z")[-1].split(".")[0]) if "_z" in n else 0)
    for sc in sorted(data_dir.glob("satellite_*.bbox.json"), key=priority):
        tile = sc.parent / sc.name.replace(".bbox.json", ".png")
        if tile.exists():
            img  = cv2.cvtColor(cv2.imread(str(tile)), cv2.COLOR_BGR2RGB)
            bbox = json.loads(sc.read_text())
            return img, bbox
    return None, None


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_trajectory(vo_records: list[dict], fit: ma.WorldFit,
                    fused_track: list[tuple], anchor_list: list[dict],
                    sat_img: np.ndarray | None, bbox: dict | None,
                    out_path: Path) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: ENU metres with VO path + fused path + anchors ──
    ax = axes[0]
    en = []
    for r in vo_records:
        p = np.array([r["cum_x"], r["cum_y"]], dtype=float)
        en.append((fit.scale_m_per_px * (fit.R @ p)) + fit.t)
    xs = [e for e, n in en]
    ys = [n for e, n in en]

    colors = cm.plasma(np.linspace(0, 1, len(xs)))
    for i in range(len(xs) - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], color=colors[i], lw=1)
    ax.scatter(xs[0],  ys[0],  c="lime", s=80, zorder=5, label="Start")
    ax.scatter(xs[-1], ys[-1], c="red",  s=80, zorder=5, label="End")

    fused_en = [ma.gps_to_local_m(lat, lon, fit.lat0, fit.lon0)
                for (_, lat, lon) in fused_track]
    ax.plot([e for e, n in fused_en], [n for e, n in fused_en],
            "-", c="cyan", lw=1.5, label="Fused (drift-corrected)")

    for a in anchor_list:
        aen = ma.gps_to_local_m(a["est_lat"], a["est_lon"], fit.lat0, fit.lon0)
        clr = "yellow" if a.get("source", "manual") == "manual" else "aqua"
        ax.scatter(*aen, c=clr, s=60, marker="^", edgecolors="k", zorder=6)
    # dummy entries for legend
    ax.scatter([], [], c="yellow", marker="^", edgecolors="k", s=60, label="Manual anchor")
    ax.scatter([], [], c="aqua",   marker="^", edgecolors="k", s=60, label="Sat anchor")

    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("VO path + fused path (colour = time)")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Right: satellite tile overlay ──
    ax2 = axes[1]
    if sat_img is not None and bbox is not None:
        h, w = sat_img.shape[:2]

        def to_px(lat, lon):
            x = (lon - bbox["lon_min"]) / (bbox["lon_max"] - bbox["lon_min"]) * w
            y = (bbox["lat_max"] - lat) / (bbox["lat_max"] - bbox["lat_min"]) * h
            return x, y

        ax2.imshow(sat_img, extent=[0, w, h, 0])
        fx = [to_px(lat, lon)[0] for (_, lat, lon) in fused_track]
        fy = [to_px(lat, lon)[1] for (_, lat, lon) in fused_track]
        ax2.plot(fx, fy, "-", c="cyan", lw=1.5, label="Fused path")
        for a in anchor_list:
            mx, my = to_px(a["est_lat"], a["est_lon"])
            clr = "yellow" if a.get("source", "manual") == "manual" else "aqua"
            ax2.plot(mx, my, "^", c=clr, ms=8, mec="k")
        ax2.set_xlim(0, w); ax2.set_ylim(h, 0); ax2.axis("off")
        ax2.set_title(f"Fused path over satellite tile ({bbox['provider']})")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No satellite tile\n(run fetch_satellite.py)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_title("Satellite overlay")

    plt.suptitle(
        f"Drone geolocalization — scale={fit.scale_m_per_px:.4f} m/px  "
        f"rot={fit.rotation_deg:.1f}°  residual={fit.residual_m:.1f} m  n={fit.n}",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ── Core fusion logic ─────────────────────────────────────────────────────────

def run_fusion(args, run_dir: Path) -> None:
    """Load anchors + VO, fuse, write results to run_dir."""
    vo_records = load_vo_csv(args.vo_csv)

    manual_dict = load_manual_anchors_json(args.anchors)
    anchor_list = ma.make_manual_anchors(manual_dict, vo_records)

    if args.sat_anchors and args.sat_anchors.exists():
        sat_ancs = load_sat_anchors_csv(args.sat_anchors)
        existing = {a["frame_idx"] for a in anchor_list}
        added    = [sa for sa in sat_ancs if sa["frame_idx"] not in existing]
        anchor_list.extend(added)
        anchor_list.sort(key=lambda a: a["frame_idx"])
        print(f"  {len(anchor_list)} anchors total "
              f"({len(manual_dict)} manual + {len(added)} satellite)")
    else:
        print(f"  {len(anchor_list)} manual anchors")

    if len(anchor_list) < 2:
        raise ValueError(f"Need ≥2 anchors, got {len(anchor_list)}")

    fit         = ma.fit_vo_to_world(anchor_list)
    fused_track = ma.fuse_vo_with_anchors(vo_records, anchor_list, fit)

    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "fused_track.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame_idx", "est_lat", "est_lon"])
        w.writeheader()
        for fi, lat, lon in fused_track:
            w.writerow({"frame_idx": fi, "est_lat": f"{lat:.7f}", "est_lon": f"{lon:.7f}"})
    print(f"  wrote {out_csv.name}  ({len(fused_track)} rows)  "
          f"scale={fit.scale_m_per_px:.4f} m/px  residual={fit.residual_m:.1f} m")

    sat_img, bbox = find_tile(args.data_dir)
    plot_trajectory(vo_records, fit, fused_track, anchor_list,
                    sat_img, bbox, run_dir / "fused_trajectory.png")

    if args.srt.exists():
        fixes    = gpsb.parse_dji_srt(args.srt, nominal_lat=NOM_LAT, nominal_lon=NOM_LON)
        max_fi   = max(fi for fi, _, _ in fused_track)
        frame_ts = gpsb.load_frame_timestamps(args.frames_dir, n_needed=max_fi + 1)
        scored, summary = gpsb.benchmark_estimates(
            list(fused_track), fixes, frame_ts, good_m=args.good_m)
        gpsb.print_summary(summary, "fused trajectory")
        gpsb.write_scored_csv(scored, run_dir / "benchmark.csv")
        gpsb.plot_report(scored, fixes, summary,
                         run_dir / "benchmark.png",
                         title="Fused trajectory vs DJI GPS")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse VO + anchors → geolocalized trajectory")
    ap.add_argument("--vo-csv",        type=Path, default=ROOT / "results/run_20260623_093123/dl_vo_log_full.csv")
    ap.add_argument("--anchors",       type=Path, default=ROOT / "data/manual_anchors.json",
                    help="JSON of {frame_idx: [lat, lon]} manual marks")
    ap.add_argument("--sat-anchors",   type=Path, default=None,
                    help="Optional sat_anchors.py CSV to augment manual marks")
    ap.add_argument("--data-dir",      type=Path, default=ROOT / "data")
    ap.add_argument("--srt",           type=Path, default=ROOT / "data/gps_data.SRT",
                    help="DJI .SRT for GPS benchmark (optional)")
    ap.add_argument("--frames-dir",    type=Path, default=ROOT / "data/frames_15hz")
    ap.add_argument("--out-dir",       type=Path, default=ROOT / "results/run_20260623_093123")
    ap.add_argument("--good-m",        type=float, default=15.0,
                    help="error threshold (m) for the GPS benchmark")
    ap.add_argument("--watch",         action="store_true",
                    help="re-fuse automatically whenever the anchor CSV changes")
    ap.add_argument("--watch-interval", type=float, default=15.0,
                    help="seconds between polls in --watch mode (default 15)")
    args = ap.parse_args()

    if not args.vo_csv.exists():
        sys.exit(f"VO CSV not found: {args.vo_csv}\nRun vo_run.py first.")
    if not args.anchors.exists():
        sys.exit(f"Manual anchors not found: {args.anchors}")

    if args.watch:
        # Live mode: write to a fixed folder so the dashboard always knows where to look
        run_dir   = args.out_dir / "fuse_live"
        last_sigs: dict[str, float] = {}
        print(f"Watch mode — polling every {args.watch_interval}s")
        print(f"Output: {run_dir}\n")
        while True:
            sigs: dict[str, float] = {}
            if args.sat_anchors and args.sat_anchors.exists():
                sigs["sat"] = args.sat_anchors.stat().st_mtime
            if args.anchors.exists():
                sigs["manual"] = args.anchors.stat().st_mtime
            if sigs != last_sigs:
                ts_str = datetime.now().strftime("%H:%M:%S")
                n_sat  = 0
                if args.sat_anchors and args.sat_anchors.exists():
                    with open(args.sat_anchors, newline="") as f:
                        n_sat = sum(1 for _ in f) - 1
                print(f"[{ts_str}] anchor change detected "
                      f"(sat={n_sat}) — re-fusing …")
                try:
                    run_fusion(args, run_dir)
                    last_sigs = dict(sigs)
                except Exception as e:
                    print(f"  skipped: {e}")
            _time.sleep(args.watch_interval)
    else:
        # Normal mode: timestamped subfolder
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.out_dir / f"fuse_{ts}"
        print(f"Saving to {run_dir}")
        run_fusion(args, run_dir)


if __name__ == "__main__":
    main()
