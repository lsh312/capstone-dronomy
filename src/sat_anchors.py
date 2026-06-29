"""
Automated satellite anchor matching using SuperPoint + LightGlue.

Normal mode: reads the completed VO CSV, matches all candidate frames at once.
Watch mode:  runs in parallel with vo_run.py — polls the VO CSV for new rows
             and matches each candidate frame as soon as VO has processed it.
             Exits automatically when vo_run.py writes its .done sentinel.

Reads:
    results/run_<ts>/dl_vo_log_full.csv   (from vo_run.py)
    data/satellite_*_z<N>.png             (from fetch_satellite.py)
    data/satellite_*_z<N>.bbox.json       (sidecar bounding box)

Writes:
    results/run_<ts>/sat_anchors_<ts>.csv

Usage:
    uv run python src/sat_anchors.py
    uv run python src/sat_anchors.py --watch-vo          # parallel with vo_run.py
    uv run python src/sat_anchors.py --min-inliers 10 --hz 0.5
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

NOM_LAT = 43.521955
NOM_LON = -5.624290
DOWNSAMPLE_SIZE = (640, 480)


def setup_device(override: str = "auto") -> torch.device:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vo_csv_safe(path: Path) -> list[dict]:
    """Read VO CSV, silently skipping any partial rows (safe for concurrent reads)."""
    rows = []
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows.append({
                        "frame_idx": int(row["frame_idx"]),
                        "cum_x":     float(row["cum_x"]),
                        "cum_y":     float(row["cum_y"]),
                    })
                except (ValueError, KeyError):
                    pass
    except OSError:
        pass
    return rows


def find_tile(data_dir: Path) -> tuple[Path, dict] | tuple[None, None]:
    def priority(p: Path) -> tuple[int, int]:
        name = p.stem
        prov = 0 if "google" in name else 1
        zoom = int(name.split("_z")[-1].split(".")[0]) if "_z" in name else 0
        return (prov, -zoom)
    for sc in sorted(data_dir.glob("satellite_*.bbox.json"), key=priority):
        tile = sc.parent / sc.name.replace(".bbox.json", ".png")
        if tile.exists():
            return tile, json.loads(sc.read_text())
    return None, None


def pixel_to_gps(px: float, py: float, img_w: int, img_h: int, bbox: dict) -> tuple[float, float]:
    lat = bbox["lat_max"] - (py / img_h) * (bbox["lat_max"] - bbox["lat_min"])
    lon = bbox["lon_min"] + (px / img_w) * (bbox["lon_max"] - bbox["lon_min"])
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def match_frame(fi: int, frame_paths: list[Path],
                extractor, matcher, sat_feats,
                sat_h: int, sat_w: int, bbox: dict,
                by_frame: dict, args) -> dict | None:
    """Match one drone frame against the satellite tile. Returns anchor row or None."""
    from lightglue.utils import rbd

    if fi >= len(frame_paths):
        return None
    if fi not in by_frame:
        return None

    img = cv2.cvtColor(cv2.imread(str(frame_paths[fi])), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, DOWNSAMPLE_SIZE, interpolation=cv2.INTER_AREA)
    drone_t     = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    drone_feats = extractor.extract(drone_t.to(next(extractor.parameters()).device))

    res = matcher({"image0": drone_feats, "image1": sat_feats})
    f0, f1, r = rbd(drone_feats), rbd(sat_feats), rbd(res)
    m = r["matches"]
    if len(m) < 4:
        return None

    pts_d  = f0["keypoints"][m[:, 0]].cpu().numpy()
    pts_s  = f1["keypoints"][m[:, 1]].cpu().numpy()
    scores = r.get("matching_scores", torch.ones(len(m))).cpu().numpy()

    src = pts_d.reshape(-1, 1, 2).astype(np.float32)
    dst = pts_s.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if (H is not None and mask is not None) else 0
    if H is None or inliers < args.min_inliers:
        return None

    dh, dw = DOWNSAMPLE_SIZE[1], DOWNSAMPLE_SIZE[0]
    centre = np.float32([[dw / 2, dh / 2]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(centre, H).reshape(2)
    est_lat, est_lon = pixel_to_gps(mapped[0], mapped[1], sat_w, sat_h, bbox)

    dist_m = haversine_m(est_lat, est_lon, NOM_LAT, NOM_LON)
    if dist_m > args.max_dist_km * 1000:
        return None

    vo_rec = by_frame[fi]
    return {
        "frame_idx":      fi,
        "est_lat":        est_lat,
        "est_lon":        est_lon,
        "good_matches":   len(m),
        "inliers":        inliers,
        "mean_score":     float(scores.mean()),
        "dist_nominal_m": dist_m,
        "vo_cum_x":       vo_rec["cum_x"],
        "vo_cum_y":       vo_rec["cum_y"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Satellite anchor matching (SuperPoint + LightGlue)")
    ap.add_argument("--vo-csv",      type=Path,  default=ROOT / "results/run_20260623_093123/dl_vo_log_full.csv")
    ap.add_argument("--frames-dir",  type=Path,  default=ROOT / "data/frames_15hz")
    ap.add_argument("--data-dir",    type=Path,  default=ROOT / "data")
    ap.add_argument("--out-dir",     type=Path,  default=ROOT / "results/run_20260623_093123")
    ap.add_argument("--hz",          type=float, default=1.0,  help="matching frequency (Hz)")
    ap.add_argument("--fps",         type=float, default=15.0, help="frame extraction rate")
    ap.add_argument("--min-inliers", type=int,   default=8)
    ap.add_argument("--max-dist-km", type=float, default=2.0)
    ap.add_argument("--max-kp",      type=int,   default=4096)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--watch-vo",    action="store_true",
                    help="run in parallel with vo_run.py, polling for new frames")
    ap.add_argument("--watch-poll",  type=float, default=10.0,
                    help="seconds between VO CSV polls in --watch-vo mode")
    args = ap.parse_args()

    device = setup_device(args.device)
    print(f"Device: {device}", flush=True)

    # Satellite tile
    tile_path, bbox = find_tile(args.data_dir)
    if tile_path is None:
        raise SystemExit("No satellite tile found — run fetch_satellite.py first.")
    print(f"Tile: {tile_path.name}  zoom={bbox['zoom']}  provider={bbox['provider']}", flush=True)

    # Load models + extract satellite features once
    from lightglue import LightGlue, SuperPoint
    torch.set_grad_enabled(False)
    extractor = SuperPoint(max_num_keypoints=args.max_kp).eval().to(device)
    matcher   = LightGlue(features="superpoint",
                          depth_confidence=0.9, width_confidence=0.95).eval().to(device)
    sat_img   = cv2.cvtColor(cv2.imread(str(tile_path)), cv2.COLOR_BGR2RGB)
    sat_h, sat_w = sat_img.shape[:2]
    sat_t     = torch.from_numpy(sat_img).float().permute(2, 0, 1)[None] / 255.0
    sat_feats = extractor.extract(sat_t.to(device))
    print(f"Satellite keypoints: {sat_feats['keypoints'].shape[1]}", flush=True)

    frame_paths = sorted(args.frames_dir.glob("*.jpg"))
    every       = max(1, int(args.fps / args.hz))

    # Output CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_dir / f"sat_anchors_{ts}.csv"
    fields  = ["frame_idx", "est_lat", "est_lon", "good_matches",
               "inliers", "mean_score", "dist_nominal_m", "vo_cum_x", "vo_cum_y"]
    csv_fh  = open(out_csv, "w", newline="")
    csv_w   = csv.DictWriter(csv_fh, fieldnames=fields)
    csv_w.writeheader()
    csv_fh.flush()
    print(f"Writing anchors to {out_csv}", flush=True)

    anchors: list[dict] = []
    t_start = time.perf_counter()

    if args.watch_vo:
        # ── Parallel mode: poll VO CSV as it grows ─────────────────────────────
        sentinel  = args.vo_csv.with_suffix(".done")
        processed: set[int] = set()

        print(f"Watch mode: polling every {args.watch_poll}s "
              f"(sentinel: {sentinel.name})\n", flush=True)

        # Wait for VO to create the CSV
        while not args.vo_csv.exists():
            time.sleep(args.watch_poll)

        while True:
            vo_rows  = load_vo_csv_safe(args.vo_csv)
            by_frame = {r["frame_idx"]: r for r in vo_rows}
            new_cands = sorted([
                r["frame_idx"] for r in vo_rows
                if r["frame_idx"] % every == 0
                and r["frame_idx"] not in processed
            ])

            for fi in new_cands:
                row = match_frame(fi, frame_paths, extractor, matcher,
                                  sat_feats, sat_h, sat_w, bbox, by_frame, args)
                if row:
                    anchors.append(row)
                    csv_w.writerow(row)
                    csv_fh.flush()
                    print(f"  anchor frame {fi:>5}  "
                          f"lat={row['est_lat']:.6f}  lon={row['est_lon']:.6f}  "
                          f"inliers={row['inliers']}", flush=True)
                processed.add(fi)

            if sentinel.exists():
                break
            time.sleep(args.watch_poll)

    else:
        # ── Normal mode: process all VO rows at once ───────────────────────────
        if not args.vo_csv.exists():
            raise SystemExit(f"VO CSV not found: {args.vo_csv}\nRun vo_run.py first.")
        vo_rows  = load_vo_csv_safe(args.vo_csv)
        by_frame = {r["frame_idx"]: r for r in vo_rows}
        candidates = [r["frame_idx"] for r in vo_rows if r["frame_idx"] % every == 0]
        print(f"Matching {len(candidates)} frames  "
              f"(every {every} frames = {args.hz} Hz)  "
              f"min_inliers={args.min_inliers}\n", flush=True)

        for i, fi in enumerate(candidates):
            row = match_frame(fi, frame_paths, extractor, matcher,
                              sat_feats, sat_h, sat_w, bbox, by_frame, args)
            if row:
                anchors.append(row)
                csv_w.writerow(row)
                csv_fh.flush()
            if (i + 1) % 20 == 0 or (i + 1) == len(candidates):
                elapsed = time.perf_counter() - t_start
                print(f"  {i+1:>4}/{len(candidates)}  anchors: {len(anchors)}  "
                      f"{elapsed/max(1,i+1)*1000:.0f} ms/frame  "
                      f"elapsed {elapsed:.0f}s", flush=True)

    csv_fh.close()
    elapsed = time.perf_counter() - t_start
    print(f"\nDone — {len(anchors)} anchors  ({elapsed:.1f}s)  -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
