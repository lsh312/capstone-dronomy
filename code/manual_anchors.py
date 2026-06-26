"""
Manual anchoring + VO-to-world alignment for the drone geolocalization pipeline.

Motivation
----------
Automated satellite anchors are noisy (~100 m median error vs DJI GPS truth) and
the old scale calibration — GPS distance / VO pixel distance, per anchor pair —
is unstable because VO drift makes the ratio explode for pairs far apart in time.

A handful of *human-verified* anchors fixes both problems. A person looks at a
few drone frames, finds each location on the satellite map / Google Earth, and
records its lat/lon. With those, we fit a single similarity transform
(scale + rotation + translation) that maps the whole VO trajectory into world
coordinates by least squares — far more robust than pairwise ratios, and it also
yields the path in real lat/lon for the dashboard.

Integrity note
--------------
Manual anchors must come from a human reading the imagery, NOT from the GPS file.
`validate_manual_anchors` uses the DJI GPS only to *check* a human's marks — it is
never used to create them. The localization pipeline itself stays GPS-free.

Anchors produced here use the same dict schema as the notebook's automated
anchors (frame_idx, est_lat, est_lon, vo_cum_x, vo_cum_y, ...), so they drop into
the existing scale / drift / fusion cells unchanged.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# Local-tangent-plane constants (match the notebook's Cell 7 sanity check).
M_PER_DEG_LAT = 111_320.0


# --------------------------------------------------------------------------- #
# Local ENU projection (equirectangular about a reference point)
# --------------------------------------------------------------------------- #
def gps_to_local_m(lat: float, lon: float,
                   lat0: float, lon0: float) -> tuple[float, float]:
    """Project lat/lon to local east/north metres about (lat0, lon0)."""
    east = (lon - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
    north = (lat - lat0) * M_PER_DEG_LAT
    return east, north


def local_m_to_gps(east: float, north: float,
                   lat0: float, lon0: float) -> tuple[float, float]:
    """Inverse of gps_to_local_m."""
    lat = lat0 + north / M_PER_DEG_LAT
    lon = lon0 + east / (M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat, lon


# --------------------------------------------------------------------------- #
# Build pipeline-compatible anchors from human marks
# --------------------------------------------------------------------------- #
def make_manual_anchors(manual_dict: dict[int, tuple[float, float]],
                        vo_records: list[dict]) -> list[dict]:
    """Turn {frame_idx: (lat, lon)} human marks into pipeline anchor dicts.

    Pulls the VO cumulative pixel position (cum_x, cum_y) for each marked frame
    from vo_records so the anchors plug straight into scale / fusion cells.
    """
    by_idx = {r["frame_idx"]: r for r in vo_records}
    anchors: list[dict] = []
    for frame_idx, (lat, lon) in sorted(manual_dict.items()):
        rec = by_idx.get(frame_idx)
        if rec is None:
            raise KeyError(
                f"frame_idx {frame_idx} not found in vo_records "
                f"(range {vo_records[0]['frame_idx']}..{vo_records[-1]['frame_idx']})")
        anchors.append({
            "frame_idx": frame_idx,
            "est_lat": lat,
            "est_lon": lon,
            "vo_cum_x": rec["cum_x"],
            "vo_cum_y": rec["cum_y"],
            "source": "manual",
        })
    return anchors


# --------------------------------------------------------------------------- #
# VO -> world similarity fit (the proper scale calibration)
# --------------------------------------------------------------------------- #
class WorldFit(NamedTuple):
    scale_m_per_px: float       # uniform scale
    R: np.ndarray               # 2x2 rotation/reflection (VO axes -> world axes)
    t: np.ndarray               # 2 translation (m), in local ENU
    lat0: float                 # reference origin for the local ENU frame
    lon0: float
    residual_m: float           # RMS fit residual at the anchors (m)
    n: int                      # number of anchors used
    rotation_deg: float         # heading of VO +x axis in world frame
    reflected: bool             # whether VO frame is mirrored vs world


def fit_vo_to_world(anchors: list[dict],
                    allow_reflection: bool = True) -> WorldFit:
    """Least-squares similarity transform mapping VO pixels -> world metres.

    Solves for (scale s, rotation R, translation t) minimising
        || s * R @ p_vo + t  -  p_world ||
    over all anchors (Umeyama, 2D). Needs >= 2 anchors; the residual is only
    meaningful with >= 3. VO image axes are often mirrored vs world (y-down),
    so reflection is allowed by default.
    """
    if len(anchors) < 2:
        raise ValueError(f"need >= 2 anchors to fit scale, got {len(anchors)}")

    lat0 = float(np.mean([a["est_lat"] for a in anchors]))
    lon0 = float(np.mean([a["est_lon"] for a in anchors]))

    src = np.array([[a["vo_cum_x"], a["vo_cum_y"]] for a in anchors], dtype=np.float64)
    dst = np.array([gps_to_local_m(a["est_lat"], a["est_lon"], lat0, lon0)
                    for a in anchors], dtype=np.float64)

    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / len(src)

    cov = (dc.T @ sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(2)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else float("nan")
    t = mu_d - scale * (R @ mu_s)

    # Fit residual (RMS over anchors).
    pred = (scale * (src @ R.T)) + t
    residual = float(np.sqrt(np.mean(((pred - dst) ** 2).sum(axis=1))))

    rotation_deg = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))
    reflected = bool(np.linalg.det(R) < 0)
    return WorldFit(scale, R, t, lat0, lon0, residual, len(anchors),
                    rotation_deg, reflected)


def vo_to_gps(cum_x: float, cum_y: float, fit: WorldFit) -> tuple[float, float]:
    """Map a VO cumulative pixel position to lat/lon using a fitted transform."""
    p = np.array([cum_x, cum_y], dtype=np.float64)
    east, north = (fit.scale_m_per_px * (fit.R @ p)) + fit.t
    return local_m_to_gps(float(east), float(north), fit.lat0, fit.lon0)


def vo_track_to_gps(vo_records: list[dict], fit: WorldFit) -> list[tuple[int, float, float]]:
    """Map an entire VO trajectory to (frame_idx, lat, lon)."""
    return [(r["frame_idx"], *vo_to_gps(r["cum_x"], r["cum_y"], fit))
            for r in vo_records]


def fuse_vo_with_anchors(vo_records: list[dict], anchors: list[dict],
                         fit: WorldFit) -> list[tuple[int, float, float]]:
    """Drift-correct the VO track using anchors, piecewise between fixes.

    A global similarity fit (`fit`) sets the overall scale + orientation, but VO
    drift means one global transform can't line the path up everywhere. Here we
    additionally cancel the residual at each anchor and linearly interpolate that
    residual correction along the track between consecutive anchors (held flat
    before the first and after the last). At each anchor the output equals the
    marked position; between anchors only the *change* in drift remains.
    """
    anchors = sorted(anchors, key=lambda a: a["frame_idx"])

    # Base world positions for the whole track and at each anchor frame.
    def base_world(rec):
        p = np.array([rec["cum_x"], rec["cum_y"]], dtype=np.float64)
        return (fit.scale_m_per_px * (fit.R @ p)) + fit.t

    a_frames = np.array([a["frame_idx"] for a in anchors], dtype=np.float64)
    by_idx = {r["frame_idx"]: r for r in vo_records}
    corrections = []
    for a in anchors:
        rec = by_idx[a["frame_idx"]]
        target = np.array(gps_to_local_m(a["est_lat"], a["est_lon"], fit.lat0, fit.lon0))
        corrections.append(target - base_world(rec))
    corrections = np.array(corrections)  # (n_anchors, 2)

    out: list[tuple[int, float, float]] = []
    for rec in vo_records:
        bw = base_world(rec)
        fi = rec["frame_idx"]
        cx = np.interp(fi, a_frames, corrections[:, 0])
        cy = np.interp(fi, a_frames, corrections[:, 1])
        east, north = bw + np.array([cx, cy])
        lat, lon = local_m_to_gps(float(east), float(north), fit.lat0, fit.lon0)
        out.append((fi, lat, lon))
    return out


# --------------------------------------------------------------------------- #
# Validate human marks against DJI GPS truth (QA only — never to create anchors)
# --------------------------------------------------------------------------- #
def validate_manual_anchors(anchors: list[dict], srt_path,
                            frames_dir, good_m: float = 15.0) -> dict:
    """Report each manual anchor's error vs DJI GPS — a check that the human
    marked accurately. Returns the gps_benchmark summary.
    """
    import gps_benchmark as gpsb

    fixes = gpsb.parse_dji_srt(srt_path)
    max_idx = max(a["frame_idx"] for a in anchors)
    frame_ts = gpsb.load_frame_timestamps(frames_dir, n_needed=max_idx + 1)
    estimates = [(a["frame_idx"], a["est_lat"], a["est_lon"]) for a in anchors]
    scored, summary = gpsb.benchmark_estimates(estimates, fixes, frame_ts, good_m=good_m)
    for a, s in zip(anchors, scored):
        a["truth_error_m"] = s.error_m
    return summary


# --------------------------------------------------------------------------- #
# Interactive helper: click locations on the satellite tile (best-effort)
# --------------------------------------------------------------------------- #
def click_anchors_on_tile(sat_tile, bbox_sat, frame_indices,
                          frame_paths=None, downsample=None) -> dict[int, tuple[float, float]]:
    """Interactively mark drone-frame locations on the satellite tile.

    For each frame index, shows the drone frame (if available) next to the
    satellite tile; click the point on the tile where the drone is. Returns
    {frame_idx: (lat, lon)} via the tile bbox. Requires an interactive
    matplotlib backend (e.g. `%matplotlib widget` / `%matplotlib qt`); falls
    back gracefully if clicking is unavailable.

    bbox_sat is (lat_min, lat_max, lon_min, lon_max), as produced by the
    notebook's tile_bbox().
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    lat_min, lat_max, lon_min, lon_max = bbox_sat
    h, w = sat_tile.shape[:2]
    marks: dict[int, tuple[float, float]] = {}

    for fi in frame_indices:
        ncols = 2 if frame_paths is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
        axes = np.atleast_1d(axes)
        if frame_paths is not None:
            drone = np.array(Image.open(frame_paths[fi]).convert("RGB"))
            axes[0].imshow(drone)
            axes[0].set_title(f"drone frame {fi}")
            axes[0].axis("off")
        tile_ax = axes[-1]
        tile_ax.imshow(sat_tile)
        tile_ax.set_title("click drone location on tile")
        tile_ax.axis("off")
        plt.tight_layout()

        pts = plt.ginput(1, timeout=0)
        plt.close(fig)
        if not pts:
            print(f"  frame {fi}: no click — skipped")
            continue
        px, py = pts[0]
        lat = lat_max - (py / h) * (lat_max - lat_min)
        lon = lon_min + (px / w) * (lon_max - lon_min)
        marks[fi] = (lat, lon)
        print(f"  frame {fi}: marked lat {lat:.6f} lon {lon:.6f}")

    return marks
