# Notebook 3 v2 — Deep Learning Geolocalization: Explainer

## Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: drone video  (30 fps, 1920×1080, ~3.8 min)                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    Cell 0b — Frame Extraction
                    Downsample to 15 Hz, cap at 1920 px wide,
                    JPEG quality 75 → ~3 430 frames on disk
                                 │
          ┌──────────────────────┴──────────────────────┐
          │                                             │
   Every frame (15 Hz)                        Every 15th frame (1 Hz)
   ── VO branch ──                            ── Satellite branch ──
          │                                             │
   Cell 4                                     Cell 6
   Resize frame to 640×480                    Fetch ESRI World Imagery tile
   SuperPoint → keypoints + descriptors       (zoom 18, 640×640 px, ~385 m coverage)
   LightGlue → match vs. previous frame       SuperPoint → keypoints (max 4 096)
   RANSAC homography → dx, dy, dyaw                     │
   Accumulate: cum_x, cum_y, cum_yaw          Cell 7
          │                                   Resize drone frame to 640×480
   Cell 5                                     SuperPoint → keypoints (max 4 096)
   Quality metrics per frame pair             LightGlue → match drone vs. satellite
   (inlier ratio, match count, LG score)      RANSAC homography → drone position on tile
          │                                   pixel_to_gps → estimated lat/lon
          │                                   GPS sanity check (< 2 km from nominal)
          │                                   → accepted anchor (or rejected)
          │                                             │
          └──────────────────────┬──────────────────────┘
                                 │
                    Cell 8 — Scale Calibration
                    GPS distance between anchor pairs
                    ÷ VO pixel distance between anchor pairs
                    → median scale  (m / px)
                                 │
                    Cell 9 — Drift Evaluation
                    At each anchor: compare VO-predicted position
                    (converted to metres) vs. GPS anchor position
                    → drift (m), drift rate (m/s)
                                 │
                    Cell 11 — Trajectory Fusion
                    For each segment between consecutive anchors:
                    linearly interpolate GPS correction error
                    and apply to raw VO positions
                    → fused trajectory (GPS-corrected)
                                 │
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUTS                                                                     │
│  • VO trajectory plot (raw + anchor overlay)                                │
│  • Fused trajectory plot (raw VO vs. corrected)                             │
│  • Visual verification panels (drone frame / sat tile / fresh sat tile)     │
│  • dl_vo_log.csv, dl_anchors_log.csv, dl_summary.json                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key design choice:** The two branches share the same SuperPoint+LightGlue stack. VO uses a lighter keypoint budget (1 024) for speed; satellite matching uses a richer budget (4 096) because the satellite tile is matched many times and benefits from denser coverage. Drift correction is applied post-hoc rather than in real-time — the fused trajectory is a smoothed offline result, not a live filter.

---

## What This Notebook Does and Why

The goal is to estimate where a drone is flying using only its onboard camera — no GPS required. We do this in two stages:

**Stage 1 — Visual Odometry (VO):** Track how the drone moves frame-to-frame by finding and matching distinctive points in consecutive video frames. Each matched-point pair tells us how the scene shifted, which we decompose into pixel-level translation and rotation. Accumulating these step-by-step displacements gives us a relative trajectory from the starting point. Because VO is purely incremental, small errors compound over time (drift).

**Stage 2 — Satellite Anchoring:** Every second, we also try to match a drone frame against a pre-fetched satellite tile of the known flight area. When enough points match (above a quality threshold), we triangulate an absolute GPS position. These GPS fixes, called anchors, are used to correct the VO drift by linearly interpolating the error between successive anchor pairs.

### Why Deep Learning Instead of Classical Features?

Notebook 2 used ORB (handcrafted binary descriptors) for VO and SIFT+FLANN for satellite matching. This notebook replaces both with learned alternatives:

| Component | Classical (NB2) | Deep Learning (NB3v2) |
|---|---|---|
| Feature extraction | ORB / SIFT | **SuperPoint** (neural net, learned float descriptors) |
| Matching | BFMatcher + Lowe ratio | **LightGlue** (graph neural net, learns which matches are reliable) |
| Satellite imagery | Google Maps Static API | **ESRI World Imagery** (free, no key, works in EU/EEA) |

SuperPoint produces richer, more repeatable keypoints across viewpoint and lighting changes. LightGlue replaces hand-tuned heuristics (Lowe ratio, cross-check) with a learned attention mechanism that directly predicts match confidence, giving cleaner matches and removing the need for manual threshold tuning.

### Pipeline Overview

```
Video (30 fps)
    → Frame extraction at 15 Hz (Cell 0b)
    → SuperPoint keypoints + descriptors on each frame (Cell 4)
    → LightGlue matching between consecutive frames → homography → dx, dy, dyaw
    → Accumulated VO trajectory (relative, pixel units)

Every 15 frames (1 Hz):
    → SuperPoint on resized drone frame + LightGlue vs. satellite tile (Cell 7)
    → RANSAC homography → estimated GPS position (anchor)
    → Sanity check: reject if >2 km from nominal

Scale calibration (Cell 8):
    → GPS distance between anchors / VO pixel distance between anchors → m/px

Drift evaluation (Cell 9):
    → Compare VO-predicted position at each anchor against GPS anchor position

Fusion (Cell 11):
    → Linearly interpolate GPS error between anchor pairs and apply to VO trajectory
```

---

## Cell 2 — Configuration Settings

### Location

| Setting | Value | Meaning |
|---|---|---|
| `LAT` | `43.521955` | Nominal latitude of the flight area (degrees) |
| `LON` | `-5.624290` | Nominal longitude of the flight area (degrees) |

These are the known coordinates embedded in the video filename. Used as the centre of the satellite tile fetch and as the reference point for the GPS sanity check.

### Visual Odometry Parameters

| Setting | Value | Meaning |
|---|---|---|
| `DOWNSAMPLE_SIZE` | `(640, 480)` | Drone frames are resized to this resolution before VO feature extraction. Smaller than the native 1920×1080 to speed up SuperPoint and reduce memory use. The aspect ratio is preserved approximately. |
| `SAT_MATCH_HZ` | `1.0` | How often (in Hz) to attempt a satellite match. At 15 fps source, this means one attempt per 15 frames. |
| `SAT_MATCH_EVERY` | `15` | Derived from `15 / SAT_MATCH_HZ`. The frame stride between satellite match attempts. |

### Satellite Tile Parameters

| Setting | Value | Meaning |
|---|---|---|
| `SAT_ZOOM` | `18` | Zoom level for the ESRI tile. Higher zoom = higher resolution, smaller geographic coverage. Zoom 18 gives roughly 0.6 m/px and covers ~385 m × 385 m. |
| `IMG_SIZE` | `640` | Width and height (px) of the fetched satellite tile. |

### SuperPoint / LightGlue Parameters

| Setting | Value | Meaning |
|---|---|---|
| `MAX_KP_VO` | `1024` | Maximum keypoints SuperPoint detects per drone frame for VO. Capped to keep per-frame inference fast (~75 ms on T4). |
| `MAX_KP_SAT` | `4096` | Maximum keypoints SuperPoint detects in the satellite tile. Higher cap because the satellite image is matched against many drone frames, so richer coverage is worth the one-time cost. |

### Satellite Matching Quality Gates

| Setting | Value | Meaning |
|---|---|---|
| `SAT_MATCH_SIZE` | `(640, 480)` | Drone frames are resized to this before satellite matching (same as VO size). Reduces the scale mismatch between a ~1920×1080 drone frame and the 640×640 satellite tile, which improves matching quality. |
| `MIN_SAT_INLIERS` | `4` (code default) / `8` (run used) | Minimum RANSAC inliers required to accept a satellite anchor. 4 is the geometric minimum to compute a homography; 8 is more conservative and reduces false-positive anchors. The notebook comment says "4 is too permissive." |
| `MAX_ANCHOR_KM` | `2.0` | GPS sanity check: estimated positions more than 2 km from the nominal coordinates are rejected as outliers, regardless of inlier count. |

---

## Evaluation Metrics

### Visual Odometry Quality (Cell 5)

These metrics tell us how reliably SuperPoint+LightGlue is tracking the drone frame-to-frame.

| Metric | What it measures |
|---|---|
| **Inlier ratio** | Fraction of LightGlue matches that survive RANSAC for each frame pair. Close to 1.0 = nearly all matches are geometrically consistent; below 0.2 = the homography failed. |
| **Failed VO pairs** | Number of frame pairs with inlier ratio < 0.2, where we fall back to zero displacement. Indicates frames where tracking breaks down (e.g., blur, abrupt motion). |
| **Good matches** | Raw count of matches LightGlue proposes before RANSAC filtering. |
| **RANSAC inliers** | Matches that are consistent with the estimated homography. |
| **Mean LightGlue confidence** | Average match score output by LightGlue (0–1). LightGlue scores near 1.0 indicate high certainty; this notebook consistently achieved 1.000. |
| **Per-frame time (ms)** | Wall-clock time per frame pair for the full VO inference (load, extract, match, homography). At 15 Hz the budget is 66.7 ms; this run averaged 75.8 ms, so it exceeds real-time on a T4 GPU. |
| **Displacement (px/step)** | Euclidean pixel displacement estimated per frame pair. Used as a sanity check — very large or zero values indicate tracking failure. |

### Satellite Anchoring Quality (Cell 7)

These metrics assess how good the individual GPS fixes are.

| Metric | What it measures |
|---|---|
| **Successful anchors / total attempts** | Fraction of satellite match attempts that pass both the inlier threshold and the GPS sanity check. Low acceptance rate (4/229 here) is expected — the satellite tile is a top-down view at fixed scale, while the drone camera sees the scene obliquely with variable altitude. |
| **Anchor inliers** | RANSAC inliers for the drone-vs-satellite homography. Higher = more reliable GPS fix. |
| **Anchor LightGlue score** | Mean match confidence for the accepted anchor matches. |
| **Distance from nominal (m)** | How far the estimated GPS position is from the known flight coordinates. Used as a quality indicator and as the cutoff for `MAX_ANCHOR_KM`. |

### Scale Calibration (Cell 8)

| Metric | What it measures |
|---|---|
| **Scale (m/px)** | Metres per VO pixel, estimated from the ratio of GPS distance to VO pixel distance between anchor pairs. Needed to convert the pixel-unit VO trajectory into real-world metres. With ≥2 anchors the median of pairwise estimates is used; with only 1 anchor the satellite tile's known m/px is used as a fallback. |
| **Implied drone footprint width (m)** | `scale × DOWNSAMPLE_WIDTH`. A rough sanity check on the scale estimate — should match the drone's expected altitude and field of view. |

### Drift (Cell 9)

| Metric | What it measures |
|---|---|
| **Drift (m)** | At each anchor after the first, the Euclidean distance between where VO predicts the drone is (converted to metres using the scale) and where the satellite anchor says it is. Accumulates over time because VO integrates small per-frame errors. |
| **Drift rate (m/s)** | Final drift divided by elapsed time. Summarises how quickly errors grow — lower is better. |

### Overall Summary (Cell 12)

Cell 12 collects all the above into a single printout and a JSON file, including: total frames, device, mean per-frame time, effective throughput (fps), failed VO pair count, mean inlier ratio, satellite anchor count, mean anchor inliers, scale estimate, and final drift.

---

## Output Files

| File | Content |
|---|---|
| `dl_vo_log_<timestamp>.csv` | Per-frame VO record: cumulative position, per-step displacement, match counts, inlier ratio |
| `dl_anchors_log_<timestamp>.csv` | Per-anchor record: frame index, estimated GPS, match quality, distance from nominal |
| `dl_summary_<timestamp>.json` | All scalar metrics for the run in one machine-readable file |
| `vo_inlier_metrics.png` | Three time-series plots: inlier ratio, match counts, LightGlue confidence |
| `vo_displacement.png` | Per-step pixel displacement over time |
| `trajectory.png` | VO trajectory coloured by time + satellite anchor positions |
| `fused_trajectory.png` | Raw VO vs. anchor-corrected trajectory overlay |
| `visual_verification.png` | Side-by-side panels: drone frame / original satellite tile with position mark / fresh satellite tile centred at estimated GPS |
