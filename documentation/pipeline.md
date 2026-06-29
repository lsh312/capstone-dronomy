# Dronomy — Geolocalization Pipeline

## Overview

The pipeline estimates the GPS trajectory of a drone from video alone, without
relying on onboard GPS. It combines **Visual Odometry (VO)** — which tracks
relative motion between frames — with **anchor points** that tie that relative
path to real-world coordinates.

```
Video
  └─ extract_frames.py    → frames at 15 Hz
       └─ vo_run.py        → relative path in pixel space (dl_vo_log_full.csv)
            ├─ sat_anchors.py   → automated GPS anchors  ╗ run in parallel;
            │                     (polls VO CSV live)    ╝ sat waits on VO
            ├─ mark_anchors.py  → manual GPS anchors (human-in-the-loop)
            └─ fuse.py          → drift-corrected, geolocalized trajectory
                                  (fuse_<ts>/fused_track.csv)
```

All steps are orchestrated by `src/run_pipeline.py`. Each run writes output to
a timestamped folder (`results/run_YYYYMMDD_HHMMSS/`) and updates
`results/latest.txt` so the dashboard and downstream scripts always find the
current results without manual configuration.

---

## Step-by-step

### 1. Frame Extraction — `src/extract_frames.py`

Extracts frames from the drone video at a fixed rate (default 15 Hz). This
decouples the pipeline from the video file format and makes random access by
frame index fast.

Output: `data/frames_15hz/frame_<ms>ms.jpg` — one JPEG per sampled frame,
named by its video timestamp in milliseconds. The timestamp in the filename is
the key used to align VO estimates with the DJI .SRT GPS ground truth.

**Flags:** `--max-seconds N` limits extraction to the first N seconds (useful
for quick test runs).

---

### 2. Satellite Tile — `src/fetch_satellite.py`

Downloads a single satellite image covering the flight area. Tries Google Maps
Static API first (requires a key in `.env`); falls back to ESRI World Imagery
automatically.

Saves two files to `data/`:
- `satellite_<provider>_z<zoom>.png` — the RGB image
- `satellite_<provider>_z<zoom>.bbox.json` — geographic bounding box
  (`lat_min`, `lat_max`, `lon_min`, `lon_max`, `zoom`, `provider`,
  `img_width`, `img_height`, `mpp`)

**The tile is static.** It is downloaded once and used as a fixed reference
throughout the pipeline. This is by design: the satellite image is treated as
ground truth — it does not change between pipeline runs.

**Coverage constraint:** at zoom 18 (~0.6 m/px), a 1280×1280 px ESRI tile
covers roughly 750×750 m. The entire flight must fit within this area. If part
of the flight extends outside the tile boundary, no satellite anchors can be
produced for those frames, and the fused trajectory will rely on VO drift
correction alone in that region. Verify coverage before running `sat_anchors.py`
by checking the bbox against the nominal flight coordinates.

**Tile selection:** when multiple tiles exist in `data/`, all scripts use the
same priority function — Google Maps preferred over ESRI, higher zoom preferred
for a given provider — so all steps always use the same tile.

---

### 3. Visual Odometry — `src/vo_run.py`

Runs SuperPoint + LightGlue feature matching on consecutive frames to estimate
how the drone moved between each pair. Accumulates per-frame displacements into
a running position in **pixel space** (`cum_x`, `cum_y`) and heading
(`cum_yaw`).

Output: `results/run_<ts>/dl_vo_log_full.csv` — one row per frame with
cumulative displacement, inlier count, and LightGlue confidence scores.

The CSV is written **incrementally** (flushed every 10 frames by default,
configurable via `VO_FLUSH_EVERY` env var). This allows the dashboard and
`sat_anchors.py` to consume results while VO is still running.

At the end, `vo_run.py` writes a sentinel file `dl_vo_log_full.done` in the
same folder. `sat_anchors.py --watch-vo` watches for this sentinel to know
when to stop polling.

VO gives the **shape** of the trajectory accurately but has no knowledge of:
- Real-world scale (metres per pixel)
- Absolute geographic position
- Absolute heading (north vs image-up)
- Drift accumulation over time

All of these are resolved by the anchoring and fusion steps.

---

### 4. Satellite Anchor Matching — `src/sat_anchors.py`

For a subset of frames (default: 1 Hz), matches the drone view against the
satellite tile using SuperPoint + LightGlue + RANSAC homography. If the match
is geometrically consistent (≥8 inliers by default, estimated position within
2 km of the nominal site), the frame centre is projected through the homography
into the satellite tile, then converted to GPS via the tile's bounding box.

Output: `results/run_<ts>/sat_anchors_<ts>.csv` — each successful anchor
written **immediately** (not at the end), so `fuse.py --watch` can pick them
up in real time.

**Parallel mode (`--watch-vo`):** when launched alongside `vo_run.py` by
`run_pipeline.py`, `sat_anchors.py` polls the VO CSV every 10 seconds
(configurable via `--watch-poll`), processes any newly available candidate
frames, and exits automatically when the `.done` sentinel appears. This means
the total time for steps 3+4 is `max(VO time, sat_anchors time)` rather than
the sum.

**Justification:** automated satellite anchoring removes the need for manual
correspondences for every GPS fix. SuperPoint is rotation- and scale-invariant
and performs well on nadir-to-nadir matching. RANSAC filters geometric outliers.

**Limitation:** match quality degrades on featureless terrain (grass, water),
with tilted cameras, or when lighting differs significantly from the satellite
image. In those cases manual anchors are used to supplement. See the matcher
comparison results in `experiments/comparison/`.

---

### 5. Manual Anchor Marking — `src/mark_anchors.py`

An interactive tool that presents each candidate drone frame side-by-side with
the satellite tile. The operator left-clicks where the drone appears on the
satellite tile; the click coordinates are converted to GPS via the tile bounding
box and saved immediately to `data/manual_anchors.json`.

Controls:
- **Left-click** on the satellite tile — mark the drone's position
- **Right-click** — skip this frame (move to the next without marking)
- **Close window** — stop early (saves all marks made so far)

**Justification:** manual anchors handle cases where automated matching fails —
oblique angles, featureless terrain, or the flight partially leaving the tile.
They also provide high-confidence fixes at key frames. The interactive step is
intentionally kept simple: marks are written on each click so no progress is
lost if the window is closed.

---

### 6. Fusion — `src/fuse.py`

Combines all anchors (manual + satellite) with the VO trajectory in two stages:

1. **Global fit** (`fit_vo_to_world`): solves for a single similarity transform
   (scale + rotation + translation) that maps VO pixel coordinates to GPS, using
   least-squares (Umeyama 2D) over all anchors. This sets the overall scale and
   orientation of the trajectory.

2. **Piecewise drift correction** (`fuse_vo_with_anchors`): between each pair
   of consecutive anchors, linearly interpolates the residual error so the path
   passes exactly through each anchor. This corrects the slow drift that VO
   accumulates over time without distorting the local path shape.

Output: `results/run_<ts>/fuse_<ts>/fused_track.csv` — GPS coordinates for
every frame. Also writes `fused_trajectory.png` and benchmark results if a
`.SRT` file is available.

**Watch mode (`--watch`):** re-runs fusion automatically whenever the satellite
anchor CSV or manual anchors file changes, writing to `results/run_<ts>/fuse_live/`.
The dashboard detects this folder and switches to the live fused result.

---

## Dashboard — `src/dashboard.py`

```bash
streamlit run src/dashboard.py
```

The dashboard reads `results/latest.txt` to locate the active run folder
automatically. It refreshes every few seconds (configurable in the sidebar).

### Metrics bar

| Field | Source |
|---|---|
| Status | whether `dl_vo_log_full.csv` is still growing |
| Frames processed | last `frame_idx` in the VO CSV |
| Progress % | frames processed / total frames on disk |
| Mean inlier ratio | avg fraction of matched keypoints surviving RANSAC |
| Mean LG score | avg LightGlue confidence score |

High inlier ratio (>0.8) and high LG score (>0.7) indicate reliable tracking.
A sustained drop in either signals a tracking failure (rapid motion, blur,
featureless region).

### Drone frame + satellite tile

Shows the most recently processed drone frame alongside the satellite tile.
Used to visually verify that VO is processing the right footage and that the
satellite tile covers the area.

### Trajectory map

**During VO run:** Shows the approximate trajectory using a fixed scale
(0.7763 m/px, from prior anchor fits at zoom 18 ESRI) applied to raw VO pixel
displacements. The shape is correct; absolute position and scale are
approximate. Title reads *"Approximate trajectory map using VO"*.

**After `fuse.py` runs:** Automatically switches to the drift-corrected,
anchor-calibrated GPS coordinates from `fused_track.csv`. Title reads
*"Trajectory map — fused (drift-corrected with anchors)"*.

The map auto-fits zoom and centre to the trajectory extent on every refresh.

### Coordinates table

Last N frames (configurable in the sidebar) with frame index, estimated time,
estimated lat/lon, raw VO pixel position, heading, inlier ratio, match count,
and LightGlue score.

---

## Results and run management

Each `python src/run_pipeline.py` invocation creates:

```
results/run_YYYYMMDD_HHMMSS/
  dl_vo_log_full.csv          # VO trajectory
  dl_vo_log_full.done         # sentinel: written when VO finishes
  sat_anchors_<ts>.csv        # satellite anchors
  fuse_<ts>/
    fused_track.csv           # final GPS trajectory
    fused_trajectory.png
    benchmark.csv / benchmark.png
  fuse_live/                  # live fusion output (overwritten on each re-fuse)
```

`results/latest.txt` always points to the most recently created run folder.
The dashboard and all downstream scripts read from this pointer, so switching
between runs only requires updating `latest.txt`.

The canonical reference run (full 3,429-frame flight, Marco's results) lives at:
- `experiments/deep_learning/results/run_20260623_093123/` — with the DL notebooks
- `results/run_20260623_093123/` — mirrored here for pipeline access via `latest.txt`

---

## Experiments

Earlier approaches and comparison work are in `experiments/`:

| Folder | Contents |
|---|---|
| `deep_learning/` | SuperPoint+LightGlue notebook (v2) + all experiment runs |
| `roma/` | RoMa dense-matcher notebook + outputs |
| `sift/` | SIFT classical-matcher scripts + outputs |
| `comparison/` | Cross-approach benchmark scripts, markdowns, and results |
