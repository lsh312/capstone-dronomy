# capstone-dronomy

GPS-free geolocalization of a drone from its onboard camera. The pipeline
estimates where the drone is over time using only video — visual odometry for
relative motion, satellite-image matching for absolute position — and validates
the result in metres against the drone's recorded DJI GPS.

**Current best result: 4.1 m median position error, 98% of the flight within
15 m**, GPS-free, using manual anchoring (see below).

---

## How it works

Two branches run over the extracted video frames:

1. **Visual odometry (VO)** — SuperPoint + LightGlue match consecutive frames;
   a homography gives per-frame translation/rotation, accumulated into a
   relative trajectory. Tracking is near-perfect on this flight (mean inlier
   ratio 0.997, 0 failed pairs / 3,429), but like all VO it drifts over time.

2. **Absolute anchoring** — periodically fix the drone's true position by
   matching a frame to satellite imagery (automated), or to **human-marked
   reference points** (manual). Anchors correct the VO drift.

The drift correction has two parts: a global **VO→world similarity fit**
(scale + rotation, by least squares over the anchors) sets overall scale and
orientation, then a **piecewise correction** between consecutive anchors removes
the residual drift a single global transform can't. The output is the flight
path in real lat/lon, overlaid on the satellite tile.

### Why manual anchoring

Automated satellite anchors are noisy — **~100 m median error** vs GPS truth,
because the low-altitude, often featureless (grass) drone view is hard to match
against a top-down satellite tile. A handful of human-marked anchors fixes both
scale and drift:

| Anchors | Median error | Within 15 m |
|---|---|---|
| Automated (ESRI tile) | ~100 m | ~4% |
| 6 manual | 13 m | 57% |
| 10 manual | 6 m | 87% |
| **14 manual (live run)** | **4.1 m** | **98%** |

Marking error carries ~1:1 into the result, so anchors should be placed on
sharp, identifiable features (road/path junctions, building corners) within a
few metres. See [`results/benchmark/manual_anchoring.md`](results/benchmark/manual_anchoring.md)
for the full analysis.

---

## Repository layout

```
code/
  extract_frames.py     # 15 Hz frame extraction (standalone)
  vo_run.py             # standalone VO runner (SuperPoint+LightGlue), MPS-stable, -> CSV
  gps_benchmark.py      # score positions (+heading) vs DJI .SRT truth (matcher-agnostic)
  manual_anchors.py     # manual anchoring: VO->world fit + piecewise fusion + validation
  matcher_comparison.py # score SIFT/LightGlue/RoMa anchors on equal footing
  heading_benchmark.py  # score est_yaw vs SRT gb_yaw (orientation)
  sift/                 # SIFT baseline scripts (classical-matcher pipeline)
notebooks/
  deep_learning_localization_v2.ipynb        # full pipeline (SuperPoint+LightGlue + manual anchoring)
  deep_learning_localization_v2_explainer.md # cell-by-cell explainer
  baseline_sift.ipynb                        # SIFT classical-matcher baseline
  roma_localize_v2.ipynb                     # RoMa dense-matcher baseline (Colab)
data/
  gps_data.SRT          # DJI onboard GPS (ground truth) — tracked
  IE_Challenge_*.MP4    # source video — gitignored (3.5 GB)
  frames_15hz/          # extracted frames — gitignored (~1.9 GB, regenerable)
results/
  nb3v2/                # SuperPoint+LightGlue pipeline outputs (VO log, plots, dashboard)
  deep-learning-*/      # earlier automated-anchor runs
  sift_baseline/        # SIFT baseline outputs (estimates, metrics, match images)
  roma_v2/              # RoMa baseline outputs (positions, fused track, plots)
  matcher_comparison.md            # 3-matcher comparison table + findings
  matcher_comparison/estimates/    # the three matchers' anchor CSVs (re-scorable)
  heading_benchmark.md             # orientation (est_yaw vs gb_yaw) results
  benchmark/
    README.md           # automated-anchor error analysis
    manual_anchoring.md # manual-anchoring method + results
```

---

## Setup

Uses [uv](https://docs.astral.sh/uv/). Dependencies (opencv, torch, lightglue,
etc.) are in `pyproject.toml`.

```bash
uv sync                 # creates .venv with all deps
```

For the notebook, select the `.venv` interpreter as the Jupyter kernel (or
`uv run jupyter lab`). On Apple Silicon the pipeline uses the MPS GPU.

---

## Running the pipeline

The source video is not in the repo (too large). Place it at
`data/IE_Challenge_lat43_521955_lon5_624290.MP4`, then:

1. **Extract frames** — notebook Cell 0b (or `code/extract_frames.py`). Writes
   ~3,430 frames at 15 Hz to `data/frames_15hz/`.

2. **Visual odometry** — run `code/vo_run.py` (writes
   `results/nb3v2/dl_vo_log_full.csv`), then load it in the notebook instead of
   the slow in-notebook VO:
   ```bash
   cd code && VO_DEVICE=mps VO_MAX_KP=256 uv run python vo_run.py
   ```
   (~10 min on MPS, stable. The 256-keypoint budget matches 1024-kp VO quality
   for frame-to-frame tracking while avoiding an MPS memory leak.)

3. **Satellite tile + automated anchors** — notebook Cells 6–7.

4. **Manual anchoring** — Cell 7c (scout grid: pick ~10–15 frames with clear
   landmarks), then Cell 7b: put `{frame_idx: (lat, lon)}` into `MANUAL_ANCHORS`
   (coordinates read from Google Earth), run → fused path, dashboard, and the
   truth-validated error.

5. **GPS benchmark** (any matcher) — Cell 9b, or:
   ```bash
   cd code && uv run python gps_benchmark.py \
       --srt ../data/gps_data.SRT \
       --anchors ../results/<run>/dl_anchors_log_<ts>.csv \
       --frames ../data/frames_15hz --out ../results/benchmark
   ```

---

## Status vs. project goals

- [x] **Visual odometry** — working, near-perfect tracking, MPS-stable runner.
- [x] **GPS ground-truth benchmark** — error now measured in metres against the
      DJI SRT; matcher-agnostic so it scores any method on equal footing.
- [x] **Manual anchoring** — 4.1 m median / 98% within 15 m, GPS-free, with the
      path-over-satellite dashboard.
- [x] **Matcher comparison** (SIFT / SuperPoint+LightGlue / RoMa) — all three
      automated matchers scored on equal footing: RoMa 65 m, SIFT 75 m, LightGlue
      109 m median; none clears the target, which is what motivates manual
      anchoring (4.1 m). See [`results/matcher_comparison.md`](results/matcher_comparison.md).
- [ ] **Higher-res / multi-tile satellite imagery** (Google Maps key rotation) —
      only needed to push *automated* anchors toward manual quality.
- [x] **Heading output** — benchmarked `est_yaw` vs SRT `gb_yaw`. Finding:
      automated orientation is **not usable** on this flight (RoMa ~64° median
      error even after best-offset correction, ~18% within 30° — barely above
      random), because the nadir camera over low-texture terrain is rotationally
      ambiguous. See [`results/heading_benchmark.md`](results/heading_benchmark.md).

---

## Notes

- **Ground truth integrity:** the DJI GPS is used only to *score* accuracy and
  *validate* manual marks — never to create anchors. The localization itself is
  GPS-free.
- **Frames and video are gitignored.** Frames regenerate from the video via
  Cell 0b; the satellite tile re-fetches from ESRI on demand.
