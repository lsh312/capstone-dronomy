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
few metres. See [`experiments/comparison/results/benchmark/manual_anchoring.md`](experiments/comparison/results/benchmark/manual_anchoring.md)
for the full analysis.

---

## Repository layout

```
src/                          # production pipeline scripts
  run_pipeline.py             # end-to-end runner (steps 1–6, parallel VO+sat)
  extract_frames.py           # 15 Hz frame extraction
  fetch_satellite.py          # download satellite tile (Google → ESRI fallback)
  vo_run.py                   # visual odometry (SuperPoint+LightGlue) → CSV
  sat_anchors.py              # automated satellite anchor matching
  mark_anchors.py             # interactive manual anchor marking
  fuse.py                     # VO + anchor fusion → geolocalized trajectory
  dashboard.py                # Streamlit live dashboard
  gps_benchmark.py            # score positions vs DJI .SRT truth (matcher-agnostic)

experiments/
  deep_learning/
    deep_learning_localization_v2.ipynb        # full pipeline notebook (SuperPoint+LightGlue)
    deep_learning_localization_v2_explainer.md # cell-by-cell explainer
    results/
      run_20260623_093123/    # final full-flight results (canonical reference)
      deep-learning-*/        # earlier automated-anchor experiment runs
      benchmark/              # GPS accuracy analysis (README + CSV + plots)
  roma/
    roma_localize_v2.ipynb    # RoMa dense-matcher baseline
    results/roma_v2/          # RoMa outputs (positions, fused track, plots)
  sift/
    baseline_sift.ipynb       # SIFT classical-matcher baseline
    *.py                      # SIFT evaluation scripts
    results/sift_baseline/    # SIFT outputs (estimates, metrics, match images)
  comparison/
    matcher_comparison.py     # score SIFT/LightGlue/RoMa anchors on equal footing
    heading_benchmark.py      # score est_yaw vs SRT gb_yaw (orientation)
    realtime_benchmark.py     # per-frame VO-style latency vs 15 Hz budget
    matcher_comparison.md     # 3-matcher comparison table + findings
    heading_benchmark.md      # orientation benchmark results
    realtime_benchmark.md     # latency vs 15 Hz budget results
    results_comparison.md     # cross-experiment summary
    results/
      matcher_comparison/     # per-matcher anchor CSVs (re-scorable)
      benchmark/              # manual anchoring analysis + plots

data/
  gps_data.SRT                # DJI onboard GPS (ground truth) — tracked
  IE_Challenge_*.MP4          # source video — gitignored (3.5 GB)
  frames_15hz/                # extracted frames — gitignored (~1.9 GB, regenerable)
  satellite_*.png/.bbox.json  # satellite tile + bounding box

results/                      # pipeline run outputs
  latest.txt                  # points to the most recent run_* folder
  run_YYYYMMDD_HHMMSS/        # one folder per pipeline run
    dl_vo_log_full.csv        # VO trajectory (one row per frame)
    sat_anchors_<ts>.csv      # automated satellite anchors
    fuse_<ts>/                # fusion outputs (fused_track.csv, plots)
    fuse_live/                # live fusion output (written by fuse --watch)

documentation/
  pipeline.md                 # detailed pipeline documentation
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
`data/IE_Challenge_lat43_521955_lon5_624290.MP4`, then run the full pipeline:

```bash
python src/run_pipeline.py
```

Each run creates a timestamped output folder (`results/run_YYYYMMDD_HHMMSS/`)
and updates `results/latest.txt` so the dashboard and downstream scripts always
find the current results automatically.

**Common flags:**

```bash
python src/run_pipeline.py --max-seconds 5      # quick test (first 5 s only)
python src/run_pipeline.py --from vo            # resume from VO step (reuse latest run)
python src/run_pipeline.py --skip-sat           # skip automated satellite anchors
python src/run_pipeline.py --no-mark            # skip interactive anchor marking
python src/run_pipeline.py --force              # re-run all steps in a new folder
```

**Steps run automatically:**

| Step | Script | Notes |
|---|---|---|
| 1. Extract frames | `extract_frames.py` | Skipped if frames already exist |
| 2. Satellite tile | `fetch_satellite.py` | Skipped if tile already exists |
| 3+4. VO + sat anchors | `vo_run.py` + `sat_anchors.py` | Run **in parallel** — sat polls VO CSV as it grows |
| 5. Manual anchors | `mark_anchors.py` | Interactive window; right-click to skip a frame |
| 6. Fuse | `fuse.py` | Combines all anchors → geolocalized trajectory |

**Live dashboard** (open in a second terminal while the pipeline runs):

```bash
streamlit run src/dashboard.py
```

The dashboard reads from `results/latest.txt` and updates automatically as VO
writes frames and fuse produces new results.

**GPS benchmark** (score any run against DJI GPS truth):

```bash
python src/gps_benchmark.py \
    --srt data/gps_data.SRT \
    --anchors results/run_<ts>/sat_anchors_<ts>.csv \
    --frames data/frames_15hz \
    --out experiments/comparison/results/benchmark
```

---

## Status vs. project goals

- [x] **Visual odometry** — working, near-perfect tracking, MPS-stable runner.
- [x] **GPS ground-truth benchmark** — error measured in metres against the
      DJI SRT; matcher-agnostic so it scores any method on equal footing.
- [x] **Manual anchoring** — 4.1 m median / 98% within 15 m, GPS-free, with
      the path-over-satellite dashboard.
- [x] **Matcher comparison** (SIFT / SuperPoint+LightGlue / RoMa) — all three
      automated matchers scored on equal footing: RoMa 65 m, SIFT 75 m,
      LightGlue 109 m median; none clears the target, which is what motivates
      manual anchoring (4.1 m). See [`experiments/comparison/matcher_comparison.md`](experiments/comparison/matcher_comparison.md).
- [x] **Higher-res / multi-tile satellite imagery** — evaluated, **not worth
      pursuing**. Automated-match error tracks terrain *matchability*, not tile
      resolution: RoMa anchors are bimodal (good frames ~760 inliers vs bad
      ~396; `corr(error, inliers) = -0.36`), and RoMa already used ~0.22 m/px
      imagery yet still hit 65 m. The bottleneck is featureless,
      rotationally-ambiguous terrain + the drone-vs-satellite appearance gap —
      more pixels don't add matchable structure to grass.
- [x] **Real-time characterization** — measured per-frame latency vs the 15 Hz
      budget. VO loop (every frame, 66.7 ms budget): ORB 7.6 ms ✅, SIFT 84 ms
      ❌, SuperPoint+LightGlue 281 ms MPS ❌ / ~76 ms CUDA ⚠️. Anchor loop
      (periodic, 1 Hz) is comfortably within budget. See
      [`experiments/comparison/realtime_benchmark.md`](experiments/comparison/realtime_benchmark.md).
- [x] **Heading output** — benchmarked `est_yaw` vs SRT `gb_yaw`. Finding:
      automated orientation is **not usable** on this flight (RoMa ~64° median
      error after best-offset correction). See
      [`experiments/comparison/heading_benchmark.md`](experiments/comparison/heading_benchmark.md).

---

## Notes

- **Ground truth integrity:** the DJI GPS is used only to *score* accuracy and
  *validate* manual marks — never to create anchors. The localization itself is
  GPS-free.
- **Frames and video are gitignored.** Frames regenerate from the video via
  `extract_frames.py`; the satellite tile re-fetches from ESRI on demand.
- **Run outputs are isolated.** Each `python src/run_pipeline.py` call writes
  to a fresh `results/run_YYYYMMDD_HHMMSS/` folder. The canonical reference
  run is `experiments/deep_learning/results/run_20260623_093123/` (also mirrored
  at `results/run_20260623_093123/`).
