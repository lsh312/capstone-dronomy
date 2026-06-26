# Project Summary — GPS-Free Drone Geolocalization

A digest of what was built, the key results, and where each piece lives in the
repo. Intended as raw material for the report.

**Goal (brief):** automatically fetch a georeferenced satellite reference image,
match real-time camera frames to it, and output absolute **position and
orientation** — GPS-free. Bonus: visual odometry from the stream, fused with the
anchors.

**Flight:** DJI video, 3,430 frames @ 15 Hz (228.7 s), nadir camera. DJI on-board
GPS (`.SRT`) is used **only to score** accuracy — never to localize.

---

## Headline results

| Capability | Result | Status |
|---|---|---|
| Position — **manual anchoring** (deployed) | **4.1 m median, 98% within 15 m** | ✅ hits "few-metre" target |
| Position — **automated** (best, RoMa) | 65 m median, 23% within 15 m | ⚠️ below target |
| Orientation (RoMa `est_yaw` vs truth) | ~64° median error, ~18% within 30° | ❌ measured, not usable |
| Visual odometry tracking | mean inlier ratio 0.997, 0 failed pairs / 3,429 | ✅ |
| Real-time (VO loop @ 15 Hz, 66.7 ms) | ORB 7.6 ms ✅ · SIFT 84 ms · SP+LG 281 ms MPS / ~76 ms CUDA | ⚠️ matcher/HW dependent |

---

## Components — what, result, where it lives

### 1. Visual odometry (relative motion)
- **What:** SuperPoint + LightGlue match consecutive frames; homography →
  per-frame translation/rotation, integrated into a relative trajectory.
- **Result:** near-perfect tracking (inlier ratio 0.997, 0 failures). Drifts over
  time like all VO; after the world-frame integration fix the global-fit residual
  is 27.4 m and drift rate 0.08 m/s.
- **Lives in:** `code/vo_run.py` (standalone, MPS-stable) and
  `notebooks/deep_learning_localization_v2.ipynb`. Output VO track:
  `results/nb3v2/dl_vo_log_full.csv`.

### 2. GPS ground-truth scorer (the measuring stick)
- **What:** matcher-agnostic benchmark; scores any `(frame, lat, lon[, yaw])`
  estimates against the DJI `.SRT` by timestamp. Reports metres / degrees.
- **Lives in:** `code/gps_benchmark.py`; truth file `data/gps_data.SRT`.

### 3. Absolute anchoring + matcher comparison
- **What:** match a frame to satellite imagery for an absolute fix. Three matchers
  compared on equal footing (no manual anchors).
- **Result:** RoMa 65 m · SIFT 75 m · SuperPoint+LightGlue 109 m median; none
  clears the target. Dense (RoMa) > classical (SIFT) > sparse-deep (LightGlue)
  for the oblique-frame-vs-top-down-tile gap.
- **Lives in:** `code/matcher_comparison.py`, write-up
  `results/matcher_comparison.md`, re-scorable CSVs in
  `results/matcher_comparison/estimates/`. Per-matcher sources:
  - SIFT: `notebooks/baseline_sift.ipynb`, `code/sift/`, `results/sift_baseline/`
  - LightGlue: `notebooks/deep_learning_localization_v2.ipynb`, `results/nb3v2/`
  - RoMa: `notebooks/roma_localize_v2.ipynb`, `results/roma_v2/`

### 4. Manual anchoring (the deployed solution)
- **What:** a few human-marked reference points + a global VO→world similarity
  fit + piecewise drift correction between anchors. GPS-free (human reads the
  imagery; GPS only scores).
- **Result:** **4.1 m median, 98% within 15 m** with 14 anchors.
- **Lives in:** `code/manual_anchors.py` + the notebook (Cell 7b/7c); write-up
  `results/benchmark/manual_anchoring.md`; dashboard
  `results/nb3v2/manual_anchored_dashboard.png`.

### 5. Orientation
- **What:** score estimated heading (`est_yaw`) vs DJI `gb_yaw`, with an
  offset-correction test to separate a fixable convention bug from genuine noise.
- **Result:** ~64° median even after best-offset correction → **not usable**.
  Cause: nadir camera over low-texture, rotationally-ambiguous terrain. Only RoMa
  logged absolute yaw; VO yaw is relative-only.
- **Lives in:** `code/heading_benchmark.py`; write-up
  `results/heading_benchmark.md`.

### 6. Real-time characterization
- **What:** per-frame latency vs budget. Two loops: VO (every frame, 66.7 ms @
  15 Hz) and anchoring (periodic, 1 Hz, ~1000 ms).
- **Result:** VO loop is the binding constraint — ORB real-time (7.6 ms), SIFT/
  SP+LG not (84 / 281 ms MPS; ~76 ms CUDA is borderline). Anchor loop fits easily.
  Real-time is a matcher/hardware choice; project ran offline on MPS.
- **Lives in:** `code/realtime_benchmark.py`; write-up
  `results/realtime_benchmark.md`.

---

## Key findings (for the discussion)

1. **Automated frame-to-satellite matching tops out at ~65 m** on this nadir,
   low-texture flight. Error tracks terrain *matchability* (inlier count), not
   tile resolution — higher-res/multi-tile imagery was evaluated and shown
   **not** to help (see README).
2. **Manual anchoring closes the gap to 4.1 m** by sidestepping terrain ambiguity
   — but it is human-assisted and post-hoc (whole-flight fusion), not autonomous
   or live.
3. **Orientation from satellite matching does not work here** — rotational
   symmetry of featureless terrain under a nadir camera makes heading near-random.
4. **Real-time is feasible but constrained** — needs a fast matcher (ORB) or a
   capable GPU for the per-frame VO loop; anchoring is never the bottleneck.

## Brief compliance

| Requirement | Status |
|---|---|
| Auto-fetch georeferenced satellite reference | ✅ done (ESRI/Google tile fetch); "most-recent/quality" not strictly verified |
| Match real-time frames → satellite | ✅ done; real-time characterized (matcher/HW dependent) |
| Absolute **position** | ✅ automated ~65 m; manual-anchored 4.1 m |
| Absolute **orientation** | ✅ measured, ❌ not usable (documented with cause) |
| *Bonus:* VO from stream + fusion | ✅ done (this is the 4.1 m path) |

## Limitations / next steps
- Autonomous (no-human) accuracy is ~65 m, not few-metre.
- Manual anchoring is offline (uses future anchors); a live variant would correct
  forward-only.
- Orientation needs an oblique camera, higher-texture imagery, or IMU/magnetometer
  fusion to become usable.

## Reproduce everything
```bash
cd code
uv run python matcher_comparison.py     # position comparison table
uv run python heading_benchmark.py      # orientation results
uv run python realtime_benchmark.py     # latency vs 15 Hz budget
```
