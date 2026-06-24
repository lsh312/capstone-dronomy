# GPS Ground-Truth Benchmark

Real positioning error of the satellite anchors, measured against the drone's
on-board **DJI GPS** track (`data/gps_data.SRT`, 6 853 fixes over 228.6 s).

This replaces the old "distance from nominal" proxy — which compared every
estimate to a *single* hardcoded coordinate and so could not distinguish a good
fix from a bad one. Error here is the haversine distance (m) between each
estimated anchor position and the interpolated true position at that frame's
timestamp.

## How to run

```bash
cd code
uv run python gps_benchmark.py \
    --srt ../data/gps_data.SRT \
    --anchors ../results/<run>/dl_anchors_log_<ts>.csv \
    --frames ../data/frames_15hz \
    --out ../results/benchmark --label <run>
```

Or it runs automatically as **Cell 9b** in `deep_learning_localization_v2.ipynb`.
The harness is matcher-agnostic: the same command scores SIFT / SURF / LightGlue
/ MatchAnything anchor logs, so method comparisons are apples-to-apples.

## Results (current deep-learning runs)

| Run | Anchors | Median err | Mean err | RMSE | p90 | Best fix | % ≤ 15 m |
|---|---|---|---|---|---|---|---|
| Run 1 (8 inl, default LG, 5 px) | 4 | **37.8 m** | 54.8 m | 70.4 m | 102.8 m | 14.3 m | 25 % |
| Run 2 (6 inl, default LG, 5 px) | 65 | 105.0 m | 132.1 m | 164.5 m | 219.9 m | 23.1 m | 0 % |
| Run 3 (8 inl, LG disabled, 8 px) | 53 | 108.6 m | 169.3 m | 385.5 m | 180.9 m | **6.8 m** | 4 % |

## Interpretation

- **Anchor quality is poor, now quantified.** Median error is ~100 m for the
  dense runs (2 & 3); even the best run is ~38 m median. The brief asks for "a
  few metres" — so automated anchoring is currently 1–2 orders of magnitude off.
  This confirms the mentor's "drone-vs-satellite comparisons are significantly
  off" with a hard number.
- **Run 3 has the single best individual fix (6.8 m)** but also the worst tail
  (RMSE 385 m) — a couple of false positives at >2 km drag the mean. High inlier
  count does **not** guarantee a correct fix (a homography can be geometrically
  consistent yet map to the wrong place on the tile).
- **The flight is spatially compact** (~300 m E–W, see the right panel of each
  plot): the true track is a tight cluster, so anchors that land hundreds of
  metres away are unambiguous false positives, not near-misses.
- **`MIN_SAT_INLIERS` ranks runs the way truth does:** Run 1 (strict, few
  anchors) is most accurate per-anchor; loosening to 6 (Run 2) adds quantity at
  the cost of accuracy. This validates "stricter is better" against ground truth.

## Why ~100 m and not a few metres

The error is dominated by the **viewpoint gap** (oblique drone camera vs.
top-down satellite tile) and **single-tile coverage** (one 385 m ESRI tile at
zoom 18 centred on nominal). Next steps that should move the number:

1. **Manual anchoring** on a few well-separated, verified frames — stabilises
   scale and gives a clean reference set.
2. **Multi-tile / higher-res imagery** (Google Earth, or Maps with key rotation)
   so matching isn't bottlenecked by the low-res single ESRI tile.
3. **Altitude-aware zoom** — the SRT carries `rel_alt`/`abs_alt`, so the
   satellite zoom can be matched to the drone footprint instead of hardcoded.

## Bonus signals available in the SRT

`gps_data.SRT` also records `gb_yaw` (gimbal yaw = **heading truth**) and
`rel_alt`/`abs_alt` (**altitude**). Heading covers the mentor's "direction is a
plus" requirement, and altitude feeds the zoom-calibration thread — both are
already parsed (altitude) or trivially addable (heading) in `gps_benchmark.py`.

## Files

- `gps_benchmark.py` (in `code/`) — the benchmark module + CLI.
- `benchmark_<run>.csv` — per-anchor: frame_idx, t_ms, est/true lat-lon, error_m, is_good.
- `benchmark_<run>.png` — error-over-time, error CDF, spatial truth-vs-estimate overlay.
