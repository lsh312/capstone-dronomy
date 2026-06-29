# Matcher Comparison — Automated Satellite Anchoring

Three feature matchers were used to estimate the drone's absolute position by
matching drone frames against satellite imagery, with **no manual anchors**. All
three are scored through the same `gps_benchmark.py` against the same DJI `.SRT`
GPS truth, so they sit on equal footing. Manual anchoring is listed last as the
non-automated baseline that actually clears the target.

Reproduce: `cd code && uv run python matcher_comparison.py`

| Matcher | Type | Anchors scored | Median err | p90 | Within 15 m |
|---|---|---|---|---|---|
| **RoMa** | dense deep | 44 | **65.0 m** | 149 m | **23 %** |
| SIFT | sparse classical | 158 | 75.1 m | 147 m | 0 % |
| SuperPoint + LightGlue | sparse deep | 218 | 109.2 m | 213 m | 1 % |
| — | | | | | |
| **Manual anchoring** | VO + human marks + fusion | 3,430 | **4.1 m** | — | **98 %** |

## Findings

1. **Dense matching wins among automated methods.** RoMa (dense) gives the
   lowest median error (65 m) and by far the most fixes within 15 m (23 %).
2. **Classical SIFT is competitive.** At 75 m median it beats SuperPoint+LightGlue
   on median, though it never lands within 15 m on this flight.
3. **Sparse deep features struggle most here.** SuperPoint+LightGlue has the
   worst median (109 m). The oblique, low-altitude, often-featureless (grass)
   drone view against a top-down satellite tile is a large viewpoint/appearance
   gap that hurts sparse keypoint matching more than dense matching.
4. **No automated matcher is good enough alone.** All three sit at 60–110 m
   median with ≤ 23 % within 15 m — which is exactly why **manual anchoring**
   (VO + a handful of human-marked anchors + piecewise drift correction) was
   added, reaching **4.1 m median, 98 % within 15 m**. Manual anchoring is a
   different *system*, not a different matcher, so it is shown separately.

## Caveat — equal scoring, not identical inputs

Each automated run used its **own satellite source, zoom, and frame sampling**
(SIFT → Google satellite; SuperPoint+LightGlue → ESRI World Imagery; RoMa → its
own tile), and a different set of candidate frames. Every estimate is scored
identically against the same GPS truth by timestamp, so this is a fair comparison
of the **three pipelines as built** — but not a perfectly controlled isolation of
the matchers alone. Re-running all three through one anchoring harness (same tile,
same frames) would remove the confound but needs each matcher's model + a GPU.

## Sources

- **SIFT** — `notebooks/baseline_sift.ipynb`, `code/sift/*.py`, full outputs in
  `results/sift_baseline/` (originally branch `initial-sift-baseline`). Estimates:
  `results/matcher_comparison/estimates/sift_anchors.csv`.
- **RoMa** — `notebooks/roma_localize_v2.ipynb`, full outputs in
  `results/roma_v2/` (originally branch `feature/roma`). Estimates:
  `results/matcher_comparison/estimates/roma_anchors.csv`.
- **SuperPoint + LightGlue** — `notebooks/deep_learning_localization_v2.ipynb`,
  `results/deep-learning-6inliers/` (218-anchor run). Estimates:
  `results/matcher_comparison/estimates/lightglue_anchors.csv`.
- **Manual anchoring** — `results/benchmark/manual_anchoring.md`.
