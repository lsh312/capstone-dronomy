# Manual Anchoring — Results

Manual anchoring replaces the noisy automated satellite anchors (~100 m median
error) with a handful of **human-marked** reference points, then drift-corrects
the VO trajectory between them. Validated end-to-end against DJI GPS truth.

## Live result (14 human-marked anchors)

Real run on the full flight (3,430 frames), VO computed fresh (SuperPoint +
LightGlue, 256 kp), anchors marked by hand in Google Earth, scored against the
DJI GPS track:

| Metric | Automated anchors | **Manual (14 anchors)** |
|---|---|---|
| Median error | ~100 m | **4.1 m** |
| Mean error | ~130 m | 5.1 m |
| RMSE | — | 6.4 m |
| p90 error | ~180 m | 10.5 m |
| Max error | >2 km | 17.2 m |
| Within 15 m | ~4 % | **98 %** (3,367 / 3,430) |

Global VO→world fit: scale 0.0232 m/px, rotation 95.6°, anchor residual 49.7 m
(that residual is pure VO drift — the piecewise correction removes it, which is
why the fused median is 4.1 m, not 49.7 m). Dashboard: the fused path traces real
roads and field edges over the satellite tile (`manual_anchored_dashboard.png`).

**The remaining error lives entirely in the gaps between anchors** — the
error-vs-frame curve sits under ~10 m except three small bumps (≈ frames 700,
2000, 3300) mid-gap. Adding one anchor in each flattens them; 14 anchors already
clears the brief's target.

## Method

1. A human marks a few drone frames on the satellite map / Google Earth and
   records each location's lat/lon (`MANUAL_ANCHORS` in Cell 7b, or click them
   on the tile with `ma.click_anchors_on_tile`).
2. **Global fit** (`fit_vo_to_world`): a single similarity transform
   (scale + rotation + translation) aligns the whole VO trajectory to world
   metres by least squares (Umeyama). This sets scale and absolute orientation.
3. **Piecewise drift correction** (`fuse_vo_with_anchors`): the residual at each
   anchor is cancelled and linearly interpolated along the track between anchors,
   removing the VO drift a single global transform can't.
4. The fused path is emitted in real lat/lon and overlaid on the satellite tile
   (the dashboard deliverable).

The DJI GPS is used **only to score** accuracy — never to create anchors. The
localization itself stays GPS-free.

## Result: error vs. number of manual anchors

Measured on the Run-3 VO trajectory (3 430 frames, 228.6 s) against DJI GPS,
with anchors placed at their true locations (the accuracy ceiling):

| Manual anchors | Global fit median | **Piecewise-fused median** | p90 | within 15 m |
|---|---|---|---|---|
| Automated (current) | — | ~100 m | ~180 m | ~4 % |
| 3 | 81 m | 81 m | 183 m | 13 % |
| 4 | 59 m | 36 m | 92 m | 15 % |
| 6 | 56 m | 13 m | 34 m | 57 % |
| 10 | 54 m | **6 m** | 18 m | 87 % |
| 15 | 54 m | **3 m** | 10 m | 95 % |

**Two findings:**

1. A single global transform plateaus at ~54 m no matter how many anchors — that
   is residual **VO drift**, which a global scale+rotation cannot absorb.
2. Piecewise correction between anchors removes it: with ~10–15 well-spread
   anchors the median error reaches the **few-metres** target the brief asks for.

## Result: how precise must the marks be?

Marking error propagates almost 1:1 to trajectory error (10 anchors, averaged):

| Marking error (per click) | Trajectory median | within 15 m |
|---|---|---|
| perfect | 6 m | 86 % |
| ± 5 m | 8 m | 84 % |
| ± 10 m | 12 m | 68 % |
| ± 20 m | 20 m | 34 % |
| ± 30 m | 31 m | 17 % |

**Practical guidance:** mark sharp, identifiable features (road/building corners,
crosswalk ends) on high-zoom imagery, where a few-metre click is realistic. Aim
for ~10–15 anchors spread evenly across the flight. The current max error
(~40 m even at 15 anchors) sits at one sharp VO drift excursion — placing an
extra anchor near it would pull that down too.

## Files

- `code/manual_anchors.py` — anchoring, VO→world similarity fit, piecewise
  fusion, validation, interactive click helper.
- Notebook **Cell 7b** — fill `MANUAL_ANCHORS`, run, get the fused GPS path +
  dashboard overlay + truth-validated error.
- `manual_anchored_dashboard.png` (written on run) — fused path over satellite.

## Bottom line for the mentor meeting

> Manual anchoring takes us from ~100 m to **4.1 m median error (98 % of the
> flight within 15 m)** — comfortably hitting the target, on a live run with 14
> hand-marked anchors. The key insight is that anchor *quality* alone wasn't
> enough; we needed enough well-spread anchors **plus** piecewise drift
> correction, because VO drift can't be removed by a single global scale. No API
> keys required.
