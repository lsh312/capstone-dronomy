# Heading / Orientation Benchmark

The brief asks for absolute **orientation** alongside position. This scores a
matcher's estimated heading (`est_yaw`) against the DJI gimbal-yaw truth
(`gb_yaw`) in the `.SRT`, using the same timestamp alignment as the position
benchmark. Circular (wrap-aware) error throughout.

Reproduce: `cd code && uv run python heading_benchmark.py`

| Matcher | Scored | Raw median err | Best constant offset | Corrected median | Within 30° |
|---|---|---|---|---|---|
| RoMa | 44 | 74.5° | 62.9° | **64.3°** | 18% |

*Only RoMa logged an absolute per-frame heading; SuperPoint+LightGlue and SIFT
did not record orientation, so they cannot be scored here.*

## Finding: automated orientation is currently unreliable

Removing the single best constant offset barely changes the error (74.5° → 64.3°).
That's the decisive test:

- **If** corrected error collapsed toward 0, the heading would be *informative
  but mis-referenced* — a fixable axis/convention bug.
- **It doesn't**, so the heading is **genuinely uninformative**. A uniformly
  random heading would give ~90° median and ~17% within 30°; RoMa's 64° / 18%
  is only marginally better than chance.

## Why

The drone camera is **nadir** (`gb_pitch ≈ -90°`, pointing straight down) over
**low-texture, rotationally-ambiguous terrain** (grass, fields). A homography
between the drone frame and a top-down satellite tile can localize *position*
reasonably (RoMa ~65 m) but cannot pin *absolute heading*: rotating a featureless
field by 30° looks much the same, so the recovered orientation is near-random.

## Note on visual-odometry yaw

The VO branch (`vo_run.py`) produces a **relative** heading (`cum_yaw`,
accumulated frame-to-frame rotation). After the world-frame integration fix it is
internally consistent and usable for *incremental* rotation, but it is not an
*absolute* heading — it has no north reference of its own. Turning VO's relative
yaw into absolute orientation would require an external heading fix (e.g. a
reliable anchor heading), which the automated satellite match does not currently
provide.

## Implication for the brief

Absolute orientation is **measured but not yet solved**. Position is delivered
(automated ~65 m; manual-anchored 4.1 m), but heading from satellite matching is
not usable on this nadir, low-texture flight. Plausible next steps: an oblique
(non-nadir) camera to break rotational symmetry, matching against higher-texture
imagery, or fusing a magnetometer/IMU heading if the platform exposes one.
