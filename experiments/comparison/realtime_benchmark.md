# Real-Time Characterization

The brief asks the system to match **real-time** frames. "Real-time" here is not
one number — the pipeline has two loops with very different budgets:

| Loop | Runs | Budget | Why |
|---|---|---|---|
| **VO** (frame→frame) | every frame | **66.7 ms** @ 15 Hz | must finish before the next frame or it falls behind the stream |
| **Anchoring** (frame→satellite) | periodically (1 Hz) | **~1000 ms** | VO carries position between anchors, so this just has to beat the anchor cadence |

The system is *designed* so the expensive part (satellite matching) runs rarely
and the cheap part (VO) runs every frame. So "is it real-time?" splits into two
questions answered separately below.

## VO loop — the hard constraint (measured, same hardware)

Per-frame cost of VO-style matching (extract + match-to-previous + RANSAC) at the
pipeline's 640×480, measured by `code/realtime_benchmark.py`:

| Matcher | Device | Mean | Median | p90 | fps | Meets 15 Hz? |
|---|---|---|---|---|---|---|
| ORB | CPU | 7.6 ms | 7.6 ms | 7.9 ms | 131 | ✅ easily |
| SIFT | CPU | 83.9 ms | 83.2 ms | 91.1 ms | 12 | ❌ |
| SuperPoint+LightGlue | MPS (this Mac) | 280.6 ms | 251.8 ms | 413 ms | 3.6 | ❌ (4× over) |
| SuperPoint+LightGlue | CUDA (logged) | 75.8 ms | — | — | 13 | ⚠️ just short (1.1× over) |

**Reading:** the *accurate* matcher (SuperPoint+LightGlue) is **not** real-time on
this Mac's MPS (4× too slow) and only *marginally* short on a CUDA GPU (~13 fps).
The only matcher that comfortably clears 15 Hz is **ORB** (131 fps) — but ORB is
the least accurate/robust. So real-time VO is a **speed-vs-accuracy trade-off**:
ORB on CPU, or SuperPoint+LightGlue on a GPU stronger than the one logged.

## Anchoring loop — the loose constraint (1 Hz, ~1000 ms budget)

| Matcher | Per-anchor latency | Fits 1 Hz cadence? |
|---|---|---|
| SIFT (frame→satellite, full-res) | 142 ms (logged) | ✅ ~7× headroom |
| SuperPoint+LightGlue (frame→satellite) | not separately logged; heavier than VO (max_kp 4096, larger tile) — order of a few hundred ms | ✅ likely |
| RoMa (dense) | not measured (`romatch` not installed; needs GPU); heaviest of the four | ⚠️ may exceed 1 s — but the cadence is tunable (anchor every few seconds) |

Because anchoring is periodic, even a slow matcher works here — you control the
cadence. SIFT already fits with large margin; for RoMa you'd simply anchor less
often and let VO bridge the gap.

## Verdict

**Real-time is achievable, but it's a hardware/matcher decision, and the current
Mac/MPS setup is not real-time for the accurate VO.**

- **Anchoring loop:** not a bottleneck — periodic, comfortably within budget for
  classical/sparse matchers; slow matchers just anchor less often.
- **VO loop:** the binding constraint. Real-time at 15 Hz needs either **ORB**
  (fast, lower accuracy) or **SuperPoint+LightGlue on a capable GPU** (the logged
  CUDA run was ~13 fps — close; a stronger GPU clears it). On Apple MPS, the
  accurate matcher runs at ~3.6 fps — usable offline, not live.

This matches the project's actual usage: the pipeline was run **offline on
recorded frames**, which is why MPS latency was acceptable. A live deployment
would pick the matcher/hardware per the table above.

Reproduce: `cd code && uv run python realtime_benchmark.py [n_frames]`
