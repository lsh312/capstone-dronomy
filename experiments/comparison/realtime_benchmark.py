"""
Real-time latency micro-benchmark.

Times the per-frame cost of frame-to-frame (VO-style) feature matching for three
matchers on the SAME hardware and resolution so they are comparable: SIFT and
ORB (classical, OpenCV/CPU) and SuperPoint+LightGlue (deep, MPS/GPU). Each
"frame" = extract features on the new frame + match to the previous frame +
RANSAC homography, which is exactly the VO inner loop.

RoMa is not timed here: `romatch` is not installed and dense matching needs a
GPU. Qualitatively it is the heaviest of the four (dense warp over the whole
image), so it is the least real-time-friendly per frame.

Budgets:
  * VO loop runs every frame -> 66.7 ms/frame at 15 Hz (the hard constraint).
  * Anchor loop runs periodically -> budget = anchor cadence (e.g. ~1000 ms at
    1 Hz), far looser, so a slow matcher can still anchor in time.

Run: cd code && uv run python realtime_benchmark.py [n_frames]
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES = sorted(glob.glob(str(ROOT / "data" / "frames_15hz" / "*.jpg")))
SIZE = (640, 480)            # pipeline VO resolution (DOWNSAMPLE_SIZE)
BUDGET_MS = 1000.0 / 15.0    # 15 Hz VO budget
WARMUP = 3                   # discard first N timings (lazy init / cache warmup)


def stats(times: np.ndarray) -> dict:
    t = times[WARMUP:]
    return {"mean": float(t.mean()), "median": float(np.median(t)),
            "p90": float(np.percentile(t, 90)), "fps": 1000.0 / float(t.mean())}


def time_classical(kind: str, grays: list, ratio: float = 0.75) -> np.ndarray:
    if kind == "sift":
        det, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
    else:
        det, bf = cv2.ORB_create(nfeatures=1024), cv2.BFMatcher(cv2.NORM_HAMMING)
    prev_kp, prev_des = det.detectAndCompute(grays[0], None)
    times = []
    for g in grays[1:]:
        t0 = time.perf_counter()
        kp, des = det.detectAndCompute(g, None)
        if des is not None and prev_des is not None and len(des) > 1 and len(prev_des) > 1:
            m = bf.knnMatch(prev_des, des, k=2)
            good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
            if len(good) >= 4:
                src = np.float32([prev_kp[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
                cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        times.append((time.perf_counter() - t0) * 1000)
        prev_kp, prev_des = kp, des
    return np.array(times)


def time_superpoint_lightglue(rgbs: list) -> tuple[np.ndarray, str]:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    torch.set_grad_enabled(False)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    ext = SuperPoint(max_num_keypoints=1024).eval().to(dev)
    mat = LightGlue(features="superpoint").eval().to(dev)

    def feats(rgb):
        t = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
        return ext.extract(t.to(dev))

    prev = feats(rgbs[0])
    times = []
    for rgb in rgbs[1:]:
        t0 = time.perf_counter()
        cur = feats(rgb)
        out = mat({"image0": prev, "image1": cur})
        _ = rbd(out)["matches"]
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        prev = cur
    return np.array(times), dev.type


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    paths = FRAMES[:n]
    grays = [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), SIZE) for p in paths]
    rgbs = [cv2.cvtColor(cv2.resize(cv2.imread(p), SIZE), cv2.COLOR_BGR2RGB) for p in paths]
    print(f"VO-style frame-to-frame latency  |  {len(paths)} frames @ {SIZE[0]}x{SIZE[1]}  "
          f"|  15 Hz budget = {BUDGET_MS:.1f} ms/frame\n")

    rows = []
    for kind, label, dev in [("sift", "SIFT", "CPU"), ("orb", "ORB", "CPU")]:
        s = stats(time_classical(kind, grays))
        rows.append((label, dev, s))
    splg_times, splg_dev = time_superpoint_lightglue(rgbs)
    rows.append(("SuperPoint+LightGlue", splg_dev.upper(), stats(splg_times)))

    print(f"{'Matcher':22} {'Device':7} {'mean':>8} {'median':>8} {'p90':>8} "
          f"{'fps':>6}  {'15 Hz?':>7}")
    print("-" * 72)
    for label, dev, s in rows:
        ok = "yes" if s["mean"] <= BUDGET_MS else "NO"
        print(f"{label:22} {dev:7} {s['mean']:>6.1f}ms {s['median']:>6.1f}ms "
              f"{s['p90']:>6.1f}ms {s['fps']:>5.1f}  {ok:>7}")
    print("-" * 72)
    print("RoMa (dense): not timed here (romatch not installed; needs GPU) — "
          "heaviest per frame.")


if __name__ == "__main__":
    main()
