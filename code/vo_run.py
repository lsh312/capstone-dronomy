"""
Standalone visual-odometry runner (SuperPoint + LightGlue), memory-stable on MPS.

Mirrors notebook Cell 4 but adds periodic MPS cache clearing so per-frame time
stays flat instead of degrading. Writes an incremental CSV checkpoint so the run
can be monitored and the notebook can just load the result (no in-notebook VO).

    uv run python vo_run.py
"""
from __future__ import annotations
import csv, gc, math, time
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

FRAMES_DIR = Path("../data/frames_15hz")
OUT_CSV    = Path("../results/nb3v2/dl_vo_log_full.csv")
DOWNSAMPLE_SIZE = (640, 480)
import os
MAX_KP_VO  = int(os.environ.get("VO_MAX_KP", "1024"))
EMPTY_EVERY = int(os.environ.get("VO_EMPTY_EVERY", "25"))   # clear MPS cache every N frames
REPORT_EVERY = 200

torch.set_grad_enabled(False)
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_VO_DEVICE = os.environ.get("VO_DEVICE", "auto")
if _VO_DEVICE != "auto":
    device = torch.device(_VO_DEVICE)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
_MAX_FRAMES = int(os.environ.get("VO_MAX_FRAMES", "0")) or None
print(f"Device: {device}  max_frames: {_MAX_FRAMES}", flush=True)

extractor = SuperPoint(max_num_keypoints=MAX_KP_VO).eval().to(device)
matcher   = LightGlue(features="superpoint", depth_confidence=0.9,
                      width_confidence=0.95).eval().to(device)


def load_img_tensor(path, size):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    return t.to(device)


def extract_sp(t):
    return extractor.extract(t)


def match_sp_lg(f0, f1):
    res = matcher({"image0": f0, "image1": f1})
    a, b, r = rbd(f0), rbd(f1), rbd(res)
    m = r["matches"]
    if len(m) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), 0, 0.0
    p0 = a["keypoints"][m[:, 0]].cpu().numpy()
    p1 = b["keypoints"][m[:, 1]].cpu().numpy()
    sc = r.get("matching_scores", torch.ones(len(m))).cpu().numpy()
    return p0, p1, len(m), float(sc.mean())


def find_homography(p0, p1):
    if len(p0) < 4:
        return None, 0
    H, mask = cv2.findHomography(p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2),
                                 cv2.RANSAC, 5.0)
    return H, (int(mask.sum()) if mask is not None else 0)


def main():
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if _MAX_FRAMES:
        frame_paths = frame_paths[:_MAX_FRAMES]
    print(f"{len(frame_paths)} frames", flush=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cum_x = cum_y = cum_yaw = 0.0
    rows = [{"frame_idx": 0, "cum_x": 0.0, "cum_y": 0.0, "cum_yaw": 0.0,
             "dx": 0.0, "dy": 0.0, "dyaw": 0.0,
             "good_matches": 0, "inliers": 0, "inlier_ratio": 1.0, "mean_score": 1.0}]

    prev = extract_sp(load_img_tensor(frame_paths[0], DOWNSAMPLE_SIZE))
    h, w = DOWNSAMPLE_SIZE[1], DOWNSAMPLE_SIZE[0]
    center = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)

    total = len(frame_paths) - 1
    timings, t_start = [], time.perf_counter()
    for i in range(total):
        t0 = time.perf_counter()
        curr = extract_sp(load_img_tensor(frame_paths[i + 1], DOWNSAMPLE_SIZE))
        p0, p1, n, score = match_sp_lg(prev, curr)
        H, inliers = find_homography(p0, p1)
        if H is not None:
            mapped = cv2.perspectiveTransform(center, H).reshape(2)
            dx, dy = mapped[0] - w / 2, mapped[1] - h / 2
            dyaw = math.degrees(math.atan2(H[1, 0], H[0, 0]))
            ratio = inliers / n if n else 0.0
        else:
            dx = dy = dyaw = 0.0; ratio = 0.0
        # Rotate the per-frame image-plane translation into a fixed world frame
        # by the heading accumulated so far, THEN integrate. Summing (dx, dy) in
        # the rotating camera frame collapses the path onto one axis and produces
        # km-scale phantom drift. cum_yaw is updated after use, so each step
        # rotates by the heading at its start.
        yaw = math.radians(cum_yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cum_x += cos_y * dx - sin_y * dy
        cum_y += sin_y * dx + cos_y * dy
        cum_yaw += dyaw
        rows.append({"frame_idx": i + 1, "cum_x": cum_x, "cum_y": cum_y, "cum_yaw": cum_yaw,
                     "dx": float(dx), "dy": float(dy), "dyaw": dyaw,
                     "good_matches": n, "inliers": inliers,
                     "inlier_ratio": ratio, "mean_score": score})
        prev = curr
        timings.append((time.perf_counter() - t0) * 1000)

        if (i + 1) % EMPTY_EVERY == 0 and device.type == "mps":
            torch.mps.empty_cache()
        if (i + 1) % REPORT_EVERY == 0:
            gc.collect()
            recent = np.mean(timings[-REPORT_EVERY:])
            print(f"  {i+1}/{total}  mean {recent:.0f} ms/frame  "
                  f"elapsed {time.perf_counter()-t_start:.0f}s", flush=True)
            _write(rows)  # checkpoint

    _write(rows)
    print(f"DONE {total} pairs in {time.perf_counter()-t_start:.0f}s  "
          f"mean {np.mean(timings):.0f} ms/frame  -> {OUT_CSV}", flush=True)


def _write(rows):
    with open(OUT_CSV, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)


if __name__ == "__main__":
    main()
