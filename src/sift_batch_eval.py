import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

VIDEO_PATH = Path("data/raw/dronomy_video.mp4")
SAT_PATH = Path("data/google_satellite.jpg")
OUTPUT_DIR = Path("outputs/sift_batch_eval")

TARGET_FPS = 15
EVAL_EVERY_N_FRAMES = 15

MAX_WIDTH = 1200
LOWE_RATIO = 0.90
RANSAC_THRESHOLD = 5.0
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resize_keep_aspect(img, max_width=MAX_WIDTH):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)))


sat = cv2.imread(str(SAT_PATH))
if sat is None:
    raise FileNotFoundError(f"Could not load satellite image: {SAT_PATH}")

sat = resize_keep_aspect(sat)
sat_gray = cv2.cvtColor(sat, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_sat, des_sat = sift.detectAndCompute(sat_gray, None)

if des_sat is None:
    raise RuntimeError("No satellite descriptors found.")

print(f"Satellite keypoints: {len(kp_sat)}")

bf = cv2.BFMatcher(cv2.NORM_L2)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_interval = max(round(video_fps / TARGET_FPS), 1)

print(f"Video FPS: {video_fps}")
print(f"Total frames: {total_frames}")
print(f"15 Hz interval: every {frame_interval} original frames")
print(f"Evaluating every {EVAL_EVERY_N_FRAMES} extracted 15Hz frames")

rows = []

extracted_idx = 0
frame_idx = 0

pbar = tqdm(total=total_frames)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % frame_interval == 0:
        if extracted_idx % EVAL_EVERY_N_FRAMES == 0:
            start_time = time.time()

            frame_resized = resize_keep_aspect(frame)
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

            success = False
            reason = "unknown"
            good_matches_count = 0
            inliers = 0
            inlier_ratio = 0.0

            if des_frame is None:
                reason = "no_descriptors"
            else:
                matches = bf.knnMatch(des_frame, des_sat, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < LOWE_RATIO * n.distance:
                        good_matches.append(m)

                good_matches_count = len(good_matches)

                if len(good_matches) < MIN_GOOD_MATCHES:
                    reason = "not_enough_matches"
                else:
                    src_pts = np.float32(
                        [kp_frame[m.queryIdx].pt for m in good_matches]
                    ).reshape(-1, 1, 2)

                    dst_pts = np.float32(
                        [kp_sat[m.trainIdx].pt for m in good_matches]
                    ).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(
                        src_pts,
                        dst_pts,
                        cv2.RANSAC,
                        RANSAC_THRESHOLD,
                    )

                    if H is None or mask is None:
                        reason = "homography_failed"
                    else:
                        inliers = int(mask.ravel().sum())
                        inlier_ratio = inliers / len(good_matches)

                        if inliers >= MIN_INLIERS:
                            success = True
                            reason = "success"
                        else:
                            reason = "not_enough_inliers"

            runtime = time.time() - start_time

            rows.append({
                "original_frame_idx": frame_idx,
                "extracted_15hz_idx": extracted_idx,
                "success": success,
                "reason": reason,
                "frame_keypoints": len(kp_frame) if des_frame is not None else 0,
                "satellite_keypoints": len(kp_sat),
                "good_matches": good_matches_count,
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "runtime_sec": runtime,
            })

        extracted_idx += 1

    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()

df = pd.DataFrame(rows)

csv_path = OUTPUT_DIR / "sift_batch_metrics.csv"
df.to_csv(csv_path, index=False)

print()
print("SIFT Batch Evaluation")
print("=====================")
print(f"Results saved to: {csv_path}")
print(f"Frames tested: {len(df)}")
print(f"Successful anchors: {df['success'].sum()}")
print(f"Success rate: {df['success'].mean():.3f}")
print(f"Mean good matches: {df['good_matches'].mean():.2f}")
print(f"Median good matches: {df['good_matches'].median():.2f}")
print(f"Mean inliers: {df['inliers'].mean():.2f}")
print(f"Median inliers: {df['inliers'].median():.2f}")
print(f"Mean inlier ratio: {df['inlier_ratio'].mean():.3f}")
print(f"Mean runtime/frame: {df['runtime_sec'].mean():.3f} sec")
print(f"Approx FPS: {1 / df['runtime_sec'].mean():.2f}")
print()
print("Failure reasons:")
print(df["reason"].value_counts())