import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH = ROOT / "data" / "raw" / "dronomy_video.mp4"
SAT_PATH = ROOT / "data" / "google_satellite.jpg"
OUTPUT_DIR = ROOT / "experiments" / "sift" / "results" / "sift_baseline" / "sift_gps_estimates"

LAT_CENTER = 43.521955
LON_CENTER = -5.624290
ZOOM = 20
SIZE = 640
SCALE = 2

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
        return img, 1.0
    scale = max_width / w
    resized = cv2.resize(img, (max_width, int(h * scale)))
    return resized, scale


def latlon_to_world_px(lat, lon, zoom):
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)

    x = 256 * (0.5 + lon / 360)
    y = 256 * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

    scale = 2 ** zoom
    return x * scale, y * scale


def world_px_to_latlon(x, y, zoom):
    scale = 2 ** zoom

    lon = (x / (256 * scale)) * 360.0 - 180.0

    n = math.pi - 2.0 * math.pi * y / (256 * scale)
    lat = math.degrees(math.atan(math.sinh(n)))

    return lat, lon


def satellite_pixel_to_latlon(px, py):
    image_size_px = SIZE * SCALE

    center_world_x, center_world_y = latlon_to_world_px(
        LAT_CENTER, LON_CENTER, ZOOM
    )

    world_x = center_world_x + (px - image_size_px / 2)
    world_y = center_world_y + (py - image_size_px / 2)

    return world_px_to_latlon(world_x, world_y, ZOOM)


sat = cv2.imread(str(SAT_PATH))
if sat is None:
    raise FileNotFoundError(f"Could not load satellite image: {SAT_PATH}")

sat, sat_scale = resize_keep_aspect(sat)

sat_gray = cv2.cvtColor(sat, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_sat, des_sat = sift.detectAndCompute(sat_gray, None)

if des_sat is None:
    raise RuntimeError("No satellite descriptors found.")

bf = cv2.BFMatcher(cv2.NORM_L2)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = max(round(video_fps / TARGET_FPS), 1)

rows = []

frame_idx = 0
extracted_idx = 0

pbar = tqdm(total=total_frames)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % frame_interval == 0:
        if extracted_idx % EVAL_EVERY_N_FRAMES == 0:
            start = time.time()

            frame_resized, frame_scale = resize_keep_aspect(frame)
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

            success = False
            reason = "unknown"
            good_matches_count = 0
            inliers = 0
            inlier_ratio = 0.0
            est_lat = None
            est_lon = None
            sat_px = None
            sat_py = None

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
                        src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD
                    )

                    if H is None or mask is None:
                        reason = "homography_failed"
                    else:
                        inliers = int(mask.ravel().sum())
                        inlier_ratio = inliers / len(good_matches)

                        if inliers >= MIN_INLIERS:
                            success = True
                            reason = "success"

                            h, w = frame_resized.shape[:2]
                            center = np.float32([[[w / 2, h / 2]]])
                            projected_center = cv2.perspectiveTransform(center, H)

                            sat_px = float(projected_center[0, 0, 0])
                            sat_py = float(projected_center[0, 0, 1])

                            # Convert back to original Google image coordinate if resized
                            sat_px_original = sat_px / sat_scale
                            sat_py_original = sat_py / sat_scale

                            est_lat, est_lon = satellite_pixel_to_latlon(
                                sat_px_original, sat_py_original
                            )
                        else:
                            reason = "not_enough_inliers"

            runtime = time.time() - start

            rows.append({
                "original_frame_idx": frame_idx,
                "extracted_15hz_idx": extracted_idx,
                "time_sec": frame_idx / video_fps,
                "success": success,
                "reason": reason,
                "frame_keypoints": len(kp_frame) if des_frame is not None else 0,
                "satellite_keypoints": len(kp_sat),
                "good_matches": good_matches_count,
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "sat_px": sat_px,
                "sat_py": sat_py,
                "estimated_lat": est_lat,
                "estimated_lon": est_lon,
                "runtime_sec": runtime,
            })

        extracted_idx += 1

    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()

df = pd.DataFrame(rows)

output_csv = OUTPUT_DIR / "sift_gps_estimates.csv"
df.to_csv(output_csv, index=False)

print()
print("SIFT GPS Estimate Generation")
print("============================")
print(f"Saved: {output_csv}")
print(f"Frames tested: {len(df)}")
print(f"Successful GPS estimates: {df['success'].sum()}")
print(f"Success rate: {df['success'].mean():.3f}")
print(f"Mean runtime/frame: {df['runtime_sec'].mean():.3f} sec")
print(f"Approx FPS: {1 / df['runtime_sec'].mean():.2f}")
print()
print("First successful estimates:")
print(df[df["success"]][["time_sec", "estimated_lat", "estimated_lon", "inliers", "inlier_ratio"]].head())