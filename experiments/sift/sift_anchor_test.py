import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DRONE_PATH = ROOT / "data" / "drone_test.jpg"
SAT_PATH = ROOT / "data" / "google_satellite.jpg"
OUTPUT_DIR = ROOT / "experiments" / "sift" / "results" / "sift_baseline" / "sift_anchor_test"

MAX_WIDTH = 1200
LOWE_RATIO = 0.90
RANSAC_THRESHOLD = 5.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resize_keep_aspect(img, max_width=MAX_WIDTH):
    h, w = img.shape[:2]

    if w <= max_width:
        return img

    scale = max_width / w
    new_size = (max_width, int(h * scale))

    return cv2.resize(img, new_size)


drone = cv2.imread(str(DRONE_PATH))
sat = cv2.imread(str(SAT_PATH))

if drone is None:
    raise FileNotFoundError(f"Could not load {DRONE_PATH}")

if sat is None:
    raise FileNotFoundError(f"Could not load {SAT_PATH}")

drone = resize_keep_aspect(drone)
sat = resize_keep_aspect(sat)

gray_drone = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
gray_sat = cv2.cvtColor(sat, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray_drone, None)
kp2, des2 = sift.detectAndCompute(gray_sat, None)

print(f"Drone keypoints: {len(kp1)}")
print(f"Satellite keypoints: {len(kp2)}")

if des1 is None or des2 is None:
    raise RuntimeError("Could not compute descriptors for one of the images.")

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < LOWE_RATIO * n.distance:
        good_matches.append(m)

print(f"Good matches: {len(good_matches)}")

drone_rgb = cv2.cvtColor(drone, cv2.COLOR_BGR2RGB)
sat_rgb = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)

match_img = cv2.drawMatches(
    drone_rgb,
    kp1,
    sat_rgb,
    kp2,
    good_matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

matches_path = OUTPUT_DIR / "sift_matches.jpg"

plt.figure(figsize=(18, 10))
plt.imshow(match_img)
plt.title("SIFT Matches")
plt.axis("off")
plt.savefig(matches_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved matches visualization to {matches_path}")

if len(good_matches) < 4:
    print("Not enough matches for homography.")
    raise SystemExit

src_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    RANSAC_THRESHOLD,
)

if H is None or mask is None:
    print("Homography estimation failed.")
    raise SystemExit

inliers = int(mask.ravel().sum())
inlier_ratio = inliers / len(good_matches)

print(f"Inliers: {inliers}")
print(f"Inlier ratio: {inlier_ratio:.3f}")

h, w = drone_rgb.shape[:2]

corners = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h],
]).reshape(-1, 1, 2)

projected = cv2.perspectiveTransform(corners, H)

sat_box = sat_rgb.copy()

sat_box = cv2.polylines(
    sat_box,
    [np.int32(projected)],
    True,
    (255, 0, 0),
    4,
)

location_path = OUTPUT_DIR / "estimated_location.jpg"

plt.figure(figsize=(10, 10))
plt.imshow(sat_box)
plt.title("Estimated Drone Location")
plt.axis("off")
plt.savefig(location_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved estimated location visualization to {location_path}")