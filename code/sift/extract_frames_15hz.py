import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw/dronomy_video.mp4")
OUTPUT_DIR = Path("data/frames_sample")

TARGET_FRAMES = [0, 100, 250, 300, 400, 500]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = total_frames / video_fps if video_fps else 0

print(f"Video FPS: {video_fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration_seconds:.2f} seconds")
print(f"Extracting {len(TARGET_FRAMES)} sample frames")

for frame_idx in TARGET_FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame {frame_idx}")
        continue

    output_path = OUTPUT_DIR / f"frame_original_{frame_idx:05d}.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"Saved {output_path}")

cap.release()

print(f"Done. Saved selected frames to {OUTPUT_DIR}")