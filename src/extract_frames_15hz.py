import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw/dronomy_video.mp4")
OUTPUT_DIR = Path("data/frames_15hz")
TARGET_FPS = 15

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

if video_fps <= 0:
    raise ValueError("Could not read video FPS.")

frame_interval = max(round(video_fps / TARGET_FPS), 1)

print(f"Target FPS: {TARGET_FPS}")
print(f"Saving every {frame_interval} frame(s)")

frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % frame_interval == 0:
        output_path = OUTPUT_DIR / f"frame_{saved_idx:05d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved_idx += 1

    frame_idx += 1

cap.release()

actual_output_fps = video_fps / frame_interval

print(f"Saved {saved_idx} frames to {OUTPUT_DIR}")
print(f"Approximate output FPS: {actual_output_fps:.2f}")
