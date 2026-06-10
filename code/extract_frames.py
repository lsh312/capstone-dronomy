"""
Extract frames from drone video at 15 Hz for computer vision analysis.
Outputs frames as JPEG images named by their timestamp in milliseconds.
"""

import cv2
import os
import sys
from pathlib import Path

VIDEO_PATH = Path("../data/IE_Challenge_lat43_521955_lon5_624290.MP4")
OUTPUT_DIR = Path("../data/frames_15hz")
TARGET_FPS = 15


def extract_frames(video_path: Path, output_dir: Path, target_fps: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open {video_path}", file=sys.stderr)
        sys.exit(1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / source_fps if source_fps > 0 else 0

    print(f"Video:      {video_path.name}")
    print(f"Source FPS: {source_fps:.3f}")
    print(f"Duration:   {duration_s:.1f}s  ({total_frames} frames)")
    print(f"Target FPS: {target_fps} Hz")
    print(f"Output dir: {output_dir}")
    expected_output = int(duration_s * target_fps)
    print(f"Expected snapshots: ~{expected_output}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Interval between sampled frames in source-frame units
    frame_interval = source_fps / target_fps
    next_sample = 0.0   # next source frame index to capture
    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_sample:
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            filename = output_dir / f"frame_{timestamp_ms:010d}ms.jpg"
            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            next_sample += frame_interval

            if saved % 150 == 0:
                pct = frame_idx / total_frames * 100 if total_frames else 0
                print(f"  {saved} frames saved  ({pct:.1f}% of video)")

        frame_idx += 1

    cap.release()
    print(f"\nDone. {saved} frames saved to {output_dir}/")


if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_DIR, TARGET_FPS)
