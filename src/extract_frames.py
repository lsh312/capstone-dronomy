"""
Extract frames from drone video at 15 Hz for computer vision analysis.
Outputs frames as JPEG images named by their timestamp in milliseconds.

Usage:
    uv run python src/extract_frames.py
    uv run python src/extract_frames.py --max-seconds 5   # test run (75 frames)
"""
import argparse
import sys
import cv2
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "IE_Challenge_lat43_521955_lon5_624290.MP4"
OUTPUT_DIR = ROOT / "data" / "frames_15hz"
TARGET_FPS = 15


def extract_frames(video_path: Path, output_dir: Path,
                   target_fps: float, max_seconds: float | None = None) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open {video_path}", file=sys.stderr)
        sys.exit(1)

    source_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / source_fps if source_fps > 0 else 0
    clip_s       = min(max_seconds, duration_s) if max_seconds else duration_s

    print(f"Video:      {video_path.name}")
    print(f"Source FPS: {source_fps:.3f}")
    print(f"Duration:   {duration_s:.1f}s  ({total_frames} frames)")
    print(f"Target FPS: {target_fps} Hz")
    print(f"Extracting: {clip_s:.1f}s"
          + ("  (full video)" if not max_seconds else f"  (--max-seconds {max_seconds})"))
    print(f"Output dir: {output_dir}")
    print(f"Expected snapshots: ~{int(clip_s * target_fps)}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = source_fps / target_fps
    next_sample    = 0.0
    saved          = 0
    frame_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if max_seconds and timestamp_ms / 1000.0 > max_seconds:
            break

        if frame_idx >= next_sample:
            filename = output_dir / f"frame_{timestamp_ms:010d}ms.jpg"
            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved     += 1
            next_sample += frame_interval

            if saved % 150 == 0:
                pct = frame_idx / total_frames * 100 if total_frames else 0
                print(f"  {saved} frames saved  ({pct:.1f}% of video)")

        frame_idx += 1

    cap.release()
    print(f"\nDone. {saved} frames saved to {output_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract frames from drone video")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop after this many seconds (default: full video)")
    ap.add_argument("--video",      type=Path, default=VIDEO_PATH)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--fps",        type=float, default=TARGET_FPS)
    args = ap.parse_args()

    extract_frames(args.video, args.output_dir, args.fps, args.max_seconds)
