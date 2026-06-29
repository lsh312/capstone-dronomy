"""
Interactive anchor marking tool.

Shows each candidate drone frame next to the satellite tile.
Click once on the satellite tile where the drone is located.
Coordinates are computed from the tile bounding box and saved
directly to data/manual_anchors.json (merges with existing marks).

Controls:
  Left-click  -> mark the drone position on the satellite tile
  Right-click -> skip this frame (mark as reviewed but don't save)
  Middle-click or close window -> quit early (saves what you have so far)

Usage:
    uv run python code/mark_anchors.py
    uv run python code/mark_anchors.py --n-frames 20
    uv run python code/mark_anchors.py --frames 200 450 1000 1300 1900
    uv run python code/mark_anchors.py --redo   # re-mark already marked frames too
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")          # works on Windows; change to "Qt5Agg" if you prefer
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_bbox(data_dir: Path) -> tuple[np.ndarray, dict]:
    """Return (rgb satellite image, bbox dict) for the best available tile."""
    def priority(p: Path) -> tuple[int, int]:
        n = p.stem
        return (0 if "google" in n else 1, -int(n.split("_z")[-1].split(".")[0]) if "_z" in n else 0)
    for sc in sorted(data_dir.glob("satellite_*.bbox.json"), key=priority):
        tile = sc.parent / sc.name.replace(".bbox.json", ".png")
        if tile.exists():
            img  = cv2.cvtColor(cv2.imread(str(tile)), cv2.COLOR_BGR2RGB)
            bbox = json.loads(sc.read_text())
            print(f"Satellite tile: {tile.name}  "
                  f"({bbox['img_width']}×{bbox['img_height']} px, "
                  f"zoom={bbox['zoom']}, provider={bbox['provider']})")
            return img, bbox
    raise SystemExit(
        "No satellite tile found — run fetch_satellite.py first."
    )


def px_to_gps(px: float, py: float, img_w: int, img_h: int, bbox: dict) -> tuple[float, float]:
    lat = bbox["lat_max"] - (py / img_h) * (bbox["lat_max"] - bbox["lat_min"])
    lon = bbox["lon_min"] + (px / img_w) * (bbox["lon_max"] - bbox["lon_min"])
    return lat, lon


def load_existing(json_path: Path) -> dict[int, list[float]]:
    if json_path.exists():
        raw = json.loads(json_path.read_text())
        return {int(k): v for k, v in raw.items()}
    return {}


def save_anchors(json_path: Path, anchors: dict[int, list[float]]) -> None:
    ordered = dict(sorted(anchors.items()))
    json_path.write_text(json.dumps(ordered, indent=2))


def pick_frame_indices(frame_paths: list[Path], n: int,
                       explicit: list[int] | None) -> list[int]:
    if explicit:
        return sorted(set(explicit))
    total = len(frame_paths)
    return sorted(set(int(i) for i in np.linspace(0, total - 1, n)))


def mark_one(frame_path: Path, frame_idx: int,
             sat_img: np.ndarray, bbox: dict,
             existing_marks: dict[int, list[float]],
             done: int, total: int) -> tuple[float, float] | None:
    """Show the frame + satellite tile and wait for a click.
    Returns (lat, lon) or None if skipped/quit."""

    sat_h, sat_w = sat_img.shape[:2]

    drone = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    drone = cv2.resize(drone, (sat_w, sat_h), interpolation=cv2.INTER_AREA)

    fig, (ax_drone, ax_sat) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Frame {frame_idx}  ({done}/{total})  —  "
        "LEFT-CLICK on satellite tile where the drone is  |  "
        "RIGHT-CLICK to skip  |  CLOSE to quit",
        fontsize=10,
    )

    ax_drone.imshow(drone)
    ax_drone.set_title(f"Drone frame #{frame_idx}")
    ax_drone.axis("off")

    ax_sat.imshow(sat_img)
    ax_sat.set_title("Satellite tile — click drone location")
    ax_sat.axis("off")

    # Draw existing marks on the satellite tile
    for fi, (lat, lon) in existing_marks.items():
        ex = (lon - bbox["lon_min"]) / (bbox["lon_max"] - bbox["lon_min"]) * sat_w
        ey = (bbox["lat_max"] - lat) / (bbox["lat_max"] - bbox["lat_min"]) * sat_h
        ax_sat.plot(ex, ey, "y^", ms=8, mec="k", mew=0.8)
        ax_sat.annotate(str(fi), (ex, ey), fontsize=6, color="yellow",
                        xytext=(3, 3), textcoords="offset points")

    plt.tight_layout()

    result: list[tuple[float, float] | None] = [None]
    quit_flag: list[bool] = [False]
    skip_flag: list[bool] = [False]

    def on_click(event):
        if event.inaxes is not ax_sat:
            return
        if event.button == 1:   # left click -> mark
            lat, lon = px_to_gps(event.xdata, event.ydata, sat_w, sat_h, bbox)
            result[0] = (lat, lon)
            ax_sat.plot(event.xdata, event.ydata, "r+", ms=20, mew=2.5)
            ax_sat.plot(event.xdata, event.ydata, "ro", ms=10, fillstyle="none", mew=1.5)
            fig.canvas.draw()
            plt.pause(0.3)
            plt.close(fig)
        elif event.button == 3:  # right click -> skip (not quit)
            skip_flag[0] = True
            plt.close(fig)

    def on_close(event):
        quit_flag[0] = True

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.show(block=True)

    if quit_flag[0] and not skip_flag[0] and result[0] is None:
        raise StopIteration("User closed window — stopping early")

    return result[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactively mark manual anchors on the satellite tile")
    ap.add_argument("--n-frames", type=int, default=15,
                    help="number of evenly-spaced frames to suggest (default 15)")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="specific frame indices to mark (overrides --n-frames)")
    ap.add_argument("--frames-dir", type=Path, default=ROOT / "data/frames_15hz")
    ap.add_argument("--data-dir",   type=Path, default=ROOT / "data")
    ap.add_argument("--out",        type=Path, default=ROOT / "data/manual_anchors.json")
    ap.add_argument("--redo", action="store_true",
                    help="re-mark frames that are already in the JSON")
    args = ap.parse_args()

    frame_paths = sorted(args.frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {args.frames_dir}")
    print(f"Found {len(frame_paths)} frames")

    sat_img, bbox = load_bbox(args.data_dir)
    sat_h, sat_w  = sat_img.shape[:2]

    existing = load_existing(args.out)
    print(f"Existing marks: {len(existing)}"
          + (f"  (frames {sorted(existing)[:5]}{'…' if len(existing)>5 else ''})"
             if existing else ""))

    indices = pick_frame_indices(frame_paths, args.n_frames, args.frames)
    if not args.redo:
        indices = [i for i in indices if i not in existing]
    print(f"Frames to mark: {len(indices)}  {indices[:10]}{'…' if len(indices)>10 else ''}\n")

    if not indices:
        print("Nothing to mark — all frames already in manual_anchors.json. "
              "Use --redo to re-mark them.")
        return

    current_marks = dict(existing)   # working copy

    try:
        for done, fi in enumerate(indices, 1):
            if fi >= len(frame_paths):
                print(f"  frame {fi}: out of range — skipping")
                continue
            result = mark_one(frame_paths[fi], fi, sat_img, bbox,
                              current_marks, done, len(indices))
            if result is not None:
                lat, lon = result
                current_marks[fi] = [round(lat, 7), round(lon, 7)]
                save_anchors(args.out, current_marks)
                print(f"  frame {fi:>5}: lat={lat:.6f}  lon={lon:.6f}  ✓ saved")
            else:
                print(f"  frame {fi:>5}: skipped")
    except StopIteration as e:
        print(f"\n{e}")

    print(f"\nDone. {len(current_marks)} total anchors in {args.out}")
    print("Run fuse.py to generate the geolocalized trajectory.")


if __name__ == "__main__":
    main()
