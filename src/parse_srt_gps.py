import re
from pathlib import Path

import pandas as pd

SRT_PATH = Path("data/raw/gps_data.SRT")
OUTPUT_PATH = Path("outputs/gps/gps_data.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

text = SRT_PATH.read_text(errors="ignore")

blocks = text.strip().split("\n\n")

rows = []

for block in blocks:
    frame_match = re.search(r"FrameCnt:\s*(\d+)", block)
    diff_time_match = re.search(r"DiffTime:\s*([\d.]+)ms", block)
    lat_match = re.search(r"latitude:\s*([-+]?\d+\.\d+)", block)
    lon_match = re.search(r"longitude:\s*([-+]?\d+\.\d+)", block)
    rel_alt_match = re.search(r"rel_alt:\s*([-+]?\d+\.\d+)", block)
    abs_alt_match = re.search(r"abs_alt:\s*([-+]?\d+\.\d+)", block)
    yaw_match = re.search(r"gb_yaw:\s*([-+]?\d+\.\d+)", block)
    pitch_match = re.search(r"gb_pitch:\s*([-+]?\d+\.\d+)", block)
    roll_match = re.search(r"gb_roll:\s*([-+]?\d+\.\d+)", block)

    if not frame_match or not lat_match or not lon_match:
        continue

    frame = int(frame_match.group(1))

    rows.append({
        "frame": frame,
        "time_sec": float(diff_time_match.group(1)) / 1000 if diff_time_match else None,
        "latitude": float(lat_match.group(1)),
        "longitude": float(lon_match.group(1)),
        "rel_alt": float(rel_alt_match.group(1)) if rel_alt_match else None,
        "abs_alt": float(abs_alt_match.group(1)) if abs_alt_match else None,
        "yaw": float(yaw_match.group(1)) if yaw_match else None,
        "pitch": float(pitch_match.group(1)) if pitch_match else None,
        "roll": float(roll_match.group(1)) if roll_match else None,
    })

df = pd.DataFrame(rows)
df = df.sort_values("frame")

df.to_csv(OUTPUT_PATH, index=False)

print(f"Parsed {len(df)} GPS rows")
print(f"Saved to {OUTPUT_PATH}")
print(df.head())