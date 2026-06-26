from pathlib import Path
import math

import pandas as pd
import numpy as np

SIFT_PATH = Path("outputs/sift_gps_estimates/sift_gps_estimates.csv")
GPS_PATH = Path("outputs/gps/gps_data.csv")
OUTPUT_PATH = Path("outputs/sift_gps_estimates/sift_gps_error_metrics.csv")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )

    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


sift = pd.read_csv(SIFT_PATH)
gps = pd.read_csv(GPS_PATH)

sift_success = sift[sift["success"] == True].copy()

# SIFT script stores original_frame_idx as zero-based.
# SRT frame numbers start at 1.
sift_success["frame"] = sift_success["original_frame_idx"] + 1

merged = sift_success.merge(
    gps[["frame", "latitude", "longitude"]],
    on="frame",
    how="left",
)

merged["error_m"] = merged.apply(
    lambda row: haversine_m(
        row["estimated_lat"],
        row["estimated_lon"],
        row["latitude"],
        row["longitude"],
    ),
    axis=1,
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_PATH, index=False)

errors = merged["error_m"].dropna()

print("SIFT GPS Benchmark")
print("==================")
print(f"Estimates scored : {len(errors)} / {len(sift)}")
print(f"Median error     : {errors.median():.1f} m")
print(f"Mean error       : {errors.mean():.1f} m")
print(f"RMSE             : {np.sqrt(np.mean(errors ** 2)):.1f} m")
print(f"p90 error        : {errors.quantile(0.90):.1f} m")
print(f"Min / max        : {errors.min():.1f} m / {errors.max():.1f} m")
print(f"Within 15 m      : {(errors <= 15).sum()} ({(errors <= 15).mean() * 100:.0f}%)")
print()
print(f"Saved detailed results to {OUTPUT_PATH}")