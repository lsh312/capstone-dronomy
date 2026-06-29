"""
Fetch a satellite tile centred on the flight's nominal GPS.

Tries Google Maps Static API first (requires MAPS_API_KEY in .env).
Falls back to ESRI World Imagery automatically if Google fails or
the key is missing — common for EEA-registered accounts where Google
blocks maptype=satellite.

Saves:
    data/satellite_<provider>_z<zoom>.png
    data/satellite_<provider>_z<zoom>.bbox.json   <- bounding box sidecar
                                                     (required by sat_anchors.py)

Usage:
    uv run python code/fetch_satellite.py
    uv run python code/fetch_satellite.py --zoom 19
    uv run python code/fetch_satellite.py --provider esri
    uv run python code/fetch_satellite.py --provider google --size 640 --scale 2
"""
from __future__ import annotations
import argparse
import io
import json
import math
import sys
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def meters_per_pixel(lat_deg: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)


def tile_bbox(lat: float, lon: float, zoom: int, logical_size: int) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) for a tile centred at lat/lon."""
    mpp  = meters_per_pixel(lat, zoom)
    half = mpp * logical_size / 2
    dlat = half / 111320
    dlon = half / (111320 * math.cos(math.radians(lat)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def fetch_google(lat: float, lon: float, zoom: int, size: int, scale: int, api_key: str) -> bytes:
    params = {
        "center":   f"{lat},{lon}",
        "zoom":     zoom,
        "size":     f"{size}x{size}",
        "scale":    scale,
        "maptype":  "satellite",
        "format":   "jpg",
        "key":      api_key,
    }
    resp = requests.get("https://maps.googleapis.com/maps/api/staticmap",
                        params=params, timeout=30)
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "")
    # Google returns HTML or a tiny "sorry" image when the maptype is blocked
    if "html" in ct or len(resp.content) < 5_000:
        raise RuntimeError(
            "Google Maps returned a non-image response — maptype=satellite may be "
            "blocked for your account (common for EEA-registered API keys). "
            "Falling back to ESRI."
        )
    return resp.content


def fetch_esri(lat: float, lon: float, zoom: int, size: int) -> bytes:
    lat_min, lat_max, lon_min, lon_max = tile_bbox(lat, lon, zoom, size)
    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        f"?bbox={lon_min},{lat_min},{lon_max},{lat_max}"
        f"&bboxSR=4326&size={size},{size}&imageSR=4326&format=jpg&transparent=false&f=image"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def save_tile(img_bytes: bytes, png_path: Path,
              lat: float, lon: float, zoom: int,
              logical_size: int, provider: str) -> None:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    img.save(png_path)

    bbox = tile_bbox(lat, lon, zoom, logical_size)
    sidecar = png_path.with_suffix(".bbox.json")
    sidecar.write_text(json.dumps({
        "lat_min":   bbox[0], "lat_max": bbox[1],
        "lon_min":   bbox[2], "lon_max": bbox[3],
        "zoom":      zoom,
        "provider":  provider,
        "img_width": w, "img_height": h,
        "mpp":       meters_per_pixel(lat, zoom),
    }, indent=2))

    mpp = meters_per_pixel(lat, zoom)
    print(f"Saved   {png_path}  ({w}×{h} px)")
    print(f"Sidecar {sidecar}")
    print(f"Coverage:   {mpp * logical_size:.0f} m × {mpp * logical_size:.0f} m")
    print(f"Resolution: {mpp:.3f} m/px (logical)  {mpp * logical_size / w:.3f} m/px (actual pixel)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch satellite tile for the drone flight area")
    ap.add_argument("--lat",      type=float,  default=43.521955)
    ap.add_argument("--lon",      type=float,  default=-5.624290)
    ap.add_argument("--zoom",     type=int,    default=18)
    ap.add_argument("--size",     type=int,    default=640,  help="logical tile size (px)")
    ap.add_argument("--scale",    type=int,    default=2,    help="Google Maps scale (1 or 2)")
    ap.add_argument("--provider", choices=["auto", "google", "esri"], default="auto")
    ap.add_argument("--out-dir",  type=Path,   default=ROOT / "data")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load .env
    api_key = ""
    try:
        from dotenv import dotenv_values
        api_key = dotenv_values(ROOT / ".env").get("MAPS_API_KEY", "")
    except ImportError:
        pass
    placeholder = "your_google_maps_static_api_key_here"
    has_key = bool(api_key) and api_key != placeholder

    # Try Google
    if args.provider in ("auto", "google"):
        if not has_key:
            if args.provider == "google":
                sys.exit("MAPS_API_KEY not set in .env — cannot use Google provider")
            print("No MAPS_API_KEY found — skipping Google, using ESRI")
        else:
            print(f"Trying Google Maps Static (zoom={args.zoom}, size={args.size}, scale={args.scale}) …")
            try:
                data = fetch_google(args.lat, args.lon, args.zoom, args.size, args.scale, api_key)
                out  = args.out_dir / f"satellite_google_z{args.zoom}.png"
                save_tile(data, out, args.lat, args.lon, args.zoom, args.size, "google")
                return
            except Exception as exc:
                if args.provider == "google":
                    sys.exit(f"Google Maps failed: {exc}")
                print(f"Google Maps failed: {exc}")

    # ESRI fallback
    print(f"Fetching ESRI World Imagery (zoom={args.zoom}, size={args.size}) …")
    try:
        data = fetch_esri(args.lat, args.lon, args.zoom, args.size)
        out  = args.out_dir / f"satellite_esri_z{args.zoom}.png"
        save_tile(data, out, args.lat, args.lon, args.zoom, args.size, "esri")
    except Exception as exc:
        sys.exit(f"ESRI fetch failed: {exc}")


if __name__ == "__main__":
    main()
