import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not API_KEY:
    raise ValueError("Missing GOOGLE_MAPS_API_KEY in .env")

LAT = 43.521955
LON = -5.624290

ZOOM = 21
SIZE = "640x640"
SCALE = 2

output_path = Path("data/google_satellite.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

url = "https://maps.googleapis.com/maps/api/staticmap"

params = {
    "center": f"{LAT},{LON}",
    "zoom": ZOOM,
    "size": SIZE,
    "scale": SCALE,
    "maptype": "satellite",
    "format": "jpg",
    "key": API_KEY,
}

response = requests.get(url, params=params)
response.raise_for_status()

output_path.write_bytes(response.content)

print(f"Saved satellite image to {output_path}")