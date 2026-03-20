"""
src/proximity.py: Haversine distance utilities and proximity filtering for attractions.
"""

import csv
from math import radians, sin, cos, sqrt, atan2


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Returns the great-circle distance in km between two WGS84 coordinates."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_attraction_coords(csv_path: str) -> list[dict]:
    """
    Loads id, name, lat, lng for every attraction that has valid coordinates.
    Returns a list of dicts: [{"id": str, "name": str, "lat": float, "lng": float}]
    """
    results = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = row.get("coordinates", "").strip()
            if not raw:
                continue
            try:
                lat, lng = [float(x.strip()) for x in raw.split(",")]
                results.append({
                    "id":   row.get("id", ""),
                    "name": row.get("name", ""),
                    "lat":  lat,
                    "lng":  lng,
                })
            except ValueError:
                continue
    return results


def find_nearby(
    origin_lat: float,
    origin_lng: float,
    attractions: list[dict],
    threshold_km: float = 3.0,
) -> tuple[list[tuple[float, dict]], tuple[float, dict]]:
    """
    Computes the distance from (origin_lat, origin_lng) to every attraction.

    Returns:
        within  — list of (distance_km, attraction) sorted nearest-first,
                  only for attractions within threshold_km
        closest — (distance_km, attraction) for the single nearest attraction
                  regardless of threshold (used for the fallback suggestion)
    """
    distances = [(haversine(origin_lat, origin_lng, a["lat"], a["lng"]), a) for a in attractions]
    within = sorted([(d, a) for d, a in distances if d <= threshold_km], key=lambda x: x[0])
    closest = min(distances, key=lambda x: x[0])
    return within, closest
