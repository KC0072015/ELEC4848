"""
Converts a place name CSV into an enriched dataset with coordinates and district.

Input CSV columns:  OBJECTID, PLACE_NAME_ID, GEO_NAME_ID, NAME_EN, NAME_TC, NAME_STATUS
Output CSV columns: OBJECTID, PLACE_NAME_ID, GEO_NAME_ID, NAME_EN, NAME_STATUS,
                    COORDINATES (lat,long), DISTRICT

Strategy:
- Groups rows by GEO_NAME_ID
- Queries the HK gov location API once per group using the Official name
- Applies the result (district + coordinates) to all aliases in the same group
- Saves a checkpoint after each group so the script can be safely interrupted and resumed
"""

import csv
import json
import time
import os
import requests
from difflib import SequenceMatcher

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_CSV       = "./src/place_names_input.csv"       # path to your input CSV
OUTPUT_CSV      = "./src/place_names_output.csv"
CHECKPOINT_FILE = "./src/place_names_checkpoint.json" # tracks completed GEO_NAME_IDs

DELAY_LOCATION  = 1.5   # seconds between locationSearch calls
DELAY_COORD     = 0.5   # seconds between coordinate conversion calls
MAX_RETRIES     = 3
RETRY_BACKOFF   = 5     # seconds to wait after a failed request before retrying

LOCATION_API = "https://www.map.gov.hk/gs/api/v1.0.0/locationSearch?q={q}"
COORD_API    = "https://www.geodetic.gov.hk/transform/v2/?inSys=hkgrid&outSys=wgsgeog&e={x}&n={y}"

# ── Helpers ──────────────────────────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_location(name_en: str) -> dict | None:
    """
    Calls the HK location search API and returns the best-matching result,
    or None if no results are found.
    """
    url = LOCATION_API.format(q=requests.utils.quote(name_en))
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # API may return a single object or a list
            results = data if isinstance(data, list) else [data]
            results = [r for r in results if r.get("nameEN")]

            if not results:
                return None

            # Pick the result whose nameEN best matches our query
            best = max(results, key=lambda r: similarity(name_en, r.get("nameEN", "")))
            return best

        except Exception as e:
            print(f"  [locationSearch] attempt {attempt}/{MAX_RETRIES} failed for '{name_en}': {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return None


def get_wgs84(x, y) -> tuple[float, float] | tuple[None, None]:
    """Converts HK80 grid coordinates to WGS84 lat/lng."""
    url = COORD_API.format(x=x, y=y)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("wgsLat"), data.get("wgsLong")
        except Exception as e:
            print(f"  [coordConvert] attempt {attempt}/{MAX_RETRIES} failed for ({x},{y}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return None, None


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(completed), f)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Read input
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    # Group rows by GEO_NAME_ID
    groups: dict[str, list[dict]] = {}
    for row in rows:
        gid = row["GEO_NAME_ID"]
        groups.setdefault(gid, []).append(row)

    print(f"Found {len(groups)} unique GEO_NAME_IDs")

    # Load checkpoint (already-processed GEO_NAME_IDs)
    completed = load_checkpoint()
    print(f"Resuming — {len(completed)} groups already completed\n")

    # Load existing output rows if resuming
    output_rows: list[dict] = []
    if os.path.exists(OUTPUT_CSV) and completed:
        with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
            output_rows = list(csv.DictReader(f))

    OUTPUT_FIELDS = [
        "OBJECTID", "PLACE_NAME_ID", "GEO_NAME_ID",
        "NAME_EN", "NAME_STATUS", "COORDINATES", "DISTRICT"
    ]

    # Process each group
    total = len(groups)
    for idx, (geo_id, group_rows) in enumerate(groups.items(), 1):
        if geo_id in completed:
            continue

        # Use the Official name for the API query; fall back to first row
        official = next((r for r in group_rows if r.get("NAME_STATUS") == "Official"), group_rows[0])
        name_en = official["NAME_EN"]

        print(f"[{idx}/{total}] GEO_NAME_ID={geo_id}  '{name_en}'")

        # 1. Location search
        location = get_location(name_en)
        time.sleep(DELAY_LOCATION)

        lat, lng, district = None, None, ""
        if location:
            x = location.get("x")
            y = location.get("y")
            district = location.get("districtEN", "")

            if x and y:
                # 2. Coordinate conversion
                lat, lng = get_wgs84(x, y)
                time.sleep(DELAY_COORD)
                print(f"  → district='{district}'  lat={lat}  lng={lng}")
            else:
                print(f"  → district='{district}'  (no coordinates returned)")
        else:
            print(f"  → no result found")

        coords_str = f"{lat},{lng}" if lat is not None and lng is not None else ""

        # Apply result to every row in this group (official + aliases)
        for row in group_rows:
            output_rows.append({
                "OBJECTID":     row["OBJECTID"],
                "PLACE_NAME_ID": row["PLACE_NAME_ID"],
                "GEO_NAME_ID":  row["GEO_NAME_ID"],
                "NAME_EN":      row["NAME_EN"],
                "NAME_STATUS":  row["NAME_STATUS"],
                "COORDINATES":  coords_str,
                "DISTRICT":     district,
            })

        # Mark complete and persist
        completed.add(geo_id)
        save_checkpoint(completed)

        # Write output incrementally (overwrite with all rows so far)
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)

    print(f"\nDone. Output written to {OUTPUT_CSV}")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file removed.")


if __name__ == "__main__":
    main()
