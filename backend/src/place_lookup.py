"""
src/place_lookup.py: Loads the HK place names dataset and detects place name
mentions in user queries, returning the matched place's coordinates.
"""

import csv
from typing import Optional

from langchain_ollama import ChatOllama
from src.config import OLLAMA_HOST, CLASSIFIER_MODEL


def load_place_names(csv_path: str) -> dict:
    """
    Loads the place names output CSV into a dict keyed by lowercase name.
    Only Official entries are loaded to avoid duplicate coordinate lookups.

    Returns: {name_lower: {"name": str, "lat": float, "lng": float, "district": str}}
    """
    places = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("NAME_STATUS", "").strip() != "Official":
                continue
            name = row.get("NAME_EN", "").strip()
            coords = row.get("COORDINATES", "").strip()
            district = row.get("DISTRICT", "").strip()
            if not name or not coords:
                continue
            try:
                lat, lng = [float(x.strip()) for x in coords.split(",")]
                places[name.lower()] = {
                    "name": name,
                    "lat": lat,
                    "lng": lng,
                    "district": district,
                }
            except ValueError:
                continue
    return places


def is_location_anchored(query: str, place_name: str) -> bool:
    """
    Uses a lightweight LLM to decide whether the user genuinely wants
    attractions near 'place_name', or is merely mentioning it as context.

    Returns True only if the intent is clearly location-anchored.
    """
    llm = ChatOllama(model=CLASSIFIER_MODEL, base_url=OLLAMA_HOST, temperature=0)
    prompt = (
        f"Does the following query intend to find places or attractions that are "
        f"physically located near or in '{place_name}'?\n"
        f"Or is '{place_name}' only mentioned as background context, a past experience, or a comparison?\n\n"
        f"Query: \"{query}\"\n\n"
        f"Reply with only YES or NO."
    )
    result = llm.invoke(prompt)
    return result.content.strip().upper().startswith("YES")


def detect_place(query: str, place_dict: dict) -> Optional[dict]:
    """
    Scans the query for a known HK place name using longest-match-first so
    that e.g. "Tsim Sha Tsui" matches before "Tsim".

    Returns the matched place dict or None if no place name is found.
    """
    q = query.lower()
    for name in sorted(place_dict.keys(), key=len, reverse=True):
        if name in q:
            return place_dict[name]
    return None
