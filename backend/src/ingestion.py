import csv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.embedding import get_embedding_func

REQUIRED_FIELDS = ["summary_desc"]

def _row_to_text(row: dict) -> str:
    parts = []
    if row.get("name"):             parts.append(f"Name: {row['name']}")
    if row.get("district"):         parts.append(f"District: {row['district']}")
    if row.get("category"):         parts.append(f"Category: {row['category']}")
    if row.get("vibes"):            parts.append(f"Vibes: {row['vibes']}")
    if row.get("environmental_tags"): parts.append(f"Environment: {row['environmental_tags']}")
    if row.get("demographic_tags"): parts.append(f"Good for: {row['demographic_tags']}")
    if row.get("summary_desc"):     parts.append(f"Overview: {row['summary_desc']}")
    if row.get("facilities_desc"):  parts.append(f"Facilities: {row['facilities_desc']}")
    if row.get("tips_desc"):        parts.append(f"Tips: {row['tips_desc']}")
    return "\n".join(parts)

def ingest_data(file_path: str, db_path="./db/chroma") -> None:
    docs = []
    skipped = 0

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not all(row.get(field, "").strip() for field in REQUIRED_FIELDS):
                skipped += 1
                continue

            lat, lng = None, None
            raw_coords = row.get("coordinates", "").strip()
            if raw_coords:
                try:
                    lat, lng = [float(x.strip()) for x in raw_coords.split(",")]
                except ValueError:
                    pass

            docs.append(Document(
                page_content=_row_to_text(row),
                metadata={
                    "id": row.get("id", ""),
                    "name": row.get("name", ""),
                    "district": row.get("district", ""),
                    "coordinates": raw_coords,
                    "lat": lat,
                    "lng": lng,
                }
            ))

    print(f"Ingested {len(docs)} attractions, skipped {skipped}.")

    Chroma.from_documents(
        docs,
        embedding=get_embedding_func(),
        persist_directory=db_path
    )