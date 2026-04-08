"""
backend/api.py: FastAPI REST API wrapping the HK Travel Guide RAG engine.

Run with:
    cd backend
    uvicorn api:app --reload --port 8000
"""

import json
import os
import uuid
from pathlib import Path
from typing import Optional

# Resolve paths relative to this file so the API can be launched from any cwd
_BACKEND_DIR = Path(__file__).parent.resolve()
os.chdir(_BACKEND_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_chroma import Chroma

from src.embedding import get_embedding_func
from src.chat_history import ChatHistory
from src.ingestion import ingest_data
from src.place_lookup import detect_place, is_location_anchored, load_place_names
from src.proximity import find_nearby, load_attraction_coords
from src.rag import stream_rag_response

# ---------------------------------------------------------------------------
# Constants (mirrors main.py)
# ---------------------------------------------------------------------------
PROXIMITY_THRESHOLD_KM = 3.0
ATTRACTIONS_CSV = "./data/attractions.csv"
PLACE_NAMES_CSV = "./data/place_names.csv"
DB_PATH = "./db/chroma"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ELEC4848 HK Travel Guide", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state (loaded once at startup)
# ---------------------------------------------------------------------------
db = Chroma(persist_directory=DB_PATH, embedding_function=get_embedding_func())
base_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
)
place_names = load_place_names(PLACE_NAMES_CSV)
attraction_coords = load_attraction_coords(ATTRACTIONS_CSV)

# session_id -> ChatHistory
sessions: dict[str, ChatHistory] = {}


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, ChatHistory]:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = ChatHistory(max_turns=100)
    return session_id, sessions[session_id]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class IngestRequest(BaseModel):
    force: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "place_names_loaded": len(place_names),
        "attractions_loaded": len(attraction_coords),
    }


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    """Re-ingest attraction data from CSV into Chroma."""
    ingest_data(ATTRACTIONS_CSV, db_path=DB_PATH)
    return {"status": "ingested"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Stream the RAG response as Server-Sent Events.

    Event types:
      - session  : sent first, contains the session_id (useful when no id was supplied)
      - message  : each token chunk from the LLM
      - proximity: optional proximity note (JSON)
      - done     : signals end of stream
      - error    : unrecoverable error (JSON with 'detail' key)
    """
    session_id, history = _get_or_create_session(req.session_id)
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="message must not be empty")

    # Resolve retriever & optional proximity context (mirrors main.py logic)
    retriever = base_retriever
    extra_context: Optional[str] = None
    proximity_note: Optional[dict] = None

    place = detect_place(query, place_names)
    if place and is_location_anchored(query, place["name"]):
        within, closest = find_nearby(
            place["lat"], place["lng"], attraction_coords, PROXIMITY_THRESHOLD_KM
        )
        if within:
            nearby_ids = [a["id"] for _, a in within]
            proximity_note = {
                "place": place["name"],
                "count": len(within),
                "threshold_km": PROXIMITY_THRESHOLD_KM,
            }
            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 20,
                    "lambda_mult": 0.7,
                    "filter": {"id": {"$in": nearby_ids}},
                },
            )
        else:
            closest_dist, closest_attr = closest
            proximity_note = {
                "place": place["name"],
                "count": 0,
                "threshold_km": PROXIMITY_THRESHOLD_KM,
                "closest": closest_attr["name"],
                "closest_km": round(closest_dist, 1),
            }
            extra_context = (
                f"PROXIMITY NOTE: The user asked about '{place['name']}'. "
                f"There are no documented attractions within {PROXIMITY_THRESHOLD_KM}km of that location. "
                f"The nearest attraction in the database is '{closest_attr['name']}' ({closest_dist:.1f}km away). "
                f"Inform the user of this, suggest '{closest_attr['name']}' as the closest option, "
                f"and suggest one attraction from the Context below that best matches their preferences."
            )

    def event_stream():
        # 1. Session id
        yield f"event: session\ndata: {session_id}\n\n"

        # 2. Optional proximity note
        if proximity_note:
            yield f"event: proximity\ndata: {json.dumps(proximity_note)}\n\n"

        # 3. Token stream
        collected: list[str] = []
        try:
            for token in stream_rag_response(
                query,
                retriever,
                history=history.turns,
                extra_context=extra_context,
            ):
                collected.append(token)
                # Newlines inside SSE data fields must be escaped
                safe = token.replace("\\", "\\\\").replace("\n", "\\n")
                yield f"event: message\ndata: {safe}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return

        history.add_turn(query, "".join(collected))

        # 4. Done sentinel
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear a session's chat history."""
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
