import sys
import time

from langchain_chroma import Chroma

from src.ingestion import ingest_data
from src.performance_measure import (
    measure_retrieval,
    print_retrieval_stats,
)
from src.rag import stream_rag_response
from src.embedding import get_embedding_func
from src.chat_history import ChatHistory
from src.place_lookup import load_place_names, detect_place, is_location_anchored
from src.proximity import load_attraction_coords, find_nearby

PROXIMITY_THRESHOLD_KM = 3.0
ATTRACTIONS_CSV = "./data/attractions.csv"
PLACE_NAMES_CSV = "./data/place_names.csv"

DISTRICTS = [
    "Central & Western", "Eastern", "Southern", "Wan Chai",
    "Kowloon City", "Kwun Tong", "Sham Shui Po", "Wong Tai Sin",
    "Yau Tsim Mong", "Islands", "Kwai Tsing", "North", "Sai Kung",
    "Sha Tin", "Tai Po", "Tsuen Wan", "Tuen Mun", "Yuen Long"
]

def extract_district(query: str, history: ChatHistory) -> str | None:
    search_text = query
    if history.turns:
        search_text += " " + history.turns[-1][0]
    for district in DISTRICTS:
        if district.lower() in search_text.lower():
            return district
    return None

def build_retrieval_query(query: str, history: ChatHistory) -> str:
    if not history.turns:
        return query
    last_user = history.turns[-1][0]
    last_assistant = history.turns[-1][1][:300]
    return f"{last_user} {last_assistant} {query}"

# 1. Setup/Load Database
# Run ingestion once, then comment it out if your CSV hasn't changed
if "-i" in sys.argv:
    ingest_data(ATTRACTIONS_CSV, db_path="./db/chroma")

db = Chroma(persist_directory="./db/chroma", embedding_function=get_embedding_func())
base_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7})
history = ChatHistory(max_turns=100)

# 2. Load place names and attraction coordinates for proximity search
place_names = load_place_names(PLACE_NAMES_CSV)
attraction_coords = load_attraction_coords(ATTRACTIONS_CSV)
print(f"Loaded {len(place_names)} place names and {len(attraction_coords)} attraction coordinates.")

# 3. Run Query
query = input("Please enter your query: ")
if not query:
    query = "Tell me your job." # Example query
elif query.lower() == "/bye":
    print("Goodbye!")
    sys.exit(0)
while query:
    # --- Proximity resolution ---
    retriever = base_retriever
    extra_context = None

    place = detect_place(query, place_names)
    if place and is_location_anchored(query, place["name"]):
        within, closest = find_nearby(place["lat"], place["lng"], attraction_coords, PROXIMITY_THRESHOLD_KM)
        if within:
            nearby_ids = [a["id"] for _, a in within]
            print(f"[Proximity] '{place['name']}' → {len(within)} attraction(s) within {PROXIMITY_THRESHOLD_KM}km")
            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7, "filter": {"id": {"$in": nearby_ids}}}
            )
        else:
            closest_dist, closest_attr = closest
            print(f"[Proximity] '{place['name']}' → no attractions within {PROXIMITY_THRESHOLD_KM}km (closest: '{closest_attr['name']}' at {closest_dist:.1f}km)")
            extra_context = (
                f"PROXIMITY NOTE: The user asked about '{place['name']}'. "
                f"There are no documented attractions within {PROXIMITY_THRESHOLD_KM}km of that location. "
                f"The nearest attraction in the database is '{closest_attr['name']}' ({closest_dist:.1f}km away). "
                f"Inform the user of this, suggest '{closest_attr['name']}' as the closest option, "
                f"and suggest one attraction from the Context below that best matches their preferences."
            )

    retrieval_query = build_retrieval_query(query, history)
    district = extract_district(query, history)
    if district:
        district_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7, "filter": {"district": district}}
        )
        docs = district_retriever.invoke(retrieval_query)
        if len(docs) >= 2:
            retriever = district_retriever
        # else: fall through to base_retriever with enriched query
        # (prompt's district rules will handle the warning to user)

    retrieval_stats = measure_retrieval(retriever, retrieval_query)
    docs = retriever.invoke(retrieval_query)
    

    print(f"Query: {query}\nResponse:")
    start = time.perf_counter()
    collected = []
    for token in stream_rag_response(
        query,
        retriever,
        history=history.turns,
        extra_context=extra_context,
        retrieval_query=retrieval_query,
    ):
        print(token, end="", flush=True)
        collected.append(token)
    print("\n")
    duration = time.perf_counter() - start

    # print_retrieval_stats(retrieval_stats)


    # print("--- Stats ---")
    # print(f"Wall time:        {duration:.3f} s")


    # print(f"\n=== Retrieved {len(docs)} docs ===")
    # for doc in docs:
    #     print(doc.page_content[:doc.page_content.find("\n")]) # Print only the first line of each retrieved doc
    #     print("---")
    history.add_turn(query, "".join(collected))

    query = input("Please enter your query (or /bye to exit): ")
    if not query:
        query = "Tell me your job." # Example query
    elif query.lower() == "/bye":
        print("Goodbye!")
        sys.exit(0)
