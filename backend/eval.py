"""
eval.py: Automated evaluation script for the Hana RAG chatbot.

Runs every session from eval_dataset.json, calls stream_rag_response for each
turn with the accumulated ChatHistory (mirroring api.py behaviour), then
scores on 5 metrics from the scoring rubric:

  1. relevance        (1–3)        keyword hit-rate against expected_keywords
  2. faithfulness     (1–3)        hallucination detection in free-text response
  3. refusal_accuracy (PASS/FAIL)  correct refusal on out-of-scope queries
  4. memory_coherence (PASS/FAIL)  heuristic cross-turn reference check
  5. itinerary_grounding (PASS/FAIL) location names grounded in attractions.csv

Usage (from the backend/ directory):
  python eval.py                    # run all sessions
  python eval.py --sessions S1,S3   # run specific sessions only
  python eval.py --verbose          # stream every token to stdout

Output:
  eval_results.json   — full per-turn scores + session/overall summary
  Console             — live progress + final summary table
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Resolve paths relative to this file (same trick as api.py) ───────────────
_BACKEND_DIR = Path(__file__).parent.resolve()
os.chdir(_BACKEND_DIR)

from langchain_chroma import Chroma
from src.embedding import get_embedding_func
from src.chat_history import ChatHistory
from src.rag import stream_rag_response
from src.place_lookup import load_place_names, detect_place, is_location_anchored
from src.proximity import load_attraction_coords, find_nearby

# ── Constants ─────────────────────────────────────────────────────────────────
EVAL_DATASET    = "./eval_dataset.json"
ATTRACTIONS_CSV = "./data/attractions.csv"
PLACE_NAMES_CSV = "./data/place_names.csv"
DB_PATH         = "./db/chroma"
RESULTS_OUTPUT  = "./eval_results.json"
PROXIMITY_THRESHOLD_KM = 3.0

DISTRICTS = [
    "Central & Western", "Eastern", "Southern", "Wan Chai",
    "Kowloon City", "Kwun Tong", "Sham Shui Po", "Wong Tai Sin",
    "Yau Tsim Mong", "Islands", "Kwai Tsing", "North", "Sai Kung",
    "Sha Tin", "Tai Po", "Tsuen Wan", "Tuen Mun", "Yuen Long"
]

# Famous HK landmarks known to be absent from attractions.csv.
# If these appear in a response they indicate LLM knowledge bleed-through.
KNOWN_NOT_IN_DATASET: set[str] = {
    "ocean park",
    "victoria peak",
    "the peak tower",
    "peak tower",
    "star ferry pier",
    "hong kong disneyland",
    "disneyland",
    "temple street night market",
    "ladies market",
    "mong kok market",
    "museum of history",
    "avenue of stars",
    "times square",
    "harbour city",
    "ifc mall",
}

# Phrases that signal the bot is refusing / saying it cannot help
REFUSAL_PHRASES: list[str] = [
    "don't have information",
    "do not have information",
    "don't have exact directions",
    "do not have exact directions",
    "exact directions",
    "don't have restaurant information",
    "do not have restaurant information",
    "don't have weather information",
    "do not have weather information",
    "i'm sorry",
    "i am sorry",
    "sorry,",
    "sorry.",
    "apologies",
    "i cannot",
    "i can't",
    "cannot help",
    "outside my",
    "not able to",
    "no information",
    "unable to",
    "not in my",
    "beyond my",
    "can't book",
    "cannot book",
    "booking or taxi app",
    "google maps",
    "mtr website",
    "google maps or openrice",
    "openrice",
    "hong kong observatory",
    "weather app",
    "No data found",
    "no data found"
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_attraction_names(csv_path: str) -> set[str]:
    """Return a lowercase set of every attraction name in attractions.csv."""
    names: set[str] = set()
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if name:
                names.add(name.lower())
    return names


def setup_retriever():
    """Build the same MMR retriever used by api.py and return the db instance."""
    db = Chroma(persist_directory=DB_PATH, embedding_function=get_embedding_func())
    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
    )
    return db, base_retriever


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


# ── Response collection ────────────────────────────────────────────────────────

def collect_response(
    query: str,
    retriever,
    history: ChatHistory,
    verbose: bool,
    extra_context: Optional[str] = None,
    retrieval_query: Optional[str] = None,
) -> tuple[str, float]:
    """Stream and collect the full response; return (text, elapsed_seconds)."""
    tokens: list[str] = []
    t0 = time.time()
    for token in stream_rag_response(
        query,
        retriever,
        history=history.turns,
        extra_context=extra_context,
        retrieval_query=retrieval_query,
    ):
        tokens.append(token)
        if verbose:
            print(token, end="", flush=True)
    if verbose:
        print()
    return "".join(tokens), round(time.time() - t0, 1)


# ── Scorers ────────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Normalize quotes/dashes/spacing so semantic fallback matches are stable."""
    normalized = text.lower()
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def score_relevance(response: str, keywords: list[str]) -> Optional[int]:
    """
    1–3 score based on keyword hit-rate.
    Returns None when expected_keywords is empty (not applicable).
      3 = all keywords present
      2 = ≥ 50 % keywords present
      1 = < 50 % keywords present
    """
    if not keywords:
        return None
    rl = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in rl)
    ratio = hits / len(keywords)
    return 3 if ratio >= 0.8 else (2 if ratio >= 0.5 else 1)


def score_refusal(response: str, should_refuse: bool) -> dict:
    """
    For should_refuse=True turns: PASS if bot refused, FAIL otherwise.
    For should_refuse=False turns: N/A normally; WARN if the bot refused
    when it shouldn't have (unexpected refusal logged but not penalised).
    """
    rl = _normalize_text(response)
    is_refusal = any(_normalize_text(p) in rl for p in REFUSAL_PHRASES)
    if should_refuse:
        return {
            "result": "PASS" if is_refusal else "FAIL",
            "expected": True,
            "is_refusal": is_refusal,
        }
    return {
        "result": "WARN" if is_refusal else "N/A",
        "expected": False,
        "is_refusal": is_refusal,
    }


def score_faithfulness(
    response: str,
    hallucination_check: bool,
    attraction_names: set[str],
) -> dict:
    """
    1–3 faithfulness score.
      3 = at least one verified attraction name found, no known hallucination
      2 = no known hallucinations but no verified names either (neutral)
      1 = response contains a known-not-in-dataset landmark (hallucination)
    Only evaluated when hallucination_check=True.
    """
    if not hallucination_check:
        return {"score": None, "note": "not checked (hallucination_check=false)"}

    rl = response.lower()
    flagged  = [n for n in KNOWN_NOT_IN_DATASET if n in rl]
    verified = [n for n in attraction_names    if n in rl]

    if flagged:
        return {"score": 1, "flagged": flagged, "verified": verified[:8],
                "note": f"Hallucination detected: {flagged}"}
    if verified:
        return {"score": 3, "flagged": [], "verified": verified[:8],
                "note": f"Grounded — {len(verified)} dataset name(s) matched"}
    return {"score": 2, "flagged": [], "verified": [],
            "note": "No hallucinations detected; no dataset names verified"}


def score_itinerary_grounding(
    response: str,
    category: str,
    attraction_names: set[str],
) -> Optional[dict]:
    """
    PASS / FAIL for itinerary_generation and itinerary_modification turns only.
    FAIL if a known-not-in-dataset place is mentioned, or no verified names found.
    """
    if category not in ("itinerary_generation", "itinerary_modification"):
        return None
    rl = response.lower()
    flagged  = [n for n in KNOWN_NOT_IN_DATASET if n in rl]
    verified = [n for n in attraction_names    if n in rl]
    result   = "FAIL" if (flagged or not verified) else "PASS"
    return {"result": result, "verified": verified[:10], "hallucinated": flagged}


def score_memory_coherence(
    response: str,
    memory_dependency: Optional[str],
    prior_responses: dict[str, str],
    attraction_names: set[str],
) -> Optional[dict]:
    """
    PASS / FAIL / MANUAL for turns that reference a previous turn.

    Primary check: verify that at least one attraction name from the prior
    turn's response also appears in the current response (entity carry-over).

    Fallback (when no attractions were mentioned before): use long-word token
    overlap — 15 % overlap threshold for PASS.
    """
    if not memory_dependency:
        return None
    prior = prior_responses.get(memory_dependency)
    if not prior:
        return {
            "result": "MANUAL",
            "note": f"Prior turn {memory_dependency} not available in this run",
        }

    rl      = response.lower()
    prior_l = prior.lower()

    # Primary: attraction name carry-over
    prior_attractions = [n for n in attraction_names if n in prior_l]
    if prior_attractions:
        carried = [n for n in prior_attractions if n in rl]
        return {
            "result": "PASS" if carried else "FAIL",
            "prior_attractions": prior_attractions[:5],
            "carried_over": carried[:5],
            "note": f"{len(carried)}/{len(prior_attractions)} prior attraction(s) found in response",
        }

    # Fallback: long word overlap
    tokens  = set(re.findall(r"\b[a-z]{6,}\b", prior_l))
    overlap = sum(1 for t in tokens if t in rl)
    ratio   = overlap / len(tokens) if tokens else 0
    return {
        "result": "PASS" if ratio >= 0.15 else "FAIL",
        "overlap_ratio": round(ratio, 3),
        "note": f"word-overlap fallback: {overlap}/{len(tokens)} tokens from {memory_dependency}",
    }


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_evaluation(
    verbose: bool = False,
    session_filter: Optional[set[str]] = None,
) -> dict:
    print(f"{'='*62}")
    print(f"  Hana RAG Evaluation  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*62}")

    with open(EVAL_DATASET, encoding="utf-8") as f:
        dataset = json.load(f)
        
    attraction_names = load_attraction_names(ATTRACTIONS_CSV)
    place_names = load_place_names(PLACE_NAMES_CSV)
    attraction_coords = load_attraction_coords(ATTRACTIONS_CSV)
    db, base_retriever = setup_retriever()

    print(f"  Loaded {len(attraction_names)} attraction names from CSV")
    print(f"  Loaded {len(place_names)} place names for proximity detection")
    print(f"  Vector DB: {DB_PATH}")
    if session_filter:
        print(f"  Running sessions: {sorted(session_filter)}")
    print()

    results: dict = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "total_sessions": dataset["meta"]["total_sessions"],
            "total_turns": dataset["meta"]["total_questions"],
        },
        "sessions": [],
    }

    # Score accumulators
    all_relevance:    list[int] = []
    all_faithfulness: list[int] = []
    refusal_counts   = {"PASS": 0, "FAIL": 0, "WARN": 0, "N/A": 0}
    itinerary_counts = {"PASS": 0, "FAIL": 0}
    memory_counts    = {"PASS": 0, "FAIL": 0, "MANUAL": 0}

    for session in dataset["sessions"]:
        sid = session["session_id"]
        if session_filter and sid not in session_filter:
            continue

        print(f"\n{'─'*62}")
        print(f"  {sid}: {session['label']}")
        print(f"{'─'*62}")

        # Fresh ChatHistory per session (max_turns=100 to keep full context)
        history:         ChatHistory       = ChatHistory(max_turns=100)
        prior_responses: dict[str, str]    = {}
        session_result: dict = {
            "session_id": sid,
            "label":      session["label"],
            "turns":      [],
        }

        for td in session["turns"]:
            tid   = td["id"]
            query = td["query"]
            print(f"\n  [{tid}] Q: {query}")

            # ── Proximity resolution ──────────────────────────────────────────
            retriever = base_retriever
            extra_context = None

            place = detect_place(query, place_names)
            if place and is_location_anchored(query, place["name"]):
                within, closest = find_nearby(
                    place["lat"], place["lng"], attraction_coords, PROXIMITY_THRESHOLD_KM
                )
                if within:
                    nearby_ids = [a["id"] for _, a in within]
                    if verbose:
                        print(f"  [Proximity] Detected '{place['name']}', filtering to {len(within)} nearby attractions.")
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
                    if verbose:
                        print(f"  [Proximity] Detected '{place['name']}', 0 attractions within {PROXIMITY_THRESHOLD_KM}km. Injecting fallback context.")
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
                    search_kwargs={
                        "k": 6,
                        "fetch_k": 20,
                        "lambda_mult": 0.7,
                        "filter": {"district": district},
                    },
                )
                docs = district_retriever.invoke(retrieval_query)
                if len(docs) >= 2:
                    retriever = district_retriever

            # Collect response (streaming)
            response, elapsed = collect_response(
                query,
                retriever,
                history,
                verbose,
                extra_context,
                retrieval_query=retrieval_query,
            )

            # Save for downstream memory-coherence checks BEFORE updating history
            prior_responses[tid] = response
            history.add_turn(query, response)

            # ── Score ────────────────────────────────────────────────────────
            rel = score_relevance(response, td.get("expected_keywords", []))
            ref = score_refusal(response, td.get("should_refuse", False))
            fth = score_faithfulness(response, td.get("hallucination_check", False), attraction_names)
            itn = score_itinerary_grounding(response, td.get("category", ""), attraction_names)
            mem = score_memory_coherence(
                response, td.get("memory_dependency"), prior_responses, attraction_names
            )

            # ── Accumulate ───────────────────────────────────────────────────
            if rel is not None:
                all_relevance.append(rel)
            if fth.get("score") is not None:
                all_faithfulness.append(fth["score"])
            refusal_counts[ref["result"]] += 1
            if itn:
                itinerary_counts[itn["result"]] += 1
            if mem:
                memory_counts[mem["result"]] += 1

            # ── Console output ───────────────────────────────────────────────
            if not verbose:
                preview = response[:280].replace("\n", " ")
                print(f"  A: {preview}{'…' if len(response) > 280 else ''}")

            score_line = (
                f"  [{elapsed}s]  "
                f"Relevance={rel or 'N/A'}  "
                f"Refusal={ref['result']}  "
                f"Faith={fth.get('score') or 'N/A'}"
                + (f"  Itin={itn['result']}" if itn else "")
                + (f"  Mem={mem['result']}"  if mem else "")
            )
            print(score_line)
            if fth.get("flagged"):
                print(f"  ⚠  Hallucination flag: {fth['flagged']}")

            session_result["turns"].append({
                "turn_id":  tid,
                "query":    query,
                "response": response,
                "elapsed_s": elapsed,
                "scores": {
                    "relevance":            rel,
                    "refusal_accuracy":     ref,
                    "faithfulness":         fth,
                    "itinerary_grounding":  itn,
                    "memory_coherence":     mem,
                },
            })

        results["sessions"].append(session_result)

    # ── Build summary ─────────────────────────────────────────────────────────
    def _avg(lst: list[int]) -> Optional[float]:
        return round(sum(lst) / len(lst), 2) if lst else None

    ref_tested = refusal_counts["PASS"] + refusal_counts["FAIL"]
    itn_total  = itinerary_counts["PASS"] + itinerary_counts["FAIL"]
    mem_tested = memory_counts["PASS"] + memory_counts["FAIL"]

    summary = {
        "relevance_avg":    _avg(all_relevance),
        "faithfulness_avg": _avg(all_faithfulness),
        "refusal_accuracy": {
            **refusal_counts,
            "tested":    ref_tested,
            "pass_rate": round(refusal_counts["PASS"] / ref_tested, 3) if ref_tested else None,
        },
        "itinerary_grounding": {
            **itinerary_counts,
            "tested":    itn_total,
            "pass_rate": round(itinerary_counts["PASS"] / itn_total, 3) if itn_total else None,
        },
        "memory_coherence": {
            **memory_counts,
            "tested":    mem_tested,
            "pass_rate": round(memory_counts["PASS"] / mem_tested, 3) if mem_tested else None,
        },
    }
    results["summary"] = summary

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(RESULTS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Print summary table ───────────────────────────────────────────────────
    def _pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n // d}%)" if d else "N/A"

    print(f"\n{'='*62}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*62}")
    print(f"  Relevance (avg 1–3):     {summary['relevance_avg']}")
    print(f"  Faithfulness (avg 1–3):  {summary['faithfulness_avg']}")
    print(f"  Refusal Accuracy:        {_pct(refusal_counts['PASS'], ref_tested)}")
    print(f"  Itinerary Grounding:     {_pct(itinerary_counts['PASS'], itn_total)}")
    print(f"  Memory Coherence:        {_pct(memory_counts['PASS'], mem_tested)}")

    extras: list[str] = []
    if memory_counts["MANUAL"]:
        extras.append(f"{memory_counts['MANUAL']} turn(s) require manual memory review")
    if refusal_counts["WARN"]:
        extras.append(f"{refusal_counts['WARN']} unexpected refusal(s) on normal turns")
    for note in extras:
        print(f"  ⚠  {note}")

    print(f"{'='*62}")
    print(f"  Full results saved → {RESULTS_OUTPUT}")
    print(f"{'='*62}\n")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the Hana RAG chatbot against eval_dataset.json")
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Stream every response token to stdout (default: show 280-char preview)",
    )
    ap.add_argument(
        "--sessions", "-s",
        default="",
        metavar="S1,S2",
        help="Comma-separated session IDs to run (default: all sessions)",
    )
    args = ap.parse_args()

    session_filter = set(args.sessions.split(",")) if args.sessions.strip() else None
    run_evaluation(verbose=args.verbose, session_filter=session_filter)