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

# 1. Setup/Load Database
# Run ingestion once, then comment it out if your CSV hasn't changed
# ingest_data("./data/attractions.csv", db_path="./db/chroma") 

db = Chroma(persist_directory="./db/chroma", embedding_function=get_embedding_func())
retriever = db.as_retriever(search_kwargs={"k": 2}) # Top 2 relevant rows
history = ChatHistory(max_turns=100)

# 2. Run Query
query = input("Please enter your query: "); 
if not query: 
    query = "Tell me your job." # Example query
elif query.lower() == "/bye": 
    print("Goodbye!")
    sys.exit(0)
while query: 
    retrieval_stats = measure_retrieval(retriever, query)

    print(f"Query: {query}\nResponse:")
    start = time.perf_counter()
    collected = []
    for token in stream_rag_response(query, retriever, history=history.format_for_prompt()):
        print(token, end="", flush=True)
        collected.append(token)
    print("\n")
    duration = time.perf_counter() - start
    print_retrieval_stats(retrieval_stats)
    print("--- Stats ---")
    print(f"Wall time:        {duration:.3f} s")
    history.add_turn(query, "".join(collected))

    query = input("Please enter your query (or /bye to exit): ")
    if not query:
        query = "Tell me your job." # Example query
    elif query.lower() == "/bye":
        print("Goodbye!")
        sys.exit(0)
