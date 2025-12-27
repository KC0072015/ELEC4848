import sys

from langchain_chroma import Chroma

from src.ingestion import ingest_data
from src.performance_measure import (
	measure_call,
	measure_retrieval,
	print_retrieval_stats,
	print_stats,
)
from src.rag import get_rag_response
from src.embedding import get_embedding_func

# 1. Setup/Load Database
# Run ingestion once, then comment it out if your CSV hasn't changed
# ingest_data("./data/attractions.csv", db_path="./db/chroma") 

db = Chroma(persist_directory="./db/chroma", embedding_function=get_embedding_func())
retriever = db.as_retriever(search_kwargs={"k": 2}) # Top 2 relevant rows

# 2. Run Query
query = input("Please enter your query: "); 
if not query: 
    query = "Tell me your job." # Example query
elif query.lower() == "/bye": 
    print("Goodbye!")
    sys.exit(0)
while query: 
    retrieval_stats = measure_retrieval(retriever, query)

    stats = measure_call(lambda: get_rag_response(query, retriever))
    response = stats["result"]

    print(f"""Query: {query}
    Response:
    {response.content}
    """)

    print_retrieval_stats(retrieval_stats)
    print_stats(stats)

    query = input("Please enter your query (or /bye to exit): ")
    if not query:
        query = "Tell me your job." # Example query
    elif query.lower() == "/bye":
        print("Goodbye!")
        sys.exit(0)
