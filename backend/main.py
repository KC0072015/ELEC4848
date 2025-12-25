from src.ingestion import ingest_data
from src.rag import get_rag_response
from langchain_chroma import Chroma
from src.embedding import get_embedding_func

# 1. Setup/Load Database
# Run ingestion once, then comment it out if your CSV hasn't changed
# ingest_data("./data/attractions.csv", db_path="./db/chroma") 

db = Chroma(persist_directory="./db/chroma", embedding_function=get_embedding_func())
retriever = db.as_retriever(search_kwargs={"k": 2}) # Top 2 relevant rows

# 2. Run Query
query = "Tell me about ayld001." # Example query based on your columns
response = get_rag_response(query, retriever)
print(f"""Query: {query}
Response:
{response.content}""")