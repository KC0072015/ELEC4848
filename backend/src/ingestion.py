'''
src/ingestion.py: Module for handling data ingestion processes.
'''

from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embedding import get_embedding_func

def ingest_data(file_path: str, db_path="./db/chroma") -> Chroma:
    # Load data from CSV
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()

    # Split long documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(documents)

    # Create or load vector store
    vector_store = Chroma.from_documents(
        documents,
        embedding=get_embedding_func(),
        persist_directory=db_path
    )

    print(f"Ingested data from {file_path} ({len(documents)} rows) into vector store at {db_path}.")
    return vector_store
