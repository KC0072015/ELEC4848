'''
src/embedding.py: Module for handling text embeddings using Ollama API.
'''

from langchain_ollama import OllamaEmbeddings
from src.config import OLLAMA_HOST, EMBEDDING_MODEL

def get_embedding_func():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST
    )