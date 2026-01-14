'''
src/rag.py: Module for handling retrieval-augmented generation (RAG) processes.
'''

from typing import Generator, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.config import OLLAMA_HOST, DEFAULT_MODEL

def get_rag_response(query: str, retriever, prompt_template_file: str = "src/prompt_template.txt", history: Optional[str] = None):
    llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_HOST, temperature=0.5)

    # Template to guide LLM
    with open(prompt_template_file, 'r') as f:
        prompt_template = ChatPromptTemplate.from_template(f.read())
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Combine context with prompt and call LLM
    chain = prompt_template | llm
    return chain.invoke({
        "context": context,
        "question": query,
        "history": history or "None"
    })


def stream_rag_response(
    query: str,
    retriever,
    prompt_template_file: str = "src/prompt_template.txt",
    history: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream the RAG response token-by-token."""

    llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_HOST, temperature=0.5)

    with open(prompt_template_file, 'r') as f:
        prompt_template = ChatPromptTemplate.from_template(f.read())

    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    chain = prompt_template | llm
    for chunk in chain.stream({
        "context": context,
        "question": query,
        "history": history or "None"
    }):
        content = getattr(chunk, "content", None)
        if content:
            yield content