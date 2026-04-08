'''
src/rag.py: Module for handling retrieval-augmented generation (RAG) processes.
'''

from typing import Generator, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.config import OLLAMA_HOST, DEFAULT_MODEL


def _build_messages(
    system_prompt: str,
    context: str,
    query: str,
    history: Optional[List[Tuple[str, str]]],
) -> list:
    """
    Build a proper chat message list:
      SystemMessage  — Hana's instructions (no placeholders)
      HumanMessage   — each user turn from history
      AIMessage      — each assistant turn from history
      HumanMessage   — current context + question (final turn)
    """
    messages: list = [SystemMessage(content=system_prompt)]
    for user_msg, ai_msg in (history or []):
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=f"## Context\n{context}\n\n## Question\n{query}"))
    return messages


def get_rag_response(
    query: str,
    retriever,
    prompt_template_file: str = "src/prompt_template.txt",
    history: Optional[List[Tuple[str, str]]] = None,
    retrieval_query: Optional[str] = None,
):
    llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_HOST, temperature=0.2)

    with open(prompt_template_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    docs = retriever.invoke(retrieval_query or query)
    context = "\n".join([doc.page_content for doc in docs])

    messages = _build_messages(system_prompt, context, query, history)
    return llm.invoke(messages)


def stream_rag_response(
    query: str,
    retriever,
    prompt_template_file: str = "src/prompt_template.txt",
    history: Optional[List[Tuple[str, str]]] = None,
    extra_context: Optional[str] = None,
    retrieval_query: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream the RAG response token-by-token."""

    llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_HOST, temperature=0.2)

    with open(prompt_template_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    docs = retriever.invoke(retrieval_query or query)
    context = "\n".join([doc.page_content for doc in docs])
    if extra_context:
        context = extra_context + "\n\n" + context

    messages = _build_messages(system_prompt, context, query, history)
    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", None)
        if content:
            yield content