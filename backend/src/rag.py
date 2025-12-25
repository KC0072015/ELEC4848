'''
src/rag.py: Module for handling retrieval-augmented generation (RAG) processes.
'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.config import OLLAMA_HOST, DEFAULT_MODEL

def get_rag_response(query: str, retriever):
    llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_HOST, temperature=1.0)

    # Template to guide LLM
    prompt_template = ChatPromptTemplate.from_template(
        '''Use only the following context to answer:
        {context}
        Question: {question}
        Answer:'''
        )
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Combine context with prompt and call LLM
    chain = prompt_template | llm
    return chain.invoke({"context": context, "question": query})