'''
src/rag.py: Module for handling retrieval-augmented generation (RAG) processes.
'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.config import OLLAMA_HOST, DEFAULT_MODEL

def get_rag_response(query: str, retriever, prompt_template_file:str="src/prompt_template.txt"):
    llm = ChatOllama(model=DEFAULT_MODEL, base_url=OLLAMA_HOST, temperature=1.0)

    # Template to guide LLM
    with open(prompt_template_file, 'r') as f:
        prompt_template = ChatPromptTemplate.from_template(f.read())
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Combine context with prompt and call LLM
    chain = prompt_template | llm
    return chain.invoke({"context": context, "question": query})