from langchain_core.runnables import Runnable, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from backend.memory_store import get_chat_history

import os


def get_lcel_qa_chain() -> Runnable:
    # Load vector DB
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vectorstore"))
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt
    template = """
    Answer the question based on the context:
    <context>
    {context}
    </context>

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Chain Steps
    llm = Ollama(model="gemma:2b")

    rag_chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain

def get_lcel_qa_chain_streaming(callbacks=None):
    # Load vector DB
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vectorstore"))
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt
    template = """
    Answer the question based on the context:
    <context>
    {context}
    </context>

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Chain Steps
    llm = OllamaLLM(model="gemma:2b", streaming=True, callbacks=callbacks)

    rag_chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
    )

    return rag_chain

def get_lcel_qa_chain_with_sources():
    # Load vector DB
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vectorstore"))
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # Prompt
    template = """
    Answer the question based on the context:
    <context>
    {context}
    </context>

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Chain Steps
    llm = ChatOllama(model="gemma:2b", streaming=True)

    rag_chain = (
            {"context": (lambda x: x["question"]) | retriever, "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
    )
    # Add memory wrapper
    chat_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_chat_history,  # Memory store function (by session_id)
        input_messages_key="question",  # Which input to track in memory
        history_messages_key="history"  # how to pass memory to chain
    )

    return retriever, chat_with_memory
