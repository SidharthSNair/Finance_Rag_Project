from operator import itemgetter

from langchain_core.runnables import Runnable, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama

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
    llm = OllamaLLM(model="llama2:7b")

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
    llm = OllamaLLM(model="llama2:7b", streaming=True, callbacks=callbacks)

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
    # template = """
    # You are a helpful assistant. Use chat history when relevant.
    #
    # Chat so far:
    # {history}
    # Answer the question based on the context:
    # <context>
    # {context}
    # </context>
    #
    # Question: {question}
    # """
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("system", "You are a helpful assistant. Use the context to answer accurately."),
        ("human",
         "Context:\n<context>\n{context}\n</context>\n\n"
         "Question: {question}")
    ])

    # Chain Steps
    llm = ChatOllama(model="llama2:7b", streaming=True)

    # rag_chain = (
    #         {"context": (lambda x: x["question"]) | retriever, "question": lambda x: x["question"]}
    #         | prompt
    #         | llm
    #         | StrOutputParser()
    # )

    # LCEL mapping — pass question, history, and ensure retriever gets a string
    rag_chain = (
        {
            "question": itemgetter("question"),
            "history": itemgetter("history"),
            "context": itemgetter("question") | retriever,
        }
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
