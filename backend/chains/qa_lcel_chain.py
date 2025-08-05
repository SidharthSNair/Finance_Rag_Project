from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
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
