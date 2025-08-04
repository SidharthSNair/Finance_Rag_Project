from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os

def get_qa_chain():
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vectorstore"))

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vector_db.as_retriever()
    llm = Ollama(model="gemma:2b")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain
