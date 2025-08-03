import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load persisted vectorstore
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "vectorstore"))
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding)


# Initialize Ollama LLM (e.g., mistral)
llm = Ollama(model="gemma:2b")

# Create QA chain using retriever
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# User query
query = input("💬 Ask a question: ")
response = qa_chain.run(query)

# Print response
print(f"\n🤖 Answer: {response}")