from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import os

from sympy.physics.units import vacuum_impedance
from tqdm import tqdm

# 1. Load documents
def load_documents(folder_path):

    all_docs = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".csv"):
            loader = CSVLoader(filepath)
        else:
            continue

        docs = loader.load()
        all_docs.extend(docs)

    return all_docs

# 2. Chunk documents
def split_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return  splitter.split_documents(docs)

# 3. Embed and save to vector DB
def create_vector_store(chunks, persist_dir="vectorstore"):


    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)

    # print("Embedding and adding chunks...")
    # for chunk in tqdm(chunks, desc="Embedding Chunks"):
    #     vector_db.add_documents([chunk])
    vector_db.persist()

    print(f"Vector store created at: {persist_dir}")

# 4. main
if __name__ == "__main__":
    print("🔹 Loading documents...")

    # Get the absolute path to the 'data' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    vector_path = os.path.abspath(os.path.join(script_dir, "..", "..", "vectorstore"))


    # Load documents using the proper path
    docs = load_documents(data_path)
    print(f"✅ Loaded {len(docs)} document(s)")

    print("🔹 Splitting documents...")
    chunks = split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    print("🔹 Creating vector store...")
    create_vector_store(chunks, persist_dir=vector_path)
    print("🎉 Done!")