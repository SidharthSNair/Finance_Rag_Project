from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

app = FastAPI()
templates = Jinja2Templates(directory="../frontend")

embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory="../vectorstore", embedding_function=embedding)
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=Ollama(model="gemma:2b"), retriever=retriever)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": ""})

@app.post("/ask", response_class=HTMLResponse)
def ask_question(request: Request, question: str = Form(...)):
    response = qa_chain.run(question)
    return templates.TemplateResponse("index.html", {"request": request, "answer": response, "question": question})