import os


from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


from chains.qa_chain import get_qa_chain

app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

qa_chain = get_qa_chain()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": ""})

@app.post("/ask", response_class=HTMLResponse)
def ask_question(request: Request, question: str = Form(...)):
    response = qa_chain.run(question)
    return templates.TemplateResponse("index.html", {"request": request, "answer": response, "question": question})