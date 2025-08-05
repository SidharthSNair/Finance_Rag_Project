import os
import shutil


from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


#from chains.qa_chain import get_qa_chain
from backend.chains.qa_lcel_chain import get_lcel_qa_chain
from backend.agents.ingest_documents import ingest_uploaded_documents


app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

qa_chain = get_lcel_qa_chain()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploaded_docs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": ""})

@app.post("/ask", response_class=HTMLResponse)
def ask_question(request: Request, question: str = Form(...)):
    qa_chain = get_lcel_qa_chain()
    response = qa_chain.invoke(question)
    return templates.TemplateResponse("index.html", {"request": request, "answer": response, "question": question})

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # Ingest uploaded documents to vectorstore
    ingest_uploaded_documents()

    # Redirect to home page after upload
    return RedirectResponse("/", status_code=303)