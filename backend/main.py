import os
import shutil


from fastapi import FastAPI, Request, Form, UploadFile, File, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from langchain.callbacks.base import BaseCallbackHandler

#from chains.qa_chain import get_qa_chain
from backend.chains.qa_lcel_chain import (
    get_lcel_qa_chain,
    get_lcel_qa_chain_streaming,
    get_lcel_qa_chain_with_sources
)
from backend.agents.ingest_documents import ingest_uploaded_documents


app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

qa_chain = get_lcel_qa_chain()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploaded_docs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.websocket.send_text(token)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": ""})


@app.post("/ask", response_class=HTMLResponse)
def ask_question(request: Request, question: str = Form(...)):
    retriever, qa_chain = get_lcel_qa_chain_with_sources()

    # Get sources separately
    docs = retriever.get_relevant_documents(question)
    sources = [doc.page_content for doc in docs]

    answer = qa_chain.invoke(question)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": answer,
            "question": question,
            "sources": sources
        }
    )

@app.websocket("/ws/ask")
async def websocket_ask(websocket: WebSocket):
    await websocket.accept()
    question = await websocket.receive_text()

    callback_handler = WebSocketCallbackHandler(websocket)
    qa_chain = get_lcel_qa_chain_streaming(callbacks=[callback_handler])

    # Run the chain (final answer will still be streamed token by token)
    await qa_chain.ainvoke(question)

    await websocket.close()

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