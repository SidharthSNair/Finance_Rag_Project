import os
import shutil


from fastapi import FastAPI, Request, Form, UploadFile, File, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from langchain.callbacks.base import AsyncCallbackHandler

#from chains.qa_chain import get_qa_chain
from backend.chains.qa_lcel_chain import (
    get_lcel_qa_chain,
    get_lcel_qa_chain_streaming,
    get_lcel_qa_chain_with_sources
)
from backend.agents.ingest_documents import ingest_uploaded_documents


app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

qa_chain = get_lcel_qa_chain()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploaded_docs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

class WebSocketCallbackHandler(AsyncCallbackHandler):
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

    retriever, qa_chain = get_lcel_qa_chain_with_sources()
    docs = retriever.get_relevant_documents(question)
    sources = [doc.page_content for doc in docs]

    callback_handler = WebSocketCallbackHandler(websocket)

    # 4. Run the LCEL chain with streaming + callback
    async for chunk in qa_chain.astream(
            question,
        config={"callbacks": [callback_handler]}
    ):
        pass  # chunks are handled by the callback handler in real time
    # 5. After streaming tokens, send the sources
    await websocket.send_json({"sources": sources})

    await websocket.close()

@app.post("/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # Ingest uploaded documents to vectorstore
    ingest_uploaded_documents()

    # Check if the request came from JavaScript (fetch)
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header:
        return JSONResponse(content={"message": "Upload successful"})

    # Default: assume browser form, redirect to home
    return RedirectResponse("/", status_code=303)