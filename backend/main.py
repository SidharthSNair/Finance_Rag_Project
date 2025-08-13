import os
import shutil

from fastapi import FastAPI, Request, Form, UploadFile, File, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from langchain.callbacks.base import AsyncCallbackHandler

from backend.chains.qa_lcel_chain import (
    get_lcel_qa_chain,
    get_lcel_qa_chain_streaming,  # kept in case you need it elsewhere
    get_lcel_qa_chain_with_sources,
)
from backend.agents.ingest_documents import ingest_uploaded_documents
from backend.agents.multi_tool_agent import get_multi_tool_agent


app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Not strictly used elsewhere right now, but OK to keep
qa_chain = get_lcel_qa_chain()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploaded_docs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

agent = get_multi_tool_agent()


class WebSocketCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.full_answer = ""  # accumulate final text

    async def on_llm_new_token(self, token: str, **kwargs):
        self.full_answer += token
        await self.websocket.send_text(token)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": ""})


@app.post("/ask", response_class=HTMLResponse)
def ask_question(request: Request, question: str = Form(...)):
    retriever, qa_chain_with_mem = get_lcel_qa_chain_with_sources()

    # ✅ Modern retriever call
    docs = retriever.invoke(question)
    sources = [doc.page_content for doc in docs]

    # ✅ Memory-wrapped chain requires dict input + session_id
    answer = qa_chain_with_mem.invoke(
        {"question": question},
        config={"configurable": {"session_id": "form-session"}},
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": answer,
            "question": question,
            "sources": sources,
        },
    )


@app.post("/agent_ask", response_class=HTMLResponse)
def agent_ask(request: Request, question: str = Form(...)):
    # AgentExecutor returns a dict when using invoke()
    result = agent.invoke({"input": question})
    answer = result["output"]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": answer, "question": question},
    )


@app.websocket("/ws/ask")
async def websocket_ask(websocket: WebSocket):
    await websocket.accept()
    question = await websocket.receive_text()

    # 1) Per-session id from WS URL (e.g., ws://.../ws/ask?session_id=abc)
    session_id = websocket.query_params.get("session_id", "default")

    # 2) Build retriever + memory-wrapped chain
    retriever, qa_chain_with_mem = get_lcel_qa_chain_with_sources()

    # 3) Get sources (modern API)
    docs = retriever.invoke(question)
    sources = [doc.page_content for doc in docs]

    # 4) Stream tokens via callback
    callback_handler = WebSocketCallbackHandler(websocket)
    async for _ in qa_chain_with_mem.astream(
        {"question": question},
        config={
            "callbacks": [callback_handler],
            "configurable": {"session_id": session_id},  # IMPORTANT for memory
        },
    ):
        pass  # tokens sent from on_llm_new_token

    # 5) After streaming, send sources + full answer
    await websocket.send_json(
        {"sources": sources, "final_answer": getattr(callback_handler, "full_answer", None)}
    )

    await websocket.close()


@app.post("/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # Ingest uploaded documents to vectorstore
    ingest_uploaded_documents()

    # If fetch() asked for JSON, reply JSON; otherwise, redirect back to home
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header:
        return JSONResponse(content={"message": "Upload successful"})

    return RedirectResponse("/", status_code=303)
