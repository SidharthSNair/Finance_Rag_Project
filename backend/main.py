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
from backend.agents.multi_tool_agent_test import get_multi_tool_agent

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

agent = get_multi_tool_agent()

class WebSocketCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.full_answer = ""  # to store the complete answer

    async def on_llm_new_token(self, token: str, **kwargs):
        self.full_answer += token
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
# @app.post("/agent_ask", response_class=HTMLResponse)
# def agent_ask(request: Request, question: str = Form(...)):
#     result = agent.invoke({"input": question})
#     # result may be a dict if return_intermediate_steps=True
#     answer = result["output"] if isinstance(result, dict) else result
#     # backend/main.py (your /agent_ask handler)
#     result = agent.invoke({"input": question})
#     answer = result["output"] if isinstance(result, dict) else result
#
#     # If you want to debug which tools got used:
#     steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
#     for action, observation in steps:
#         print("TOOL CALLED:", action.tool, "ARGS:", action.tool_input)
#         print("OBSERVATION:", str(observation)[:200], "...\n")
#
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request, "answer": answer, "question": question}
#     )

@app.post("/agent_ask", response_class=HTMLResponse)
def agent_ask(request: Request, question: str = Form(...)):
    # AgentExecutor returns a dict when using invoke()
    result = agent.invoke({"input": question})
    answer = result["output"]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": answer, "question": question}
    )

@app.websocket("/ws/ask")
async def websocket_ask(websocket: WebSocket):
    await websocket.accept()
    question = await websocket.receive_text()

    # 1) Pull per-session id from the WS URL, e.g. ws://.../ws/ask?session_id=abc
    session_id = websocket.query_params.get("session_id", "default")

    # 2) Build retriever + memory-wrapped chain
    retriever, qa_chain = get_lcel_qa_chain_with_sources()

    # 3) Get sources (use invoke to avoid the deprecation warning)
    docs = retriever.invoke(question)
    sources = [doc.page_content for doc in docs]

    # 4) Stream tokens to the browser via callback
    callback_handler = WebSocketCallbackHandler(websocket)
    async for _ in qa_chain.astream(
        {"question": question},  # <-- pass a dict if your chain expects {question,...}
        config={
            "callbacks": [callback_handler],
            "configurable": {"session_id": session_id},  # <-- IMPORTANT for memory
        },
    ):
        pass  # tokens are sent from on_llm_new_token

    # 5) After streaming, send sources and (optionally) the full answer
    await websocket.send_json({
        "sources": sources,
        "final_answer": getattr(callback_handler, "full_answer", None)
    })

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
