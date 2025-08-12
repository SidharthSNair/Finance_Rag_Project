# backend/tools/rag_tool.py
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from backend.chains.qa_lcel_chain import get_lcel_qa_chain_with_sources

def _rag_answer(q: str) -> str:
    retriever, chain = get_lcel_qa_chain_with_sources()
    # pass as dict because your chain expects {"question": ...}
    return chain.invoke({"question": q})

class RAGInput(BaseModel):
    q: str = Field(description="A natural language question about the uploaded Rundeck docs")

RAG_TOOL = StructuredTool.from_function(
    name="RAG_QA",
    func=_rag_answer,
    args_schema=RAGInput,
    description=(
        "Use ONLY for questions about Rundeck or anything in the uploaded documentation. "
        "If the user asks about Rundeck setup, config, plugins, troubleshooting, or errors, "
        "call this tool with the user's question in 'q'."
    ),
)
