# backend/tools/rag_tool.py
from langchain.tools import Tool
from backend.chains.qa_lcel_chain import get_lcel_qa_chain_with_sources

def rag_answer(question: str) -> str:
    retriever, chain = get_lcel_qa_chain_with_sources()
    return chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": "agent-session"}}
    )

RAG_TOOL = Tool.from_function(
    name="RAG_QA",
    func=rag_answer,
    description=(
        "Use for questions about Rundeck or anything in the uploaded documentation. "
        "Input should be the user's natural-language question."
    ),
)
