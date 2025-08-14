# backend/tools/rag_tool.py
from langchain.tools import Tool
from backend.chains.qa_lcel_chain import get_lcel_qa_chain_with_sources

def rag_answer(q: str) -> str:
    """Return an answer from the vectorstore given a plain-text question."""
    retriever, chain = get_lcel_qa_chain_with_sources()
    # your LCEL chain expects {"question": "..."}
    answer =  chain.invoke(
        {"question": q},
        config={"configurable": {"session_id": "agent-session"}}
    )
    # Get top-k docs that were retrieved (same retriever)
    top_docs = retriever.invoke(q)  # returns a list of Documents
    src_list = []
    for i, d in enumerate(top_docs[:3], start=1):
        meta = d.metadata or {}
        name = meta.get("source") or meta.get("file_path") or meta.get("path") or "unknown"
        snippet = (d.page_content or "").strip().replace("\n", " ")
        snippet = snippet[:220] + ("…" if len(snippet) > 220 else "")
        src_list.append(f"{i}. {name} — “{snippet}”")

    sources_text = "\n\nSources:\n" + ("\n".join(src_list) if src_list else "No sources found.")
    return f"{answer}{sources_text}"


RAG_TOOL = Tool.from_function(
    name="RAG_QA",
    func=rag_answer,
    description=(
        "Use for questions about Rundeck or anything in the uploaded documents. "
        "Input must be the user's question as a SINGLE plain string (no JSON)."
    ),
    infer_schema=False,   # <- important: keeps it as a single string input
    return_direct=True,
)
