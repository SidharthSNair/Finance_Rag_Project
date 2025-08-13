# backend/agents/multi_tool_agent.py
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from backend.chains.qa_lcel_chain import get_lcel_qa_chain  # the simple RAG chain (no memory wrapper)

def _rag_answer(q: str) -> str:
    """
    Accepts a plain question string and returns an answer from the RAG chain.
    IMPORTANT: pass a string to the chain that expects a string.
    """
    rag_chain = get_lcel_qa_chain()         # your non-memory LCEL chain
    return rag_chain.invoke(q)              # chain is built to accept a bare string

RAG_QA_TOOL = Tool(
    name="RAG_QA",
    func=_rag_answer,
    description=(
        "Use this to answer questions about the uploaded Rundeck docs. "
        "Input must be a single plain string question, e.g. 'How do I configure node executors?'."
    ),
)

def get_multi_tool_agent():
    llm = ChatOllama(model="gemma:2b")
    agent = initialize_agent(
        tools=[RAG_QA_TOOL],                     # keep ONLY this tool until it works
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent
