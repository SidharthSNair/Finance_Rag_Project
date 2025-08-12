# backend/agents/multi_tool_agent.py
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from backend.tools.rag_tool import RAG_TOOL
from backend.tools.search_tools import WEB_SEARCH_TOOL   # your existing web tool
from backend.tools.python_tool import PYTHON_TOOL        # your existing python tool

SYSTEM_MSG = (
    "You are an assistant with tools. "
    "If the question is about Rundeck or appears answerable from the uploaded docs, "
    "you MUST call the tool RAG_QA first. "
    "Only use WebSearch when the docs are insufficient or the user asks for web info. "
    "Use PythonREPL only for calculations or code execution. "
    "Always show final concise answer."
)

def get_multi_tool_agent():
    llm = ChatOllama(model="gemma:2b")  # consider a slightly larger model like llama3.1:8b if available
    tools = [RAG_TOOL, WEB_SEARCH_TOOL, PYTHON_TOOL]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        agent_kwargs={"system_message": SYSTEM_MSG},
    )
    return agent
