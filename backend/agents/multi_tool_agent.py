# backend/agents/multi_tool_agent.py
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from backend.tools.rag_tool import RAG_TOOL
# (optionally add) from backend.tools.search_tools import WEB_SEARCH_TOOL
# (optionally add) from backend.tools.python_tool import PYTHON_TOOL

SYSTEM_MSG = (
    "You have tools. If the question is about Rundeck or likely answered by the uploaded docs, "
    "you MUST call RAG_QA with the exact user question. "
    "Only use WebSearch if the docs are insufficient. "
    "Use Python only for calculations or code execution."
)

def get_multi_tool_agent():
    llm = ChatOllama(model="gemma:2b")  # bump to llama3.1:8b-instruct if available
    tools = [RAG_TOOL]  # add others later after confirming RAG works
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM_MSG},
        return_intermediate_steps=True,
    )
    return agent
