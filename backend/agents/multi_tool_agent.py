# backend/agents/multi_tool_agent.py
#from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from langchain import hub
from langchain.agents import initialize_agent, AgentType
from backend.tools.rag_tool import RAG_TOOL
# (optionally add) from backend.tools.search_tools import WEB_SEARCH_TOOL
# (optionally add) from backend.tools.python_tool import PYTHON_TOOL

prefix = (
    "You can use tools. To answer a user question, ALWAYS call a tool first. "
    "The tool RAG_QA takes exactly one argument q which is the question string."
)

def get_multi_tool_agent():
    llm = OllamaLLM(model="llama2:7b")  # bump to llama3.1:8b-instruct if available
    tools = [RAG_TOOL]  # add others later after confirming RAG works
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": prefix},
        return_intermediate_steps=True,
    )
    return agent
