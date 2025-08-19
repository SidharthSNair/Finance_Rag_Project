# backend/agents/multi_tool_agent_test.py
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.prompts import PromptTemplate

from backend.tools.rag_tool import RAG_TOOL
from backend.tools.sql_db_tool import DB_TOOL  # keep if you added it

REACT_TEMPLATE = """You are a careful, tool-using assistant. You MUST call a tool before answering.

You can use ONLY these tools:
{tools}

Valid tool names (MUST match exactly): {tool_names}

CRITICAL FORMAT (no extra words, no JSON):
Thought: <your brief reasoning>
Action: <EXACT tool name from {tool_names} — only the name, nothing else>
Action Input: <the user's question as a SINGLE plain string>
Observation: <tool output>
...(repeat Thought/Action/Action Input/Observation if needed)
Thought: Do I have enough information?
Final Answer: <your final answer to the user in plain English>

❌ BAD:
Action: Use DB_QA to run a query
Action: RAG_QA(q: "what is...")

✅ GOOD:
Action: DB_QA
Action Input: What is the ticker for Apple Inc.?

Begin!

Question: {input}
{agent_scratchpad}
"""

def get_multi_tool_agent():
    llm = OllamaLLM(model="llama2:7b", temperature=0)

    # include whichever tools you want available
    tools = [RAG_TOOL, DB_TOOL]  # or just [RAG_TOOL]

    prompt = PromptTemplate(
        template=REACT_TEMPLATE,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    )

    # Build the runnable ReAct agent
    agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Wrap it in an executor
    agent = AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,
        max_iterations=4,          # prevents endless loops
        handle_parsing_errors=True # lets it recover from minor format slips
    )
    return agent
