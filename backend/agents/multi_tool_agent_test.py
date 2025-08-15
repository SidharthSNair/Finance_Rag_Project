# backend/agents/multi_tool_agent.py
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from backend.tools.rag_tool import RAG_TOOL
from backend.tools.sql_db_tool import DB_TOOL

# Single, strict format. No signatures, no JSON for tool names.
REACT_PROMPT = PromptTemplate.from_template("""
You are a careful assistant that MUST use tools before answering.

TOOLS:
- RAG_QA: use for questions answered by the uploaded docs (e.g., "What is Rundeck?")
- DB_QA: use for questions about the local SQLite DB (companies, tickers, prices). Input is the user's question as a single plain string.

RULES:
- The only valid tool names are exactly: RAG_QA or DB_QA (no parentheses, no types).
- Always pick ONE tool first.
- Follow this format EXACTLY:

Thought: <brief reasoning>
Action: <RAG_QA or DB_QA>
Action Input: <the user's question as a plain string>
Observation: <tool result>
Thought: <do you now have enough to answer?>
Final Answer: <your final answer>

If the user asks about tickers, symbols, rows, SQL, companies, or prices -> use DB_QA.
If the user asks conceptual questions from docs -> use RAG_QA.

User question: {input}
""")

def get_multi_tool_agent():
    llm = OllamaLLM(model="llama2:7b", temperature=0)

    tools = [RAG_TOOL, DB_QA := DB_TOOL]  # both tools

    # Build a ReAct agent with OUR prompt (prevents signature hallucinations)
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    # Wrap in an executor; cap iterations to avoid loops
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=3,                # avoid infinite loops
        early_stopping_method="force",   # force stop if still confused
    )
    return executor
