# backend/agents/multi_tool_agent.py
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from backend.tools.rag_tool import RAG_TOOL

PROMPT_PREFIX = """You are an assistant that MUST use tools before answering.

You have ONE tool:
- RAG_QA: input is a SINGLE plain string (the user's question). Do NOT send JSON. Do NOT add fields.

CRITICAL RULES:
- The only valid tool name is exactly: RAG_QA  (no parentheses, no argument types)
- Follow this format exactly (no extra text):

Thought: <your brief reasoning>
Action: RAG_QA
Action Input: <the user's question as a plain string>
Observation: <tool result>
Thought: <do you now have enough to answer?>
Final Answer: <your final answer>

Never answer without calling RAG_QA at least once.
"""

def get_multi_tool_agent():
    llm = OllamaLLM(model="llama2:7b", temperature=0)  # keep it deterministic
    tools = [RAG_TOOL]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": PROMPT_PREFIX},
        max_iterations=2,  # <-- cap retries
        early_stopping_method="force",
        return_intermediate_steps=True,
    )
    return agent
