# backend/agents/multi_tool_agent_test.py
from typing import List, Union, Optional
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import AgentOutputParser  # pydantic BaseModel
from pydantic import Field

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

class LenientReActParser(AgentOutputParser):
    # ✅ declare as a Pydantic field so you can set it during construction
    valid_tools: List[str] = Field(default_factory=list)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Very lightweight normalizer:
        - Finds 'Action:' and 'Action Input:' lines
        - Strips any extra words like 'Use' or '(q: ...)'
        - Maps to the nearest valid tool name if the model added junk
        - If it can't find an Action, tries to return a Final Answer
        """
        # split non-empty trimmed lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        action_line: Optional[str] = next((l for l in lines if l.lower().startswith("action:")), None)
        input_line: Optional[str]  = next((l for l in lines if l.lower().startswith("action input:")), None)

        # If the model already produced a Final Answer, return it
        final_line = next((l for l in lines if l.lower().startswith("final answer:")), None)
        if final_line and not action_line:
            final = final_line.split(":", 1)[1].strip()
            return AgentFinish(return_values={"output": final}, log=text)

        if not action_line:
            # No action, but no final answer either — ask it to try again
            raise ValueError("Could not find 'Action:' in the output.")

        # Extract raw action name and normalize
        raw_action = action_line.split(":", 1)[1].strip()
        # Common junk cleanup: remove “Use ” prefix, drop anything after space/paren/colon
        raw_action = raw_action.replace("Use ", "").strip()
        raw_action = raw_action.split()[0].split("(")[0].split(":")[0].strip()

        # Snap to a valid tool name if it’s close
        action = next((t for t in self.valid_tools if t.lower() == raw_action.lower()), None)
        if action is None and self.valid_tools:
            # last-resort: pick first tool to avoid "not a valid tool"
            action = self.valid_tools[0]

        # Extract action input (default to empty)
        action_input = ""
        if input_line:
            action_input = input_line.split(":", 1)[1].strip()
            # Strip quotes if the model wrapped it
            if (action_input.startswith(("'", '"')) and
                action_input.endswith(("'", '"')) and len(action_input) >= 2):
                action_input = action_input[1:-1]

        return AgentAction(tool=action, tool_input=action_input, log=text)

def get_multi_tool_agent():
    llm = OllamaLLM(model="llama2:7b", temperature=0)
    tools = [RAG_TOOL, DB_TOOL]   # include the tools you want
    tool_names = [t.name for t in tools]

    prompt = PromptTemplate(
        template=REACT_TEMPLATE,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    )

    # ✅ pass a constructed parser with the tool names injected
    output_parser = LenientReActParser(valid_tools=tool_names)

    agent_runnable = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=output_parser,
    )

    agent = AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
    )
    return agent
