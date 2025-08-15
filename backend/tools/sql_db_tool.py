# backend/tools/db_tool.py
import os
from textwrap import dedent

from langchain.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain


DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "finance.db")
)

def db_answer(q: str) -> str:
    """Answer DB questions. Input must be a plain English question."""
    try:
        # Deterministic SQL generation
        llm = OllamaLLM(model="llama2:7b", temperature=0)

        # Add a few sample rows per table so the LLM understands schema better
        db = SQLDatabase.from_uri(
            f"sqlite:///{DB_PATH}",
            sample_rows_in_table_info=3,
        )

        # Natural language -> SQL
        nl_to_sql = create_sql_query_chain(llm, db)
        sql = nl_to_sql.invoke({"question": q}).strip()

        # Guardrail: only allow SELECT
        if not sql.lower().lstrip().startswith("select"):
            return dedent(f"""\
                SQL (blocked):
                {sql}

                Only read-only SELECT queries are allowed from this tool.
            """)

        # Execute and trim long outputs
        raw = db.run(sql)  # returns a stringified result set
        # Keep the first ~1000 chars so UI stays snappy
        results = raw if len(raw) <= 1000 else (raw[:1000] + "\n... [truncated]")

        return f"SQL:\n{sql}\n\nRESULTS:\n{results}"

    except Exception as e:
        # Never crash the agent; return the error text as the tool response
        return f"[DB TOOL ERROR] {type(e).__name__}: {e}"

DB_TOOL = Tool.from_function(
    name="DB_QA",
    func=db_answer,
    description=(
        "Ask questions about companies/prices stored in the local SQLite DB. "
        "Input is a SINGLE plain string (e.g., 'What is the ticker for Apple Inc.?')."
    ),
    infer_schema=False,   # keep it a single string input (no JSON signatures)
    return_direct=False,
)
