# backend/tools/sql_db_tool.py
import os
import re
from typing import Optional

from langchain.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "finance.db")
)

_SELECT_RE = re.compile(r"(?is)\bselect\b.*?;", re.DOTALL)

def _extract_first_select(sql_text: str) -> Optional[str]:
    """
    Pull out the first complete SELECT ... ; statement from any text the model returns.
    Returns None if no SELECT is found.
    """
    if not sql_text:
        return None
    m = _SELECT_RE.search(sql_text)
    if not m:
        return None
    sql = m.group(0).strip()
    # Optional: normalize whitespace
    sql = re.sub(r"\s+", " ", sql)
    return sql

def db_answer(q: str) -> str:
    """
    Answer DB questions. Input must be a plain English question.
    We only allow read-only SELECT queries.
    """
    llm = OllamaLLM(model="llama2:7b", temperature=0)
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    # Let the LLM write SQL for the given question
    nl_to_sql = create_sql_query_chain(llm, db)
    raw_sql = nl_to_sql.invoke({"question": q})

    # Sanitize: extract only the first SELECT ... ; from the model output
    safe_sql = _extract_first_select(raw_sql)

    if not safe_sql or not safe_sql.lower().strip().startswith("select"):
        return (
            "SQL (blocked):\n"
            f"Question: {q}\n\n"
            f"ModelOutput:\n{raw_sql}\n\n"
            "Reason: I could not find a clean SELECT statement to run. "
            "Please rephrase the question."
        )

    # Run only the sanitized SELECT
    try:
        rows = db.run(safe_sql)  # returns a stringified table / rows
    except Exception as e:
        return (
            "SQL (error while executing):\n"
            f"{safe_sql}\n\n"
            f"Error: {e}"
        )

    return (
        "SQL (executed safely):\n"
        f"{safe_sql}\n\n"
        f"RESULTS:\n{rows}"
    )

DB_TOOL = Tool.from_function(
    name="DB_QA",
    func=db_answer,
    description=(
        "Ask questions about companies/prices stored in the local SQLite DB. "
        "Input is a SINGLE plain string."
    ),
    infer_schema=False,
    return_direct=True,
)
