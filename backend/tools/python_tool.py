# backend/tools/python_tool.py
from langchain_experimental.tools import PythonREPLTool

PYTHON_TOOL = PythonREPLTool()
PYTHON_TOOL.name = "PythonREPL"
PYTHON_TOOL.description = (
    "Execute Python code for calculations or small text processing. "
    "Input should be valid Python. Returns stdout or result."
)
