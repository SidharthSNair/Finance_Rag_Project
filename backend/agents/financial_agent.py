from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Ollama
from backend.chains.qa_chain import get_qa_chain

# 1. Define the QA chain for document-based answers
qa_chain = get_qa_chain()

# 2. Create the tool with a proper detailed description
qa_tool = Tool.from_function(
    name="RundeckRAGTool",
    func=qa_chain.run,
    description=(
        "Use this tool to answer questions related to Rundeck, its setup, configuration, and troubleshooting. "
        "The input should be a natural language question about Rundeck. "
        "The tool returns answers retrieved using a Retrieval-Augmented Generation (RAG) system."
    )
)

# 3. Initialize the agent with the tool
def get_financial_agent():
    llm = Ollama(model="gemma:2b")
    tools = [qa_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent
