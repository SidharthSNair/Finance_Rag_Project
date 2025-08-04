from backend.agents.financial_agent import get_financial_agent

agent = get_financial_agent()

query = input("💬 Ask a finance question: ")

response = agent.run(query)

# 4. Display response
print(f"\n🤖 Agent says: {response}")