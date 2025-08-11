from langchain_community.chat_message_histories import ChatMessageHistory

# Simple in-memory store (can replace with Redis, DB later)
history_store = {}

def get_chat_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]
