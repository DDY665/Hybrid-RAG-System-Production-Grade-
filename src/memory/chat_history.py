"""Session-scoped chat history store for conversational memory."""

from langchain_community.chat_message_histories import ChatMessageHistory


_SESSION_STORE: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return the chat history for a session, creating it if needed."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]
