"""Persistent chat history store for conversational memory."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from pydantic import PrivateAttr


def _history_root() -> Path:
    """Resolve the directory used to persist chat history."""
    env_value = os.getenv("RAG_HISTORY_DIR")
    if env_value:
        return Path(env_value)

    return Path(__file__).resolve().parents[2] / ".rag" / "history"


def _session_history_path(session_id: str) -> Path:
    safe_session_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in session_id
    )
    return _history_root() / f"{safe_session_id}.json"


class PersistentChatMessageHistory(ChatMessageHistory):
    """File-backed chat history compatible with LangChain session memory."""

    session_id: str
    _history_file: Path = PrivateAttr()

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id=session_id)
        self._history_file = _session_history_path(session_id)
        self._load()

    def _load(self) -> None:
        if not self._history_file.exists():
            return

        with self._history_file.open("r", encoding="utf-8") as handle:
            raw_messages = json.load(handle)

        self.messages = messages_from_dict(raw_messages)

    def _save(self) -> None:
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        with self._history_file.open("w", encoding="utf-8") as handle:
            json.dump(messages_to_dict(self.messages), handle, indent=2)

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        self._save()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        super().add_messages(messages)
        self._save()

    def clear(self) -> None:
        super().clear()
        self._save()


_SESSION_STORE: dict[str, PersistentChatMessageHistory] = {}


def get_session_history(session_id: str) -> PersistentChatMessageHistory:
    """Return a persistent chat history for the given session id."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = PersistentChatMessageHistory(session_id)
    return _SESSION_STORE[session_id]
