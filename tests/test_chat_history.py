"""Unit tests for persistent chat history."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.memory.chat_history import PersistentChatMessageHistory


class PersistentChatHistoryTests(unittest.TestCase):
    def test_history_round_trip_persists_messages_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_root = Path(temp_dir)

            with patch.dict("os.environ", {"RAG_HISTORY_DIR": str(history_root)}):
                first_history = PersistentChatMessageHistory("session-1")
                first_history.add_message(HumanMessage(content="hello"))
                first_history.add_message(AIMessage(content="hi there"))

                second_history = PersistentChatMessageHistory("session-1")

                self.assertEqual(len(second_history.messages), 2)
                self.assertEqual(second_history.messages[0].content, "hello")
                self.assertEqual(second_history.messages[1].content, "hi there")

                history_file = history_root / "session-1.json"
                self.assertTrue(history_file.exists())

    def test_session_id_is_sanitized_for_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_root = Path(temp_dir)

            with patch.dict("os.environ", {"RAG_HISTORY_DIR": str(history_root)}):
                history = PersistentChatMessageHistory("one/two:three")
                history.add_message(HumanMessage(content="hello"))

                self.assertTrue((history_root / "one_two_three.json").exists())


if __name__ == "__main__":
    unittest.main()
