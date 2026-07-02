"""Unit tests for the main RAG helper functions."""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from main import (
    _format_llm_error,
    _grounding_gate,
    _normalize_citations,
    _maybe_answer_from_history,
    _validate_question,
)


class MainHelperTests(unittest.TestCase):
    def test_normalize_citations_splits_multiple_pages(self) -> None:
        answer = "See [source=doc.pdf, page=15, 7, 18] for details."

        normalized = _normalize_citations(answer)

        self.assertEqual(
            normalized,
            "See [source=doc.pdf, page=15] [source=doc.pdf, page=7] [source=doc.pdf, page=18] for details.",
        )

    def test_grounding_gate_rejects_unknown_citations(self) -> None:
        answer = "This is supported by [source=other.pdf, page=2]."

        gated = _grounding_gate(answer, {("doc.pdf", "1")})

        self.assertEqual(
            gated,
            "I don't have enough information in the provided documents.",
        )

    def test_grounding_gate_accepts_allowed_citation(self) -> None:
        answer = "This is supported by [source=doc.pdf, page=1]."

        gated = _grounding_gate(answer, {("doc.pdf", "1")})

        self.assertEqual(gated, answer)

    def test_format_llm_error_handles_over_capacity(self) -> None:
        class DummyError(Exception):
            pass

        message = _format_llm_error(DummyError("Error code: 503 - over capacity"))

        self.assertIn("over capacity", message.lower())

    def test_validate_question_rejects_empty_input(self) -> None:
        issues = _validate_question("   ")

        self.assertTrue(issues)
        self.assertIn("empty", " ".join(issues).lower())

    def test_maybe_answer_from_history_returns_previous_question(self) -> None:
        history = [
            HumanMessage(content="what is the doc about?"),
            AIMessage(content="answer one"),
            HumanMessage(content="who is the candidate?"),
            AIMessage(content="answer two"),
        ]

        answer = _maybe_answer_from_history("what was my last question?", history)

        self.assertEqual(
            answer, "The last question you asked was: who is the candidate?"
        )

    def test_maybe_answer_from_history_returns_second_last_question(self) -> None:
        history = [
            HumanMessage(content="first question"),
            AIMessage(content="answer one"),
            HumanMessage(content="second question"),
            AIMessage(content="answer two"),
            HumanMessage(content="third question"),
        ]

        answer = _maybe_answer_from_history("what was the 2nd last question?", history)

        self.assertEqual(
            answer,
            "The second-to-last question you asked was: second question",
        )


if __name__ == "__main__":
    unittest.main()
