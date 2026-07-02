"""Evaluation harness for the RAG pipeline.

The harness is intentionally lightweight:
- runs the existing RAG chain against a list of questions
- records latency
- checks citation presence
- optionally checks expected answer phrases and citation sources

Adapt the sample cases to your own PDF corpus.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from main import build_rag_chain


_CITATION_PATTERN = re.compile(r"\[source=(?P<source>.+?), page=(?P<page>.+?)\]")


@dataclass(frozen=True)
class EvalCase:
    """One evaluation prompt and the expected grounded behavior."""

    name: str
    question: str
    expected_contains: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    min_citations: int = 1


@dataclass(frozen=True)
class EvalResult:
    """Outcome for a single evaluation case."""

    name: str
    question: str
    answer: str
    latency_seconds: float
    citations_found: tuple[str, ...] = ()
    passed: bool = False
    checks: tuple[str, ...] = ()


def _extract_citations(answer: str) -> list[tuple[str, str]]:
    return [
        (match.group("source"), match.group("page"))
        for match in _CITATION_PATTERN.finditer(answer)
    ]


def load_eval_cases(file_path: Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file.

    Expected schema:
    [
      {
        "name": "...",
        "question": "...",
        "expected_contains": ["..."],
        "expected_sources": ["..."],
        "min_citations": 1
      }
    ]
    """
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as handle:
        raw_cases = json.load(handle)

    cases: list[EvalCase] = []
    for item in raw_cases:
        cases.append(
            EvalCase(
                name=item["name"],
                question=item["question"],
                expected_contains=tuple(item.get("expected_contains", [])),
                expected_sources=tuple(item.get("expected_sources", [])),
                min_citations=int(item.get("min_citations", 1)),
            )
        )

    return cases


def default_eval_cases() -> list[EvalCase]:
    """A small starter suite. Replace with questions tailored to your PDFs."""
    return [
        EvalCase(
            name="citation_grounding",
            question="Answer the question using the provided documents and include source citations.",
            expected_contains=("source", "page"),
            min_citations=1,
        ),
        EvalCase(
            name="grounded_refusal",
            question="If the answer is not in the documents, say so clearly.",
            expected_contains=("I don't have enough information",),
            min_citations=0,
        ),
    ]


def evaluate_case(rag_chain, case: EvalCase, session_id: str) -> EvalResult:
    """Run one evaluation case against the live RAG chain."""
    start_time = time.perf_counter()
    answer = rag_chain.invoke(
        {"question": case.question},
        config={"configurable": {"session_id": session_id}},
    )
    latency_seconds = time.perf_counter() - start_time

    citations = _extract_citations(answer)
    checks: list[str] = []
    passed = True

    if len(citations) < case.min_citations:
        passed = False
        checks.append(
            f"expected at least {case.min_citations} citation(s), found {len(citations)}"
        )

    if case.expected_contains:
        normalized_answer = answer.lower()
        missing_phrases = [
            phrase
            for phrase in case.expected_contains
            if phrase.lower() not in normalized_answer
        ]
        if missing_phrases:
            passed = False
            checks.append(f"missing expected phrase(s): {', '.join(missing_phrases)}")

    if case.expected_sources:
        cited_sources = {source for source, _ in citations}
        missing_sources = [
            source for source in case.expected_sources if source not in cited_sources
        ]
        if missing_sources:
            passed = False
            checks.append(f"missing expected source(s): {', '.join(missing_sources)}")

    return EvalResult(
        name=case.name,
        question=case.question,
        answer=answer,
        latency_seconds=latency_seconds,
        citations_found=tuple(f"{source}#page={page}" for source, page in citations),
        passed=passed,
        checks=tuple(checks),
    )


def run_eval_suite(
    docs_dir: Path,
    index_dir: Path,
    cases: Iterable[EvalCase],
) -> list[EvalResult]:
    """Build the RAG chain once and run the entire case suite."""
    rag_chain = build_rag_chain(docs_dir, index_dir)

    results: list[EvalResult] = []
    for index, case in enumerate(cases, start=1):
        session_id = f"eval-{index}"
        results.append(evaluate_case(rag_chain, case, session_id=session_id))

    return results


def render_report(results: Iterable[EvalResult]) -> str:
    """Render a readable text report for the evaluation run."""
    results_list = list(results)
    total = len(results_list)
    passed = sum(1 for result in results_list if result.passed)

    lines = [
        f"Evaluation summary: {passed}/{total} passed",
        "",
    ]

    for result in results_list:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.name}")
        lines.append(f"  question: {result.question}")
        lines.append(f"  latency_seconds: {result.latency_seconds:.3f}")
        lines.append(f"  citations: {', '.join(result.citations_found) or 'none'}")
        if result.checks:
            for check in result.checks:
                lines.append(f"  check: {check}")
        lines.append(f"  answer: {result.answer}")
        lines.append("")

    return "\n".join(lines).rstrip()
