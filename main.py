"""Minimal production-grade RAG CLI using Groq + HuggingFace embeddings."""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import typer

from src.embeddings.hf_faiss import (
    build_hf_embeddings,
    build_source_fingerprint,
    load_or_build_faiss_index,
)
from src.ingest.pipeline import run_ingestion
from src.memory.chat_history import get_session_history
from src.retrieve.hybrid import HybridRetriever


logger = logging.getLogger(__name__)
MAX_QUESTION_LENGTH_DEFAULT = 4000
_NOISY_LOGGERS = (
    "httpx",
    "sentence_transformers",
    "faiss",
    "groq",
)


_CITATION_PATTERN = re.compile(r"\[source=.+?, page=.+?\]")
_CITATION_EXTRACT_PATTERN = re.compile(
    r"\[source=(?P<source>.+?), page=(?P<page>.+?)\]"
)
_CITATION_SEGMENT_PATTERN = re.compile(
    r"\[source=(?P<source>.+?), page=(?P<page>.+?)\]"
)
_HISTORY_META_QUERY_PATTERN = re.compile(
    r"(?:last|2nd last|second last|previous|prior).*(?:question|ask|asked)",
    re.IGNORECASE,
)


app = typer.Typer(
    add_completion=False,
    help="Hybrid RAG CLI for asking questions over local PDFs.",
    no_args_is_help=False,
)


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    docs_dir: Path
    index_dir: Path


def _resolve_project_paths(
    docs_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> ProjectPaths:
    project_root = Path(__file__).resolve().parent
    resolved_docs_dir = docs_dir or (project_root / "docs")
    resolved_index_dir = index_dir or (project_root / ".rag" / "faiss")
    return ProjectPaths(
        project_root=project_root,
        docs_dir=resolved_docs_dir,
        index_dir=resolved_index_dir,
    )


def _format_env_value(name: str) -> str:
    value = os.getenv(name)
    return value if value else "<not set>"


def _get_max_question_length() -> int:
    return _get_env_int("RAG_MAX_QUESTION_LENGTH", MAX_QUESTION_LENGTH_DEFAULT)


def _validate_question(question: str) -> list[str]:
    issues: list[str] = []
    max_length = _get_max_question_length()

    if not question.strip():
        issues.append("question cannot be empty")

    if len(question) > max_length:
        issues.append(
            f"question is too long ({len(question)} characters, max {max_length})"
        )

    return issues


def _configure_logging() -> None:
    level_name = os.getenv("RAG_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _validate_environment() -> list[str]:
    issues: list[str] = []

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        issues.append("GROQ_API_KEY is not set")

    try:
        _load_retrieval_config()
    except ValueError as exc:
        issues.append(f"invalid retrieval config: {exc}")

    return issues


def _validate_paths(paths: ProjectPaths) -> list[str]:
    issues: list[str] = []

    if not paths.docs_dir.exists():
        issues.append(f"docs folder not found at {paths.docs_dir}")
    elif not paths.docs_dir.is_dir():
        issues.append(f"docs path is not a directory: {paths.docs_dir}")
    elif not any(paths.docs_dir.rglob("*.pdf")):
        issues.append(f"no PDF files found in {paths.docs_dir}")

    return issues


def _print_status(paths: ProjectPaths) -> None:
    typer.echo("Hybrid RAG status")
    typer.echo(f"  project_root: {paths.project_root}")
    typer.echo(f"  docs_dir: {paths.docs_dir}")
    typer.echo(f"  index_dir: {paths.index_dir}")
    typer.echo(f"  GROQ_API_KEY: {'set' if os.getenv('GROQ_API_KEY') else 'missing'}")
    typer.echo(f"  RAG_HISTORY_DIR: {_format_env_value('RAG_HISTORY_DIR')}")
    typer.echo(f"  RAG_DENSE_K: {_format_env_value('RAG_DENSE_K')}")
    typer.echo(f"  RAG_SPARSE_K: {_format_env_value('RAG_SPARSE_K')}")
    typer.echo(f"  RAG_FINAL_K: {_format_env_value('RAG_FINAL_K')}")
    typer.echo(f"  RAG_MMR_LAMBDA: {_format_env_value('RAG_MMR_LAMBDA')}")
    typer.echo(f"  RAG_HYBRID_WEIGHTS: {_format_env_value('RAG_HYBRID_WEIGHTS')}")
    typer.echo(f"  RAG_LOG_LEVEL: {_format_env_value('RAG_LOG_LEVEL')}")
    typer.echo(
        f"  RAG_MAX_QUESTION_LENGTH: {_format_env_value('RAG_MAX_QUESTION_LENGTH')}"
    )

    path_issues = _validate_paths(paths)
    env_issues = _validate_environment()

    if not path_issues and not env_issues:
        typer.secho("  readiness: ready", fg=typer.colors.GREEN)
        return

    typer.secho("  readiness: needs attention", fg=typer.colors.YELLOW)
    for issue in path_issues + env_issues:
        typer.echo(f"  - {issue}")


def _print_validation_failures(issues: list[str]) -> None:
    typer.secho("Configuration check failed.", fg=typer.colors.RED, err=True)
    for issue in issues:
        typer.echo(f"  - {issue}", err=True)


def _format_llm_error(exc: Exception) -> str:
    message = str(exc)
    lowered_message = message.lower()

    if (
        "503" in lowered_message
        or "over capacity" in lowered_message
        or "service unavailable" in lowered_message
    ):
        return (
            "The Groq model is currently unavailable or over capacity. "
            "Please retry in a moment."
        )

    if "401" in lowered_message or "unauthorized" in lowered_message:
        return "Groq authentication failed. Check that GROQ_API_KEY is set correctly."

    return f"Model request failed: {message}"


def _extract_recent_questions(history_messages: list, count: int) -> list[str]:
    questions = [
        message.content.strip()
        for message in history_messages
        if getattr(message, "type", "") == "human" and str(message.content).strip()
    ]
    if count <= 0:
        return []
    return questions[-count:]


def _maybe_answer_from_history(question: str, history_messages: list) -> str | None:
    normalized_question = question.lower().strip()

    if not _HISTORY_META_QUERY_PATTERN.search(normalized_question):
        return None

    recent_questions = _extract_recent_questions(history_messages, 3)
    if not recent_questions:
        return "I don't have enough information in the conversation history."

    if "2nd" in normalized_question or "second last" in normalized_question:
        if len(recent_questions) < 2:
            return "I don't have enough information in the conversation history."
        return f"The second-to-last question you asked was: {recent_questions[-2]}"

    return f"The last question you asked was: {recent_questions[-1]}"


def _load_rag_chain(
    paths: ProjectPaths,
) -> tuple[RunnableWithMessageHistory, Callable[[str, list], list]]:
    try:
        return build_rag_chain(paths.docs_dir, index_dir=paths.index_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_single_question(
    rag_chain: RunnableWithMessageHistory,
    retrieve_docs,
    question: str,
    session_id: str,
) -> None:
    question_issues = _validate_question(question)
    if question_issues:
        typer.secho("Invalid question.", fg=typer.colors.RED, err=True)
        for issue in question_issues:
            typer.echo(f"  - {issue}", err=True)
        return

    current_history = list(get_session_history(session_id).messages)
    history_answer = _maybe_answer_from_history(question, current_history)
    if history_answer is not None:
        typer.echo(history_answer)
        return

    try:
        logger.info("question submitted", extra={"session_id": session_id})
        streamed_answer_parts: list[str] = []
        chunks = rag_chain.stream(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        for chunk in chunks:
            streamed_answer_parts.append(chunk)

        answer_text = "".join(streamed_answer_parts).strip()
    except Exception as exc:  # pragma: no cover - exercised through CLI smoke tests
        logger.warning("llm request failed", exc_info=True)
        typer.secho(_format_llm_error(exc), fg=typer.colors.YELLOW, err=True)
        return

    normalized_answer = _normalize_citations(answer_text)
    current_history = list(get_session_history(session_id).messages)
    validation_docs = retrieve_docs(question, current_history)
    allowed_citations = _extract_allowed_citations(validation_docs)

    gated_answer = _grounding_gate(normalized_answer, allowed_citations)
    if gated_answer != normalized_answer:
        typer.secho(gated_answer, fg=typer.colors.YELLOW)
        return

    typer.echo(gated_answer)


def _run_chat_session(
    docs_dir: Optional[Path],
    index_dir: Optional[Path],
    session_id: str,
) -> None:
    paths = _resolve_project_paths(docs_dir=docs_dir, index_dir=index_dir)
    issues = _validate_paths(paths) + _validate_environment()
    if issues:
        _print_validation_failures(issues)
        raise typer.Exit(code=1)

    try:
        logger.info("starting chat session", extra={"session_id": session_id})
        rag_chain, retrieve_docs = _load_rag_chain(paths)
    except typer.BadParameter as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(
        "Hybrid RAG ready. Use /status for a readiness check, or type 'exit' to quit.",
        fg=typer.colors.GREEN,
    )
    typer.echo("Tip: use the ask command for one-off questions.")

    while True:
        try:
            question = typer.prompt("Question", default="").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        _run_single_question(rag_chain, retrieve_docs, question, session_id)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    docs_dir: Optional[Path] = typer.Option(None, help="Directory containing PDFs."),
    index_dir: Optional[Path] = typer.Option(None, help="Directory for FAISS cache."),
    session_id: str = typer.Option("default", help="Conversation session id."),
) -> None:
    """Run chat mode by default when no subcommand is provided."""
    if ctx.invoked_subcommand is not None:
        return

    load_dotenv()
    _configure_logging()
    _run_chat_session(docs_dir=docs_dir, index_dir=index_dir, session_id=session_id)


@app.command()
def status() -> None:
    """Show readiness, paths, and environment checks."""
    load_dotenv()
    _configure_logging()
    paths = _resolve_project_paths()
    _print_status(paths)


@app.command()
def build_index(
    docs_dir: Optional[Path] = typer.Option(None, help="Directory containing PDFs."),
    index_dir: Optional[Path] = typer.Option(None, help="Directory for FAISS cache."),
) -> None:
    """Build or refresh the local FAISS index."""
    load_dotenv()
    _configure_logging()
    paths = _resolve_project_paths(docs_dir=docs_dir, index_dir=index_dir)
    issues = _validate_paths(paths)
    if issues:
        _print_validation_failures(issues)
        raise typer.Exit(code=1)

    try:
        logger.info(
            "building local FAISS index", extra={"docs_dir": str(paths.docs_dir)}
        )
        chunks = run_ingestion(input_dir=paths.docs_dir)
        if not chunks:
            raise ValueError(f"No PDF content found in: {paths.docs_dir}")

        embeddings = build_hf_embeddings()
        source_fingerprint = build_source_fingerprint(paths.docs_dir)
        load_or_build_faiss_index(
            documents=chunks,
            index_dir=paths.index_dir,
            embeddings=embeddings,
            source_fingerprint=source_fingerprint,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("FAISS index is ready.", fg=typer.colors.GREEN)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask over the documents."),
    docs_dir: Optional[Path] = typer.Option(None, help="Directory containing PDFs."),
    index_dir: Optional[Path] = typer.Option(None, help="Directory for FAISS cache."),
    session_id: str = typer.Option("default", help="Conversation session id."),
) -> None:
    """Ask one question and print a grounded answer."""
    load_dotenv()
    _configure_logging()
    paths = _resolve_project_paths(docs_dir=docs_dir, index_dir=index_dir)
    issues = _validate_paths(paths) + _validate_environment()
    if issues:
        _print_validation_failures(issues)
        raise typer.Exit(code=1)

    question_issues = _validate_question(question)
    if question_issues:
        _print_validation_failures(question_issues)
        raise typer.Exit(code=1)

    rag_chain, retrieve_docs = _load_rag_chain(paths)
    _run_single_question(rag_chain, retrieve_docs, question, session_id)


@app.command()
def chat(
    docs_dir: Optional[Path] = typer.Option(None, help="Directory containing PDFs."),
    index_dir: Optional[Path] = typer.Option(None, help="Directory for FAISS cache."),
    session_id: str = typer.Option("default", help="Conversation session id."),
) -> None:
    """Start the interactive chat loop."""
    load_dotenv()
    _configure_logging()
    _run_chat_session(docs_dir=docs_dir, index_dir=index_dir, session_id=session_id)


def _format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        formatted.append(f"[source={source}, page={page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def _has_valid_citations(answer: str) -> bool:
    return bool(_CITATION_PATTERN.search(answer))


def _extract_allowed_citations(docs) -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    for doc in docs:
        source = str(doc.metadata.get("source", "unknown"))
        page = str(doc.metadata.get("page", "n/a"))
        allowed.add((source, page))
    return allowed


def _extract_answer_citations(answer: str) -> set[tuple[str, str]]:
    citations: set[tuple[str, str]] = set()
    for match in _CITATION_EXTRACT_PATTERN.finditer(answer):
        citations.add((match.group("source"), match.group("page")))
    return citations


def _normalize_citations(answer: str) -> str:
    """Split malformed page lists into single-page citations.

    Example:
    [source=foo.pdf, page=15, 7, 18] -> [source=foo.pdf, page=15] [source=foo.pdf, page=7] [source=foo.pdf, page=18]
    """

    def _replace(match: re.Match[str]) -> str:
        source = match.group("source").strip()
        page_text = match.group("page").strip()
        pages = [page.strip() for page in re.split(r"[,;]", page_text) if page.strip()]

        if len(pages) <= 1:
            return f"[source={source}, page={page_text}]"

        normalized_parts = [f"[source={source}, page={page}]" for page in pages]
        return " ".join(normalized_parts)

    return _CITATION_SEGMENT_PATTERN.sub(_replace, answer)


def _grounding_gate(answer: str, allowed_citations: set[tuple[str, str]]) -> str:
    if not _has_valid_citations(answer):
        return "I don't have enough information in the provided documents."

    if not allowed_citations:
        return "I don't have enough information in the provided documents."

    cited_sources = _extract_answer_citations(answer)
    if not cited_sources:
        return "I don't have enough information in the provided documents."

    if not cited_sources.issubset(allowed_citations):
        return "I don't have enough information in the provided documents."

    return answer.strip()


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _get_env_weight_pair(default: tuple[float, float]) -> tuple[float, float]:
    value = os.getenv("RAG_HYBRID_WEIGHTS")
    if not value:
        return default

    first, second = value.split(",", maxsplit=1)
    return float(first.strip()), float(second.strip())


def _load_retrieval_config() -> dict:
    return {
        "dense_k": _get_env_int("RAG_DENSE_K", 8),
        "sparse_k": _get_env_int("RAG_SPARSE_K", 8),
        "final_k": _get_env_int("RAG_FINAL_K", 6),
        "mmr_lambda": _get_env_float("RAG_MMR_LAMBDA", 0.5),
        "weights": _get_env_weight_pair((0.5, 0.5)),
    }


def build_rag_chain(input_dir: Path, index_dir: Path):
    chunks = run_ingestion(input_dir=input_dir)
    if not chunks:
        raise ValueError(f"No PDF content found in: {input_dir}")

    retrieval_config = _load_retrieval_config()

    embeddings = build_hf_embeddings()
    source_fingerprint = build_source_fingerprint(input_dir)
    vectorstore, loaded_from_disk = load_or_build_faiss_index(
        documents=chunks,
        index_dir=index_dir,
        embeddings=embeddings,
        source_fingerprint=source_fingerprint,
    )

    if loaded_from_disk:
        print(f"Loaded FAISS index from: {index_dir}")
    else:
        print(f"Built and saved FAISS index to: {index_dir}")

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=chunks,
        embeddings=embeddings,
        dense_k=retrieval_config["dense_k"],
        sparse_k=retrieval_config["sparse_k"],
        final_k=retrieval_config["final_k"],
        mmr_lambda=retrieval_config["mmr_lambda"],
        weights=retrieval_config["weights"],
    )

    query_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    answer_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        streaming=True,
    )

    query_rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user's latest question into a standalone search query for PDF retrieval. Use the conversation history when needed, and return only the rewritten query.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grounded document QA assistant. Answer ONLY from the provided context. If the answer is not present in context, respond exactly: I don't have enough information in the provided documents. Keep the answer concise. Add citations in this format: [source=<path>, page=<number>]. Do not invent facts or citations.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )

    query_rewrite_chain = query_rewrite_prompt | query_llm | StrOutputParser()

    def _retrieve_docs(question: str, chat_history: list) -> list:
        search_query = question
        if chat_history:
            search_query = query_rewrite_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )

        return retriever.invoke(search_query)

    def _build_context(inputs: dict) -> str:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        docs = _retrieve_docs(question, chat_history)
        inputs["_retrieved_docs"] = docs
        return _format_docs(docs)

    answer_chain = (
        {
            "context": RunnableLambda(_build_context),
            "question": RunnableLambda(lambda inputs: inputs["question"]),
            "chat_history": RunnableLambda(
                lambda inputs: inputs.get("chat_history", [])
            ),
        }
        | answer_prompt
        | answer_llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        answer_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ), _retrieve_docs


def main() -> None:
    app()


if __name__ == "__main__":
    main()
