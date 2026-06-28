"""Minimal production-grade RAG CLI using Groq + HuggingFace embeddings."""

from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from src.embeddings.hf_faiss import build_hf_embeddings, load_or_build_faiss_index
from src.ingest.pipeline import run_ingestion
from src.memory.chat_history import get_session_history
from src.retrieve.hybrid import HybridRetriever


_CITATION_PATTERN = re.compile(r"\[source=.+?, page=.+?\]")


def _format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        formatted.append(f"[source={source}, page={page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def _has_valid_citations(answer: str) -> bool:
    return bool(_CITATION_PATTERN.search(answer))


def _grounding_gate(answer: str) -> str:
    if not _has_valid_citations(answer):
        return "I don't have enough information in the provided documents."

    return answer.strip()


def build_rag_chain(input_dir: Path, index_dir: Path):
    chunks = run_ingestion(input_dir=input_dir)
    if not chunks:
        raise ValueError(f"No PDF content found in: {input_dir}")

    embeddings = build_hf_embeddings()
    vectorstore, loaded_from_disk = load_or_build_faiss_index(
        documents=chunks,
        index_dir=index_dir,
        embeddings=embeddings,
    )

    if loaded_from_disk:
        print(f"Loaded FAISS index from: {index_dir}")
    else:
        print(f"Built and saved FAISS index to: {index_dir}")

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=chunks,
        embeddings=embeddings,
        dense_k=8,
        sparse_k=8,
        final_k=6,
        mmr_lambda=0.5,
        weights=(0.5, 0.5),
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

    def _build_context(inputs: dict) -> str:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        search_query = question
        if chat_history:
            search_query = query_rewrite_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )

        docs = retriever.invoke(search_query)
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
    )


def main() -> None:
    load_dotenv()
    project_root = Path(__file__).resolve().parent
    docs_dir = project_root / "docs"
    index_dir = project_root / ".rag" / "faiss"

    if not docs_dir.exists() or not docs_dir.is_dir():
        print(f"docs folder not found at: {docs_dir}")
        print("Create the docs folder and place PDF files inside it.")
        return

    try:
        rag_chain = build_rag_chain(docs_dir, index_dir=index_dir)
    except ValueError:
        print(f"No PDF files found in: {docs_dir}")
        print("Add at least one PDF to the docs folder and run again.")
        return

    session_id = "default"
    print(
        "Hybrid RAG (FAISS + BM25 + Ensemble + MMR + memory + streaming) ready. Type 'exit' to quit."
    )
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        print("\nAnswer:", end=" ", flush=True)
        streamed_answer_parts: list[str] = []

        chunks = rag_chain.stream(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        for chunk in chunks:
            streamed_answer_parts.append(chunk)
            print(chunk, end="", flush=True)
        print()

        answer_text = "".join(streamed_answer_parts).strip()
        gated_answer = _grounding_gate(answer_text)
        if gated_answer != answer_text:
            print("Grounding check: response lacked valid source/page citations.")
            print(f"Final: {gated_answer}")


if __name__ == "__main__":
    main()
