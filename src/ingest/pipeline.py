"""Ingestion orchestration pipeline."""

from pathlib import Path

from langchain_core.documents import Document

from src.ingest.chunking import split_documents
from src.ingest.loader_pdf import load_pdf_documents
from src.ingest.ocr_fallback import extract_text_with_ocr


def _needs_ocr(docs: list[Document]) -> bool:
    """Heuristic for scanned/poorly extracted PDFs."""
    if not docs:
        return True

    short_pages = 0
    for doc in docs:
        text_len = len(doc.page_content.strip())
        if text_len < 40:
            short_pages += 1

    return (short_pages / max(len(docs), 1)) >= 0.6


def _discover_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    return sorted(p for p in input_dir.rglob("*.pdf") if p.is_file())


def run_ingestion(
    input_dir: Path,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Orchestrate load -> OCR fallback -> split pipeline.

    Loads all PDFs under input_dir recursively and returns chunked documents.
    """
    pdf_paths = _discover_pdfs(input_dir)
    if not pdf_paths:
        return []

    all_documents: list[Document] = []

    for pdf_path in pdf_paths:
        docs = load_pdf_documents([pdf_path])
        if _needs_ocr(docs):
            ocr_docs = extract_text_with_ocr(pdf_path)
            if ocr_docs:
                docs = ocr_docs

        all_documents.extend(docs)

    return split_documents(
        all_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
