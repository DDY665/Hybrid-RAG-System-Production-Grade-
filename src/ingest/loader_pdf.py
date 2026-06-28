"""PDF loading utilities for ingestion."""

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_documents(pdf_paths: Iterable[Path]) -> list[Document]:
    """Load PDFs into page-level LangChain Document objects.

    Metadata is normalized to always include a string source path and page number.
    """
    loaded_documents: list[Document] = []

    for pdf_path in pdf_paths:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            continue

        loader = PyPDFLoader(str(pdf_path))
        page_docs = loader.load()

        for doc in page_docs:
            metadata = dict(doc.metadata)
            metadata["source"] = str(pdf_path)
            metadata["page"] = int(metadata.get("page", 0))
            metadata["loader"] = "pypdf"
            doc.metadata = metadata

        loaded_documents.extend(page_docs)

    return loaded_documents
