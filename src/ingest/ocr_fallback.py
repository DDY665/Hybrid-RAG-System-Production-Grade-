"""OCR fallback helpers for scanned PDFs."""

from pathlib import Path

import pytesseract
from langchain_core.documents import Document
from pdf2image import convert_from_path


def extract_text_with_ocr(pdf_path: Path, dpi: int = 300) -> list[Document]:
    """Run OCR extraction for scanned PDF pages.

    Returns one Document per page with source/page metadata.
    """
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return []

    images = convert_from_path(str(pdf_path), dpi=dpi)
    ocr_docs: list[Document] = []

    for idx, image in enumerate(images):
        text = pytesseract.image_to_string(image).strip()
        if not text:
            continue

        ocr_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": idx,
                    "loader": "ocr",
                },
            )
        )

    return ocr_docs
