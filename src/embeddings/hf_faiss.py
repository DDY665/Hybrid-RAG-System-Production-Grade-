"""HuggingFace embedding and FAISS index utilities."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_hf_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Create the HuggingFace embeddings model used by the RAG pipeline."""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_faiss_index(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """Build an in-memory FAISS index from chunked documents."""
    if not documents:
        raise ValueError("Cannot build FAISS index from an empty document list")

    embedding_model = embeddings or build_hf_embeddings()
    return FAISS.from_documents(documents, embedding_model)


def save_faiss_index(vectorstore: FAISS, index_dir: Path) -> None:
    """Persist a FAISS index locally."""
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))


def load_faiss_index(
    index_dir: Path,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS | None:
    """Load a persisted FAISS index if it exists."""
    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        return None

    embedding_model = embeddings or build_hf_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def load_or_build_faiss_index(
    documents: list[Document],
    index_dir: Path,
    embeddings: HuggingFaceEmbeddings | None = None,
    force_rebuild: bool = False,
) -> tuple[FAISS, bool]:
    """Return a local FAISS index, loading cache when available.

    Returns:
        (vectorstore, loaded_from_disk)
    """
    embedding_model = embeddings or build_hf_embeddings()

    if not force_rebuild:
        cached = load_faiss_index(index_dir=index_dir, embeddings=embedding_model)
        if cached is not None:
            return cached, True

    vectorstore = build_faiss_index(documents=documents, embeddings=embedding_model)
    save_faiss_index(vectorstore=vectorstore, index_dir=index_dir)
    return vectorstore, False
