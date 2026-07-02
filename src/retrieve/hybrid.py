"""Hybrid retrieval with LangChain BM25 + FAISS ensemble fusion and MMR reranking."""
import numpy as np
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class HybridRetriever:
    """Combine FAISS dense retrieval + BM25 sparse retrieval via ensemble fusion."""

    def __init__(
        self,
        vectorstore: FAISS,
        documents: list[Document],
        embeddings: Embeddings,
        dense_k: int = 8,
        sparse_k: int = 8,
        final_k: int = 6,
        mmr_lambda: float = 0.5,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        if not documents:
            raise ValueError("HybridRetriever requires a non-empty document list")

        self.embeddings = embeddings
        self.final_k = final_k
        self.mmr_lambda = mmr_lambda
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": dense_k})
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = sparse_k

        self._ensemble = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=[weights[0], weights[1]],
        )

    def invoke(self, query: str) -> list[Document]:
        """Return final fused documents for a query after MMR reranking."""
        docs = self._ensemble.invoke(query)

        if len(docs) <= self.final_k:
            return docs

        query_embedding = np.array(
            self.embeddings.embed_query(query),
            dtype=np.float32,
        )

        doc_embeddings = np.array(
            self.embeddings.embed_documents(
                [doc.page_content for doc in docs]
            ),
            dtype=np.float32,
        )

        selected_indices = maximal_marginal_relevance(
            query_embedding,
            doc_embeddings,
            k=self.final_k,
            lambda_mult=self.mmr_lambda,
        )

        return [docs[idx] for idx in selected_indices]
