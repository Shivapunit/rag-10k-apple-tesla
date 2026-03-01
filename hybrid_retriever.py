"""
Hybrid Retrieval Module
Combines BM25 (keyword matching) + Vector Search (semantic similarity)
for improved retrieval accuracy on financial documents
"""

import logging
from typing import List, Tuple, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and vector search.

    Strategy:
    1. BM25 retrieves top-k keyword matches (exact/phrase matches)
    2. Vector search retrieves top-k semantic matches
    3. Combine and re-rank using weighted scoring
    4. Return merged top-k results
    """

    def __init__(
        self,
        vectorstore: FAISS,
        documents: List[Document],
        top_k: int = 5,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        """
        Initialize hybrid retriever

        Args:
            vectorstore (FAISS): Vector database for semantic search
            documents (List[Document]): Original documents for BM25
            top_k (int): Number of final results to return
            bm25_weight (float): Weight for BM25 scores (0-1)
            vector_weight (float): Weight for vector scores (0-1)
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Initialize BM25
        self._initialize_bm25()

        logger.info(
            f"Hybrid retriever initialized: "
            f"BM25 weight={bm25_weight}, Vector weight={vector_weight}"
        )

    def _initialize_bm25(self):
        """Initialize BM25 index from documents"""
        try:
            # Tokenize documents for BM25
            tokenized_docs = [
                self._tokenize(doc.page_content)
                for doc in self.documents
            ]

            # Build BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index built for {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on whitespace
        tokens = text.lower().split()
        # Remove punctuation and short tokens
        tokens = [t.strip('.,!?;:') for t in tokens if len(t) > 2]
        return tokens

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores or max(scores) == 0:
            return [0.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score == 0:
            return [1.0] * len(scores)

        return [
            (s - min_score) / (max_score - min_score)
            for s in scores
        ]

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid approach

        Args:
            query (str): Search query

        Returns:
            List[Document]: Top-k merged and re-ranked documents
        """
        try:
            logger.info(f"Hybrid retrieval for query: {query[:50]}...")

            # Step 1: BM25 retrieval (keyword matching)
            bm25_results = self._retrieve_bm25(query)
            logger.info(f"BM25 retrieved {len(bm25_results)} results")

            # Step 2: Vector search (semantic matching)
            vector_results = self._retrieve_vector(query)
            logger.info(f"Vector search retrieved {len(vector_results)} results")

            # Step 3: Merge and re-rank
            merged_results = self._merge_and_rerank(bm25_results, vector_results)
            logger.info(f"Merged and re-ranked to {len(merged_results)} results")

            return merged_results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback to vector search only
            return self._retrieve_vector(query)

    def _retrieve_bm25(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve using BM25 (keyword matching)

        Returns:
            List of (Document, score) tuples
        """
        if not self.bm25:
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:self.top_k * 2]

            # Return documents with scores
            results = [
                (self.documents[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0
            ]

            return results

        except Exception as e:
            logger.warning(f"BM25 retrieval error: {e}")
            return []

    def _retrieve_vector(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve using vector similarity search

        Returns:
            List of (Document, score) tuples
        """
        try:
            # Similarity search with scores
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=self.top_k * 2
            )

            # Convert to (Document, score) tuples
            # Note: FAISS returns distance, convert to similarity (1 / (1 + distance))
            return [
                (doc, 1.0 / (1.0 + score))
                for doc, score in results
            ]

        except Exception as e:
            logger.error(f"Vector retrieval error: {e}")
            return []

    def _merge_and_rerank(
        self,
        bm25_results: List[Tuple[Document, float]],
        vector_results: List[Tuple[Document, float]]
    ) -> List[Document]:
        """
        Merge BM25 and vector results, re-rank by combined score

        Args:
            bm25_results: Results from BM25 retrieval
            vector_results: Results from vector search

        Returns:
            Top-k merged and re-ranked documents
        """
        # Create document to score mapping
        doc_scores: Dict[str, Tuple[Document, float, float]] = {}

        # Add BM25 scores
        bm25_scores = [score for _, score in bm25_results]
        normalized_bm25 = self._normalize_scores(bm25_scores)

        for (doc, score), norm_score in zip(bm25_results, normalized_bm25):
            doc_id = doc.metadata.get('source', id(doc))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = (doc, norm_score * self.bm25_weight, 0.0)
            else:
                _, bm25_w, vector_w = doc_scores[doc_id]
                doc_scores[doc_id] = (doc, norm_score * self.bm25_weight, vector_w)

        # Add vector scores
        vector_scores = [score for _, score in vector_results]
        normalized_vector = self._normalize_scores(vector_scores)

        for (doc, score), norm_score in zip(vector_results, normalized_vector):
            doc_id = doc.metadata.get('source', id(doc))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = (doc, 0.0, norm_score * self.vector_weight)
            else:
                _, bm25_w, _ = doc_scores[doc_id]
                doc_scores[doc_id] = (doc, bm25_w, norm_score * self.vector_weight)

        # Calculate combined scores and sort
        scored_docs = [
            (doc, bm25_w + vector_w, bm25_w, vector_w)
            for doc, bm25_w, vector_w in doc_scores.values()
        ]

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Log score breakdown for top results
        logger.info(f"Top re-ranked results:")
        for i, (doc, combined, bm25, vector) in enumerate(scored_docs[:self.top_k]):
            logger.info(
                f"  {i+1}. Combined={combined:.3f} (BM25={bm25:.3f}, Vector={vector:.3f}) "
                f"- {doc.metadata.get('document', 'Unknown')[:30]}..."
            )

        # Return top-k documents
        return [doc for doc, _, _, _ in scored_docs[:self.top_k]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Hybrid Retriever module loaded")
    print("Use with RAGPipeline for hybrid search capabilities")

