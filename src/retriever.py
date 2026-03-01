"""
Retriever Module
Handles document retrieval from vector store.
"""

from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import logging

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieve relevant documents from vector store."""

    def __init__(self, vectorstore: Chroma, top_k: int = 4):
        """
        Initialize the retriever.

        Args:
            vectorstore: Chroma vector store instance
            top_k: Number of top documents to retrieve
        """
        self.vectorstore = vectorstore
        self.top_k = top_k

        # Create retriever with MMR (Maximal Marginal Relevance)
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": top_k,
                "fetch_k": top_k * 2,  # Fetch more candidates for MMR
            }
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query

        Returns:
            List of relevant documents
        """
        try:
            logger.info(f"Retrieving documents for query: {query[:50]}...")
            results = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.

        Args:
            query: Search query

        Returns:
            List of (Document, score) tuples
        """
        try:
            logger.info(f"Retrieving documents with scores for: {query[:50]}...")
            results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            raise

    def retrieve_with_filter(self,
                            query: str,
                            filter_dict: Optional[dict] = None) -> List[Document]:
        """
        Retrieve documents with metadata filtering.

        Args:
            query: Search query
            filter_dict: Metadata filter (e.g., {"ticker": "AAPL"})

        Returns:
            List of relevant documents
        """
        try:
            if filter_dict:
                logger.info(f"Retrieving documents with filter: {filter_dict}")

            # Chroma uses where clause for filtering
            results = self.vectorstore.similarity_search(
                query,
                k=self.top_k,
                where=filter_dict
            )
            logger.info(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving filtered documents: {str(e)}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Retriever module initialized")

