"""
Vector Store Module
Handles embedding generation and vector database operations.
"""

from typing import List, Optional
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage embeddings and vector database operations."""

    def __init__(self,
                 persist_dir: str = "./chroma_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "10k_documents"):
        """
        Initialize vector store manager.

        Args:
            persist_dir: Directory to persist vector database
            embedding_model: HuggingFace embedding model name
            collection_name: Name of the vector collection
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize vector store
        self.vectorstore = None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vector store.

        Args:
            documents: List of Document objects to add
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return

        try:
            logger.info(f"Adding {len(documents)} documents to vector store...")

            if self.vectorstore is None:
                # Create new vector store
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                    collection_name=self.collection_name,
                )
                logger.info(f"Created new vector store with {len(documents)} documents")
            else:
                # Add to existing vector store
                self.vectorstore.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to existing vector store")

            # Persist to disk
            self.vectorstore.persist()
            logger.info("Vector store persisted to disk")

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk.

        Returns:
            Chroma vector store object or None
        """
        try:
            logger.info(f"Loading vector store from {self.persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            logger.info("Vector store loaded successfully")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"Could not load vector store: {str(e)}")
            return None

    def get_vectorstore(self) -> Optional[Chroma]:
        """
        Get the current vector store, loading if necessary.

        Returns:
            Chroma vector store object or None
        """
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = VectorStoreManager()
    print("Vector store manager initialized")

