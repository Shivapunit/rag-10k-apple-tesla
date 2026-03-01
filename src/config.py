"""
Configuration Module
Centralized configuration for the RAG system.
"""

from pathlib import Path
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration for RAG system."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DOCUMENTS_DIR = PROJECT_ROOT / "documents"
    CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

    # Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    VECTORSTORE_COLLECTION = "10k_documents"
    VECTORSTORE_PERSIST_DIR = str(CHROMA_DB_DIR)

    # Retrieval
    RETRIEVER_TOP_K = 4

    # LLM
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return True


if __name__ == "__main__":
    config = Config()
    print(f"Documents directory: {config.DOCUMENTS_DIR}")
    print(f"Vector store directory: {config.CHROMA_DB_DIR}")
    print(f"LLM Model: {config.LLM_MODEL}")

