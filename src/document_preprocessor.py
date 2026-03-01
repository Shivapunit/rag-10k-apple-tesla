"""

    print(f"Preprocessor initialized with chunk_size={preprocessor.chunk_size}")
    preprocessor = DocumentPreprocessor()
    logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":


        return chunked_docs
        logger.info(f"Preprocessing complete: {len(chunked_docs)} chunks ready for embedding")

        chunked_docs = self.chunk_documents(documents)
        # Chunk documents

            doc.page_content = self.clean_text(doc.page_content)
        for doc in documents:
        # Clean documents
        """
            Preprocessed and chunked documents
        Returns:

            documents: Raw documents
        Args:

        Apply full preprocessing pipeline.
        """
    def preprocess(self, documents: List[Document]) -> List[Document]:
    
        return text.strip()
        text = text.replace("\x00", "")
        # Remove common artifacts
        text = " ".join(text.split())
        # Remove excessive whitespace
        """
            Cleaned text
        Returns:

            text: Raw text
        Args:

        Clean and normalize text.
        """
    def clean_text(self, text: str) -> str:
    
        return chunked_docs
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        
                chunked_docs.append(new_doc)
                )
                    }
                        "total_chunks": len(chunks),
                        "chunk_index": i,
                        **doc.metadata,
                    metadata={
                    page_content=chunk,
                new_doc = Document(
            for i, chunk in enumerate(chunks):
            # Create new documents for each chunk
            
            chunks = self.splitter.split_text(doc.page_content)
            # Split document into chunks
        for doc in documents:
        
        chunked_docs = []
        """
            List of chunked Document objects
        Returns:

            documents: List of Document objects
        Args:

        Split documents into chunks while preserving metadata.
        """
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
    
        )
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
        self.splitter = RecursiveCharacterTextSplitter(
        # Create text splitter optimized for financial documents
        
        self.chunk_overlap = chunk_overlap
        self.chunk_size = chunk_size
        """
            chunk_overlap: Number of overlapping characters between chunks
            chunk_size: Number of characters per chunk
        Args:

        Initialize the preprocessor.
        """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
    
    """Preprocess and chunk documents for RAG."""
class DocumentPreprocessor:


logger = logging.getLogger(__name__)

import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

"""
Handles chunking and preprocessing of documents while preserving context.
Document Preprocessor Module
