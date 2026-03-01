"""
PDF Ingestion and Document Chunking Module
Handles PDF parsing, text extraction, chunking, and metadata preservation
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Ingest and process PDF documents for RAG system"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize document ingester

        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter optimized for financial documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        logger.info(f"Initialized ingester with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def ingest_from_directory(self, directory: str) -> List[Document]:
        """
        Ingest all PDF files from a directory

        Args:
            directory: Path to directory containing PDFs

        Returns:
            List of Document objects with metadata
        """
        dir_path = Path(directory)
        pdf_files = sorted(dir_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to ingest")

        all_documents = []
        for pdf_file in pdf_files:
            try:
                documents = self._ingest_pdf(str(pdf_file))
                all_documents.extend(documents)
                logger.info(f"Ingested {len(documents)} chunks from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error ingesting {pdf_file.name}: {e}")
                continue

        logger.info(f"Total documents ingested: {len(all_documents)}")
        return all_documents

    def _ingest_pdf(self, file_path: str) -> List[Document]:
        """
        Ingest a single PDF file

        Args:
            file_path: Path to PDF file

        Returns:
            List of chunked documents with metadata
        """
        logger.info(f"Ingesting PDF: {file_path}")

        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        if not pages:
            logger.warning(f"No pages loaded from {file_path}")
            return []

        # Extract document metadata from filename
        file_name = Path(file_path).stem
        doc_metadata = self._extract_metadata_from_filename(file_name)

        # Clean and split text
        chunked_documents = []
        for page_num, page in enumerate(pages, 1):
            # Clean page content
            cleaned_text = self._clean_text(page.page_content)

            if not cleaned_text:
                continue

            # Split into chunks
            chunks = self.splitter.split_text(cleaned_text)

            # Create documents with metadata
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    **doc_metadata,
                    "page": page_num,
                    "chunk_idx": chunk_idx,
                    "source": file_path,
                }

                # Extract item number from chunk if available
                item_match = re.search(r"Item\s+(\d+[A-Z]*)", chunk[:200])
                if item_match:
                    metadata["item"] = f"Item {item_match.group(1)}"

                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                chunked_documents.append(doc)

        logger.info(f"Created {len(chunked_documents)} chunks from {Path(file_path).name}")
        return chunked_documents

    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF filename

        Expected format: DOCUMENT_YEAR_TYPE.pdf
        Examples:
        - Apple_2024_10K.pdf -> Apple, 2024, 10K
        - Tesla_2023_10K.pdf -> Tesla, 2023, 10K

        Args:
            filename: PDF filename without extension

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "document": "Unknown",
            "year": "Unknown",
            "filing_type": "10-K",
        }

        # Try various filename patterns
        # Pattern 1: Company_Year_Type (e.g., Apple_2024_10K)
        match = re.match(r"([A-Za-z]+)[_\-](\d{4})[_\-](\d+[A-Z]*)", filename, re.IGNORECASE)
        if match:
            company, year, filing_type = match.groups()
            metadata["document"] = f"{company} {filing_type}"
            metadata["year"] = year
            metadata["filing_type"] = filing_type
            return metadata

        # Pattern 2: Company 10K Year (e.g., "10-Q4-2024-As-Filed")
        if "10-" in filename.upper():
            if "apple" in filename.lower() or "aapl" in filename.lower():
                metadata["document"] = "Apple 10-K"
                if "2024" in filename or "2024" in filename:
                    metadata["year"] = "2024"
            elif "tesla" in filename.lower() or "tsla" in filename.lower():
                metadata["document"] = "Tesla 10-K"
                if "2023" in filename:
                    metadata["year"] = "2023"

        # Extract year from filename
        year_match = re.search(r"20\d{2}", filename)
        if year_match:
            metadata["year"] = year_match.group()

        return metadata

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove page numbers, headers/footers
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Skip lines that are just page numbers
            if re.match(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", line):
                continue

            # Skip lines that are mainly whitespace
            if len(line.strip()) < 5:
                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def estimate_tokens(self, text: str, model: str = "gpt2") -> int:
        """
        Estimate token count (rough approximation)

        Args:
            text: Text to estimate
            model: Model for estimation (rough approximation)

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test ingestion
    ingester = DocumentIngester()
    documents = ingester.ingest_from_directory("data")

    print(f"\nIngested {len(documents)} documents")
    if documents:
        print(f"\nFirst document sample:")
        print(f"  Content: {documents[0].page_content[:200]}...")
        print(f"  Metadata: {documents[0].metadata}")

