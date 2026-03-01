"""
RAG Pipeline - Core answer_question() implementation
Retrieval-Augmented Generation for 10-K financial documents

Supports:
- Pure vector search (FAISS)
- Hybrid retrieval (BM25 + FAISS)
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os
import requests

# Vector store and embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# LLM imports
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Document processing
from ingest import DocumentIngester

# Hybrid retrieval
try:
    from hybrid_retriever import HybridRetriever
    HYBRID_RETRIEVAL_AVAILABLE = True
except ImportError:
    HYBRID_RETRIEVAL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("rank-bm25 not available. Hybrid retrieval disabled.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OllamaAPILLM:
    """Ollama API-based LLM client with authentication"""

    def __init__(self, api_key: Optional[str] = None, api_url: str = "https://api.ollama.com/v1/chat/completions"):
        """
        Initialize Ollama API client

        Args:
            api_key: API key (if None, tries to get from OLLAMA_API_KEY env var)
            api_url: Ollama API endpoint
        """
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.api_url = api_url
        self.temperature = 0.3

        if not self.api_key:
            logger.warning("OLLAMA_API_KEY not set. API calls will fail.")

    def invoke(self, prompt: str) -> str:
        """
        Call Ollama API with authentication

        Args:
            prompt: The prompt to send to LLM

        Returns:
            str: Generated response from LLM
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama3:8b-instruct",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return "Error: Request timed out"
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama API")
            return "Error: Could not connect to API"
        except requests.exceptions.HTTPError as e:
            logger.error(f"API error: {e}")
            return f"Error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"



class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for 10-K documents"""

    # Custom prompt for financial QA
    FINANCIAL_QA_PROMPT = """You are an expert financial analyst answering questions about 10-K SEC filings.

IMPORTANT RULES:
1. Answer ONLY based on the provided context from the documents
2. If the answer is not in the context, respond with: "Not specified in the document."
3. For questions outside the scope of these documents, respond with: "This question cannot be answered based on the provided documents."
4. Always cite your sources in the format: [Document Name, Item X, p. YY]
5. Be precise with numbers, percentages, and dates
6. Use professional financial analysis tone

Context from 10-K documents:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        data_dir: str = "data",
        vector_store_dir: str = "vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistral",
        top_k: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_hybrid_retrieval: bool = True,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        use_api: bool = False,
        ollama_api_key: Optional[str] = None,
        ollama_api_url: str = "https://api.ollama.com/v1/chat/completions",
    ):
        """
        Initialize RAG pipeline

        Args:
            data_dir: Directory containing PDF files
            vector_store_dir: Directory for FAISS vector store
            embedding_model: HuggingFace embedding model name
            llm_model: Local LLM model (Ollama)
            top_k: Number of chunks to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_hybrid_retrieval: Use BM25 + Vector hybrid search
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            use_api: Use Ollama API instead of local
            ollama_api_key: API key for Ollama API (if None, uses OLLAMA_API_KEY env var)
            ollama_api_url: Ollama API endpoint
        """
        self.data_dir = Path(data_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_retrieval = use_hybrid_retrieval and HYBRID_RETRIEVAL_AVAILABLE
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.use_api = use_api
        self.ollama_api_key = ollama_api_key
        self.ollama_api_url = ollama_api_url

        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.hybrid_retriever = None
        self.qa_chain = None
        self.ingested_documents = []

        self._initialize()

    def _initialize(self):
        """Initialize embedding and LLM models"""
        logger.info("Initializing RAG components...")

        # Initialize embeddings
        logger.info(f"Loading embeddings model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # Initialize LLM (Ollama - local or API)
        logger.info(f"Initializing LLM: {self.llm_model_name}")

        if self.use_api:
            # Use Ollama API with authentication
            logger.info("Using Ollama API with authentication")
            try:
                self.llm = OllamaAPILLM(
                    api_key=self.ollama_api_key,
                    api_url=self.ollama_api_url
                )
                # Test API connection
                test_response = self.llm.invoke("Hi")
                if "Error" in test_response:
                    raise Exception(f"API Error: {test_response}")
                logger.info("Ollama API connected successfully")
            except Exception as e:
                logger.error(f"Ollama API initialization error: {e}")
                raise
        else:
            # Use local Ollama (must be running locally)
            logger.info("Using local Ollama instance")
            try:
                self.llm = Ollama(model=self.llm_model_name, temperature=0.3)
                # Test LLM connection
                _ = self.llm.invoke("Hi")
                logger.info(f"Local Ollama {self.llm_model_name} ready")
            except Exception as e:
                logger.error(f"Local Ollama initialization error: {e}")
                logger.info("Make sure Ollama is running locally with: ollama run mistral")
                raise

    def is_indexed(self) -> bool:
        """Check if vector store already exists"""
        return (self.vector_store_dir / "index.faiss").exists()

    def build_index(self) -> bool:
        """
        Build vector index from PDF documents

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Building index from documents in {self.data_dir}")

            if not self.data_dir.exists():
                logger.error(f"Data directory not found: {self.data_dir}")
                return False

            # Ingest documents
            ingester = DocumentIngester(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            documents = ingester.ingest_from_directory(str(self.data_dir))

            if not documents:
                logger.error("No documents were ingested")
                return False

            logger.info(f"Ingested {len(documents)} chunks from PDFs")

            # Store documents for hybrid retrieval
            self.ingested_documents = documents

            # Create vector store
            logger.info("Creating FAISS vector store...")
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)

            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # Save vector store
            self.vectorstore.save_local(str(self.vector_store_dir))
            logger.info(f"Vector store saved to {self.vector_store_dir}")

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )

            # Initialize hybrid retriever if available
            if self.use_hybrid_retrieval:
                try:
                    logger.info("Initializing hybrid retriever (BM25 + Vector)...")
                    self.hybrid_retriever = HybridRetriever(
                        vectorstore=self.vectorstore,
                        documents=documents,
                        top_k=self.top_k,
                        bm25_weight=self.bm25_weight,
                        vector_weight=self.vector_weight
                    )
                    logger.info("Hybrid retriever initialized successfully")
                except Exception as e:
                    logger.warning(f"Could not initialize hybrid retriever: {e}")
                    logger.info("Falling back to vector search only")
                    self.use_hybrid_retrieval = False

            logger.info("Index built successfully")
            return True

        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False

    def load_index(self) -> bool:
        """
        Load existing vector store from disk

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_indexed():
                logger.warning("Vector store not found. Call build_index() first.")
                return False

            logger.info(f"Loading vector store from {self.vector_store_dir}")
            self.vectorstore = FAISS.load_local(
                str(self.vector_store_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )

            logger.info("Vector store loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

    def get_chunk_count(self) -> int:
        """Get total number of chunks in vector store"""
        if self.vectorstore:
            return self.vectorstore.index.ntotal
        return 0

    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.3,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Core function as specified in assignment.

        Args:
            query (str): The user question about Apple or Tesla 10-K filings
            top_k (Optional[int]): Number of chunks to retrieve (uses default if None)
            temperature (float): LLM temperature (0=deterministic, 1=random)
            return_sources (bool): Whether to return source documents

        Returns:
            dict: {
                "answer": "Answer text or 'This question cannot be answered based on the provided documents.'",
                "sources": [
                    {
                        "document": "Apple 10-K",
                        "item": "Item 8",
                        "page": "282",
                        "content": "retrieved chunk text"
                    }
                ]
            }
        """
        try:
            # Load index if not already loaded
            if not self.vectorstore:
                if not self.load_index():
                    if not self.build_index():
                        return {
                            "answer": "Error: Could not initialize vector store",
                            "sources": []
                        }

            # Update temperature in LLM
            if self.llm:
                self.llm.temperature = temperature

            # Retrieve relevant chunks
            logger.info(f"Retrieving documents for query: {query[:100]}")
            k = top_k if top_k else self.top_k

            retrieved_docs = self.retriever.get_relevant_documents(query)[:k]

            if not retrieved_docs:
                return {
                    "answer": "No relevant documents found in the database.",
                    "sources": []
                }

            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Create prompt
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=self.FINANCIAL_QA_PROMPT
            )

            # Generate answer
            logger.info("Generating answer with LLM...")
            formatted_prompt = prompt.format(context=context, question=query)
            answer = self.llm.invoke(formatted_prompt)

            # Extract sources
            sources = []
            if return_sources:
                for doc in retrieved_docs:
                    sources.append({
                        "document": doc.metadata.get("document", "Unknown"),
                        "item": doc.metadata.get("item", "Unknown"),
                        "page": str(doc.metadata.get("page", "N/A")),
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })

            result = {
                "answer": answer.strip(),
                "sources": sources
            }

            logger.info("Answer generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }

    def answer_multiple_questions(
        self,
        questions: List[str],
        save_to_json: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions and optionally save results to JSON

        Args:
            questions: List of questions to answer
            save_to_json: Optional file path to save results

        Returns:
            List of answer dictionaries
        """
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.answer_question(question)
            results.append({
                "question_id": i,
                "question": question,
                **result
            })

        if save_to_json:
            output_path = Path(save_to_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {save_to_json}")

        return results


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()

    if not pipeline.is_indexed():
        print("Building index...")
        pipeline.build_index()

    # Test questions
    test_query = "What was Apple's total revenue for FY 2024?"
    result = pipeline.answer_question(test_query)

    print("\nQuery:", test_query)
    print("\nAnswer:", result["answer"])
    print("\nSources:", result["sources"])
