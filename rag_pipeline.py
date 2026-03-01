"""
RAG Pipeline - Core answer_question() implementation
Retrieval-Augmented Generation for 10-K financial documents

Supports:
- Pure vector search (FAISS)
- Hybrid retrieval (BM25 + FAISS)
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import os
import requests
import pickle

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to import Embeddings base class
try:
    from langchain.schema.embeddings import Embeddings
except ImportError:
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError:
        class Embeddings:
            pass

# Vector store and embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except Exception:
    HuggingFaceEmbeddings = None
    EMBEDDINGS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("HuggingFaceEmbeddings not available. Will use TF-IDF fallback for embeddings.")

try:
    from langchain_community.vectorstores import FAISS
    VECTORSTORE_AVAILABLE = True
except Exception:
    FAISS = None
    VECTORSTORE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("FAISS (langchain_community.vectorstores) not available. Using simple in-memory fallback vector store.")

from langchain.schema import Document

# LLM imports - guard optional dependencies
try:
    from langchain_community.llms import Ollama
except Exception:
    Ollama = None
    logger = logging.getLogger(__name__)
    logger.warning("langchain_community.llms.Ollama not available. Local Ollama mode will be disabled.")

# Document processing
from ingest import DocumentIngester
from pageindex_retriever import PageIndexRetriever

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


class MockLLM:
    """
    A fallback mock LLM that provides a placeholder response
    when the real LLM is unavailable.
    """
    def __init__(self):
        self.temperature = 0.3

    def invoke(self, prompt: str) -> str:
        return (
            "**[System Notification]**\n\n"
            "The local AI model (Ollama) is currently unreachable. "
            "However, the system has successfully retrieved the relevant documents for your query. "
            "Please review the **Sources** section below to find the specific answer."
        )


class HuggingFaceAPILLM:
    """HuggingFace Inference API client"""
    def __init__(self, api_token: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.api_token = api_token
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.temperature = 0.3
        
    def invoke(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": self.temperature,
                "return_full_text": False
            }
        }
        try:
            logger.info(f"Calling HuggingFace API: {self.api_url}")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            # HF returns list of dicts: [{'generated_text': '...'}]
            if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                return result[0]['generated_text']
            elif isinstance(result, dict) and 'error' in result:
                return f"HuggingFace API Error: {result['error']}"
            return str(result)
        except Exception as e:
            logger.error(f"HF API error: {e}")
            return f"Error calling HuggingFace API: {str(e)}"


class OllamaAPILLM:
    """Ollama API-based LLM client with authentication"""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize Ollama API client

        Args:
            api_key: API key (if None, tries to get from OLLAMA_API_KEY env var)
            api_url: Ollama API endpoint (if None, uses environment variable or default)
        """
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.api_url = api_url or os.getenv("OLLAMA_API_URL") or "http://localhost:11434/api/chat"
        self.temperature = 0.3

        if not self.api_key and not self.api_url.startswith("http://localhost"):
            logger.warning("OLLAMA_API_KEY not set. Remote API calls will fail. Set OLLAMA_API_KEY environment variable or pass api_key parameter.")

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
                "Content-Type": "application/json"
            }

            # Add authorization only if API key is set and not localhost
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            logger.info(f"Calling External LLM API at: {self.api_url}")

            # Check for OpenAI-compatible endpoint (v1/chat/completions)
            if "/v1/chat/completions" in self.api_url:
                payload = {
                    "model": "llama3:8b-instruct", # Default model, can be adjusted
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"Error: Unexpected response format from API: {result}"
            
            else:
                # Standard Ollama API format
                payload = {
                    "model": "llama3:8b-instruct",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature
                }

                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()

                # Handle different response formats
                result = response.json()
                if "response" in result:
                    return result["response"]
                elif "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "")
                else:
                    logger.warning(f"Unexpected response format: {result}")
                    return str(result)

        except requests.exceptions.Timeout:
            logger.error("API request timed out (60s)")
            return "Error: Request timed out. Server may be slow or overloaded."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to API at {self.api_url}: {e}")
            return f"Error: Could not connect to {self.api_url}. Check your URL and network."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(f"Unauthorized (401): API key may be invalid or missing. URL: {self.api_url}")
                return "Error: 401 Unauthorized. Check your OLLAMA_API_KEY."
            elif e.response.status_code == 404:
                logger.error(f"Not Found (404): Endpoint {self.api_url} does not exist.")
                return f"Error: 404 Not Found. Check your API URL: {self.api_url}"
            else:
                logger.error(f"API error {e.response.status_code}: {e}")
                return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"


class FallbackEmbeddings(Embeddings):
    """Simple TF-IDF based embeddings wrapper for environments without sentence-transformers."""
    def __init__(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as e:
            logger.error("scikit-learn not available. Install 'scikit-learn' or 'sentence-transformers'.")
            raise
        self.vectorizer = TfidfVectorizer(max_features=2048)
        self._fit = False
        self._corpus = []
        self._embeddings = None

    def fit(self, documents: List[str]):
        # Fit the TF-IDF vectorizer on the provided documents
        self._corpus = documents
        self._embeddings = self.vectorizer.fit_transform(documents)
        self._fit = True

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if not self._fit:
            # Fit on input documents if not pre-fitted
            self.fit(documents)
        emb = self.vectorizer.transform(documents)
        return emb.toarray().tolist()

    def embed_query(self, query: str) -> List[float]:
        if not self._fit:
            # Fit on small corpus to avoid failure
            self.fit([query])
        emb = self.vectorizer.transform([query])
        return emb.toarray()[0].tolist()
    
    def __call__(self, text: Any) -> List[float]:
        """Make the object callable to satisfy some FAISS implementations"""
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)
        
    def embed(self, text: str) -> List[float]:
        """Alias for embed_query"""
        return self.embed_query(text)

    def save(self, path: Path):
        try:
            with open(path / "tfidf_model.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"Saved TF-IDF model to {path}")
        except Exception as e:
            logger.error(f"Failed to save TF-IDF model: {e}")

    def load(self, path: Path):
        model_path = path / "tfidf_model.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self._fit = True
                logger.info(f"Loaded TF-IDF model from {path}")
            except Exception as e:
                logger.error(f"Failed to load TF-IDF model: {e}")
        else:
            logger.warning(f"TF-IDF model not found at {model_path}")


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
        ollama_api_url: str = None,
        hf_token: Optional[str] = None,
        page_level: bool = False,
        retrieval_mode: str = "hybrid",  # options: 'pageindex', 'vector', 'hybrid'
    ):
        """
        Initialize RAG pipeline
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
        self.ollama_api_url = ollama_api_url or os.getenv("OLLAMA_API_URL") or "http://localhost:11434/api/chat"
        self.hf_token = hf_token
        self.page_level = page_level
        self.retrieval_mode = retrieval_mode
        self.page_retriever = None

        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.hybrid_retriever = None
        self.qa_chain = None
        self.ingested_documents = []

        # Initialize safely
        try:
            self._initialize()
        except Exception as e:
            logger.error(f"RAGPipeline initialization failed: {e}")
            # Continue with uninitialized components (will fail gracefully later)

    def _initialize(self):
        """Initialize embedding and LLM models"""
        logger.info("Initializing RAG components...")

        # Initialize embeddings
        logger.info(f"Loading embeddings model: {self.embedding_model_name}")
        try:
            if EMBEDDINGS_AVAILABLE:
                try:
                    self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
                    logger.info("Loaded HuggingFaceEmbeddings successfully")
                except Exception as e:
                    logger.warning(f"HuggingFaceEmbeddings instantiation failed: {e}")
                    logger.info("Falling back to TF-IDF embeddings (scikit-learn)")
                    self.embeddings = FallbackEmbeddings()
            else:
                # Use TF-IDF fallback
                self.embeddings = FallbackEmbeddings()
                logger.info("Using TF-IDF fallback embeddings (scikit-learn)")
        except Exception as e:
            logger.error(f"Embeddings initialization error: {e}")
            # Do not raise here, allow pipeline to init without embeddings (will fail later if needed)
            self.embeddings = None

        # Initialize LLM
        logger.info(f"Initializing LLM: {self.llm_model_name}")
        self.llm = None
        
        # 1. Try HuggingFace API if token provided
        if self.hf_token:
            try:
                self.llm = HuggingFaceAPILLM(api_token=self.hf_token)
                logger.info("Initialized HuggingFace API LLM")
            except Exception as e:
                logger.error(f"HuggingFace API initialization error: {e}")
        
        # 2. Try Ollama API if configured
        elif self.use_api:
            try:
                self.llm = OllamaAPILLM(api_key=self.ollama_api_key, api_url=self.ollama_api_url)
                test_response = self.llm.invoke("Hi")
                if isinstance(test_response, str) and test_response.lower().startswith("error"):
                    logger.warning(f"Ollama API test response indicates error: {test_response}")
                    # Fallback to MockLLM if API fails
                    logger.info("Switching to MockLLM due to API error.")
                    self.llm = MockLLM()
                else:
                    logger.info("Ollama API connected successfully")
            except Exception as e:
                logger.error(f"Ollama API initialization error: {e}. Switching to MockLLM.")
                self.llm = MockLLM()
        
        # 3. Try Local Ollama
        else:
            if Ollama is not None:
                try:
                    self.llm = Ollama(model=self.llm_model_name, temperature=0.3)
                    # Test connection
                    self.llm.invoke("Hi")
                    logger.info(f"Local Ollama {self.llm_model_name} ready")
                except Exception as e:
                    logger.error(f"Local Ollama initialization error: {e}. Switching to MockLLM.")
                    self.llm = MockLLM()
            else:
                logger.info("Local Ollama client not available. Switching to MockLLM.")
                self.llm = MockLLM()

    def is_indexed(self) -> bool:
        """Check if vector store already exists"""
        faiss_path = (self.vector_store_dir / "index.faiss")
        fallback_path = (self.vector_store_dir / "simple_vector_store.pkl")
        return faiss_path.exists() or fallback_path.exists()

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

            if not self.embeddings:
                logger.error("Embeddings model not initialized. Cannot build index.")
                return False

            # Ingest documents
            ingester = DocumentIngester(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                page_level=self.page_level  # Pass page_level to ingester
            )
            documents = ingester.ingest_from_directory(str(self.data_dir))

            if not documents:
                logger.error("No documents were ingested")
                return False

            logger.info(f"Ingested {len(documents)} chunks from PDFs")

            # Store documents for hybrid retrieval
            self.ingested_documents = documents

            # Create vector store
            logger.info("Creating vector store...")
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)

            # Use VectorStoreClass which may be FAISS or our SimpleVectorStore
            self.vectorstore = VectorStoreClass.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # Save vector store
            self.vectorstore.save_local(str(self.vector_store_dir))
            
            # Save embeddings model if it's FallbackEmbeddings
            if isinstance(self.embeddings, FallbackEmbeddings):
                self.embeddings.save(self.vector_store_dir)
                
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

            # Initialize page_index retriever on page-level documents
            try:
                logger.info("Initializing page-level retriever (PageIndex/BM25)")
                self.page_retriever = PageIndexRetriever(self.ingested_documents)
            except Exception as e:
                logger.warning(f"Could not initialize page retriever: {e}")
                self.page_retriever = None

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

            if not self.embeddings:
                logger.error("Embeddings model not initialized. Cannot load index.")
                return False

            logger.info(f"Loading vector store from {self.vector_store_dir}")
            
            # Load embeddings model if it's FallbackEmbeddings
            if isinstance(self.embeddings, FallbackEmbeddings):
                self.embeddings.load(self.vector_store_dir)
            
            # Use VectorStoreClass.load_local if available (FAISS or SimpleVectorStore)
            try:
                self.vectorstore = VectorStoreClass.load_local(
                    str(self.vector_store_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                # Fallback: try SimpleVectorStore loader
                from pathlib import Path as _Path
                if (_Path(self.vector_store_dir) / 'simple_vector_store.pkl').exists():
                    self.vectorstore = VectorStoreClass.load_local(str(self.vector_store_dir), self.embeddings)
                else:
                    raise

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )

            # Initialize page retriever if in pageindex mode
            if self.retrieval_mode == 'pageindex':
                try:
                    logger.info("Initializing page-level retriever (PageIndex/BM25)")
                    self.page_retriever = PageIndexRetriever(self.ingested_documents)
                except Exception as e:
                    logger.warning(f"Could not initialize page retriever: {e}")
                    self.page_retriever = None

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

            # 1. Retrieval
            retrieved_docs_with_scores = []
            try:
                logger.info(f"Retrieving documents for query: {query[:100]}")
                k = top_k if top_k else self.top_k

                if self.retrieval_mode == 'pageindex':
                    if not self.page_retriever:
                        return {"answer": "Error: PageIndex retriever not available.", "sources": []}
                    # PageIndex returns (doc, score)
                    retrieved_docs_with_scores = self.page_retriever.retrieve(query, k=k)
                
                elif self.retrieval_mode == 'vector':
                    # Try to get scores if supported
                    if hasattr(self.vectorstore, "similarity_search_with_score"):
                        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
                    else:
                        # Fallback to just docs
                        if hasattr(self.retriever, 'invoke'):
                            docs = self.retriever.invoke(query)[:k]
                        else:
                            docs = self.retriever.get_relevant_documents(query)[:k]
                        retrieved_docs_with_scores = [(d, None) for d in docs]
                
                else:  # hybrid
                    # Combine pageindex and vector results; simple merge and dedupe
                    results = []
                    if self.page_retriever:
                        page_hits = self.page_retriever.retrieve(query, k=k)
                        results.extend(page_hits) # already (doc, score)
                    
                    # Use hybrid retriever if available, otherwise vector retriever
                    if self.hybrid_retriever:
                        # Hybrid retriever now returns (doc, score)
                        vector_hits = self.hybrid_retriever.retrieve(query)
                    else:
                        if hasattr(self.vectorstore, "similarity_search_with_score"):
                            vector_hits = self.vectorstore.similarity_search_with_score(query, k=k)
                        else:
                            if hasattr(self.retriever, 'invoke'):
                                docs = self.retriever.invoke(query)[:k]
                            else:
                                docs = self.retriever.get_relevant_documents(query)[:k]
                            vector_hits = [(d, None) for d in docs]
                    
                    results.extend(vector_hits)
                    
                    # Dedupe preserving order
                    seen = set()
                    deduped = []
                    for item in results:
                        try:
                            # Handle both (doc, score) and doc
                            if isinstance(item, tuple):
                                doc = item[0]
                                score = item[1]
                            else:
                                doc = item
                                score = None
                            
                            # Robustness check for metadata
                            if not hasattr(doc, 'metadata'):
                                logger.warning(f"Skipping item with no metadata: {type(doc)}")
                                continue

                            doc_id = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk_idx', 0))
                            if doc_id not in seen:
                                seen.add(doc_id)
                                deduped.append((doc, score))
                        except Exception as e:
                            logger.error(f"Error processing item in dedupe: {e}")
                            continue

                    retrieved_docs_with_scores = deduped[:k]

                if not retrieved_docs_with_scores:
                    return {
                        "answer": "No relevant documents found in the database.",
                        "sources": []
                    }
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                return {
                    "answer": f"Error retrieving documents: {str(e)}",
                    "sources": []
                }

            # 2. Extract Sources and Context
            sources = []
            context_docs = []
            
            if return_sources:
                for item in retrieved_docs_with_scores:
                    if isinstance(item, tuple):
                        doc = item[0]
                        score = item[1]
                    else:
                        doc = item
                        score = None
                    
                    context_docs.append(doc)
                    
                    source_path = doc.metadata.get("source", "")
                    source_file = Path(source_path).name if source_path else "Unknown"
                    
                    sources.append({
                        "document": doc.metadata.get("document", "Unknown"),
                        "source_file": source_file,
                        "item": doc.metadata.get("item", "Unknown"),
                        "page": str(doc.metadata.get("page", "N/A")),
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "score": float(score) if score is not None else None
                    })

            # 3. Generation
            answer = "LLM not available"
            if self.llm:
                try:
                    # Build context from retrieved documents
                    context = "\n\n".join([doc.page_content for doc in context_docs])
                    
                    # Format the prompt directly (no dependency on langchain PromptTemplate)
                    formatted_prompt = self.FINANCIAL_QA_PROMPT.format(context=context, question=query)
                    logger.info("Generating answer with LLM...")
                    answer = self.llm.invoke(formatted_prompt)
                except Exception as e:
                    logger.error(f"LLM generation error: {e}")
                    answer = f"Error generating answer: {str(e)}"
            else:
                answer = "LLM not available. Showing retrieved documents."

            return {
                "answer": answer.strip(),
                "sources": sources
            }

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


# Fallback simple vector store implementation (uses numpy + pickle)
if not VECTORSTORE_AVAILABLE:
    import numpy as _np
    import pickle as _pickle
    from types import SimpleNamespace as _SimpleNamespace

    class SimpleVectorStore:
        """Minimal in-memory vector store with cosine similarity retriever."""
        def __init__(self, docs: List[Any], embeddings_matrix: List[List[float]], embedding_obj: Any):
            self.docs = docs
            self.embeddings = _np.array(embeddings_matrix)
            self.embedding_obj = embedding_obj
            self.index = _SimpleNamespace(ntotal=len(docs))

        @classmethod
        def from_documents(cls, documents: List[Any], embedding: Any):
            # Extract texts
            texts = []
            for d in documents:
                if hasattr(d, 'page_content'):
                    texts.append(d.page_content)
                elif isinstance(d, dict) and 'content' in d:
                    texts.append(d['content'])
                else:
                    texts.append(str(d))
            embs = embedding.embed_documents(texts)
            return cls(documents, embs, embedding)

        def save_local(self, path: str):
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            with open(path_obj / 'simple_vector_store.pkl', 'wb') as f:
                _pickle.dump({'docs': self.docs, 'embeddings': self.embeddings}, f)

        @classmethod
        def load_local(cls, path: str, embedding: Any, allow_dangerous_deserialization: bool = True):
            path_obj = Path(path)
            with open(path_obj / 'simple_vector_store.pkl', 'rb') as f:
                data = _pickle.load(f)
            return cls(data['docs'], data['embeddings'], embedding)

        def as_retriever(self, search_type: str = 'similarity', search_kwargs: dict = None):
            if search_kwargs is None:
                search_kwargs = {'k': 5}
            k = int(search_kwargs.get('k', 5))
            return SimpleRetriever(self, k)
            
        def similarity_search_with_score(self, query: str, k: int = 5):
            """Return docs and similarity scores"""
            try:
                qvec = _np.array(self.embedding_obj.embed_query(query))
            except Exception:
                # fallback to zero vector
                if self.embeddings.size > 0:
                    qvec = _np.zeros(self.embeddings.shape[1])
                else:
                    return []

            if self.embeddings.size == 0:
                return []

            emb = self.embeddings
            qnorm = _np.linalg.norm(qvec) + 1e-12
            norms = _np.linalg.norm(emb, axis=1) + 1e-12
            dots = emb.dot(qvec)
            sims = dots / (norms * qnorm)
            
            # Top k
            idx = _np.argsort(-sims)[:k]
            
            results = []
            for i in idx:
                results.append((self.docs[i], float(sims[i])))
            
            return results

    class SimpleRetriever:
        def __init__(self, vectorstore: SimpleVectorStore, k: int = 5):
            self.vs = vectorstore
            self.k = k

        def get_relevant_documents(self, query: str):
            results = self.vs.similarity_search_with_score(query, k=self.k)
            return [doc for doc, score in results]
            
        def invoke(self, query: str):
            """Alias for get_relevant_documents to match LangChain API"""
            return self.get_relevant_documents(query)

    # Set the vectorstore class to the fallback
    VectorStoreClass = SimpleVectorStore
else:
    VectorStoreClass = FAISS


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
