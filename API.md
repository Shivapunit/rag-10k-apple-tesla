"""
API DOCUMENTATION
RAG Pipeline - Core Interface
"""

## RAGPipeline Class

Main class for the RAG system. Handles document ingestion, indexing, retrieval, and LLM-based answer generation.

### Import
```python
from rag_pipeline import RAGPipeline
```

### Constructor
```python
RAGPipeline(
    data_dir: str = "data",
    vector_store_dir: str = "vector_store",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "mistral",
    top_k: int = 5,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
)
```

**Parameters:**
- `data_dir` (str): Directory containing PDF files. Default: `"data"`
- `vector_store_dir` (str): Directory for FAISS index. Default: `"vector_store"`
- `embedding_model` (str): HuggingFace embedding model. Default: `"sentence-transformers/all-MiniLM-L6-v2"`
- `llm_model` (str): Ollama model name (mistral, llama2, phi, neural-chat). Default: `"mistral"`
- `top_k` (int): Number of chunks to retrieve per query. Default: `5`
- `chunk_size` (int): Characters per chunk. Default: `500`
- `chunk_overlap` (int): Overlap between chunks. Default: `100`

**Example:**
```python
rag = RAGPipeline(
    data_dir="documents",
    top_k=3,
    chunk_size=400
)
```

---

## Methods

### 1. build_index()
Build vector index from PDF documents.

```python
def build_index(self) -> bool:
    """
    Build FAISS index from PDFs in data_dir.

    Returns:
        bool: True if successful, False otherwise

    Raises:
        Exception: On PDF parsing or embedding errors
    """
```

**Example:**
```python
rag = RAGPipeline()
success = rag.build_index()
if success:
    print(f"Index built with {rag.get_chunk_count()} chunks")
```

**Time:** 2-5 minutes (depends on PDF size and CPU)

**Output:** Creates `vector_store/` directory with:
- `index.faiss` - Vector database
- `index.pkl` - Metadata

---

### 2. load_index()
Load existing vector index from disk.

```python
def load_index(self) -> bool:
    """
    Load pre-built FAISS index from disk.

    Returns:
        bool: True if successful, False otherwise
    """
```

**Example:**
```python
rag = RAGPipeline()
if rag.is_indexed():
    rag.load_index()
else:
    rag.build_index()
```

**Time:** <1 second

---

### 3. is_indexed()
Check if vector index exists on disk.

```python
def is_indexed(self) -> bool:
    """
    Check if vector store exists.

    Returns:
        bool: True if index found
    """
```

**Example:**
```python
if rag.is_indexed():
    print("Using cached index")
else:
    print("Building new index")
```

---

### 4. answer_question()
**THE MAIN INTERFACE** - Answer a question using the RAG pipeline.

```python
def answer_question(
    self,
    query: str,
    top_k: Optional[int] = None,
    temperature: float = 0.3,
    return_sources: bool = True
) -> Dict[str, Any]:
    """
    Answer a question using the RAG pipeline.

    Args:
        query (str): The user question about Apple or Tesla 10-K filings
        top_k (Optional[int]): Number of chunks to retrieve (uses default if None)
        temperature (float): LLM temperature (0=deterministic, 1=random)
        return_sources (bool): Whether to return source documents

    Returns:
        dict: {
            "answer": "Answer text or 'This question cannot be answered...'",
            "sources": [
                {
                    "document": "Apple 10-K",
                    "item": "Item 8",
                    "page": "282",
                    "content": "retrieved chunk text"
                }
            ]
        }

    Raises:
        Exception: On LLM or retrieval errors
    """
```

**Example 1: Simple Query**
```python
rag = RAGPipeline()
rag.build_index()

result = rag.answer_question("What was Apple's FY2024 revenue?")
print(result["answer"])
# Output: $391,036 million for the fiscal year ended September 28, 2024...

print(result["sources"])
# Output: [{"document": "Apple 10-K", "item": "Item 8", "page": "282", ...}]
```

**Example 2: Adjust Parameters**
```python
# Retrieve more context
result = rag.answer_question(query, top_k=10)

# Deterministic mode (no randomness)
result = rag.answer_question(query, temperature=0.0)

# Creative mode (more varied responses)
result = rag.answer_question(query, temperature=0.9)

# Don't include sources
result = rag.answer_question(query, return_sources=False)
```

**Example 3: Out-of-Scope Handling**
```python
# This question is not in the documents
result = rag.answer_question("Tesla's stock price forecast for 2025?")
print(result["answer"])
# Output: This question cannot be answered based on the provided documents.

print(result["sources"])
# Output: []  (empty list)
```

**Return Value Structure:**
```python
{
    "answer": "...",  # str: LLM-generated answer or refusal
    "sources": [      # list: Retrieved document chunks
        {
            "document": "Apple 10-K",      # str: Source document name
            "item": "Item 8",              # str: SEC item reference
            "page": "282",                 # str: Page number
            "content": "..."               # str: First 200 chars of chunk
        }
    ]
}
```

**Behavior:**
- **Auto-loads index** if not already loaded
- **Auto-builds index** if it doesn't exist (first time only)
- **Enforces citations** - Answer must reference retrieved chunks
- **Handles out-of-scope** - Refuses questions not in documents
- **Streams from LLM** - May take 3-15 seconds depending on query complexity

---

### 5. answer_multiple_questions()
Answer multiple questions and optionally save results.

```python
def answer_multiple_questions(
    self,
    questions: List[str],
    save_to_json: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Answer multiple questions and optionally save to JSON.

    Args:
        questions (List[str]): List of questions
        save_to_json (Optional[str]): File path to save results (e.g., "results.json")

    Returns:
        List[Dict]: List of answer dictionaries
    """
```

**Example:**
```python
questions = [
    "What was Apple's revenue?",
    "What was Tesla's revenue?",
    "Who is Tesla's CEO?"
]

results = rag.answer_multiple_questions(
    questions,
    save_to_json="my_results.json"
)

# results = [
#     {"question_id": 1, "answer": "...", "sources": [...]},
#     {"question_id": 2, "answer": "...", "sources": [...]},
#     {"question_id": 3, "answer": "...", "sources": [...]},
# ]
```

**File Output (JSON):**
```json
[
  {
    "question_id": 1,
    "question": "What was Apple's revenue?",
    "answer": "$391,036 million...",
    "sources": [...]
  }
]
```

---

### 6. get_chunk_count()
Get total number of chunks in vector store.

```python
def get_chunk_count(self) -> int:
    """
    Get total chunks in FAISS index.

    Returns:
        int: Number of embedded chunks
    """
```

**Example:**
```python
rag = RAGPipeline()
rag.build_index()
print(f"Total chunks: {rag.get_chunk_count()}")
# Output: Total chunks: 2547
```

---

## DocumentIngester Class

Handles PDF parsing, chunking, and preprocessing.

### Import
```python
from ingest import DocumentIngester
```

### Constructor
```python
DocumentIngester(chunk_size: int = 500, chunk_overlap: int = 100)
```

### Methods

#### ingest_from_directory()
```python
def ingest_from_directory(self, directory: str) -> List[Document]:
    """
    Ingest all PDFs from directory and create chunks.

    Args:
        directory (str): Path to directory with PDFs

    Returns:
        List[Document]: Chunked documents with metadata
    """
```

**Example:**
```python
ingester = DocumentIngester(chunk_size=500)
documents = ingester.ingest_from_directory("data")
print(f"Created {len(documents)} chunks")
```

---

## Complete Example Workflows

### Workflow 1: Quick Start
```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Build index (first time only)
rag.build_index()

# Answer questions
result = rag.answer_question("What was Apple's revenue?")
print(result["answer"])
```

### Workflow 2: Batch Processing
```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.build_index()

# Read questions from file
with open("questions.txt") as f:
    questions = [line.strip() for line in f]

# Answer all
results = rag.answer_multiple_questions(questions, save_to_json="results.json")

# Review
for r in results:
    print(f"Q{r['question_id']}: {r['answer'][:100]}...")
```

### Workflow 3: Custom Configuration
```python
from rag_pipeline import RAGPipeline

# Use different parameters
rag = RAGPipeline(
    data_dir="/path/to/pdfs",
    vector_store_dir="/tmp/vectors",
    chunk_size=300,     # Smaller chunks
    chunk_overlap=50,   # Less overlap
    top_k=3             # Retrieve fewer docs
)

rag.build_index()
result = rag.answer_question("Question", top_k=5)  # Override for this query
```

### Workflow 4: Web Interface (Streamlit)
```bash
streamlit run app.py
```
Opens interactive dashboard at http://localhost:8501

### Workflow 5: Tests (All 13 Questions)
```bash
python test_runner.py
```
Outputs `test_results.json`

---

## Error Handling

### Common Errors

**Error: "ModuleNotFoundError: No module named 'langchain'"**
```bash
pip install -r requirements.txt
```

**Error: "Ollama connection refused"**
- Make sure Ollama is running:
```bash
ollama run mistral
```

**Error: "No PDFs found in data/"**
- Check directory exists and has PDF files
- File must end in `.pdf` (case-sensitive on Linux)

**Error: "CUDA out of memory"**
- Use smaller model: `ollama run phi`
- Or reduce `top_k` parameter
- Or reduce `chunk_size`

### Exception Handling
```python
from rag_pipeline import RAGPipeline

try:
    rag = RAGPipeline()
    rag.build_index()
    result = rag.answer_question("Query")
except Exception as e:
    print(f"Error: {e}")
    # result = {"answer": "Error occurred", "sources": []}
```

---

## Performance Tuning

### Faster Responses
```python
# Retrieve fewer chunks
rag = RAGPipeline(top_k=2)  # Instead of 5

# Reduce context window
result = rag.answer_question(query, top_k=2)
```

### Better Accuracy
```python
# Retrieve more chunks
rag = RAGPipeline(top_k=8)

# Lower temperature (more deterministic)
result = rag.answer_question(query, temperature=0.1)
```

### Lower Memory Usage
```python
# Smaller chunks = fewer embeddings
rag = RAGPipeline(chunk_size=300, chunk_overlap=50)

# Smaller embedding model (but less accurate)
# Edit rag_pipeline.py to change embedding_model
```

---

## Configuration Reference

### rag_pipeline.py Parameters
- `data_dir`: Where PDFs are stored
- `vector_store_dir`: Where FAISS index is saved
- `embedding_model`: HuggingFace model ID
- `llm_model`: Ollama model (mistral, llama2, phi, neural-chat)
- `top_k`: Chunks to retrieve (3-10 recommended)
- `chunk_size`: Characters per chunk (300-1000)
- `chunk_overlap`: Overlap between chunks (50-200)

### Typical Configurations

**Fast (Low Accuracy)**
```python
rag = RAGPipeline(top_k=2, chunk_size=300)
```

**Balanced (Recommended)**
```python
rag = RAGPipeline(top_k=5, chunk_size=500)
```

**Accurate (Slow)**
```python
rag = RAGPipeline(top_k=8, chunk_size=700)
```

---

**For more information, see:**
- README.md - Full documentation
- design_report.md - Architecture decisions
- GETTING_STARTED.md - Quick setup guide

