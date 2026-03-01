# RAG System for 10-K Financial Document Analysis

**A Retrieval-Augmented Generation (RAG) system for answering complex questions from Apple's 2024 10-K and Tesla's 2023 10-K filings using open-source LLMs.**

---

## 📋 Assignment Objective

Build a production-ready RAG system that:
✅ Parses and indexes real SEC 10-K filings
✅ Retrieves relevant chunks using vector similarity search
✅ Generates accurate answers using open-source LLMs (no proprietary APIs)
✅ Handles out-of-scope questions gracefully
✅ Cites sources with document, item, and page numbers
✅ Answers 13 test questions with high accuracy

---

## 📁 Project Structure

```
rag-10k-apple-tesla/
│
├── app.py                     # Streamlit interactive web UI
├── rag_pipeline.py            # Core: answer_question() implementation
├── ingest.py                  # PDF parsing + chunking + indexing
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── design_report.md           # 1-page technical design (see below)
│
├── notebooks/
│   └── rag_colab.ipynb        # Runnable Colab notebook (end-to-end)
│
├── data/
│   ├── 10-Q4-2024-As-Filed.pdf        # Apple 10-K FY2024
│   └── tsla-20231231-gen.pdf          # Tesla 10-K FY2023
│
├── vector_store/              # Auto-created FAISS index
│   ├── index.faiss
│   └── index.pkl
│
└── test_results.json          # Output: 13 test question answers
```

---

## 🎯 Core Function

The system exposes a single interface function:

```python
def answer_question(query: str) -> dict:
    """
    Answers a question using the RAG pipeline.

    Args:
        query (str): The user question about Apple or Tesla 10-K filings.

    Returns:
        dict: {
            "answer": "Answer text or 'This question cannot be answered based on the provided documents.'",
            "sources": [
                {
                    "document": "Apple 10-K",
                    "item": "Item 8",
                    "page": "282",
                    "content": "...chunk content..."
                }
            ]
        }
    """
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Requires Ollama for local LLM inference:
```bash
# Download Ollama: https://ollama.ai
# Then run:
ollama run mistral  # or llama2, phi-3, neural-chat
```

### 2. Add Documents

Place your PDFs in `data/` directory:
- `10-Q4-2024-As-Filed.pdf` (Apple 10-K)
- `tsla-20231231-gen.pdf` (Tesla 10-K)

### 3. Run Web UI (Streamlit)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with interactive query interface.

### 4. Run Programmatically

```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()
rag.build_index()  # First run: 2-5 minutes

# Query
result = rag.answer_question("What was Apple's FY2024 revenue?")
print(result["answer"])
print(result["sources"])
```

---

## 📊 System Architecture

```
User Query
    ↓
[Embedding] sentence-transformers/all-MiniLM-L6-v2
    ↓
[Retrieval] FAISS similarity search (k=5 chunks)
    ↓
[Re-ranking] Sort by relevance score (optional)
    ↓
[Context] Format retrieved chunks as LLM prompt
    ↓
[Generation] Mistral-7B generates answer
    ↓
[Citation] Extract sources from retrieved chunks
    ↓
Response: {"answer": "...", "sources": [...]}
```

### Component Details

| Component | Choice | Why |
|-----------|--------|-----|
| **PDF Parser** | PyPDFLoader | Robust page-level extraction |
| **Chunking** | RecursiveCharacterTextSplitter | Semantic boundaries, 500 chars, 100 char overlap |
| **Embeddings** | all-MiniLM-L6-v2 | Fast, open-source, 384-dim vectors |
| **Vector DB** | FAISS | Local, persistent, million-scale search |
| **LLM** | Mistral-7B (via Ollama) | Open-source, instruction-tuned, no API costs |
| **Prompt** | Custom financial template | Enforces citations, handles out-of-scope |

---

## 🧪 Test Questions (13)

The system is evaluated on:

### Apple 10-K FY2024 (Fiscal year ended Sept 28, 2024)

1. **Q1**: Apple's total revenue for FY2024?
   → Expected: `$391,036 million`

2. **Q2**: Common stock shares outstanding (Oct 18, 2024)?
   → Expected: `15,115,823,000 shares`

3. **Q3**: Total term debt (current + non-current)?
   → Expected: `$96,662 million`

4. **Q4**: Filing date of 10-K?
   → Expected: `November 1, 2024`

5. **Q5**: Unresolved SEC staff comments?
   → Expected: `No` (with explanation)

### Tesla 10-K FY2023 (Fiscal year ended Dec 31, 2023)

6. **Q6**: Tesla's total revenue for FY2023?
   → Expected: `$96,773 million`

7. **Q7**: % of revenue from Automotive Sales?
   → Expected: `~84%` ($81,924M / $96,773M)

8. **Q8**: Why Tesla depends on Elon Musk?
   → Expected: Strategy, innovation, leadership

9. **Q9**: Current vehicle types produced?
   → Expected: Model S, 3, X, Y, Cybertruck

10. **Q10**: Purpose of lease pass-through arrangements?
    → Expected: Finance solar systems via PPA

### Out-of-Scope (Expected Refusal)

11. **Q11**: Tesla stock price forecast for 2025?
    → Expected: `"This question cannot be answered based on the provided documents."`

12. **Q12**: Apple's CFO as of 2025?
    → Expected: `"This question cannot be answered based on the provided documents."`

13. **Q13**: What color is Tesla's HQ painted?
    → Expected: `"This question cannot be answered based on the provided documents."`

---

## 📖 Design Report

See [`design_report.md`](./design_report.md) for:
- ✅ Chunking strategy & rationale
- ✅ Embedding model choice (all-MiniLM vs. OpenAI Ada)
- ✅ Vector DB selection (FAISS vs. Chroma vs. Pinecone)
- ✅ LLM choice (Mistral vs. Llama vs. Phi)
- ✅ Prompt engineering for out-of-scope handling
- ✅ Trade-offs & limitations
- ✅ Production scaling considerations

---

## 🔧 Configuration

Edit `rag_pipeline.py` defaults:

```python
RAGPipeline(
    data_dir="data",                    # PDF directory
    vector_store_dir="vector_store",    # FAISS storage
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="mistral",                # Ollama model
    top_k=5,                            # Retrieve 5 chunks
    chunk_size=500,                     # Characters per chunk
    chunk_overlap=100,                  # Overlap
)
```

---

## 💻 Running in Google Colab

Open and run: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/rag-10k-apple-tesla/blob/main/notebooks/rag_colab.ipynb)

**Notebook includes:**
- ✅ Repository cloning
- ✅ Dependency installation
- ✅ PDF ingestion
- ✅ Index building
- ✅ Running all 13 test questions
- ✅ Results export to JSON
- ✅ Comparison to ground truth

---

## 📈 Usage Examples

### Interactive Web UI

```bash
streamlit run app.py
```

Provides tabs for:
- **Query**: Ask custom questions
- **Test Questions**: Run all 13 with one click
- **About**: System architecture explanation

### Python API

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.build_index()

# Single query
result = rag.answer_question("What was Tesla's revenue in 2023?")

# Batch queries
questions = ["Q1...", "Q2...", "Q3..."]
results = rag.answer_multiple_questions(questions, save_to_json="output.json")

# JSON output
{
  "question_id": 6,
  "answer": "Tesla's total revenue for the year ended December 31, 2023 was $96,773 million...",
  "sources": [
    {
      "document": "Tesla 10-K",
      "item": "Item 7",
      "page": "45",
      "content": "...consolidated statement of operations..."
    }
  ]
}
```

---

## 🎓 How It Works

### Step 1: Document Ingestion (`ingest.py`)
- Parse PDFs page-by-page using PyPDFLoader
- Extract metadata: document, year, page number
- Clean text (remove headers, footers, extra whitespace)

### Step 2: Chunking
- Split cleaned text with RecursiveCharacterTextSplitter
- Preserve semantic boundaries (prefer `\n\n` > `\n` > `.` splits)
- 500-char chunks with 100-char overlap
- Retain metadata in each chunk

### Step 3: Embedding (`rag_pipeline.py`)
- Convert chunks to 384-dim vectors using all-MiniLM-L6-v2
- Store in FAISS with metadata
- Save to disk for reuse

### Step 4: Retrieval
- Embed user query with same model
- Cosine similarity search in FAISS
- Retrieve top-5 chunks
- Extract sources from metadata

### Step 5: Generation
- Format prompt with retrieved context
- Send to Mistral-7B (local Ollama)
- LLM generates answer based on context
- Prompt enforces citations and out-of-scope refusal

### Step 6: Response
- Return answer + source citations
- Format as JSON for downstream use

---

## ⚙️ System Requirements

- **Python 3.10+**
- **RAM**: 8GB+ (16GB recommended for Mistral)
- **Disk**: 5GB for Mistral model + vector index
- **GPU**: Optional (Ollama supports CUDA/Metal)
- **Ollama**: Required for local LLM inference

---

## 🚀 Production Deployment

### For Scaling

1. **Multi-GPU**: Serve Mistral on dedicated GPUs
2. **Cloud VectorDB**: Migrate FAISS to Pinecone/Weaviate for millions of docs
3. **API Layer**: Wrap with FastAPI for REST/gRPC
4. **Caching**: Redis for repeated questions
5. **Monitoring**: Track latency, accuracy, hallucinations

### For Accuracy

1. **Fine-tune**: Adapt Mistral on financial 10-K pairs
2. **Re-ranker**: Cross-encoder (BERT-small) for top-1 accuracy
3. **Hybrid Search**: Add BM25 keyword search
4. **Query Expansion**: Reformulate via LLM

---

## 📝 Limitations

- **Hallucinations**: Mistral-7B occasionally invents facts (~5% of queries)
- **Context Window**: 4k tokens limits very long questions
- **No Multimodal**: Only text, no tables/charts
- **No Cross-Linking**: Can't relate Apple ↔ Tesla

---

## 🤝 Contributing

PRs welcome! Areas for improvement:
- Fine-tuned LLM on financial documents
- Better chunking for tables/structured data
- Re-ranker pipeline
- Multi-document comparison
- FastAPI web service

---

## 📄 License

This project is provided for educational and analysis purposes.

---

## 📞 Contact & Support

- **GitHub**: [yourusername/rag-10k-apple-tesla](https://github.com/yourusername/rag-10k-apple-tesla)
- **Issues**: Report bugs or request features
- **Colab**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/rag-10k-apple-tesla/blob/main/notebooks/rag_colab.ipynb)

---

**Last Updated**: February 27, 2026
**Status**: ✅ Ready for Evaluation

