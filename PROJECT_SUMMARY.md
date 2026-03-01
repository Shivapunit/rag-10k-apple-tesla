"""
PROJECT STRUCTURE & SUMMARY
RAG System for 10-K Financial Documents
"""

## ✅ COMPLETE PROJECT STRUCTURE

```
rag-10k-apple-tesla/
│
├── 📄 Core Application Files
│   ├── app.py                      # Streamlit interactive web UI
│   ├── rag_pipeline.py             # Core RAG logic + answer_question()
│   ├── ingest.py                   # PDF parsing, chunking, indexing
│   ├── test_runner.py              # Run all 13 evaluation questions
│   └── main.py                     # Legacy (can be deleted)
│
├── 📋 Documentation
│   ├── README.md                   # Full project documentation
│   ├── GETTING_STARTED.md          # Quick start guide (this file)
│   ├── design_report.md            # 1-page technical design
│   └── .env.example                # Environment variables template
│
├── 📁 Data & Storage
│   ├── data/                       # Your 10-K PDFs go here
│   │   ├── README.md               # Instructions for PDF files
│   │   ├── 10-Q4-2024-As-Filed.pdf # Apple 10-K FY2024 (TO ADD)
│   │   └── tsla-20231231-gen.pdf   # Tesla 10-K FY2023 (TO ADD)
│   │
│   └── vector_store/               # Auto-created FAISS index
│       ├── index.faiss             # Vector database (created on first run)
│       └── index.pkl               # Metadata pickle (created on first run)
│
├── 📚 Notebooks
│   ├── notebooks/
│   │   └── rag_colab.ipynb         # Google Colab notebook (end-to-end)
│   │
│   └── notebooks/README.md
│
├── ⚙️ Configuration & Setup
│   ├── requirements.txt            # Python dependencies (pip install -r)
│   ├── setup.sh                    # Quick start script (macOS/Linux)
│   ├── setup.bat                   # Quick start script (Windows)
│   ├── .gitignore                  # Git ignore patterns
│   └── .env.example                # Environment variables template
│
├── 🔧 Legacy Code (Not Used)
│   └── src/                        # Old modular code (can be deleted)
│       ├── config.py
│       ├── document_loader.py
│       ├── document_preprocessor.py
│       ├── qa_chain.py
│       ├── retriever.py
│       ├── vector_store.py
│       └── __init__.py
│
└── 📊 Output Files (Generated)
    └── test_results.json           # Results from test_runner.py
```

---

## 🎯 KEY FILES EXPLAINED

### 1. **rag_pipeline.py** (CORE)
```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.build_index()

# THE REQUIRED FUNCTION:
result = rag.answer_question("What was Apple's revenue?")
# Returns: {"answer": "...", "sources": [...]}
```
- Orchestrates entire RAG pipeline
- Implements `answer_question()` - the main interface
- Handles embedding, retrieval, LLM generation
- Loads/manages FAISS vector store

### 2. **ingest.py** (DATA PROCESSING)
- Parses PDFs (Apple 10-K, Tesla 10-K)
- Splits into 500-character chunks (semantic overlap)
- Extracts metadata (document, page, item)
- Creates embeddings with sentence-transformers
- Stores vectors in FAISS

### 3. **app.py** (WEB INTERFACE)
- Streamlit dashboard
- Query tab: Ask single questions
- Test Questions tab: Run all 13 at once
- About tab: System architecture explanation
- Export results as JSON

### 4. **test_runner.py** (EVALUATION)
```bash
python test_runner.py
```
- Runs all 13 test questions
- Saves results to `test_results.json`
- Compares to ground truth
- Outputs statistics

### 5. **design_report.md** (DOCUMENTATION)
- 1-page technical report
- Chunking strategy & rationale
- Embedding model selection (why all-MiniLM vs Ada)
- Vector DB choice (why FAISS)
- LLM selection (why Mistral, not GPT-4)
- Prompt engineering approach
- Out-of-scope handling
- Production scaling considerations

### 6. **notebooks/rag_colab.ipynb** (CLOUD SETUP)
- Google Colab compatible
- Clone repository
- Install dependencies
- Build index
- Run 13 tests
- Download results JSON

---

## 📦 DEPENDENCIES

```
langchain==0.1.14                   # LLM orchestration
langchain-community==0.0.38         # Community integrations
pypdf==4.0.1                        # PDF parsing
faiss-cpu==1.7.4                    # Vector database (local)
sentence-transformers==2.2.2        # Embeddings
streamlit==1.28.1                   # Web UI
python-dotenv==1.0.0                # Environment variables
pydantic==2.5.0                     # Data validation
numpy, pandas, requests, tqdm       # Utilities
```

**External Requirement:**
- **Ollama** (https://ollama.ai) - Local LLM server
- Commands: `ollama run mistral` (or llama2, phi, neural-chat)

---

## 🚀 QUICK START (3 STEPS)

### Step 1: Install
```bash
pip install -r requirements.txt
ollama run mistral
```

### Step 2: Add PDFs
```
data/
├── 10-Q4-2024-As-Filed.pdf
└── tsla-20231231-gen.pdf
```

### Step 3: Run
```bash
# Option A: Web UI
streamlit run app.py

# Option B: Tests
python test_runner.py

# Option C: Python
python -c "from rag_pipeline import RAGPipeline; rag = RAGPipeline(); print(rag.answer_question('Your question'))"
```

---

## 🧪 EVALUATION QUESTIONS (13)

### Apple 10-K (Q1-Q5)
1. Total revenue FY2024 → $391,036 million
2. Common shares outstanding → 15,115,823,000
3. Total term debt → $96,662 million
4. Filing date → November 1, 2024
5. Unresolved SEC comments? → No

### Tesla 10-K (Q6-Q10)
6. Total revenue FY2023 → $96,773 million
7. % from automotive sales → ~84%
8. Why dependent on Elon Musk? → Strategy, innovation, leadership
9. Current vehicles? → Model S, 3, X, Y, Cybertruck
10. Lease pass-through purpose? → Finance solar systems

### Out-of-Scope (Q11-Q13)
11. Tesla stock forecast 2025? → "Cannot be answered"
12. Apple CFO 2025? → "Cannot be answered"
13. Tesla HQ color? → "Cannot be answered"

---

## 📊 SYSTEM ARCHITECTURE

```
PDFs (10-K filings)
  ↓
[ingest.py] - Parse & Chunk
  ↓
500-char chunks with metadata
  ↓
[sentence-transformers] - Embed (384-dim)
  ↓
[FAISS] - Vector Store (local)
  ↓
User Query
  ↓
[sentence-transformers] - Embed query
  ↓
[FAISS] - Similarity search (k=5)
  ↓
Retrieved chunks
  ↓
[Ollama/Mistral-7B] - Generate answer
  ↓
Answer + Source Citations
```

**Key Design Decisions:**
- **Chunking**: 500 chars (semantic boundaries)
- **Embeddings**: all-MiniLM-L6-v2 (fast, open-source)
- **Vector DB**: FAISS (local, no cloud dependency)
- **LLM**: Mistral-7B (open-source, no API costs)
- **Retrieval**: Top-5 chunks via cosine similarity
- **No Re-ranker**: Simplicity (can add for production)

---

## 🔄 WORKFLOW

### First Time (5-10 minutes)
```
1. Install dependencies (pip install -r requirements.txt)
2. Download LLM (ollama run mistral)
3. Add PDFs to data/
4. Build index (rag.build_index()) ← Takes 2-5 minutes
5. Query (rag.answer_question(...))
```

### Subsequent Runs (Fast)
```
1. Load existing FAISS index (rag.load_index())
2. Query instantly (rag.answer_question(...))
```

### On First Query
- Auto-detects if index exists
- If not found, builds automatically
- Saves to disk for reuse

---

## 📈 EXPECTED RESULTS

### Performance on Test Questions
- **Q1-Q10** (Factual, In-Scope): 80-95% accuracy
  - Exact numbers: 95%+ accuracy
  - Explanations: 80%+ accuracy
  - Sources: Correct citations

- **Q11-Q13** (Out-of-Scope): 100% refusal rate
  - Must correctly refuse stock forecasts
  - Must correctly refuse 2025 information
  - Must correctly refuse non-document questions

### Latency
- First query: 5-15 seconds (includes LLM generation)
- Subsequent queries: Same (stateless)
- Index building: 2-5 minutes (first time only)

### Memory Usage
- FAISS index: ~500MB
- Embeddings model: ~384MB
- Mistral LLM: ~3.8GB
- **Total**: ~5GB RAM needed

---

## 🎓 OUTPUT FORMAT

### Single Query
```python
result = rag.answer_question("What was Apple's revenue?")

print(result)
# {
#     "answer": "$391,036 million for the fiscal year ended...",
#     "sources": [
#         {
#             "document": "Apple 10-K",
#             "item": "Item 8",
#             "page": "282",
#             "content": "...chunk text..."
#         }
#     ]
# }
```

### Batch (test_runner.py)
```json
[
    {
        "question_id": 1,
        "answer": "$391,036 million",
        "sources": ["Apple 10-K - Item 8 - p. 282"]
    },
    {
        "question_id": 11,
        "answer": "This question cannot be answered based on the provided documents.",
        "sources": []
    }
]
```

---

## 🔗 NEXT STEPS

1. **Setup**: Follow GETTING_STARTED.md
2. **Understand**: Read design_report.md
3. **Test**: Run `python test_runner.py`
4. **Interact**: Run `streamlit run app.py`
5. **Deploy**: See README.md "Production Deployment"

---

## 📚 DOCUMENTATION HIERARCHY

1. **README.md** ← Start here (comprehensive overview)
2. **GETTING_STARTED.md** ← Quick setup (5 minutes)
3. **design_report.md** ← Technical deep-dive (1 page)
4. **notebooks/rag_colab.ipynb** ← Cloud implementation (Colab)

---

## ✅ CHECKLIST BEFORE SUBMISSION

- [ ] PDFs in data/ directory
- [ ] `pip install -r requirements.txt` successful
- [ ] `ollama run mistral` running
- [ ] `python test_runner.py` produces test_results.json
- [ ] Answers match ground truth (Q1-Q10)
- [ ] Out-of-scope questions refused (Q11-Q13)
- [ ] GitHub repo public & README visible
- [ ] Colab notebook link in README
- [ ] design_report.md explains all choices
- [ ] All files committed to git

---

**Ready to submit!** 🚀

