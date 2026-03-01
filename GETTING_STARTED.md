# Getting Started Guide

## Quick Setup (5 minutes)

### Prerequisites
- Python 3.10+
- Ollama installed (from https://ollama.ai)
- 8GB+ RAM
- Internet connection for first dependency download

### Step 1: Install Dependencies

```bash
# On Windows
pip install -r requirements.txt

# On macOS/Linux
pip install -r requirements.txt
```

### Step 2: Download LLM Model (Ollama)

```bash
# Install Ollama from: https://ollama.ai

# Download Mistral model (2.2GB)
ollama run mistral

# OR use Llama 2 (4GB)
ollama run llama2

# OR use Phi-3 (smallest, ~2GB)
ollama run phi
```

Ollama will start a local LLM server at `http://localhost:11434`

### Step 3: Add PDF Documents

Place these files in the `data/` directory:

```
data/
├── 10-Q4-2024-As-Filed.pdf    # Apple 10-K FY2024
└── tsla-20231231-gen.pdf      # Tesla 10-K FY2023
```

**Where to download:**
- Apple 10-K: https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-24-000123&xbrl_type=v
- Tesla 10-K: https://www.sec.gov/cgi-bin/viewer?action=view&cik=1018724&accession_number=0000101968-24-000007&xbrl_type=v

### Step 4: Run the System

#### Option A: Interactive Web UI (Recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

#### Option B: Run Tests
```bash
python test_runner.py
```
Outputs `test_results.json` with answers to 13 questions

#### Option C: Use Python API
```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.build_index()  # First time only (takes 2-5 min)

result = rag.answer_question("What was Apple's FY2024 revenue?")
print(result["answer"])
print(result["sources"])
```

---

## Project Files Explained

| File | Purpose |
|------|---------|
| `rag_pipeline.py` | **Core RAG logic** - `answer_question()` function |
| `ingest.py` | PDF parsing, chunking, metadata extraction |
| `app.py` | Streamlit web UI (interactive dashboard) |
| `test_runner.py` | Run 13 evaluation questions |
| `design_report.md` | Technical design decisions |
| `README.md` | Full documentation |
| `requirements.txt` | Python dependencies |
| `data/` | Your PDF files go here |
| `vector_store/` | Auto-created FAISS index (local) |
| `notebooks/rag_colab.ipynb` | Google Colab notebook |

---

## Architecture at a Glance

```
PDFs (data/)
    ↓ [ingest.py]
Chunks + Metadata
    ↓ [sentence-transformers]
384-dim Embeddings
    ↓ [FAISS]
Vector Store (local)
    ↓
User Query
    ↓ [rag_pipeline.py]
Embed Query
    ↓ [FAISS Search]
Top-5 Chunks Retrieved
    ↓ [Ollama/Mistral]
Answer Generated with Sources
```

---

## Answering Questions

### Function Signature
```python
def answer_question(query: str) -> dict:
    return {
        "answer": "...",
        "sources": [
            {
                "document": "Apple 10-K",
                "item": "Item 8",
                "page": "282",
                "content": "..."
            }
        ]
    }
```

### Example Queries

**In-Scope (Answerable):**
```
Q: "What was Apple's total revenue for FY 2024?"
A: "$391,036 million" [with sources]

Q: "What percentage of Tesla's revenue came from automotive sales?"
A: "~84% ($81,924M / $96,773M)" [with sources]
```

**Out-of-Scope (Refusal):**
```
Q: "What is Tesla's stock price forecast for 2025?"
A: "This question cannot be answered based on the provided documents."

Q: "What color is Tesla's headquarters?"
A: "This question cannot be answered based on the provided documents."
```

---

## Troubleshooting

### "Ollama connection error"
```bash
# Make sure Ollama is running:
ollama run mistral

# In a separate terminal, test:
curl http://localhost:11434/api/generate -d '{"model":"mistral", "prompt":"hi"}'
```

### "No PDFs found in data/"
```bash
# Check files exist:
ls -la data/
# Should show:
# 10-Q4-2024-As-Filed.pdf
# tsla-20231231-gen.pdf
```

### "ModuleNotFoundError: No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "FAISS build error on Windows"
```bash
# Use CPU version:
pip install faiss-cpu

# If issues persist:
pip uninstall faiss-cpu
pip install faiss-cpu==1.7.4
```

### Memory issues
- Reduce `top_k` parameter from 5 to 3
- Use `phi` model instead of `mistral` (smaller)
- Reduce chunk size from 500 to 300

---

## Performance

### Typical Latencies
- **Indexing (first run)**: 2-5 minutes (depends on CPU/GPU)
- **Query embedding**: 50-100ms
- **FAISS retrieval**: 10-20ms
- **LLM generation**: 2-10 seconds (depends on model)
- **Total per query**: 3-15 seconds

### Index Size
- FAISS index: ~500MB for 10-K documents
- Embedding model: 384MB (all-MiniLM-L6-v2)
- LLM model: 3.8GB (Mistral-7B)
- **Total**: ~5GB on disk

---

## What's Next?

### For Evaluation
1. Run `python test_runner.py`
2. Check results in `test_results.json`
3. Compare to ground truth in `design_report.md`

### For Development
- Modify `rag_pipeline.py` to adjust retrieval or prompting
- Edit `ingest.py` to change chunking strategy
- Use Streamlit app to test interactively

### For Production
- Deploy with FastAPI (see: `app.py` wrapper)
- Scale FAISS to Pinecone/Weaviate
- Add re-ranker for higher accuracy
- Fine-tune LLM on financial documents

---

## Documentation

- **Design Report**: `design_report.md` - Technical architecture
- **README**: `README.md` - Full documentation
- **This Guide**: `GETTING_STARTED.md` - Quick start
- **Colab Notebook**: `notebooks/rag_colab.ipynb` - Cloud setup

---

## Questions?

- Check `README.md` for detailed docs
- Review `design_report.md` for architecture
- See `notebooks/rag_colab.ipynb` for examples
- Test with `python test_runner.py`

---

**Happy querying!** 🚀

