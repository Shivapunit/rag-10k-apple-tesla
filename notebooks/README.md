# Notebooks

## rag_colab.ipynb - Google Colab Notebook

A complete, runnable end-to-end RAG system for Google Colab or Jupyter environments.

### Features
- ✅ Clone GitHub repository
- ✅ Install all dependencies
- ✅ Ingest and process PDFs
- ✅ Build vector index (FAISS)
- ✅ Run all 13 test questions
- ✅ Export results as JSON
- ✅ Compare to ground truth answers

### How to Use

#### Option 1: Google Colab (Recommended)
1. Click the Colab badge in the notebook
2. Or visit: https://colab.research.google.com and upload `rag_colab.ipynb`
3. Run all cells in order
4. Download `test_results.json`

#### Option 2: Local Jupyter
```bash
jupyter notebook rag_colab.ipynb
```

#### Option 3: Kaggle
1. Upload to Kaggle Notebooks
2. Enable internet and GPU
3. Run cells

### What It Does

**Section 1: Setup (1 cell)**
- Clone GitHub repo
- List repository structure

**Section 2: Dependencies (1 cell)**
- Install requirements from `requirements.txt`
- Install transformers for local inference

**Section 3: Imports (1 cell)**
- Import RAG pipeline
- Configure logging

**Section 4: Document Check (2 cells)**
- Verify PDFs are present in data/
- Show file sizes
- Provide download instructions if missing

**Section 5: Initialize (2 cells)**
- Create RAGPipeline instance
- Check if index exists
- Build or load FAISS index

**Section 6: Test Single (1 cell)**
- Run one example question
- Display answer + sources
- Verify system works

**Section 7: Run All Tests (2 cells)**
- Process all 13 questions
- Show progress bar
- Display results

**Section 8: Results (2 cells)**
- Show results as table
- Display full JSON
- Save to file
- Download from Colab

**Section 9: Detailed Review (1 cell)**
- Print each answer with sources
- Review for accuracy

**Section 10: Evaluation (1 cell)**
- Compare to ground truth answers
- Show expected vs. actual

### Requirements

- **Python 3.10+** (Colab has 3.10)
- **Internet**: For cloning repo + downloading dependencies
- **Storage**: 5GB for LLM models + indexes
- **Time**:
  - Setup: 2 minutes
  - Build index: 2-5 minutes
  - Run 13 tests: 5-15 minutes
  - **Total**: 10-25 minutes first run

### Expected Output

```json
[
  {
    "question_id": 1,
    "answer": "$391,036 million for the fiscal year ended September 28, 2024...",
    "sources": ["Apple 10-K - Item 8 - p. 282"]
  },
  ...
]
```

### Troubleshooting

**"ModuleNotFoundError"**
- Run `pip install -q langchain langchain-community ...` cell again

**"PDFs not found"**
- Upload `10-Q4-2024-As-Filed.pdf` and `tsla-20231231-gen.pdf` to Colab
- Then restart runtime and re-run

**"Ollama connection error"**
- Colab doesn't have Ollama installed
- Use lightweight LLM alternatives in the notebook
- Or run locally with `jupyter notebook`

**"Out of memory"**
- Use smaller LLM (Phi instead of Mistral)
- Or reduce `top_k` from 5 to 3
- Use Colab Pro for more RAM

### Tips

1. **Save Checkpoints**: Colab can disconnect. Save vectors to Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Copy vector_store to Drive
   !cp -r vector_store /content/drive/MyDrive/
   ```

2. **Reuse Index**: Download vector_store after first run to avoid rebuilding

3. **GPU Acceleration**: Enable GPU in Colab Runtime → Change runtime type

4. **Batch Mode**: Process multiple queries in a loop with progress bar

### Advanced Usage

#### Custom Prompt
```python
rag.qa_chain.custom_prompt = "..."
result = rag.answer_question("query")
```

#### Adjust Retrieval
```python
result = rag.answer_question("query", top_k=3)  # Get 3 chunks instead of 5
```

#### Save to Drive
```python
from google.colab import drive, files
files.download('test_results.json')
```

### Next Steps

1. **Test Locally**: Run `python test_runner.py` on your machine
2. **Deploy**: Use Streamlit with `streamlit run app.py`
3. **Integrate**: Use `from rag_pipeline import RAGPipeline` in your code
4. **Improve**: Modify `rag_pipeline.py` and `ingest.py` for better results

---

**Happy experimenting!** 🚀

