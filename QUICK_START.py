#!/usr/bin/env python3
"""
RAG SYSTEM - QUICK START
Minimal setup guide for rapid evaluation
"""

QUICK_START = """
╔════════════════════════════════════════════════════════════════════════╗
║                   RAG SYSTEM - QUICK START (5 MINUTES)                ║
╚════════════════════════════════════════════════════════════════════════╝

STEP 1: INSTALL (2 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pip install -r requirements.txt
ollama run mistral

STEP 2: ADD DOCUMENTS (1 minute)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Download from SEC EDGAR:
  • Apple 10-K: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K
  • Tesla 10-K: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1018724&type=10-K

Place in: data/
  ✅ 10-Q4-2024-As-Filed.pdf
  ✅ tsla-20231231-gen.pdf

STEP 3: RUN TESTS (2 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python test_runner.py

Output: test_results.json (contains answers to all 13 questions)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALTERNATIVE OPTIONS:

🌐 WEB INTERFACE (Interactive)
   streamlit run app.py
   → Opens http://localhost:8501

☁️ GOOGLE COLAB (Cloud - No Setup)
   1. Open: notebooks/rag_colab.ipynb
   2. Click "Open in Colab"
   3. Run all cells
   4. Download results

🐍 PYTHON API
   from rag_pipeline import RAGPipeline
   rag = RAGPipeline()
   result = rag.answer_question("What was Apple's revenue?")
   print(result["answer"])
   print(result["sources"])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOCUMENTATION:

📖 README.md                - Complete overview
⚡ GETTING_STARTED.md      - Detailed setup
🏗️ design_report.md       - Architecture & decisions
📚 API.md                  - API reference
🎉 COMPLETION_SUMMARY.md   - What's included

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY FEATURES:

✅ answer_question(query) -> {"answer": "...", "sources": [...]}
✅ FAISS vector store (local, fast, persistent)
✅ Mistral-7B LLM via Ollama (open-source, no API costs)
✅ 13 test questions (Q1-Q10 answerable, Q11-Q13 refusal)
✅ Source citations with [Document, Item, Page]
✅ Production-ready code with full error handling
✅ Google Colab notebook (cloud deployment)
✅ Streamlit web UI (interactive dashboard)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TROUBLESHOOTING:

❓ "ModuleNotFoundError"
   → Run: pip install -r requirements.txt

❓ "Ollama connection error"
   → Make sure Ollama is running in another terminal

❓ "No PDFs found"
   → Check data/ directory has Apple & Tesla 10-Ks

❓ "Out of memory"
   → Reduce top_k: RAGPipeline(top_k=2)
   → Use smaller model: ollama run phi

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPECTED OUTPUT (test_results.json):

[
  {
    "question_id": 1,
    "answer": "$391,036 million for the fiscal year ended September 28, 2024...",
    "sources": ["Apple 10-K - Item 8 - p. 282"]
  },
  {
    "question_id": 2,
    "answer": "15,115,823,000 shares...",
    "sources": ["Apple 10-K - first paragraph"]
  },
  ...
  {
    "question_id": 11,
    "answer": "This question cannot be answered based on the provided documents.",
    "sources": []
  }
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

READY? Let's go! 🚀

   1. pip install -r requirements.txt && ollama run mistral
   2. Place PDFs in data/
   3. python test_runner.py

✅ That's it! Your RAG system is ready.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

if __name__ == "__main__":
    print(QUICK_START)

