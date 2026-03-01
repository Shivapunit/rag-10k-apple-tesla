#!/bin/bash
# Quick start script for RAG 10-K system

echo "=========================================="
echo "RAG System for 10-K Financial Documents"
echo "Quick Start Setup"
echo "=========================================="

# Step 1: Create virtual environment
echo -e "\n[1/4] Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 2: Install dependencies
echo -e "\n[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Download Ollama (interactive)
echo -e "\n[3/4] LLM Setup"
echo "This system requires Ollama for local LLM inference."
echo "Visit: https://ollama.ai and download Ollama for your OS."
echo ""
read -p "After installing Ollama, press Enter to continue..."
read -p "Then run 'ollama run mistral' in another terminal and press Enter here..."

# Step 4: Run tests
echo -e "\n[4/4] Running test questions..."
python test_runner.py

echo -e "\n=========================================="
echo "✅ Setup complete!"
echo ""
echo "To start the web UI, run:"
echo "  streamlit run app.py"
echo ""
echo "To run tests again:"
echo "  python test_runner.py"
echo ""
echo "To query programmatically:"
echo "  python -c \"from rag_pipeline import RAGPipeline; rag = RAGPipeline(); print(rag.answer_question('Your question here'))\""
echo "=========================================="

