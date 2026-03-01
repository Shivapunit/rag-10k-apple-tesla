@echo off
REM Quick start script for RAG 10-K system (Windows)

echo.
echo ==========================================
echo RAG System for 10-K Financial Documents
echo Quick Start Setup (Windows)
echo ==========================================

REM Step 1: Create virtual environment
echo.
echo [1/4] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Step 2: Install dependencies
echo.
echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Step 3: Download Ollama (interactive)
echo.
echo [3/4] LLM Setup
echo This system requires Ollama for local LLM inference.
echo Visit: https://ollama.ai and download Ollama for Windows.
echo.
pause
echo Please open another terminal window and run: ollama run mistral
echo Then come back and press any key when Ollama is running...
pause

REM Step 4: Run tests
echo.
echo [4/4] Running test questions...
python test_runner.py

echo.
echo ==========================================
echo [OK] Setup complete!
echo.
echo To start the web UI, run:
echo   streamlit run app.py
echo.
echo To run tests again:
echo   python test_runner.py
echo.
echo To query programmatically, run:
echo   python -c "from rag_pipeline import RAGPipeline; rag = RAGPipeline(); print(rag.answer_question('Your question here'))"
echo ==========================================
echo.
pause

