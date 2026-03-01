#!/usr/bin/env python3
"""
Quick example: Using RAG system with Ollama API

Two ways to use:
1. Local Ollama: No setup needed, just run 'ollama run mistral'
2. Ollama API: Use with API key for cloud deployment
"""

import os
from rag_pipeline import RAGPipeline

# ============================================================================
# EXAMPLE 1: Using Local Ollama (Default)
# ============================================================================
print("=" * 80)
print("EXAMPLE 1: Local Ollama Mode")
print("=" * 80)

# Create pipeline with local Ollama
rag_local = RAGPipeline(
    use_api=False  # Use local Ollama
)

# Build index (first run only)
rag_local.build_index()

# Ask a question
result = rag_local.answer_question("What was Apple's revenue for FY 2024?")
print("\nAnswer:", result["answer"])
print("Sources:", result["sources"])


# ============================================================================
# EXAMPLE 2: Using Ollama API (Cloud/Remote)
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: Ollama API Mode (with Authentication)")
print("=" * 80)

# Method 1: Pass API key directly
api_key = "your_api_key_here"  # Replace with your actual key

rag_api = RAGPipeline(
    use_api=True,  # Use Ollama API
    ollama_api_key=api_key,
    ollama_api_url="https://api.ollama.com/v1/chat/completions"
)

# Method 2: Use environment variable (recommended for security)
# Set in your shell: export OLLAMA_API_KEY="your_api_key_here"
# Then just use: use_api=True (no need to pass api_key)

rag_api_env = RAGPipeline(
    use_api=True  # Will auto-load from OLLAMA_API_KEY env variable
)

# Build index
rag_api.build_index()

# Ask a question
result_api = rag_api.answer_question("What was Tesla's revenue for 2023?")
print("\nAnswer:", result_api["answer"])
print("Sources:", result_api["sources"])


# ============================================================================
# EXAMPLE 3: Using in Streamlit App
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Streamlit App (app.py)")
print("=" * 80)

print("""
The Streamlit app supports both modes via toggle:

1. In sidebar, toggle "Use Ollama API" ON/OFF
2. If ON: Enter your API key in the password field
3. If OFF: Uses local Ollama (must be running)

The app automatically handles both modes!
""")


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
print("\n" + "=" * 80)
print("ENVIRONMENT SETUP")
print("=" * 80)

setup_local = """
LOCAL OLLAMA:
  1. Install: https://ollama.ai
  2. Run: ollama run mistral
  3. Then use the RAG system
"""

setup_api = """
OLLAMA API:
  1. Get your API key from your Ollama API provider
  2. Option A - Pass directly:
     RAGPipeline(use_api=True, ollama_api_key="your_key")
  
  2. Option B - Use environment variable:
     export OLLAMA_API_KEY="your_key"
     RAGPipeline(use_api=True)  # Auto-loads from env
"""

print(setup_local)
print(setup_api)


# ============================================================================
# SECURITY BEST PRACTICES
# ============================================================================
print("\n" + "=" * 80)
print("SECURITY BEST PRACTICES")
print("=" * 80)

security_tips = """
✅ DO:
  • Store API key in environment variable
  • Use .env file (add to .gitignore)
  • Never hardcode API key in source code
  • Use st.secrets in Streamlit Cloud

❌ DON'T:
  • Commit API key to git
  • Log API key to console
  • Share API key in code
  • Put API key in public repositories

USING .env FILE:
  1. Create .env file in project root
  2. Add: OLLAMA_API_KEY=your_key_here
  3. Add .env to .gitignore
  4. Use: os.getenv("OLLAMA_API_KEY")

USING STREAMLIT SECRETS:
  1. In Streamlit Cloud: Settings > Secrets
  2. Add: OLLAMA_API_KEY = "your_key"
  3. Access: st.secrets["OLLAMA_API_KEY"]
"""

print(security_tips)

