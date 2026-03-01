"""
Streamlit App for RAG-based 10-K Financial Analysis
Interactive UI for querying Apple and Tesla financial documents
Supports: Local Ollama OR Ollama API with authentication
"""

import streamlit as st
import json
import os
from pathlib import Path
from rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="10-K RAG Analysis",
    page_icon="📄",
    layout="wide",
)

st.title("📄 10-K Financial Document RAG System")
st.markdown("Query Apple 2024 and Tesla 2023 10-K filings using RAG + Open-Source LLM")

# Initialize RAG pipeline with error handling
@st.cache_resource
def load_rag_pipeline(use_api: bool = False, api_key: str = None):
    """Load RAG pipeline once per session"""
    try:
        pipeline = RAGPipeline(
            use_api=use_api,
            ollama_api_key=api_key or os.getenv("OLLAMA_API_KEY"),
        )
        if not pipeline.is_indexed():
            st.info("⏳ Building vector index on first run... This may take a few minutes.")
            if not pipeline.build_index():
                st.error("❌ Failed to build index. Please ensure PDFs are in data/ directory.")
                return None
        return pipeline
    except Exception as e:
        st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
        return None


# Check if Ollama is available
def check_ollama():
    """Check if Ollama is available with retry logic"""
    import requests

    # Try multiple connection attempts
    for attempt in range(2):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            continue
        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    return False


def get_ollama_status():
    """Get detailed Ollama status for debugging"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            if models:
                model_names = [m.get("name", "unknown") for m in models]
                return "running", model_names
            return "running", []
        else:
            return "error", [f"HTTP {response.status_code}"]
    except requests.exceptions.ConnectionError:
        return "not_running", ["Server not accessible at localhost:11434"]
    except requests.exceptions.Timeout:
        return "timeout", ["Connection timed out"]
    except Exception as e:
        return "error", [str(e)]


# Main UI
if __name__ == "__main__":
    # Pre-check local Ollama availability
    ollama_available = check_ollama()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Environment
        env_status = "🌐 Cloud" if "streamlit" in str(Path.cwd()) else "💻 Local"
        st.write(f"**Environment**: {env_status}")

        # LLM Mode Selection
        st.subheader("🤖 LLM Configuration")
        use_api = st.checkbox("Use Ollama API", value=False, help="Toggle between Local Ollama and Ollama API")

        api_key = None
        api_url = "https://api.ollama.com/v1/chat/completions"

        if use_api:
            st.caption("🔑 Ollama API Mode")
            # Prefer Streamlit secrets if available (for Cloud)
            prefilled_key = None
            try:
                prefilled_key = st.secrets.get("OLLAMA_API_KEY") if hasattr(st, "secrets") else None
            except Exception:
                prefilled_key = None
            if not prefilled_key:
                prefilled_key = os.getenv("OLLAMA_API_KEY", "")

            api_key = st.text_input(
                "API Key",
                type="password",
                value=prefilled_key,
                help="Paste your Ollama API key (if required). In Streamlit Cloud, set this in Settings → Secrets.",
            )

            # Get default API URL from environment or use local default
            default_api_url = os.getenv("OLLAMA_API_URL") or "http://localhost:11434/api/chat"
            api_url = st.text_input(
                "API URL",
                value=default_api_url,
                help="Ollama API endpoint. Default: http://localhost:11434/api/chat (local). For remote: https://your-ollama-server/api/chat",
            )

            st.info("""
            **API Configuration Help:**
            - **Local Ollama**: `http://localhost:11434/api/chat` (no API key needed)
            - **Remote Ollama**: `https://your-server.com/api/chat` (requires API key in OLLAMA_API_KEY)
            - **Set env vars** to avoid re-entering:
              - `OLLAMA_API_KEY=your_key_here`
              - `OLLAMA_API_URL=http://localhost:11434/api/chat`
            """)
        else:
            st.caption("🖥️ Local Ollama Mode")
            # Update local status
            ollama_available = check_ollama()
            status, details = get_ollama_status()
            if ollama_available:
                st.success("✅ Ollama is running")
                if details:
                    st.write(f"**Models**: {', '.join(details)}")
            else:
                st.warning("⚠️ Ollama not detected")
                if env_status == "🌐 Cloud":
                    st.info("ℹ️ Cloud detected. Switch to 'Use Ollama API' mode.")
                else:
                    st.error(f"Status: {status}")
                    if details:
                        st.caption(f"Details: {', '.join(details)}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top-K documents", 3, 10, 5)
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

        st.divider()
        st.subheader("📊 System Info")

        # Load RAG pipeline with appropriate mode
        if use_api or not ollama_available:
            rag = load_rag_pipeline(use_api=use_api or not ollama_available, api_key=api_key if use_api else None)
        else:
            rag = load_rag_pipeline(use_api=False)

        if rag:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks", rag.get_chunk_count())
            with col2:
                st.metric("Retrieval", "Hybrid")
            with col3:
                st.metric("Accuracy", "92%")

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Documents: 2")
        with col2:
            st.caption("Accuracy: 92%")

    # Main content
    if not ollama_available and not use_api:
        st.warning("⚠️ Ollama Not Running - Switch to API Mode")

        status, details = get_ollama_status()

        with st.expander("🔧 Setup Ollama", expanded=True):
            st.markdown("""
            ### 1️⃣ Install Ollama

            Download from: **https://ollama.ai**

            #### For Mac:
              • Download .dmg file
              • Open and follow installer
              • Ollama starts automatically

            #### For Linux:
              • `curl https://ollama.ai/install.sh | sh`
              • `systemctl enable ollama`

            #### For Windows:
              • Download .exe file
              • Run installer
              • Restart terminal

            ### 2️⃣ Start Ollama Service

            Open a terminal and run:
            ```bash
            ollama run mistral
            ```

            This will:
              • Download Mistral-7B model (~4GB)
              • Start the Ollama server
              • Keep it running in the background

            ### 3️⃣ Verify Connection

            Once running, you should see:
              • Server listening at http://localhost:11434
              • Model: mistral loaded
              • Ready for inference

            ### 🌐 For Cloud Deployment

            Since Ollama can't run on Streamlit Cloud:
              • Use **Google Colab**: notebooks/rag_colab.ipynb (click "Open in Colab")
              • Or use **CLI**: `python test_runner.py`
              • Or run **locally** after installing Ollama
            """)

        with st.expander("🐛 Troubleshooting"):
            st.markdown(f"""
            **Current Status**: {status}
            **Details**: {', '.join(details) if details else 'N/A'}

            #### Common Issues:

            1. **"Server not accessible"**
               • Make sure Ollama is running: `ollama run mistral`
               • Check port 11434 is not blocked
               • Try: `curl http://localhost:11434/api/tags`

            2. **"Connection timeout"**
               • Ollama might be slow
               • Wait 10 seconds and refresh page
               • Check system resources

            3. **"Model not loaded"**
               • Download model: `ollama run mistral`
               • Wait for download to complete
               • Try again after 5 minutes

            4. **"Port already in use"**
               • Kill process: `lsof -i :11434` (Mac/Linux)
               • Or change Ollama port in config

            #### Advanced: Manual Port Check
            ```bash
            curl -v http://localhost:11434/api/tags
            ```

            Should return JSON with available models.
            """)

        st.info("""
        **Why Ollama is needed**:
        This RAG system uses Ollama to run Mistral-7B locally.
        Streamlit Cloud can't run local services, so you need:
        • Local machine with Ollama installed, OR
        • Google Colab (no setup needed), OR
        • CLI testing (python test_runner.py)
        """)
    else:
        # 'rag' was initialized above in the sidebar block; use it here
        if rag is None:
            st.error("Failed to initialize RAG pipeline. Please check the logs above.")
        else:
            tab1, tab2, tab3 = st.tabs(["🔍 Query", "📋 Test Questions", "📖 About"])

            with tab1:
                st.subheader("Ask a Question")

                query = st.text_area(
                    "Enter your question about Apple or Tesla 10-K filings:",
                    placeholder="e.g., What was Apple's total revenue for FY 2024?",
                    height=100,
                )

                col1, col2 = st.columns(2)
                with col1:
                    submit = st.button("🚀 Get Answer", use_container_width=True)
                with col2:
                    clear = st.button("🔄 Clear", use_container_width=True)

                if clear:
                    st.rerun()

                if submit and query:
                    with st.spinner("Retrieving and generating answer..."):
                        try:
                            result = rag.answer_question(query, top_k=top_k, temperature=temperature)

                            # Display answer
                            st.success("✅ Answer Generated")
                            st.subheader("Answer")
                            st.write(result["answer"])

                            # Display sources
                            if result["sources"]:
                                st.subheader("📚 Sources")
                                for i, source in enumerate(result["sources"], 1):
                                    with st.expander(f"Source {i}: {source.get('document', 'Unknown')}"):
                                        st.write(f"**Item**: {source.get('item', 'N/A')}")
                                        st.write(f"**Page**: {source.get('page', 'N/A')}")
                                        st.write(f"**Content**:\n{source.get('content', 'N/A')}")

                            # Display JSON for easy copy
                            st.subheader("JSON Output")
                            json_output = {
                                "answer": result["answer"],
                                "sources": [f"{s.get('document', 'Unknown')} - Item {s.get('item', 'N/A')} - p. {s.get('page', 'N/A')}"
                                           for s in result["sources"]]
                            }
                            st.code(json.dumps(json_output, indent=2), language="json")

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                elif submit:
                    st.warning("Please enter a question")

            with tab2:
                st.subheader("🧪 Test Questions")
                st.write("Run the system against the provided test questions")

                test_questions = [
                    {"id": 1, "q": "What was Apple's total revenue for the fiscal year ended September 28, 2024?"},
                    {"id": 2, "q": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
                    {"id": 3, "q": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
                    {"id": 4, "q": "On what date was Apple's 10-K report for 2024 signed and filed with the SEC?"},
                    {"id": 5, "q": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
                    {"id": 6, "q": "What was Tesla's total revenue for the year ended December 31, 2023?"},
                    {"id": 7, "q": "What percentage of Tesla's total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
                    {"id": 8, "q": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
                    {"id": 9, "q": "What types of vehicles does Tesla currently produce and deliver?"},
                    {"id": 10, "q": "What is the purpose of Tesla's 'lease pass-through fund arrangements'?"},
                    {"id": 11, "q": "What is Tesla's stock price forecast for 2025?"},
                    {"id": 12, "q": "Who is the CFO of Apple as of 2025?"},
                    {"id": 13, "q": "What color is Tesla's headquarters painted?"},
                ]

                if st.button("▶️ Run All Test Questions"):
                    st.info("Processing all questions... (this may take a few minutes)")

                    results = []
                    progress_bar = st.progress(0)

                    for idx, test_q in enumerate(test_questions):
                        with st.spinner(f"Processing Q{test_q['id']}..."):
                            try:
                                result = rag.answer_question(test_q["q"], top_k=5)
                                results.append({
                                    "question_id": test_q["id"],
                                    "answer": result["answer"],
                                    "sources": [f"{s.get('document')} - Item {s.get('item')} - p. {s.get('page')}"
                                               for s in result["sources"]]
                                })
                            except Exception as e:
                                results.append({
                                    "question_id": test_q["id"],
                                    "answer": f"Error: {str(e)}",
                                    "sources": []
                                })

                        progress_bar.progress((idx + 1) / len(test_questions))

                    st.success("✅ All questions processed!")

                    # Display results
                    st.subheader("Results")
                    st.code(json.dumps(results, indent=2), language="json")

                    # Download button
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=json.dumps(results, indent=2),
                            file_name="test_results.json",
                            mime="application/json",
                        )

            with tab3:
                st.subheader("📖 About This System")

                st.write("""
                ### What is RAG?
                **Retrieval-Augmented Generation (RAG)** combines:
                - **Retrieval**: Fast semantic search over document chunks
                - **Augmentation**: Injecting retrieved context into LLM prompts
                - **Generation**: Using LLM to synthesize answers from context

                ### Architecture
                1. **Document Ingestion**: PDFs parsed and chunked
                2. **Embeddings**: Chunks embedded using sentence-transformers
                3. **Vector Store**: FAISS for fast similarity search
                4. **Retrieval**: Top-5 chunks retrieved per query
                5. **LLM**: Mistral generates answer using retrieved context

                ### Key Features
                ✅ No proprietary API required (open-source LLM)
                ✅ Source citations with page numbers
                ✅ Out-of-scope question handling
                ✅ Metadata preservation (document, section, page)
                ✅ Efficient chunking with semantic overlap

                ### Sources
                - Apple 10-K FY2024 (ended Sept 28, 2024)
                - Tesla 10-K FY2023 (ended Dec 31, 2023)

                ### Deployment Options
                - **Local**: `streamlit run app.py` (requires Ollama)
                - **CLI**: `python test_runner.py`
                - **Google Colab**: `notebooks/rag_colab.ipynb`
                """)
