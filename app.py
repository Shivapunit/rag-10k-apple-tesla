"""
Streamlit App for RAG-based 10-K Financial Analysis
Interactive UI for querying Apple and Tesla financial documents
"""

import streamlit as st
import json
from pathlib import Path
from rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="10-K RAG Analysis",
    page_icon="📄",
    layout="wide"
)

st.title("📄 10-K Financial Document RAG System")
st.markdown("Query Apple 2024 and Tesla 2023 10-K filings using RAG + Open-Source LLM")

# Initialize RAG pipeline with error handling
@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline once per session"""
    try:
        pipeline = RAGPipeline()
        if not pipeline.is_indexed():
            st.info("⏳ Building vector index on first run... This may take a few minutes.")
            if not pipeline.build_index():
                st.error("❌ Failed to build index. Please ensure PDFs are in data/ directory.")
                return None
        return pipeline
    except Exception as e:
        st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
        st.write("Make sure Ollama is running: `ollama run mistral`")
        return None

# Check if Ollama is available
def check_ollama():
    """Check if Ollama is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

# Main UI
if __name__ == "__main__":
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check environment
        env_status = "🌐 Cloud" if "streamlit" in str(Path.cwd()) else "💻 Local"
        st.write(f"**Environment**: {env_status}")

        ollama_available = check_ollama()
        if ollama_available:
            st.success("✅ Ollama is running")
        else:
            st.warning("⚠️ Ollama not detected")
            if env_status == "🌐 Cloud":
                st.info("ℹ️ Streamlit Cloud doesn't support Ollama. Please use local deployment or Google Colab.")

        if ollama_available:
            top_k = st.slider("Top-K documents to retrieve", 3, 10, 5)
            temperature = st.slider("LLM Temperature (0=deterministic)", 0.0, 1.0, 0.3)

        st.divider()
        st.subheader("📊 System Info")
        if ollama_available:
            rag = load_rag_pipeline()
            if rag:
                st.write(f"**Embedded Chunks**: {rag.get_chunk_count()}")
        st.write(f"**Documents**: Apple 10-K 2024, Tesla 10-K 2023")

    # Main content
    if not ollama_available:
        st.error("❌ Ollama LLM Server Not Found")

        with st.expander("📖 How to Use This App", expanded=True):
            st.markdown("""
            ### Option 1: Local Deployment (Full Features) ✅
            
            1. **Install Ollama**: https://ollama.ai
            2. **Start LLM**: `ollama run mistral`
            3. **Run App**: `streamlit run app.py`
            4. Open: http://localhost:8501
            
            ### Option 2: Google Colab (Cloud, No Setup) ✅
            
            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/rag-10k-apple-tesla/blob/main/notebooks/rag_colab.ipynb)
            
            1. Click badge above
            2. Run all cells
            3. Test questions answered with full accuracy
            
            ### Option 3: Command Line (Works Anywhere) ✅
            
            ```bash
            python test_runner.py
            ```
            
            Generates `test_results.json` with all 13 questions answered.
            """)

        st.info("""
        **Why Ollama is needed**: Streamlit Cloud can't run local LLM servers. 
        Use local deployment or cloud alternatives above.
        """)
    else:
        rag = load_rag_pipeline()

        if rag is None:
            st.error("Failed to initialize RAG pipeline. Please check the logs above.")
        else:
            tab1, tab2, tab3 = st.tabs(["🔍 Query", "📋 Test Questions", "📖 About"])

            with tab1:
                st.subheader("Ask a Question")

                query = st.text_area(
                    "Enter your question about Apple or Tesla 10-K filings:",
                    placeholder="e.g., What was Apple's total revenue for FY 2024?",
                    height=100
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
                            mime="application/json"
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

