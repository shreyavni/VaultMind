import os
import tempfile
import time

import streamlit as st

from core import create_vector_database, retrieve_documents, generate_answer
from evaluator import evaluate_answer


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="VaultMind",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# Styling
# =====================================================

st.markdown(
    """
    <style>

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #9ca3af;
        margin-bottom: 14px;
    }

    .badge-row span {
        display: inline-block;
        font-size: 12px;
        font-weight: 600;
        color: #d1d5db;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 999px;
        padding: 4px 12px;
        margin-right: 8px;
        margin-top: 4px;
    }

    .info-card {
        padding: 22px 26px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #9ca3ag;
    }

    .doc-card {
        padding: 14px 16px;
        border-radius: 10px;
        background: black;
        border: 1px solid #bbf7d0;
        margin-bottom: 12px;
    }

    .status-pill {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 999px;
    }

    .status-high   { background:#dcfce7; color:#166534; }
    .status-medium { background:#fef9c3; color:#854d0e; }
    .status-low    { background:#fee2e2; color:#991b1b; }

    .latency-tag {
        font-size: 12px;
        color: #6b7280;
        margin-left: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Session State
# =====================================================

defaults = {
    "document_processed": False,
    "document_name": None,
    "chunk_count": 0,
    "process_time": 0.0,
    "vector_store": None,
    "messages": [],  # list of {role, content, score, status, latency, context, documents}
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def status_class(status: str) -> str:
    return {
        "Highly Grounded": "status-high",
        "Moderately Grounded": "status-medium",
        "Weakly Grounded": "status-low",
        "Potentially Ungrounded": "status-low",
    }.get(status, "status-medium")


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.markdown("### 📄 Document Upload")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    process_document = st.button("⚡ Process Document", use_container_width=True, type="primary")

    st.divider()

    if st.session_state.document_processed:
        st.markdown(
            f"""
            <div class="doc-card">
            <b>📎 {st.session_state.document_name}</b><br>
            {st.session_state.chunk_count} chunks · indexed in {st.session_state.process_time:.1f}s
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.rerun()
        st.divider()

    st.info(
        "🔐 **Privacy Mode Active**\n\n"
        "All documents, embeddings, queries, and AI responses are "
        "processed locally. No external API calls are made."
    )

    st.caption("Built with Streamlit · LangChain · Ollama · ChromaDB")


# =====================================================
# Header
# =====================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔒 VaultMind </div>
        <div class="hero-subtitle">Offline-first Retrieval-Augmented Generation for sensitive documents</div>
        <div class="badge-row">
            <span>100% Local Inference</span>
            <span>Zero API Calls</span>
            <span>Groundedness Scoring</span>
            <span>ChromaDB Vector Store</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Process PDF
# =====================================================

if uploaded_file and process_document:
    progress = st.progress(0, text="Reading PDF...")
    start_time = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        progress.progress(30, text="Chunking text...")
        progress.progress(55, text="Generating local embeddings...")

        vector_store, chunk_count = create_vector_database(temp_pdf_path)

        progress.progress(90, text="Indexing in ChromaDB...")
        os.unlink(temp_pdf_path)

        elapsed = time.perf_counter() - start_time

        st.session_state.vector_store = vector_store
        st.session_state.document_processed = True
        st.session_state.document_name = uploaded_file.name
        st.session_state.chunk_count = chunk_count
        st.session_state.process_time = elapsed
        st.session_state.messages = []

        progress.progress(100, text="Done")
        time.sleep(0.2)
        progress.empty()

        st.toast(f"Indexed {chunk_count} chunks in {elapsed:.1f}s", icon="✅")
        st.rerun()

    except Exception as error:
        progress.empty()
        st.error(f"Error processing document: {error}")


# =====================================================
# Main Interface
# =====================================================

if not st.session_state.document_processed:

    st.markdown(
        """
        <div class="info-card">

        #### 🚀 Get started

        Upload a sensitive PDF from the sidebar to build a completely
        local knowledge base, then ask questions about it below.

        **Pipeline:** PDF → Chunking → Local Embeddings → ChromaDB →
        Similarity Search → Local Llama Inference → Groundedness Evaluation

        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.subheader("💬 Ask Questions About Your Document")

    # ---- Render chat history ----
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                pill = status_class(message["status"])
                st.markdown(
                    f"""
                    <span class="status-pill {pill}">{message['status']} · {message['score']}%</span>
                    <span class="latency-tag">⏱ {message['latency']:.1f}s</span>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("📚 View retrieved context"):
                    for index, document in enumerate(message["documents"], start=1):
                        st.markdown(f"**Chunk {index}**")
                        st.write(document.page_content)
                        if document.metadata:
                            st.caption(f"Source: {document.metadata}")

    # ---- New query ----
    query = st.chat_input("Ask a question about your document...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Searching local knowledge base..."):
                try:
                    documents = retrieve_documents(st.session_state.vector_store, query, k=4)
                    result = generate_answer(query, documents)
                    evaluation = evaluate_answer(result.answer, result.context)

                    st.write(result.answer)

                    pill = status_class(evaluation["status"])
                    st.markdown(
                        f"""
                        <span class="status-pill {pill}">{evaluation['status']} · {evaluation['score']}%</span>
                        <span class="latency-tag">⏱ {result.latency_seconds:.1f}s</span>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("📚 View retrieved context"):
                        for index, document in enumerate(documents, start=1):
                            st.markdown(f"**Chunk {index}**")
                            st.write(document.page_content)
                            if document.metadata:
                                st.caption(f"Source: {document.metadata}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "score": evaluation["score"],
                            "status": evaluation["status"],
                            "latency": result.latency_seconds,
                            "documents": documents,
                        }
                    )

                except Exception as error:
                    st.error(f"Error generating response: {error}")