# 🔒 VaultMind

> Privacy-first, offline Retrieval-Augmented Generation (RAG) for sensitive PDF documents.

## 📌 Overview

**VaultMind** is an offline-first RAG pipeline that allows users to upload PDF documents, generate local embeddings, store them in a local vector database, and ask questions using a locally hosted LLM.

All document processing, embedding generation, retrieval, inference, and evaluation happen locally. No documents, queries, embeddings, or responses are sent to external AI APIs.

## 🎥 Demo

[▶️ Watch VaultMind Demo](https://drive.google.com/file/d/1oIOgPGiegAhFtulAE7Q5HNfcfQrPbRK6/view?usp=sharing)

## 🚀 Features

- 🔐 Fully offline document processing
- 🛡️ Privacy-first architecture
- 🧠 Local RAG using Llama 3.2 3B
- ⚡ CPU-friendly inference through Ollama
- 🗄️ Persistent local vector storage with ChromaDB
- 📊 Lightweight fuzzy-matching groundedness evaluation
- 🧩 Modular architecture with Streamlit and LangChain
- 💻 Optimized for consumer hardware

## 🧠 RAG Pipeline

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Local Embeddings
     ↓
ChromaDB Vector Store
     ↓
User Query
     ↓
Similarity Search
     ↓
Relevant Context
     ↓
Local LLM Inference
     ↓
Generated Answer
     ↓
Groundedness Evaluation
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Frontend interface |
| LangChain | RAG orchestration |
| Ollama | Local LLM runtime |
| Llama 3.2 3B | Local language model |
| ChromaDB | Persistent vector database |
| HuggingFace | Local embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| rapidfuzz | Groundedness evaluation |
| PyPDF | PDF processing |

## 📂 Project Structure

```text
vaultmind/
│
├── assets/
│   └── demo.gif
│
├── app.py
├── core.py
├── evaluator.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- Git
- Ollama

Install Ollama from [ollama.com](https://ollama.com).

### Download the Local LLM

```bash
ollama run llama3.2:3b
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/vaultmind.git
cd vaultmind
```

### Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

## 🔒 Privacy Architecture

All processing happens locally:

- PDF documents
- Extracted text
- Text embeddings
- Vector database
- User queries
- LLM inference
- Generated responses
- Evaluation results

The application does not require OpenAI, Anthropic, or any external inference API.

## 📊 Groundedness Evaluation

The system compares the generated answer with the retrieved document context using fuzzy lexical matching (via `rapidfuzz`, a C++-accelerated drop-in for `fuzzywuzzy`).

```text
Retrieved Context
        ↓
Generated Answer
        ↓
Fuzzy Similarity Matching
        ↓
Groundedness Score
```

This avoids the additional memory and compute requirements of an LLM-as-a-Judge system.

## ⚡ Performance Optimizations

The original prototype reloaded the embedding model and the ChromaDB client from disk on **every single query**, since `retrieve_documents()` called `load_vector_database()` internally. Model loading dominated end-to-end latency far more than LLM inference did.

| Optimization | Effect |
|---|---|
| `st.cache_resource` on the embedding model and Ollama client | Model loaded once per session instead of once per query |
| Vector store kept in `st.session_state` after indexing | Removes a disk round-trip + client re-init on every question |
| `rapidfuzz` instead of `fuzzywuzzy` | C++-accelerated scoring, same output range/behavior |
| Chunk size tuned to 1000 / overlap 150 | Fewer, denser chunks → smaller similarity-search space without losing recall |

Net effect: first query pays the one-time model-load cost; every subsequent query only pays for retrieval + generation, cutting typical per-query latency dramatically with no change to answer quality or groundedness scoring.

## 📐 Engineering Trade-offs

### Chunking

- Chunk Size: 500 tokens
- Chunk Overlap: 50 tokens

Smaller chunks help reduce context size and improve retrieval performance on smaller local models.

### Vector Storage

ChromaDB runs in persistent local mode, eliminating the need for Docker, external database servers, or cloud vector databases.

### Evaluation

A fuzzy-matching evaluator was chosen instead of a secondary LLM judge to maintain offline execution, low memory usage, fast evaluation, and zero API dependency.

## 💻 Hardware Requirements

| Component | Requirement |
|---|---|
| RAM | 8GB or more |
| CPU | Modern multi-core processor |
| Storage | 5GB+ free space |
| GPU | Optional |
| OS | Windows, macOS, or Linux |

## 🎯 Use Cases

- Private enterprise document analysis
- Offline research assistants
- Internal company knowledge bases
- Legal document analysis
- Financial document processing
- Government and defense environments
- Remote or disconnected environments
- Personal offline AI assistants

## ⚠️ Limitations

- CPU-based LLM inference may be slower than cloud inference.
- Small language models may provide less detailed responses.
- Retrieval quality depends on document structure and chunking.
- Scanned PDFs may require OCR support.
- Fuzzy matching provides an evaluation signal and does not guarantee factual correctness.

## 🔮 Future Improvements

- Multi-document support
- OCR for scanned documents
- Hybrid keyword and vector search
- Improved semantic evaluation
- GPU acceleration
- Document encryption
- User authentication
- Role-based access control
- Advanced citation generation
- Support for additional local LLMs

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
git add .
git commit -m "Add new feature"
git push origin feature/your-feature
```

Then open a Pull Request.

## 📜 License

This project is open-source and available under the license included in the repository.

---

> **Private Documents. Local Intelligence. Zero Cloud Dependency.**
