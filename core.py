import os
import shutil
import time
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document


# -----------------------------
# Configuration
# -----------------------------

PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@dataclass
class GenerationResult:
    answer: str
    context: str
    latency_seconds: float


# -----------------------------
# Embedding Model (cached — loaded once per session)
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads the embedding model locally, once per session.
    No external API calls are required.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -----------------------------
# Local LLM (cached — client built once per session)
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_llm() -> OllamaLLM:
    """
    Loads the Llama model client through Ollama, once per session.
    """
    return OllamaLLM(model=LLM_MODEL, temperature=0.1)


# -----------------------------
# Load and Split PDF
# -----------------------------

def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    Loads a PDF and splits it into smaller overlapping chunks.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


# -----------------------------
# Create Vector Database
# -----------------------------

def create_vector_database(pdf_path: str) -> Tuple[Chroma, int]:
    """
    Creates a persistent local ChromaDB vector store from a PDF.
    Returns the live vector store object so it can be cached in
    session_state and reused for every subsequent query (no disk reload).
    """
    documents = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()

    # Remove old database so stale collections don't accumulate
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="vault_mind",
    )

    return vector_store, len(documents)


# -----------------------------
# Retrieve Relevant Documents
# -----------------------------

def retrieve_documents(vector_store: Chroma, query: str, k: int = 4) -> List[Document]:
    """
    Retrieves the most relevant chunks from an already-loaded vector store.
    Accepting `vector_store` as a parameter (instead of reloading from disk)
    is the single biggest latency win in this pipeline.
    """
    return vector_store.similarity_search(query, k=k)


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(query: str, documents: List[Document]) -> GenerationResult:
    """
    Generates an answer using only the retrieved document context,
    and reports wall-clock latency for display in the UI.
    """
    context = "\n\n".join(document.page_content for document in documents)

    prompt = f"""You are a private offline document assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, respond with:
"I could not find this information in the uploaded document."

Do not use outside knowledge.
Do not invent facts.
Keep the answer clear and concise.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    llm = get_llm()

    start = time.perf_counter()
    response = llm.invoke(prompt)
    latency = time.perf_counter() - start

    return GenerationResult(answer=response, context=context, latency_seconds=latency)