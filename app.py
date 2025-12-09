# app.py
import os
from typing import List

import streamlit as st

from config import (
    UPLOAD_DIR,
    VECTORDB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)
from loaders.pdf_loader import load_pdfs_from_directory
from utils.text_splitter import split_documents
from vectorstore.store import build_or_load_vectorstore
from llm.local_llm import get_local_llm
from rag.chain import build_rag_chain


def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    """Save uploaded Streamlit files into UPLOAD_DIR."""
    for f in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, f.name)
        with open(file_path, "wb") as out:
            out.write(f.getbuffer())


def main():
    st.set_page_config(page_title="RAG PDF Q&A (Ollama)", layout="wide")

    # --- Hero ---
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(135deg, #0d6efd33, #6610f233);
            padding: 28px 32px;
            border-radius: 14px;
            border: 1px solid #e6e6f0;
        }
        .pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: #0d6efd22;
            color: #0d6efd;
            font-weight: 600;
            font-size: 13px;
            margin-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="pill">RAG • LangChain • Chroma • Ollama</div>
            <h1 style="margin-top:8px; margin-bottom:6px;">📄🔍 PDF Resume Q&A</h1>
            <p style="color:#4a4a4a; font-size:16px; margin-bottom:0;">
                Upload resumes, build embeddings locally, and ask questions with your Ollama model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --- Upload + Settings ---
    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Drop one or more PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are saved locally and ingested into Chroma.",
        )
        ingest_clicked = st.button("🚀 Ingest & Build Vector DB", type="primary", use_container_width=True)

    with right:
        st.subheader("⚙️ Parameters")
        chunk_size = st.slider(
            "Chunk size",
            min_value=200,
            max_value=2500,
            value=CHUNK_SIZE,
            step=100,
        )
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=1000,
            value=CHUNK_OVERLAP,
            step=50,
        )
        top_k = st.slider(
            "Top-K retrieved chunks",
            min_value=1,
            max_value=10,
            value=TOP_K,
            step=1,
        )
        st.info(
            "Uses local Ollama (see `config.py`) and SentenceTransformers embeddings. "
            "All data stays on your machine."
        )

    if ingest_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before ingesting.")
        else:
            with st.spinner("Saving files..."):
                save_uploaded_files(uploaded_files)

            with st.spinner("Loading & splitting PDFs..."):
                docs = load_pdfs_from_directory(UPLOAD_DIR)
                chunks = split_documents(
                    docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            st.success(f"Loaded {len(docs)} pages → {len(chunks)} chunks.")

            with st.spinner("Building vector store (Chroma)..."):
                build_or_load_vectorstore(
                    docs=chunks,
                    persist_directory=VECTORDB_DIR,
                )

            st.success("Vector DB built and persisted successfully! ✅")

    # ---------- Ingestion step ----------
    if ingest_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before ingesting.")
        else:
            with st.spinner("Saving files..."):
                save_uploaded_files(uploaded_files)

            with st.spinner("Loading & splitting PDFs..."):
                docs = load_pdfs_from_directory(UPLOAD_DIR)
                chunks = split_documents(
                    docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            st.write(f"Loaded {len(docs)} pages → {len(chunks)} chunks.")

            with st.spinner("Building vector store (Chroma)..."):
                vs = build_or_load_vectorstore(
                    docs=chunks,
                    persist_directory=VECTORDB_DIR,
                )

            st.success("Vector DB built and persisted successfully! ✅")

    st.markdown("## 💬 Ask questions about your PDFs")

    user_question = st.text_input(
        "Type your question:",
        placeholder="e.g. What is the main conclusion of the report?",
    )

    if st.button("Get Answer", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Loading LLM & vector store..."):
                    llm = get_local_llm()
                    vs = build_or_load_vectorstore(
                        docs=None,
                        persist_directory=VECTORDB_DIR,
                    )
                    rag_chain, retriever = build_rag_chain(
                        llm=llm,
                        vectorstore=vs,
                        top_k=top_k,
                    )

                with st.spinner("Thinking..."):
                    answer = rag_chain.invoke(user_question)
                    st.markdown("### ✅ Answer")
                    st.write(answer)

                    # Show sources (retrieved docs)
                    st.markdown("### 📚 Top Sources")
                    docs = retriever.invoke(user_question)
                    for i, d in enumerate(docs, start=1):
                        st.markdown(f"**Source {i}**")
                        meta = d.metadata or {}
                        source_name = meta.get("source", "Unknown")
                        page = meta.get("page", "N/A")
                        st.caption(f"{source_name} (page {page})")
                        st.write(d.page_content[:700] + ("..." if len(d.page_content) > 700 else ""))

            except FileNotFoundError:
                st.error("Vector DB not found. Please ingest PDFs first.")
            except Exception as e:
                st.error(f"Error while answering: {e}")

    st.markdown("---")
    st.caption(
        "Pipeline: PDFs → text chunks → embeddings (SentenceTransformers) "
        "→ Chroma vector DB → retriever → local LLM (Ollama)."
    )


if __name__ == "__main__":
    main()

# streamlit run app.py