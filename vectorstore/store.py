# vectorstore/store.py
import os
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from embeddings.embedding import get_embedding_model
from config import VECTORDB_DIR


def build_or_load_vectorstore(
    docs: Optional[list[Document]] = None,
    persist_directory: str = VECTORDB_DIR,
) -> VectorStore:
    """
    If docs is provided, create/update vectorstore from docs and persist.
    If docs is None, load existing persistent vectorstore.
    """
    embedding = get_embedding_model()

    if docs:
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory,
        )
        vs.persist()
        return vs

    # Load existing
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector DB directory '{persist_directory}' does not exist. "
            "Ingest PDFs first."
        )

    return Chroma(
        embedding_function=embedding, # embedding return after vectorize
        persist_directory=persist_directory,
    )
