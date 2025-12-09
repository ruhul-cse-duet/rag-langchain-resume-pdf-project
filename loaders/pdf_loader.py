# loaders/pdf_loader.py
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdfs_from_directory(directory: str) -> List[Document]:
    """Load all PDF files from a directory into LangChain Documents."""
    documents: List[Document] = []
    for fname in os.listdir(directory):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(directory, fname)
        loader = PyPDFLoader(fpath)
        docs = loader.load()
        # Add filename into metadata for later citation
        for d in docs:
            d.metadata.setdefault("source", fname)
        documents.extend(docs)
    return documents
