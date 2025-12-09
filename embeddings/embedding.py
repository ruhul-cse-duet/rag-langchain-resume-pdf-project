# embeddings/embedding.py
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a singleton HuggingFaceEmbeddings instance.

    Uses sentence-transformers/all-MiniLM-L6-v2 (fast, free).
    """
    model_kwargs = {"device": "cuda" if _has_gpu() else "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
