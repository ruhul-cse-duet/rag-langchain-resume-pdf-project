# llm/local_llm.py
from functools import lru_cache

from langchain_community.llms import Ollama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL


@lru_cache(maxsize=1)
def get_local_llm() -> Ollama:
    """
    Return a singleton Ollama LLM client.

    Assumes Ollama is installed and running locally.
    """
    return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
