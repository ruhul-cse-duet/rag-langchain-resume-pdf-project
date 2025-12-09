# config.py
import os

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
VECTORDB_DIR = os.path.join(DATA_DIR, "vectordb")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)

# -------- Embeddings --------
# Small, fast free embedding model (CPU/GPU via PyTorch)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 4

# -------- LLM (Ollama) --------
# Ensure Ollama is running locally: https://ollama.com
# List models: `ollama list`; pull: `ollama pull llama3`
OLLAMA_MODEL = "tinyllama"
OLLAMA_BASE_URL = "http://localhost:11434"
