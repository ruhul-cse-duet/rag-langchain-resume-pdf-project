# RAG PDF Q&A (Local LLM via Ollama)

A Streamlit + LangChain app that ingests PDF resumes, builds a local Chroma vector store with SentenceTransformers embeddings, and answers questions using a **local Ollama model** (default `llama3`). Everything runs locally; ensure the Ollama daemon is running.

## 🚀 Features

- PDF upload (multi-file) and ingestion
- Chunking with configurable size/overlap
- Embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Vector store persisted with Chroma
- Local LLM via Ollama (default: `llama3`)
- Source snippets displayed with page numbers
- Docker and GitHub Actions support

## 📋 Prerequisites

- Python 3.10+ (Dockerfile uses 3.10)
- Ollama installed and running locally (`ollama --version`)
- Docker (optional)  
- GPU optional; pull a GPU-friendly model in Ollama if available

## 🛠️ Installation

### Quick Start (local)

1) Create venv & install deps  
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Ensure an Ollama model is pulled (default `llama3`):
```bash
ollama pull llama3
```

3) Run Streamlit  
```bash
streamlit run app.py
```
Open http://localhost:8501.

> Quick helpers: `run.bat` (Windows) or `run.sh` (Unix). Make sure Ollama is running.

## 🐳 Docker

Build:  
```bash
docker build -t utils-resume-chatbot .
```
Run:  
```bash
docker run -p 8501:8501 utils-resume-chatbot
```
Persist Chroma DB:  
```bash
docker run -p 8501:8501 -v $(pwd)/data/vectordb:/app/data/vectordb utils-resume-chatbot
```
GPU models (if you use Ollama GPU models):  
```bash
docker run --gpus all -p 8501:8501 utils-resume-chatbot
```

## 📖 Usage

1) Upload one or more PDFs (sidebar)  
2) Click **“Ingest PDFs & Build Vector DB”**  
3) Ask a question in the main panel and click **“Get Answer”**  
4) View retrieved sources with page numbers

### Example Questions

- "What are the key skills mentioned in this resume?"
- "What is the candidate's work experience?"
- "What educational background does the candidate have?"
- "What are the main responsibilities mentioned?"
- "What technologies or tools are mentioned?"

## 🏗️ Architecture (code refs)

- `loaders/pdf_loader.py` — loads PDFs with `PyPDFLoader`, adds filename metadata
- `utils/text_splitter.py` — `RecursiveCharacterTextSplitter`
- `embeddings/embedding.py` — cached `HuggingFaceEmbeddings` (MiniLM)
- `vectorstore/store.py` — build/load Chroma persistent store
- `llm/local_llm.py` — Ollama client using `config.py`
- `rag/chain.py` — simple RAG pipeline (retriever → prompt → LLM)
- `app.py` — Streamlit UI wiring everything together

## 📁 Project Structure

```
.
├── app.py                  # Streamlit UI
├── config.py               # Paths + hyperparams + model path
├── loaders/pdf_loader.py   # PDF ingestion
├── utils/text_splitter.py  # Chunking
├── embeddings/embedding.py # Embedding model
├── vectorstore/store.py    # Chroma persistence
├── llm/local_llm.py        # LlamaCpp wrapper
├── rag/chain.py            # RAG chain
├── data/                   # uploads + vectordb
├── storage/faiss_index/    # legacy FAISS artifacts (if any)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

Key knobs in `config.py` / UI:
- `CHUNK_SIZE` / `CHUNK_OVERLAP`
- `TOP_K` (retrieval)
- `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `TOP_K`, chunk sizes

## 🐛 Troubleshooting

### Ollama not reachable
- Make sure the Ollama service is running: `ollama list` should work.
- Update `OLLAMA_BASE_URL` in `config.py` if running remotely or in Docker.

### Retrieval quality
- Tune chunk size/overlap and `TOP_K` in the sidebar.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📦 GitHub Setup

### Initializing Git Repository

1. **Initialize git repository**
   ```bash
   git init
   ```

2. **Add all files**
   ```bash
   git add .
   ```

3. **Create initial commit**
   ```bash
   git commit -m "Initial commit: RAG Resume Chatbot with Streamlit"
   ```

4. **Add remote repository** (replace with your GitHub repo URL)
   ```bash
   git remote add origin https://github.com/yourusername/rag-resume-chatbot.git
   ```

5. **Push to GitHub**
   ```bash
   git branch -M main
   git push -u origin main
   ```

### GitHub Repository Structure

Make sure your GitHub repository includes:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `Dockerfile` - Docker configuration
- ✅ `docker-compose.yml` - Docker Compose setup
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules
- ✅ `.dockerignore` - Docker ignore rules
- ✅ `LICENSE` - License file

## 📧 Contact
[Md Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/); \
Email: ruhul.cse.duet@gmail.com

For questions or issues, please open an issue on GitHub.

---

The first run downloads embedding weights; keep an internet connection for that step.

