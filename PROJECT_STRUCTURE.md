# Project Structure

```
RAG Langchain chatbot PDF CV/
├── app.py                    # Streamlit UI (ingest + Q&A)
├── config.py                 # Paths and model/config settings
├── loaders/
│   └── pdf_loader.py         # PDF loader (PyPDFLoader)
├── utils/
│   └── text_splitter.py      # RecursiveCharacterTextSplitter
├── embeddings/
│   └── embedding.py          # HuggingFaceEmbeddings (cached)
├── vectorstore/
│   └── store.py              # Chroma build/load
├── llm/
│   └── local_llm.py          # Ollama client
├── rag/
│   └── chain.py              # RAG chain wiring
├── data/
│   ├── uploads/              # Uploaded PDFs
│   └── vectordb/             # Chroma persistence
├── storage/faiss_index/      # Legacy FAISS artifacts (if any)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── code/                     # Original notebook
    └── rag-langchain-chatbot-google-flan-t5-base.ipynb
```

## Notes
- Configure Ollama model in `config.py` (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`).
- Vector DB persists to `data/vectordb/` so ingestion is reused between runs.

