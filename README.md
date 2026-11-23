## Local AI Document & Legal Diligence Copilot (RAG Pipeline)

It is a private AI assistant that runs completely offline on your own computer to read, analyze, and answer questions about your sensitive documents without your data ever touching the internet.

---

## Key Features
- **Zero Cloud Leakage:** Fully local inference using Ollama (`llama3.1:8b`) and local vector embeddings (`nomic-embed-text`).
- **Sub-Second Semantic Retrieval:** Powered by an embedded ChromaDB vector store.
- **Dynamic PDF Ingestion:** Upload any multi-page PDF via the UI; the pipeline extracts text, chunks it, and builds vector indices in real-time.
- **Strict Evidence Drawer:** Answers questions citing the exact section and chunk, displaying raw source excerpts to prevent hallucinations.
- **Interactive UI:** Built on Chainlit with real-time token streaming and document attachments.

---

## Architecture

```
[ User PDF Upload / Query ]
            │
            ▼
[ Text Extraction & Chunking (pypdf) ]
            │
            ▼
[ Local Vector Embeddings (nomic-embed-text) ]
            │
            ▼
[ Persistent Vector Database (ChromaDB) ]
            │ (Top-k Relevant Chunks Retrieved)
            ▼
[ Augmented Context Prompt ]
            │
            ▼
[ Local LLM Inference (Llama 3.1 8B via Ollama) ]
            │
            ▼
[ Chainlit UI Streaming Output with Evidence Card ]
```

---

##  Tech Stack
- **LLM Engine:** Ollama (`llama3.1:8b`)
- **Embedding Model:** `nomic-embed-text` (768 dimensions)
- **Vector Database:** ChromaDB
- **Document Parser:** `pypdf`
- **Frontend / Chat Framework:** Chainlit
- **Runtime Environment:** Python 3.10+ / Windows PowerShell

---

##  Quickstart Setup

### 1. Prerequisites
Install Ollama and pull the required models:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate.ps1
pip install chainlit chromadb ollama pypdf
```

### 3. Run the Copilot
```bash
chainlit run app.py -w
```
Open `http://localhost:8000` in your browser.     
