# 🧠 Offline Hybrid RAG System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red?style=for-the-badge)](https://qdrant.tech)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?style=for-the-badge)](https://ollama.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-orange?style=for-the-badge)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

**A fully offline, private Retrieval-Augmented Generation (RAG) system that answers questions strictly from your own documents — zero internet required.**

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Usage](#-usage) · [Configuration](#-configuration) · [Troubleshooting](#-troubleshooting)

</div>

---

## 📌 Overview

**Offline Hybrid RAG System** is a production-ready, 100% local AI knowledge assistant. It combines semantic vector search with a locally hosted large language model to let you query any collection of documents privately — no external APIs, no data leaving your machine.

Built for researchers, developers, and privacy-conscious users who want the power of AI-driven document Q&A without cloud dependency.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔒 **100% Offline** | No internet required after initial model download |
| 🗂️ **Smart Chunking** | Documents split into 300–500 word semantic chunks |
| 🔍 **Semantic Search** | Cosine similarity search via sentence-transformers embeddings |
| 💾 **Vector Storage** | Local Qdrant vector database (embedded, no server needed) |
| 🤖 **Local LLM** | Ollama-powered generation (Llama2, Mistral, Phi, etc.) |
| ✅ **Strict Grounding** | Refuses to answer if the context doesn't support it |
| 🖥️ **Dual Interface** | Both CLI and Streamlit web UI included |
| 📊 **Source Transparency** | Every answer shows which documents were used |

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE HYBRID RAG SYSTEM                    │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  User Input  │───▶│  RAG Pipeline│───▶│  Answer + Sources│  │
│  │  (Question)  │    │  Orchestrator│    │   (Grounded)     │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────┘  │
│                             │                                   │
│           ┌─────────────────┼─────────────────┐                │
│           ▼                 ▼                 ▼                │
│  ┌────────────────┐ ┌──────────────┐ ┌───────────────────┐    │
│  │   Embedder     │ │  Retriever   │ │   Ollama LLM      │    │
│  │ (Sentence      │ │  (Top-K      │ │ (llama2/mistral/  │    │
│  │  Transformers) │ │  Cosine Sim) │ │  phi3/codellama)  │    │
│  └────────┬───────┘ └──────┬───────┘ └───────────────────┘    │
│           │                │                                   │
│           ▼                ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Qdrant Vector Store (Local / Embedded)      │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Collection: documents                           │   │  │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │   │  │
│  │  │  │ Vector   │ │ Payload  │ │    Metadata       │ │   │  │
│  │  │  │ (384-dim)│ │ (text)   │ │ (source, chunk_id)│ │   │  │
│  │  │  └──────────┘ └──────────┘ └──────────────────┘ │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Indexing Pipeline

```
.txt Files
    │
    ▼
┌─────────────────┐
│ DocumentLoader  │  ── Reads UTF-8 text files
│                 │  ── Splits into 300–500 word chunks
│                 │  ── Assigns unique chunk_id + source
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Embedder     │  ── Loads sentence-transformers model
│ (all-MiniLM-   │  ── Generates 384-dim dense vectors
│   L6-v2)       │  ── Batch processing (32 texts/batch)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   VectorStore   │  ── Creates/updates Qdrant collection
│   (Qdrant)      │  ── Stores vectors + text + metadata
│                 │  ── COSINE distance metric
└─────────────────┘
```

### Query Pipeline

```
User Question
    │
    ▼
┌─────────────────┐
│    Embedder     │  ── Encodes question into vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retriever     │  ── Cosine similarity search
│                 │  ── Returns Top-K chunks (default: 3)
│                 │  ── Includes similarity scores
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │  ── Formats context from chunks
│                 │  ── Builds strict grounding prompt
│                 │  ── Calls Ollama for generation
│                 │  ── Validates answer is grounded
└────────┬────────┘
         │
         ▼
Answer + Sources
```

### Component Relationships

```
main.py / streamlit_app.py
         │
         ├── DocumentLoader  (src/document_loader.py)
         │       └── Document (dataclass: text, source, chunk_id)
         │
         ├── Embedder        (src/embedder.py)
         │       └── SentenceTransformer model
         │
         ├── VectorStore     (src/vector_store.py)
         │       └── QdrantClient (local embedded mode)
         │
         ├── Retriever       (src/retriever.py)
         │       └── uses Embedder + VectorStore
         │
         └── RAGPipeline     (src/rag_pipeline.py)
                 └── uses Retriever + Ollama
```

---

## 📁 Project Structure

```
offline-hybrid-rag-system/
│
├── 📂 docs/                        # Your document corpus
│   ├── machine_learning.txt        # Sample: ML concepts
│   └── python_programming.txt      # Sample: Python basics
│
├── 📂 src/                         # Core source modules
│   ├── __init__.py                 # Package exports
│   ├── document_loader.py          # Load & chunk .txt files
│   ├── embedder.py                 # Sentence-transformers wrapper
│   ├── vector_store.py             # Qdrant CRUD operations
│   ├── retriever.py                # Semantic search logic
│   └── rag_pipeline.py             # RAG orchestration + Ollama
│
├── 📂 qdrant_db/                   # Auto-created: local vector DB
│
├── main.py                         # CLI entry point
├── streamlit_app.py                # Web UI (Streamlit)
├── test_setup.py                   # Installation verifier
├── requirements.txt                # Python dependencies
│
├── README.md                       # This file
├── QUICKSTART.md                   # 5-step setup guide
├── STREAMLIT_GUIDE.md              # Web UI documentation
└── TROUBLESHOOTING.md              # Common issues & fixes
```

---

## 🚀 Quick Start

### Prerequisites

You need three things running before you start:

#### 1. Qdrant (Vector Database)

**Recommended — Docker:**
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Alternative — Standalone Binary:**
Download from [github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases) and run the executable.

> ✅ Verify: open `http://localhost:6333` in your browser.

#### 2. Ollama (Local LLM)

1. Download from [ollama.ai/download](https://ollama.ai/download)
2. Install and launch Ollama
3. Pull a model:
```bash
ollama pull llama2        # Default (4GB, balanced)
ollama pull mistral       # Better reasoning (4GB)
ollama pull phi           # Lightweight (1.5GB, fast)
ollama pull llama3        # Latest Llama (8GB)
```

> ✅ Verify: `ollama list` should show your downloaded models.

#### 3. Python 3.8+

```bash
python --version   # Should be 3.8 or higher
```

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Bhogeswarareddy/offline-hybrid-rag-system.git
cd offline-hybrid-rag-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify everything is working
python test_setup.py
```

---

### Run It

```bash
# Step 1: Index your documents
python main.py --index

# Step 2: Ask a question
python main.py --query "What is machine learning?"

# Step 3: Start interactive chat
python main.py --interactive
```

Or launch the **web interface**:
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## 💻 Usage

### CLI Reference

```bash
python main.py [MODE] [OPTIONS]
```

**Modes:**

| Mode | Command | Description |
|---|---|---|
| Index | `--index` | Process and store documents |
| Query | `--query "question"` | Ask a single question |
| Interactive | `--interactive` | Start a chat session |

**Options:**

| Option | Default | Description |
|---|---|---|
| `--docs-dir` | `docs` | Directory with .txt files |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `--llm-model` | `llama2` | Ollama model name |
| `--collection` | `documents` | Qdrant collection name |
| `--top-k` | `3` | Number of chunks to retrieve |
| `--force-recreate` | `False` | Delete and rebuild index |
| `--verbose` | `False` | Show retrieved chunks + prompt |

**Examples:**

```bash
# Index documents from a custom folder
python main.py --index --docs-dir my_papers --force-recreate

# Query with verbose output (shows retrieved chunks)
python main.py --query "What is deep learning?" --verbose

# Interactive mode with a better model
python main.py --interactive --llm-model mistral --top-k 5

# Use a higher-quality embedding model
python main.py --index --embedding-model all-mpnet-base-v2
```

---

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

**Interface features:**
- 📤 Index your documents with one click
- 💬 Chat interface with full message history
- 🔍 Expand "Retrieved Chunks" to see exactly what text was used
- 📚 Source badges showing which files contributed to the answer
- ⚙️ Sidebar controls for models, Top-K, and collection name
- 📊 Live system status (collection name, total chunks, status)

---

### Interactive Mode Commands

While in `--interactive` mode:

| Input | Action |
|---|---|
| Any question | Get an answer |
| `verbose` | Toggle showing retrieved chunks |
| `quit` / `exit` / `q` | Exit the session |

---

## ⚙️ Configuration

### Embedding Models

| Model | Size | Speed | Quality | Use When |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 80MB | ⚡ Fast | Good | Default, most use cases |
| `all-mpnet-base-v2` | 420MB | Medium | Better | Quality matters more than speed |
| `all-distilroberta-v1` | 290MB | Medium | Good | Balanced alternative |

### LLM Models (Ollama)

| Model | Size | Speed | Use When |
|---|---|---|---|
| `llama2` | 4GB | Medium | Default general use |
| `mistral` | 4GB | Fast | Better instruction following |
| `phi` | 1.5GB | ⚡ Fast | Limited RAM |
| `llama3` | 8GB | Slow | Best quality (powerful machine) |
| `codellama` | 4GB | Medium | Code-heavy documents |

### Chunk Size Tuning

Edit `document_loader.py` or pass args:
```python
DocumentLoader(docs_dir="docs", min_words=300, max_words=500)
```

| min_words | max_words | Result |
|---|---|---|
| 100 | 200 | More chunks, finer granularity |
| 300 | 500 | Default — balanced |
| 500 | 800 | Fewer chunks, more context per chunk |

---

## 🔬 How It Works

### 1. Document Loading
`.txt` files are read with UTF-8 encoding. Text is cleaned (whitespace normalized), then split into sentence-aware chunks between `min_words` and `max_words`. Each chunk gets a UUID and stores its source filename.

### 2. Embedding Generation
Each chunk is passed through a `SentenceTransformer` model, producing a 384-dimensional dense vector that captures the chunk's semantic meaning. All embeddings are stored in memory as numpy arrays.

### 3. Vector Storage
Qdrant stores each vector alongside its payload (raw text, source filename, chunk ID). It uses **cosine distance** as the similarity metric, making retrieval robust to document length variation.

### 4. Semantic Retrieval
When a user asks a question, it's embedded using the same model. Qdrant performs an approximate nearest-neighbor search, returning the Top-K chunks with the highest cosine similarity scores.

### 5. Grounded Generation
The retrieved chunks are concatenated into a context block and injected into a strict prompt template that instructs the LLM to answer only from the provided context. If the answer isn't present, the system explicitly responds: *"Answer not found in the provided documents."*

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to Qdrant` | Qdrant not running | `docker run -p 6333:6333 qdrant/qdrant` |
| `Cannot connect to Ollama` | Ollama not started | Open Ollama app or run `ollama serve` |
| `Collection not found` | Documents not indexed | Run `python main.py --index` first |
| `No .txt files found` | Wrong directory | Check `--docs-dir` path and file extensions |
| `Model not found` | Model not pulled | Run `ollama pull llama2` |
| `Out of memory` | Model too large | Use `--llm-model phi` (1.5GB) |
| `Slow responses` | Heavy model or high Top-K | Reduce `--top-k` or switch to `phi`/`mistral` |
| Poor answer quality | Low retrieval relevance | Increase `--top-k` or use `all-mpnet-base-v2` |

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for full solutions including CUDA errors, encoding issues, and memory tuning.

---

## 📦 Dependencies

```
sentence-transformers==2.2.2    # Embedding generation
qdrant-client==1.7.0            # Vector database client
ollama==0.1.6                   # Local LLM interface
streamlit==1.29.0               # Web UI framework
langchain==0.1.0                # (Optional utilities)
python-dotenv==1.0.0            # Environment config
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🗺️ Roadmap

- [ ] PDF document support
- [ ] Multi-collection management
- [ ] Chat history persistence across sessions
- [ ] Hybrid search (BM25 + vector)
- [ ] Document metadata filtering
- [ ] REST API endpoint
- [ ] Docker Compose one-command setup

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Bhogeswara Reddy**

- GitHub: [@Bhogeswarareddy](https://github.com/Bhogeswarareddy)
- Repository: [offline-hybrid-rag-system](https://github.com/Bhogeswarareddy/offline-hybrid-rag-system)

---

<div align="center">

⭐ **If this project helped you, consider starring the repository!** ⭐

*Built with ❤️ using Python, Qdrant, Ollama, and Streamlit*

</div>
