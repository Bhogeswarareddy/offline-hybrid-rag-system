# Quick Start Guide

Get your local RAG system running in 5 steps!

## Step 1: Install Prerequisites

### Qdrant (Vector Database)
**Option A - Docker (Recommended):**
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Option B - Standalone Binary:**
1. Download from: https://github.com/qdrant/qdrant/releases
2. Extract and run the executable
3. Access at: http://localhost:6333

### Ollama (Local LLM)
1. Download from: https://ollama.ai/download
2. Install and launch Ollama
3. Pull a model:
```bash
ollama pull llama2
```
Other options: `mistral`, `llama3`, `phi`

## Step 2: Install Python Dependencies

```bash
cd C:\Users\HP\OneDrive\Desktop\LOCAL_ADV_Rag
pip install -r requirements.txt
```

This installs:
- sentence-transformers (embeddings)
- qdrant-client (vector database)
- ollama (LLM interface)

## Step 3: Verify Setup

```bash
python test_setup.py
```

This checks:
- Python version
- Dependencies
- Qdrant connection
- Ollama connection
- Sample documents

## Step 4: Index Your Documents

The `docs/` folder already contains sample documents. Index them:

```bash
python main.py --index
```

**To add your own documents:**
1. Place `.txt` files in the `docs/` folder
2. Run indexing again with `--force-recreate`:
```bash
python main.py --index --force-recreate
```

## Step 5: Ask Questions!

### Single Question:
```bash
python main.py --query "What is machine learning?"
```

### Interactive Mode:
```bash
python main.py --interactive
```

Commands in interactive mode:
- Ask any question
- Type `verbose` to see retrieved chunks
- Type `quit` or `exit` to stop

## Example Session

```bash
# Index documents
python main.py --index

# Ask a question
python main.py --query "What are the types of machine learning?"

# Start chatting
python main.py --interactive
```

## Troubleshooting

### "Cannot connect to Qdrant"
- Ensure Qdrant is running on port 6333
- Test: Open http://localhost:6333 in browser

### "Cannot connect to Ollama"
- Ensure Ollama is installed and running
- Test: Run `ollama list` in terminal

### "No models found"
- Pull a model: `ollama pull llama2`
- Verify: `ollama list`

### "Collection not found"
- Run indexing first: `python main.py --index`

## Advanced Usage

### Use Different Models

**Embedding Model:**
```bash
python main.py --index --embedding-model "all-mpnet-base-v2"
```

**LLM Model:**
```bash
python main.py --interactive --llm-model "mistral"
```

### Retrieve More Chunks

```bash
python main.py --query "Your question" --top-k 5
```

### Verbose Mode

See what's happening under the hood:
```bash
python main.py --query "Your question" --verbose
```

## What Happens When You Run

1. **Indexing (`--index`):**
   - Loads `.txt` files from `docs/`
   - Splits into 300-500 word chunks
   - Generates embeddings
   - Stores in Qdrant with metadata

2. **Querying (`--query`):**
   - Converts your question to embedding
   - Finds top-3 similar chunks (cosine similarity)
   - Sends chunks + question to Ollama
   - Returns grounded answer

3. **Interactive Mode:**
   - Keeps connection open
   - Process multiple questions
   - Toggle verbose mode on/off

## Next Steps

- Add your own documents to `docs/`
- Try different embedding models
- Experiment with different LLMs
- Adjust chunk size in `document_loader.py`
- Modify the RAG prompt in `rag_pipeline.py`

## File Structure

```
LOCAL_ADV_Rag/
├── docs/                      # Your .txt documents
│   ├── machine_learning.txt   # Sample doc 1
│   └── python_programming.txt # Sample doc 2
├── src/                       # Source code
│   ├── document_loader.py     # Load & chunk docs
│   ├── embedder.py            # Generate embeddings
│   ├── vector_store.py        # Qdrant operations
│   ├── retriever.py           # Retrieve chunks
│   └── rag_pipeline.py        # RAG logic
├── main.py                    # Entry point
├── test_setup.py              # Verification script
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
└── QUICKSTART.md             # This file
```

Happy RAG-ing! 🚀
