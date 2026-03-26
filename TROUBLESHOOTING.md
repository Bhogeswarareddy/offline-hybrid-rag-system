# Troubleshooting Guide

## Common Issues and Solutions

### 1. Qdrant Connection Issues

**Error:** `Cannot connect to Qdrant` or `Connection refused`

**Solutions:**

A. **Check if Qdrant is running:**
```bash
curl http://localhost:6333/
```
If this fails, Qdrant is not running.

B. **Start Qdrant (Docker):**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

C. **Start Qdrant (Binary):**
Download from https://github.com/qdrant/qdrant/releases and run the executable

D. **Check firewall:**
Ensure port 6333 is not blocked by firewall

---

### 2. Ollama Connection Issues

**Error:** `Cannot connect to Ollama`

**Solutions:**

A. **Check if Ollama is installed:**
```bash
ollama --version
```

B. **Check if Ollama is running:**
```bash
ollama list
```

C. **Start Ollama:**
- On Windows: Run Ollama from Start Menu
- On Mac/Linux: `ollama serve`

D. **Pull a model if none exist:**
```bash
ollama pull llama2
```

---

### 3. No Models Available

**Error:** `No models found` or `Model 'llama2' not found`

**Solutions:**

A. **List available models:**
```bash
ollama list
```

B. **Pull required model:**
```bash
ollama pull llama2
```

C. **Try alternative models:**
```bash
ollama pull mistral    # Smaller, faster
ollama pull llama3     # Newer version
ollama pull phi        # Very small model
```

D. **Use different model in command:**
```bash
python main.py --interactive --llm-model "mistral"
```

---

### 4. Embedding Model Download Issues

**Error:** Download fails or takes too long

**Solutions:**

A. **Check internet connection** (needed for first-time download)

B. **Use smaller model:**
```bash
python main.py --index --embedding-model "paraphrase-MiniLM-L3-v2"
```

C. **Wait for download** - First run downloads ~80MB model

D. **Check HuggingFace access:**
Visit https://huggingface.co/sentence-transformers

---

### 5. No Documents Found

**Error:** `No .txt files found in 'docs' directory`

**Solutions:**

A. **Check docs folder exists:**
```bash
dir docs  # Windows
ls docs   # Mac/Linux
```

B. **Ensure files have .txt extension**
- Not .doc, .docx, or .pdf
- Files must be plain text

C. **Add sample documents:**
Sample documents are already in `docs/` folder

---

### 6. Collection Not Found

**Error:** `Collection not found. Please run with --index first`

**Solutions:**

A. **Index documents first:**
```bash
python main.py --index
```

B. **Verify collection exists in Qdrant:**
Open http://localhost:6333/dashboard in browser

C. **Recreate collection:**
```bash
python main.py --index --force-recreate
```

---

### 7. Import Errors

**Error:** `ModuleNotFoundError: No module named 'X'`

**Solutions:**

A. **Install requirements:**
```bash
pip install -r requirements.txt
```

B. **Install specific package:**
```bash
pip install sentence-transformers
pip install qdrant-client
pip install ollama
```

C. **Check Python version:**
```bash
python --version  # Should be 3.8+
```

D. **Use pip3 if needed:**
```bash
pip3 install -r requirements.txt
```

---

### 8. Memory Issues

**Error:** `Out of memory` or system freezes

**Solutions:**

A. **Use smaller embedding model:**
```bash
python main.py --index --embedding-model "paraphrase-MiniLM-L3-v2"
```

B. **Use smaller LLM:**
```bash
ollama pull phi  # ~1.5GB
python main.py --interactive --llm-model "phi"
```

C. **Process fewer chunks:**
Modify `top_k` parameter:
```bash
python main.py --query "question" --top-k 2
```

D. **Increase chunk size to reduce total chunks:**
Edit `document_loader.py` and change `min_words=500, max_words=800`

---

### 9. Slow Performance

**Issue:** System is very slow

**Solutions:**

A. **Use GPU if available:**
Sentence-transformers automatically use GPU if available

B. **Reduce batch size in embedder.py:**
```python
embeddings = embedder.embed_batch(texts, batch_size=16)  # Default is 32
```

C. **Use smaller models:**
- Embedding: `all-MiniLM-L6-v2` (current) is already fast
- LLM: `phi` or `llama2` for faster responses

D. **Reduce retrieval count:**
```bash
python main.py --query "question" --top-k 2
```

---

### 10. Poor Answer Quality

**Issue:** Answers are not accurate or relevant

**Solutions:**

A. **Increase retrieved chunks:**
```bash
python main.py --query "question" --top-k 5
```

B. **Use better embedding model:**
```bash
python main.py --index --embedding-model "all-mpnet-base-v2" --force-recreate
```

C. **Improve document chunking:**
Edit chunk size in `main.py` or `document_loader.py`

D. **Use better LLM:**
```bash
ollama pull llama3
python main.py --interactive --llm-model "llama3"
```

E. **Add more context to documents:**
Ensure your .txt files have comprehensive information

---

### 11. "Answer not found" for Known Information

**Issue:** System says answer not found even though it's in documents

**Solutions:**

A. **Check if documents are indexed:**
```bash
python main.py --index --verbose
```

B. **Verify retrieval:**
```bash
python main.py --query "your question" --verbose
```
Check if relevant chunks are being retrieved

C. **Improve embeddings:**
Use better embedding model:
```bash
python main.py --index --embedding-model "all-mpnet-base-v2" --force-recreate
```

D. **Increase top-k:**
```bash
python main.py --query "question" --top-k 5 --verbose
```

---

### 12. Unicode/Encoding Errors

**Error:** Encoding issues with special characters

**Solutions:**

A. **Ensure UTF-8 encoding:**
Save .txt files with UTF-8 encoding

B. **Check file in text editor:**
Open files in Notepad++ or VS Code, check encoding

C. **Re-save with UTF-8:**
Most editors have "Save with Encoding" option

---

## Getting Help

If you're still stuck:

1. **Check verbose mode:**
```bash
python main.py --query "question" --verbose
```

2. **Run test script:**
```bash
python test_setup.py
```

3. **Check Qdrant dashboard:**
http://localhost:6333/dashboard

4. **Check Ollama:**
```bash
ollama list
```

5. **Review error messages carefully** - they usually indicate the exact issue

---

## Useful Commands

### Verify Everything
```bash
python test_setup.py
```

### Clean Start
```bash
# Delete and recreate collection
python main.py --index --force-recreate

# Or use Qdrant dashboard to delete collection
# Then re-run indexing
python main.py --index
```

### Check Collection Info
```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
info = client.get_collection("documents")
print(info)
```

### Test Individual Components
```bash
# Test document loader
cd src
python document_loader.py

# Test embedder
python embedder.py

# Test vector store
python vector_store.py
```
