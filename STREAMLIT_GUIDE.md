# 🤖 Local RAG System - Streamlit Web Interface

A beautiful web interface for your Local RAG (Retrieval-Augmented Generation) system.

## ✨ Features

- 🎨 **Modern UI**: Clean, professional interface
- 💬 **Chat Interface**: Interactive Q&A with your documents
- 📊 **Live Statistics**: Real-time system status and metrics
- 🔍 **Source Transparency**: See which documents were used for answers
- 📚 **Chunk Viewer**: Inspect retrieved document chunks
- ⚙️ **Configurable**: Adjust models and settings on the fly
- 🚀 **Easy to Use**: No coding required

## 🚀 Quick Start

### 1. Install Streamlit (if not already installed)

```bash
pip install streamlit
```

Or update all requirements:

```bash
pip install -r requirements.txt
```

### 2. Make Sure Prerequisites Are Running

**Qdrant Database:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Ollama:**
```bash
ollama pull llama2
```

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📖 How to Use

### Step 1: Prepare Your Documents
1. Add `.txt` files to the `docs/` folder
2. The app will show available documents on the welcome screen

### Step 2: Index Your Documents
1. Open the sidebar (left panel)
2. Click **"Index Documents"** button
3. Wait for processing to complete (you'll see a progress indicator)

### Step 3: Initialize the System
1. Click **"Initialize System"** in the sidebar
2. Wait for initialization (loads models and connects to database)
3. Status will change to ✅ "System Ready"

### Step 4: Start Asking Questions!
1. Type your question in the chat input at the bottom
2. Press Enter or click Send
3. View the answer with sources

## 🎛️ Configuration Options

### Sidebar Settings

**Embedding Model:**
- `all-MiniLM-L6-v2` (Default - Fast, lightweight)
- `all-mpnet-base-v2` (Better quality, slower)
- `all-distilroberta-v1` (Balanced)

**LLM Model:**
- `llama2` (Default - General purpose)
- `mistral` (Better reasoning)
- `llama3` (Latest version)
- `codellama` (Good for code questions)

**Top-K Retrieval:**
- Slider from 1-10
- Controls how many document chunks to retrieve
- Default: 3
- Higher = more context, but slower

**Collection Name:**
- Name of Qdrant collection
- Default: "documents"
- Change if you want multiple separate indexes

## 🖼️ Interface Overview

### Welcome Screen
- Quick start guide
- List of available documents
- System status

### Chat Interface
- User questions and AI answers
- **Sources**: Click to see which documents were used
- **Retrieved Chunks**: View exact text passages used

### Sidebar
- Configuration settings
- Index documents button
- Initialize system button
- Reset chat button
- System status metrics

## 💡 Tips for Best Results

### For Your Sir's Demonstration:

1. **Prepare Sample Documents:**
   ```
   docs/
   ├── machine_learning.txt
   ├── python_programming.txt
   └── deep_learning.txt
   ```

2. **Index Before Demo:**
   - Run indexing before the presentation
   - This saves time during the demo

3. **Prepare Good Questions:**
   ```
   ✅ "What is supervised learning?"
   ✅ "Explain the difference between lists and tuples in Python"
   ✅ "How do neural networks work?"
   ```

4. **Show Key Features:**
   - Click "Sources" to show transparency
   - Click "Retrieved Chunks" to show how it works
   - Try a question that's NOT in docs to show it refuses

5. **Demonstrate Configuration:**
   - Change Top-K to show different results
   - Show multiple embedding models

## 🎯 Demonstration Script

### Opening (2 minutes)
1. Show welcome screen
2. Explain: "This is a RAG system - answers only from provided documents"
3. Show available documents in sidebar

### Indexing (3 minutes)
1. Click "Index Documents"
2. Explain: "Converting text to vectors and storing in database"
3. Show completion message with chunk count

### Initialize (1 minute)
1. Click "Initialize System"
2. Show status change to "System Ready"

### Q&A Demo (5 minutes)
1. Ask 3-4 questions
2. For each answer:
   - Show the answer
   - Click "Sources" - show which files were used
   - Click "Retrieved Chunks" - show exact text used

### Advanced Features (3 minutes)
1. Change Top-K slider - show how it affects results
2. Ask a question NOT in documents - show it refuses to answer
3. Show chat history scrolling

### Conclusion (1 minute)
- Emphasize: 100% local, private, no internet needed
- All answers are grounded in provided documents

## 🔧 Troubleshooting

### "Collection not found" Error
**Solution:** Click "Index Documents" first

### "Cannot connect to Ollama" Error
**Solution:** 
```bash
ollama serve
ollama pull llama2
```

### "Qdrant connection error"
**Solution:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Slow Response Times
**Solutions:**
- Use smaller embedding model: `all-MiniLM-L6-v2`
- Reduce Top-K to 2 or 1
- Use lighter LLM: `llama2` instead of `llama3`

### App Won't Start
**Check:**
```bash
# Verify Streamlit is installed
streamlit --version

# Reinstall if needed
pip install streamlit --upgrade
```

## 📱 Keyboard Shortcuts

- `Ctrl + R` - Refresh/Reload app
- `Enter` - Send message in chat
- `Esc` - Close expanders

## 🎨 Customization

### Change Colors
Edit the CSS in `streamlit_app.py`:
```python
.main-header {
    color: #1F77B4;  # Change this hex color
}
```

### Add More Models
Edit the selectbox options:
```python
embedding_model = st.selectbox(
    "Embedding Model",
    ["all-MiniLM-L6-v2", "YOUR-MODEL-HERE"]
)
```

## 📊 System Requirements

- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for models
- **CPU**: Any modern processor (GPU optional)
- **Browser**: Chrome, Firefox, Safari, Edge

## 🌟 Key Highlights for Sir

1. **Professional UI** - Production-ready interface
2. **Real-time Processing** - Watch the system work
3. **Transparency** - See sources and chunks used
4. **Configurable** - Adjust settings without coding
5. **Chat History** - Maintains conversation context
6. **Error Handling** - Clear error messages
7. **Progress Indicators** - Know what's happening
8. **Responsive Design** - Works on different screen sizes

## 🎓 Educational Value

This Streamlit app demonstrates:
- Full-stack RAG implementation
- Modern web UI development
- Real-time AI integration
- Database management
- User experience design
- Production-ready deployment

Perfect for showing understanding of:
- RAG architecture
- Vector databases
- Embeddings
- LLM integration
- Web development
- System design

---

**Built with ❤️ using Streamlit, Qdrant, and Ollama**
