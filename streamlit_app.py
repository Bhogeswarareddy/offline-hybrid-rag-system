"""
Streamlit Web Interface for Local RAG Application
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_loader import DocumentLoader
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1F77B4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1F77B4;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    .source-badge {
        background-color: #e1f5ff;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.embedder = None
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.rag = None
    st.session_state.chat_history = []
    st.session_state.collection_info = None

def initialize_system(embedding_model, llm_model, collection_name, top_k):
    """Initialize the RAG system components"""
    try:
        with st.spinner("🔄 Initializing RAG system..."):
            # Initialize embedder
            st.session_state.embedder = Embedder(model_name=embedding_model)
            
            # Initialize vector store
            st.session_state.vector_store = VectorStore(collection_name=collection_name)
            
            # Check if collection exists
            if not st.session_state.vector_store.collection_exists():
                st.error("❌ Collection not found! Please index documents first.")
                return False
            
            # Get collection info
            st.session_state.collection_info = st.session_state.vector_store.get_collection_info()
            
            # Initialize retriever
            st.session_state.retriever = Retriever(
                st.session_state.embedder,
                st.session_state.vector_store,
                top_k=top_k
            )
            
            # Initialize RAG pipeline
            st.session_state.rag = RAGPipeline(
                st.session_state.retriever,
                llm_model=llm_model
            )
            
            st.session_state.initialized = True
            return True
            
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        return False

def index_documents(docs_dir, embedding_model, collection_name, force_recreate):
    """Index documents from docs directory"""
    try:
        with st.spinner("📚 Loading documents..."):
            loader = DocumentLoader(docs_dir=docs_dir, min_words=300, max_words=500)
            documents = loader.load_documents()
            
            if not documents:
                st.error("❌ No documents found. Add .txt files to the 'docs' folder.")
                return False
            
            st.info(f"✅ Loaded {len(documents)} chunks from documents")
        
        with st.spinner("🧠 Initializing embedder..."):
            embedder = Embedder(model_name=embedding_model)
        
        with st.spinner("💾 Connecting to Qdrant..."):
            vector_store = VectorStore(collection_name=collection_name)
            vector_store.create_collection(
                vector_size=embedder.get_embedding_dimension(),
                force_recreate=force_recreate
            )
        
        with st.spinner("🔄 Generating embeddings... This may take a moment..."):
            texts = [doc.text for doc in documents]
            embeddings = embedder.embed_batch(texts, show_progress=False)
        
        with st.spinner("💾 Storing in database..."):
            metadata = [
                {"chunk_id": doc.chunk_id, "source": doc.source}
                for doc in documents
            ]
            vector_store.add_documents(embeddings, texts, metadata)
        
        info = vector_store.get_collection_info()
        st.success(f"✅ Indexing complete! Total chunks: {info['points_count']}")
        return True
        
    except Exception as e:
        st.error(f"❌ Error during indexing: {e}")
        return False

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🤖 Local RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Retrieval-Augmented Generation powered by AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("🔧 Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"],
            help="Model for generating text embeddings"
        )
        
        llm_model = st.selectbox(
            "LLM Model",
            ["llama2", "mistral", "codellama", "llama3"],
            help="Ollama model for answer generation"
        )
        
        collection_name = st.text_input(
            "Collection Name",
            value="documents",
            help="Qdrant collection name"
        )
        
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of chunks to retrieve"
        )
        
        st.divider()
        
        # Document indexing section
        st.subheader("📚 Document Indexing")
        docs_dir = st.text_input("Documents Directory", value="docs")
        force_recreate = st.checkbox("Force Recreate Collection", value=False)
        
        if st.button("🔄 Index Documents", use_container_width=True):
            if index_documents(docs_dir, embedding_model, collection_name, force_recreate):
                st.session_state.initialized = False  # Force re-initialization
                st.rerun()
        
        st.divider()
        
        # Initialize/Reset button
        if st.button("🚀 Initialize System", use_container_width=True):
            if initialize_system(embedding_model, llm_model, collection_name, top_k):
                st.success("✅ System initialized successfully!")
                st.rerun()
        
        if st.session_state.initialized:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.divider()
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.initialized:
            st.success("✅ System Ready")
            if st.session_state.collection_info:
                info = st.session_state.collection_info
                st.metric("Collection", info['name'])
                st.metric("Total Chunks", info['points_count'])
                st.metric("Status", info['status'])
        else:
            st.warning("⚠️ System Not Initialized")
    
    # Main content area
    if not st.session_state.initialized:
        # Welcome screen
        st.info("👈 Please initialize the system using the sidebar")
        
        # Quick start guide
        st.markdown("### 🚀 Quick Start Guide")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Step 1: Prepare Documents**
            - Add `.txt` files to the `docs/` folder
            - Each file will be chunked automatically
            
            **Step 2: Index Documents**
            - Click "Index Documents" in the sidebar
            - Wait for processing to complete
            """)
        
        with col2:
            st.markdown("""
            **Step 3: Initialize System**
            - Click "Initialize System" in the sidebar
            - System will load and be ready to use
            
            **Step 4: Ask Questions**
            - Type your questions in the chat
            - Get answers based on your documents
            """)
        
        # Display sample documents if any exist
        docs_path = Path("docs")
        if docs_path.exists():
            txt_files = list(docs_path.glob("*.txt"))
            if txt_files:
                st.markdown("### 📄 Available Documents")
                for file in txt_files:
                    st.write(f"- {file.name}")
    
    else:
        # Chat interface
        st.markdown("### 💬 Chat with Your Documents")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    if 'sources' in message:
                        with st.expander("📚 Sources"):
                            for source in message['sources']:
                                st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)
                    if 'retrieved_chunks' in message and message.get('show_chunks'):
                        with st.expander("🔍 Retrieved Chunks"):
                            for i, chunk in enumerate(message['retrieved_chunks'], 1):
                                st.markdown(f"**Chunk {i}** (Score: {chunk['score']:.4f})")
                                st.markdown(f"*Source: {chunk['source']}*")
                                st.text(chunk['text'][:300] + "...")
                                st.divider()
        
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = st.session_state.rag.query(user_question, verbose=False)
                        
                        # Display answer
                        st.write(result['answer'])
                        
                        # Display sources
                        with st.expander("📚 Sources"):
                            for source in result['sources']:
                                st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)
                        
                        # Display retrieved chunks
                        with st.expander("🔍 Retrieved Chunks"):
                            for i, chunk in enumerate(result['retrieved_chunks'], 1):
                                st.markdown(f"**Chunk {i}** (Score: {chunk['score']:.4f})")
                                st.markdown(f"*Source: {chunk['source']}*")
                                st.text(chunk['text'][:300] + "...")
                                st.divider()
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': result['answer'],
                            'sources': result['sources'],
                            'retrieved_chunks': result['retrieved_chunks']
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
