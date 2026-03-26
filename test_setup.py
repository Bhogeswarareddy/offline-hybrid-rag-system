"""
Setup and Test Script
Run this to verify your installation and test the RAG system
"""

import sys
import subprocess
import os


def check_python_version():
    """Check if Python version is adequate"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_qdrant():
    """Check if Qdrant is accessible"""
    print("\nChecking Qdrant connection...")
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running on port 6333")
            return True
        else:
            print(f"⚠️ Qdrant responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")
        print("\nTo start Qdrant:")
        print("  Option 1 (Docker): docker run -p 6333:6333 qdrant/qdrant")
        print("  Option 2 (Binary): Download from https://github.com/qdrant/qdrant/releases")
        return False


def check_ollama():
    """Check if Ollama is accessible"""
    print("\nChecking Ollama connection...")
    try:
        import ollama
        models = ollama.list()
        print("✅ Ollama is running")
        
        # Check for available models
        model_names = [m['name'] for m in models.get('models', [])]
        if model_names:
            print(f"   Available models: {', '.join(model_names)}")
        else:
            print("⚠️ No models found. Pull a model with: ollama pull llama2")
        
        return True
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\nTo install Ollama:")
        print("  1. Download from https://ollama.ai/download")
        print("  2. Install and run Ollama")
        print("  3. Pull a model: ollama pull llama2")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking Python dependencies...")
    required = {
        'sentence_transformers': 'sentence-transformers',
        'qdrant_client': 'qdrant-client',
        'ollama': 'ollama',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_documents():
    """Check if sample documents exist"""
    print("\nChecking documents...")
    docs_dir = "docs"
    
    if not os.path.exists(docs_dir):
        print(f"❌ Directory '{docs_dir}' not found")
        return False
    
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"⚠️ No .txt files found in '{docs_dir}'")
        print("   Sample documents have been created for you.")
        return True
    
    print(f"✅ Found {len(txt_files)} document(s):")
    for f in txt_files:
        file_path = os.path.join(docs_dir, f)
        size = os.path.getsize(file_path)
        print(f"   - {f} ({size:,} bytes)")
    
    return True


def run_test():
    """Run a quick test of the RAG system"""
    print("\n" + "="*80)
    print("RUNNING TEST")
    print("="*80 + "\n")
    
    try:
        # Import after checking dependencies
        sys.path.insert(0, 'src')
        from document_loader import DocumentLoader
        from embedder import Embedder
        from vector_store import VectorStore
        from retriever import Retriever
        from rag_pipeline import RAGPipeline
        
        # Load documents
        print("1. Loading documents...")
        loader = DocumentLoader(docs_dir="docs", min_words=100, max_words=300)
        documents = loader.load_documents()
        
        if not documents:
            print("❌ No documents to test with")
            return False
        
        # Initialize components
        print("\n2. Initializing embedder...")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        
        print("\n3. Connecting to Qdrant...")
        vector_store = VectorStore(collection_name="test_collection")
        
        print("\n4. Creating collection...")
        vector_store.create_collection(
            vector_size=embedder.get_embedding_dimension(),
            force_recreate=True
        )
        
        print("\n5. Generating embeddings...")
        texts = [doc.text for doc in documents]
        embeddings = embedder.embed_batch(texts[:5], show_progress=False)  # Test with first 5
        
        print("\n6. Storing in Qdrant...")
        metadata = [
            {"chunk_id": doc.chunk_id, "source": doc.source} 
            for doc in documents[:5]
        ]
        vector_store.add_documents(embeddings, texts[:5], metadata)
        
        print("\n7. Testing retrieval...")
        retriever = Retriever(embedder, vector_store, top_k=2)
        results = retriever.retrieve("What is machine learning?")
        
        print(f"\n   Retrieved {len(results)} chunks")
        for i, r in enumerate(results, 1):
            print(f"   {i}. Score: {r['score']:.4f} | Source: {r['source']}")
        
        print("\n8. Testing RAG pipeline...")
        rag = RAGPipeline(retriever, llm_model="llama2")
        result = rag.query("What is Python?", verbose=False)
        
        print("\n" + "="*80)
        print("TEST RESULT:")
        print("="*80)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['sources']}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("LOCAL RAG SYSTEM - SETUP VERIFICATION")
    print("="*80 + "\n")
    
    all_good = True
    
    # Run checks
    all_good &= check_python_version()
    all_good &= check_dependencies()
    all_good &= check_qdrant()
    all_good &= check_ollama()
    all_good &= check_documents()
    
    if not all_good:
        print("\n" + "="*80)
        print("⚠️ SETUP INCOMPLETE")
        print("="*80)
        print("Please fix the issues above before proceeding.")
        return
    
    print("\n" + "="*80)
    print("✅ ALL CHECKS PASSED")
    print("="*80)
    
    # Ask if user wants to run test
    response = input("\nWould you like to run a quick test? (y/n): ").strip().lower()
    
    if response == 'y':
        run_test()
    else:
        print("\nYou're ready to use the RAG system!")
        print("\nNext steps:")
        print("  1. Index your documents: python main.py --index")
        print("  2. Ask questions: python main.py --query 'Your question'")
        print("  3. Interactive mode: python main.py --interactive")


if __name__ == "__main__":
    main()
