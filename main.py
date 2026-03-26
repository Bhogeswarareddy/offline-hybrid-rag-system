r"""
Main Entry Point for Local RAG Application
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_loader import DocumentLoader
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from rag_pipeline import RAGPipeline


def index_documents(docs_dir: str = "docs", 
                    embedding_model: str = "all-MiniLM-L6-v2",
                    collection_name: str = "documents",
                    force_recreate: bool = False):
    """
    Index all documents from docs directory
    
    Args:
        docs_dir: Directory containing .txt files
        embedding_model: Sentence-transformers model name
        collection_name: Qdrant collection name
        force_recreate: If True, recreate collection from scratch
    """
    print("\n" + "="*80)
    print("INDEXING DOCUMENTS")
    print("="*80 + "\n")
    
    try:
        # Load documents
        print("Step 1: Loading documents...")
        loader = DocumentLoader(docs_dir=docs_dir, min_words=300, max_words=500)
        documents = loader.load_documents()
        
        if not documents:
            print("❌ No documents found. Please add .txt files to the 'docs' folder.")
            return False
        
        # Initialize embedder
        print("\nStep 2: Initializing embedder...")
        embedder = Embedder(model_name=embedding_model)
        
        # Initialize vector store
        print("\nStep 3: Connecting to Qdrant...")
        vector_store = VectorStore(collection_name=collection_name)
        
        # Create collection
        print("\nStep 4: Creating/updating collection...")
        vector_store.create_collection(
            vector_size=embedder.get_embedding_dimension(),
            force_recreate=force_recreate
        )
        
        # Generate embeddings
        print("\nStep 5: Generating embeddings...")
        texts = [doc.text for doc in documents]
        embeddings = embedder.embed_batch(texts, show_progress=True)
        
        # Prepare metadata
        metadata = [
            {"chunk_id": doc.chunk_id, "source": doc.source} 
            for doc in documents
        ]
        
        # Add to vector store
        print("\nStep 6: Storing in Qdrant...")
        vector_store.add_documents(embeddings, texts, metadata)
        
        # Verify
        info = vector_store.get_collection_info()
        print("\n" + "="*80)
        print("✅ INDEXING COMPLETE")
        print("="*80)
        print(f"Collection: {info['name']}")
        print(f"Total chunks indexed: {info['points_count']}")
        print(f"Status: {info['status']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False


def query_documents(question: str,
                   embedding_model: str = "all-MiniLM-L6-v2",
                   llm_model: str = "llama2",
                   collection_name: str = "documents",
                   top_k: int = 3,
                   verbose: bool = False):
    """
    Query the RAG system
    
    Args:
        question: User's question
        embedding_model: Sentence-transformers model name
        llm_model: Ollama model name
        collection_name: Qdrant collection name
        top_k: Number of chunks to retrieve
        verbose: Show detailed output
    """
    try:
        # Initialize components
        embedder = Embedder(model_name=embedding_model)
        vector_store = VectorStore(collection_name=collection_name)
        
        # Check if collection exists
        if not vector_store.collection_exists():
            print("❌ Collection not found. Please run with --index first.")
            return
        
        # Create retriever and RAG pipeline
        retriever = Retriever(embedder, vector_store, top_k=top_k)
        rag = RAGPipeline(retriever, llm_model=llm_model)
        
        # Get answer
        result = rag.query(question, verbose=verbose)
        
        # Display result
        if not verbose:
            print("\n" + "="*80)
            print("QUESTION:")
            print("="*80)
            print(question)
            print("\n" + "="*80)
            print("ANSWER:")
            print("="*80)
            print(result['answer'])
            print("\n📚 Sources:", ", ".join(result['sources']))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode(embedding_model: str = "all-MiniLM-L6-v2",
                    llm_model: str = "llama2",
                    collection_name: str = "documents",
                    top_k: int = 3):
    """
    Run interactive Q&A session
    
    Args:
        embedding_model: Sentence-transformers model name
        llm_model: Ollama model name
        collection_name: Qdrant collection name
        top_k: Number of chunks to retrieve
    """
    try:
        # Initialize components
        print("Initializing RAG system...")
        embedder = Embedder(model_name=embedding_model)
        vector_store = VectorStore(collection_name=collection_name)
        
        # Check if collection exists
        if not vector_store.collection_exists():
            print("❌ Collection not found. Please run with --index first.")
            return
        
        # Get collection info
        info = vector_store.get_collection_info()
        print(f"\nLoaded collection: {info['name']}")
        print(f"Total chunks: {info['points_count']}\n")
        
        # Create retriever and RAG pipeline
        retriever = Retriever(embedder, vector_store, top_k=top_k)
        rag = RAGPipeline(retriever, llm_model=llm_model)
        
        # Run interactive mode
        rag.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Local RAG Application - Answer questions from your documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python main.py --index
  
  # Ask a question
  python main.py --query "What is machine learning?"
  
  # Interactive mode
  python main.py --interactive
  
  # Verbose output
  python main.py --query "What is deep learning?" --verbose
  
  # Custom models
  python main.py --index --embedding-model "all-mpnet-base-v2"
  python main.py --interactive --llm-model "mistral"
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--index', action='store_true', 
                           help='Index documents from docs folder')
    mode_group.add_argument('--query', type=str, metavar='QUESTION',
                           help='Ask a single question')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Start interactive Q&A session')
    
    # Configuration options
    parser.add_argument('--docs-dir', type=str, default='docs',
                       help='Directory containing .txt files (default: docs)')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence-transformers model (default: all-MiniLM-L6-v2)')
    parser.add_argument('--llm-model', type=str, default='llama2',
                       help='Ollama model name (default: llama2)')
    parser.add_argument('--collection', type=str, default='documents',
                       help='Qdrant collection name (default: documents)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of chunks to retrieve (default: 3)')
    parser.add_argument('--force-recreate', action='store_true',
                       help='Force recreate collection when indexing')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.index:
        success = index_documents(
            docs_dir=args.docs_dir,
            embedding_model=args.embedding_model,
            collection_name=args.collection,
            force_recreate=args.force_recreate
        )
        sys.exit(0 if success else 1)
    
    elif args.query:
        query_documents(
            question=args.query,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            collection_name=args.collection,
            top_k=args.top_k,
            verbose=args.verbose
        )
    
    elif args.interactive:
        interactive_mode(
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            collection_name=args.collection,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()
