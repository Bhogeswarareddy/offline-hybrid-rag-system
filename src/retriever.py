"""
Retriever Module
Retrieves relevant chunks for a query
"""

from typing import List, Dict
import numpy as np
from embedder import Embedder
from vector_store import VectorStore


class Retriever:
    """Retrieves relevant document chunks for queries"""
    
    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 3):
        """
        Initialize retriever
        
        Args:
            embedder: Embedder instance for query encoding
            vector_store: VectorStore instance for searching
            top_k: Number of chunks to retrieve
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query: User's question
            
        Returns:
            List of dicts with 'text', 'source', 'chunk_id', and 'score'
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=self.top_k)
        
        return results
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            results: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source: {result['source']}]\n{result['text']}"
            )
        
        return "\n\n".join(context_parts)


if __name__ == "__main__":
    # Test retriever
    from document_loader import DocumentLoader
    
    # Load documents
    loader = DocumentLoader(docs_dir="../docs")
    documents = loader.load_documents()
    
    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore()
    
    # Create collection and add documents
    vector_store.create_collection(
        vector_size=embedder.get_embedding_dimension(),
        force_recreate=True
    )
    
    texts = [doc.text for doc in documents]
    embeddings = embedder.embed_batch(texts)
    metadata = [{"chunk_id": doc.chunk_id, "source": doc.source} for doc in documents]
    
    vector_store.add_documents(embeddings, texts, metadata)
    
    # Test retrieval
    retriever = Retriever(embedder, vector_store, top_k=3)
    
    query = "What is machine learning?"
    results = retriever.retrieve(query)
    
    print(f"Query: {query}\n")
    print("Retrieved chunks:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Text: {result['text'][:200]}...")
    
    print("\n" + "="*80)
    print("Formatted context:")
    print("="*80)
    print(retriever.format_context(results))
