"""
Embedder Module
Generates embeddings using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class Embedder:
    """Generates embeddings for text using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with a sentence-transformers model
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
                       This is a lightweight, fast model good for semantic search
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of embeddings
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of embeddings"""
        return self.embedding_dim


if __name__ == "__main__":
    # Test the embedder
    embedder = Embedder()
    
    # Test single embedding
    text = "What is machine learning?"
    embedding = embedder.embed_text(text)
    print(f"\nSingle embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing handles text"
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
