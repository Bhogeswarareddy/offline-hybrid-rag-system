"""
Local RAG Application
A fully local Retrieval-Augmented Generation system
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .document_loader import DocumentLoader, Document
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .rag_pipeline import RAGPipeline

__all__ = [
    'DocumentLoader',
    'Document',
    'Embedder',
    'VectorStore',
    'Retriever',
    'RAGPipeline',
]
