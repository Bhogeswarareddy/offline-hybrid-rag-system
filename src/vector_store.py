"""
Vector Store Module
Manages Qdrant vector database operations (Local Embedded Mode)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import numpy as np
import os


class VectorStore:
    """Manages Qdrant vector database for storing and searching embeddings"""

    def __init__(self, collection_name: str = "documents", db_path: str = "qdrant_db"):
        """
        Initialize Qdrant client in LOCAL embedded mode

        Args:
            collection_name: Name of the collection to use
            db_path: Local storage folder for Qdrant
        """
        self.collection_name = collection_name

        # Ensure DB folder exists
        os.makedirs(db_path, exist_ok=True)

        # Local embedded Qdrant (no server required)
        self.client = QdrantClient(path=db_path)

        print(f"Using local Qdrant database at: {db_path}")

    # ------------------------------------------------------------------
    # COLLECTION MANAGEMENT
    # ------------------------------------------------------------------

    def create_collection(self, vector_size: int, force_recreate: bool = False):
        """
        Create a new collection in Qdrant

        Args:
            vector_size: Dimensionality of vectors
            force_recreate: If True, delete existing collection
        """

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if force_recreate:
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Collection '{self.collection_name}' already exists")
                return

        # Create collection
        print(f"Creating collection: {self.collection_name}")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        print(f"Collection created with vector size: {vector_size}")

    # ------------------------------------------------------------------
    # ADD DOCUMENTS
    # ------------------------------------------------------------------

    def add_documents(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: List[Dict]
    ):
        """
        Add documents with embeddings to the collection

        Args:
            embeddings: numpy array (n_docs, vector_size)
            texts: List of text strings
            metadata: List of metadata dicts
        """

        # ✅ Correct length validation
        if not (len(embeddings) == len(texts) == len(metadata)):
            raise ValueError(
                "embeddings, texts, and metadata must have same length"
            )

        points = []

        for idx, (embedding, text, meta) in enumerate(
            zip(embeddings, texts, metadata)
        ):
            payload = {
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", str(idx))
            }

            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=payload
            )

            points.append(point)

        # Upload in batches
        batch_size = 100

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        print(f"Added {len(points)} documents to collection")

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query_embedding: Query vector
            top_k: Number of results

        Returns:
            List of search results
        """

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        formatted_results = []

        for result in results:
            formatted_results.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "chunk_id": result.payload["chunk_id"],
                "score": result.score
            })

        return formatted_results

    # ------------------------------------------------------------------
    # COLLECTION INFO
    # ------------------------------------------------------------------

    def get_collection_info(self) -> Dict:
        """Get collection statistics"""

        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }

        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # CHECK COLLECTION
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """Check if collection exists"""

        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)


# ----------------------------------------------------------------------
# TEST BLOCK
# ----------------------------------------------------------------------

if __name__ == "__main__":

    from embedder import Embedder

    # Initialize
    embedder = Embedder()
    vector_store = VectorStore()

    # Create collection
    vector_store.create_collection(
        vector_size=embedder.get_embedding_dimension(),
        force_recreate=True
    )

    # Test documents
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing focuses on text and speech."
    ]

    embeddings = embedder.embed_batch(texts)

    metadata = [
        {"chunk_id": "1", "source": "ml_basics.txt"},
        {"chunk_id": "2", "source": "ml_basics.txt"},
        {"chunk_id": "3", "source": "nlp_intro.txt"}
    ]

    vector_store.add_documents(embeddings, texts, metadata)

    # Test search
    query = "What is deep learning?"
    query_embedding = embedder.embed_text(query)

    results = vector_store.search(query_embedding, top_k=2)

    print("\nSearch results:")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Text: {result['text']}")

    # Collection info
    info = vector_store.get_collection_info()
    print(f"\nCollection info: {info}")
