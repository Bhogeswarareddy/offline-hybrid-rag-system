"""
RAG Pipeline Module
Complete RAG system with retrieval and generation
"""

import ollama
from typing import List, Dict, Optional
from retriever import Retriever


class RAGPipeline:
    """Complete RAG pipeline with strict grounding"""
    
    def __init__(self, retriever: Retriever, llm_model: str = "llama2"):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Retriever instance
            llm_model: Ollama model name (must be pulled first)
        """
        self.retriever = retriever
        self.llm_model = llm_model
        
        # Verify Ollama connection
        try:
            ollama.list()
            print(f"Connected to Ollama. Using model: {llm_model}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama: {e}")
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create strict RAG prompt
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Answer ONLY using the context below. If the answer is not present in the context, you MUST say: "Answer not found in the provided documents."

Do not use any external knowledge. Base your answer strictly on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def query(self, question: str, verbose: bool = False) -> Dict:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            verbose: If True, print retrieved chunks and prompt
            
        Returns:
            Dict with 'answer', 'sources', and 'retrieved_chunks'
        """
        # Retrieve relevant chunks
        results = self.retriever.retrieve(question)
        
        if verbose:
            print("\n" + "="*80)
            print("RETRIEVED CHUNKS:")
            print("="*80)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.4f} | Source: {result['source']}")
                print(f"   {result['text'][:300]}...")
        
        # Format context
        context = self.retriever.format_context(results)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        if verbose:
            print("\n" + "="*80)
            print("PROMPT SENT TO LLM:")
            print("="*80)
            print(prompt)
            print("\n" + "="*80)
            print("GENERATING ANSWER...")
            print("="*80 + "\n")
        
        # Generate answer using Ollama
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt
            )
            answer = response['response'].strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"
        
        # Extract unique sources
        sources = list(set([r['source'] for r in results]))
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': results
        }
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*80)
        print("RAG SYSTEM - INTERACTIVE MODE")
        print("="*80)
        print("Ask questions about your documents. Type 'quit' or 'exit' to stop.")
        print("Type 'verbose' to toggle detailed output.\n")
        
        verbose = False
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if question.lower() == 'verbose':
                    verbose = not verbose
                    print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                    continue
                
                # Get answer
                result = self.query(question, verbose=verbose)
                
                # Display answer
                print("\n" + "="*80)
                print("📝 ANSWER:")
                print("="*80)
                print(result['answer'])
                
                if not verbose:
                    print("\n📚 Sources:", ", ".join(result['sources']))
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Test RAG pipeline
    from document_loader import DocumentLoader
    from embedder import Embedder
    from vector_store import VectorStore
    from retriever import Retriever
    
    # Setup
    loader = DocumentLoader(docs_dir="../docs")
    documents = loader.load_documents()
    
    embedder = Embedder()
    vector_store = VectorStore()
    
    # Create and populate collection
    vector_store.create_collection(
        vector_size=embedder.get_embedding_dimension(),
        force_recreate=True
    )
    
    texts = [doc.text for doc in documents]
    embeddings = embedder.embed_batch(texts)
    metadata = [{"chunk_id": doc.chunk_id, "source": doc.source} for doc in documents]
    
    vector_store.add_documents(embeddings, texts, metadata)
    
    # Create RAG pipeline
    retriever = Retriever(embedder, vector_store, top_k=3)
    rag = RAGPipeline(retriever, llm_model="llama2")
    
    # Test query
    result = rag.query("What is machine learning?", verbose=True)
    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print("="*80)
    print(result['answer'])
    print("\nSources:", result['sources'])
