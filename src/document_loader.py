"""
Document Loader Module
Loads .txt files from docs folder and splits them into chunks
"""

import os
import re
from typing import List, Dict
import uuid


class Document:
    """Represents a document chunk with metadata"""
    
    def __init__(self, text: str, source: str, chunk_id: str):
        self.text = text
        self.source = source
        self.chunk_id = chunk_id
    
    def __repr__(self):
        return f"Document(chunk_id={self.chunk_id}, source={self.source}, text_length={len(self.text)})"


class DocumentLoader:
    """Loads and chunks documents from a directory"""
    
    def __init__(self, docs_dir: str = "docs", min_words: int = 300, max_words: int = 500):
        self.docs_dir = docs_dir
        self.min_words = min_words
        self.max_words = max_words
    
    def load_documents(self) -> List[Document]:
        """Load all .txt files from docs directory"""
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Documents directory '{self.docs_dir}' not found")
        
        documents = []
        txt_files = [f for f in os.listdir(self.docs_dir) if f.endswith('.txt')]
        
        if not txt_files:
            raise ValueError(f"No .txt files found in '{self.docs_dir}' directory")
        
        print(f"Found {len(txt_files)} .txt file(s)")
        
        for filename in txt_files:
            filepath = os.path.join(self.docs_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks
                chunks = self._chunk_text(content)
                
                # Create Document objects
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    doc = Document(text=chunk, source=filename, chunk_id=chunk_id)
                    documents.append(doc)
                
                print(f"Loaded '{filename}': {len(chunks)} chunks")
            
            except Exception as e:
                print(f"Error loading '{filename}': {e}")
                continue
        
        print(f"Total chunks created: {len(documents)}")
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of min_words to max_words"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds max_words, save current chunk
            if current_word_count + sentence_words > self.max_words and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_word_count = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
            # If we've reached min_words and sentence ends properly, we can chunk
            if current_word_count >= self.min_words:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_word_count = 0
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


if __name__ == "__main__":
    # Test the document loader
    loader = DocumentLoader()
    docs = loader.load_documents()
    
    print(f"\nSample chunks:")
    for doc in docs[:3]:
        print(f"\n{doc}")
        print(f"Text preview: {doc.text[:200]}...")
