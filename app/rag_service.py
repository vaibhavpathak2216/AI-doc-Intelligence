# app/rag_service.py

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.pdf_service import extract_text_from_pdf, chunk_text
from dotenv import load_dotenv

load_dotenv()

# Load embedding model (runs locally, no API needed)
# This model converts text into numbers (vectors)
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# In-memory storage for our documents
# In production, this would be a real vector database
document_store = {
    "chunks": [],      # Original text chunks
    "embeddings": None, # Vectors stored in FAISS
    "index": None      # FAISS index for fast search
}


def add_document(pdf_path: str) -> int:
    """
    Process a PDF and add it to our vector store.
    
    Steps:
    1. Extract text from PDF
    2. Split into chunks
    3. Create embeddings (convert to numbers)
    4. Store in FAISS index
    
    Returns:
        Number of chunks created
    """
    # Step 1: Extract text
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Chunk the text
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings
    print("Creating embeddings...")
    embeddings = embedding_model.encode(chunks)
    
    # Step 4: Store in FAISS
    dimension = embeddings.shape[1]  # Size of each embedding vector
    
    if document_store["index"] is None:
        # Create new FAISS index
        document_store["index"] = faiss.IndexFlatL2(dimension)
    
    # Add to store
    document_store["chunks"].extend(chunks)
    document_store["index"].add(np.array(embeddings, dtype=np.float32))
    
    print(f"Added {len(chunks)} chunks to vector store")
    return len(chunks)


def search_similar_chunks(query: str, top_k: int = 3) -> list[str]:
    """
    Find the most relevant chunks for a query.
    
    Steps:
    1. Convert query to embedding
    2. Search FAISS for similar vectors
    3. Return matching text chunks
    
    Args:
        query: User's question
        top_k: Number of chunks to return
        
    Returns:
        List of relevant text chunks
    """
    if document_store["index"] is None:
        return []
    
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])
    
    # Search FAISS index
    distances, indices = document_store["index"].search(
        np.array(query_embedding, dtype=np.float32), 
        top_k
    )
    
    # Return matching chunks
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(document_store["chunks"]):
            relevant_chunks.append(document_store["chunks"][idx])
    
    return relevant_chunks