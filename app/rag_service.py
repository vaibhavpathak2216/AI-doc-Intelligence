# app/rag_service.py

import os
import numpy as np
import faiss
from app.pdf_service import extract_text_from_pdf, chunk_text
from dotenv import load_dotenv

load_dotenv()

# Global variables - model loads lazily on first use
embedding_model = None

# In-memory storage
document_store = {
    "chunks": [],
    "index": None
}


def get_embedding_model():
    """Load embedding model only when first needed."""
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")
    return embedding_model


def add_document(pdf_path: str) -> int:
    """Process a PDF and add it to our vector store."""
    model = get_embedding_model()
    
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings...")
    embeddings = model.encode(chunks)
    
    dimension = embeddings.shape[1]
    
    if document_store["index"] is None:
        document_store["index"] = faiss.IndexFlatL2(dimension)
    
    document_store["chunks"].extend(chunks)
    document_store["index"].add(np.array(embeddings, dtype=np.float32))
    
    print(f"Added {len(chunks)} chunks to vector store")
    return len(chunks)


def search_similar_chunks(query: str, top_k: int = 3) -> list[str]:
    """Find the most relevant chunks for a query."""
    if document_store["index"] is None:
        return []
    
    model = get_embedding_model()
    query_embedding = model.encode([query])
    
    distances, indices = document_store["index"].search(
        np.array(query_embedding, dtype=np.float32),
        top_k
    )
    
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(document_store["chunks"]):
            relevant_chunks.append(document_store["chunks"][idx])
    
    return relevant_chunks