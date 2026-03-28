# app/rag_service.py

import os
import numpy as np
from app.pdf_service import extract_text_from_pdf, chunk_text
from dotenv import load_dotenv

load_dotenv()

# In-memory storage
document_store = {
    "chunks": [],
    "embeddings": []
}


def get_embedding(text: str) -> np.ndarray:
    """Lightweight hash-based embedding."""
    import hashlib
    words = text.lower().split()
    vector = np.zeros(384)
    for i, word in enumerate(words):
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[hash_val % 384] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def add_document(pdf_path: str) -> int:
    """Process a PDF and add it to our vector store."""
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    embeddings = [get_embedding(chunk) for chunk in chunks]

    document_store["chunks"].extend(chunks)
    document_store["embeddings"].extend(embeddings)

    print(f"Added {len(chunks)} chunks to vector store")
    return len(chunks)


def search_similar_chunks(query: str, top_k: int = 3) -> list[str]:
    """Find the most relevant chunks for a query."""
    if not document_store["chunks"]:
        return []

    query_embedding = get_embedding(query)

    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in document_store["embeddings"]
    ]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [document_store["chunks"][i] for i in top_indices]