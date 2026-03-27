# app/pdf_service.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Full text content as a string
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text()
    
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Why overlap? So we don't lose context at chunk boundaries.
    
    Args:
        text: Full document text
        chunk_size: Characters per chunk
        overlap: Characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)
        
        # Move forward but keep overlap for context
        start = end - overlap
    
    return chunks