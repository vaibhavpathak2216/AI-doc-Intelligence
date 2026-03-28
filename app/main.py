# app/main.py

import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Security
from pydantic import BaseModel
from app.gpt_service import ask_gpt, ask_gpt_with_context
from app.rag_service import add_document, search_similar_chunks
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 1. Initialize the FastAPI app first
app = FastAPI(
    title="Document Intelligence API",
    description="AI-powered API that answers questions from your documents using RAG",
    version="2.0.0"
)

# 2. Mount static files so CSS/JS can load
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Key security
API_KEY = os.getenv("APP_API_KEY", "dev-secret-key-123")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key for protected endpoints."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    return api_key

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence API",
    description="AI-powered API that answers questions from your documents using RAG",
    version="2.0.0"
)

# ---------------------------
# Request & Response Models
# ---------------------------

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_found: int

# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Document Intelligence API v2.0 is live!",
        "endpoints": ["/ask", "/upload", "/query", "/history"]
    }
@app.get("/ui")
def serve_ui():
    """Serve the chat UI"""
    return FileResponse("static/index.html")

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Ask a general question to the AI (no document needed)"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    answer = ask_gpt(request.question)
    return AnswerResponse(question=request.question, answer=answer)


@app.post("/upload")
def upload_document(file: UploadFile = File(...), api_key: str = Security(verify_api_key)):
    """
    Upload a PDF document to the knowledge base.
    The document will be processed and stored for querying.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process and add to vector store
    chunks_created = add_document(file_path)
    
    return {
        "message": "Document uploaded and processed successfully",
        "filename": file.filename,
        "chunks_created": chunks_created
    }


@app.post("/query", response_model=RAGResponse)
def query_document(request: RAGRequest, api_key: str = Security(verify_api_key)):
    """
    Ask a question about uploaded documents.
    Uses RAG to find relevant sections and answer accurately.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Step 1: Find relevant chunks from vector store
    relevant_chunks = search_similar_chunks(request.question, top_k=3)
    
    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a document first using /upload"
        )
    
    # Step 2: Generate answer using retrieved context
    result = ask_gpt_with_context(request.question, relevant_chunks)
    
    return RAGResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_found=len(relevant_chunks)
    )


@app.get("/history")
def get_history():
    """Get basic stats about uploaded documents"""
    from app.rag_service import document_store
    
    total_chunks = len(document_store["chunks"])
    
    return {
        "total_chunks_in_memory": total_chunks,
        "has_documents": total_chunks > 0,
        "message": "Upload PDFs using /upload endpoint to add documents"
    }