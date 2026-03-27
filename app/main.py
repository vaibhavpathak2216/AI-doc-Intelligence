# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.gpt_service import ask_gpt

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence API",
    description="AI-powered API that answers questions using GPT",
    version="1.0.0"
)

# Requesting & Responding Models

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# API Endpoints

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Document Intelligence API is live!"
    }

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Takes a question and returns GPT's answer."""

    # Validate input
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # Get answer from GPT
    answer = ask_gpt(request.question)

    return AnswerResponse(
        question=request.question,
        answer=answer
    )