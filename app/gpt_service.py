# app/gpt_service.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_gpt(question: str) -> str:
    """Send a question to LLM and return the answer."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful enterprise AI assistant specialized in document analysis and business insights."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


def ask_gpt_with_context(question: str, context_chunks: list[str]) -> dict:
    """
    Answer a question using retrieved document context (RAG).
    
    This is the core of RAG - we give GPT the relevant 
    document chunks so it can answer accurately.
    
    Args:
        question: User's question
        context_chunks: Relevant text from the document
        
    Returns:
        Dict with answer and sources
    """
    # Combine chunks into context
    context = "\n\n---\n\n".join(context_chunks)
    
    # Build prompt with context
    prompt = f"""You are an AI assistant analyzing documents.
    
Use ONLY the following document excerpts to answer the question.
If the answer is not in the excerpts, say "I couldn't find this in the document."

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Provide a clear, concise answer and mention which part of the document supports it."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system", 
                "content": "You are a precise document analysis assistant. Only answer based on provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,  # Lower = more focused/accurate
        max_tokens=800
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": context_chunks  # Return chunks as sources
    }