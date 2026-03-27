# app/gpt_service.py

import os
from groq import Groq
from dotenv import load_dotenv

# Load secret keys from .env file
load_dotenv()

# Create Groq client (free alternative to OpenAI)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_gpt(question: str) -> str:
    """Send a question to LLM and return the answer."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",      # Free Llama3 model on Groq
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