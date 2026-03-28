# 🤖 AI Document Intelligence System

An enterprise-grade RAG (Retrieval-Augmented Generation) API that lets you upload documents and ask questions about them using AI.

## 🚀 Live Demo
**API URL:** https://ai-doc-intelligence-p20d.onrender.com/docs

## 🏗️ Architecture
```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store
                                                              ↓
User Question → Embedding → Similarity Search → Top 3 Chunks → LLM → Answer + Sources
```

## ⚡ Features
- 📄 PDF document ingestion
- 🔍 Vector similarity search using FAISS
- 🤖 RAG-powered Q&A with source citations
- 🔐 API Key authentication
- 📊 RESTful API with auto-generated docs

## 🛠️ Tech Stack
- **Framework:** FastAPI
- **LLM:** Llama 3 via Groq
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** FAISS
- **PDF Processing:** PyMuPDF
- **Deployment:** Render

## 📡 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /upload | Upload a PDF document |
| POST | /query | Ask questions about uploaded docs |
| POST | /ask | General AI Q&A |
| GET | /history | View document store stats |

## 🔐 Authentication
Protected endpoints require an API key in the header:
```
X-API-Key: your-api-key
```

## 🏃 Run Locally
```bash
git clone https://github.com/vaibhavpathak2216/AI-doc-Intelligence.git
cd AI-doc-Intelligence
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 💡 Use Cases
- Enterprise document Q&A
- Contract analysis
- Invoice processing
- Knowledge base search
- Oracle Fusion document intelligence