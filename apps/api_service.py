# apps/api_service.py
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os

from ai_gateway.client import AIGatewayClient
from vector_db.pgvector_client import PGVectorClient
from agents.rag_agent import query_rag_agent

app = FastAPI(title="AI Gateway Service")

# Initialize clients
gateway_client = AIGatewayClient()
pgvector_client = PGVectorClient()

# Models
class TextGenerationRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    model_family: Optional[str] = "openai"

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "text-embedding-3-large"

class DocumentSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class RagQueryRequest(BaseModel):
    question: str

# Routes
@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        response = gateway_client.generate_text(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model_family=request.model_family
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        embeddings = gateway_client.generate_embeddings(
            texts=request.texts,
            model=request.model
        )
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(request: DocumentSearchRequest):
    try:
        # Generate embedding for the query
        query_embedding = gateway_client.generate_embeddings(texts=[request.query])[0]
        
        # Search for similar documents
        results = pgvector_client.search_similar(
            query_embedding=query_embedding,
            limit=request.limit
        )
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
async def rag_query(request: RagQueryRequest):
    try:
        answer = query_rag_agent(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)