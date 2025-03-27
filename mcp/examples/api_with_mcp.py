# api_with_mcp.py
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel
import pandas as pd
import json

from mcp_implementation import MCPHandler
from fastmcp_implementation import FastMCPHandler
from rag_with_mcp import RAGWithMCP
from structured_data_with_fastmcp import StructuredDataHandler

app = FastAPI(title="Model Context Protocol API")

# Models
class ContextItem(BaseModel):
    type: str
    content: Any
    metadata: Optional[Dict[str, Any]] = None

class MessageItem(BaseModel):
    role: str
    content: str
    contexts: Optional[List[ContextItem]] = None

class GenerateRequest(BaseModel):
    messages: List[MessageItem]
    model: Optional[str] = "gpt4o"
    model_family: Optional[str] = "openai"

class GenerateResponse(BaseModel):
    content: str

class RAGRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class TableDataRequest(BaseModel):
    query: str
    table_data: List[Dict[str, Any]]
    table_name: Optional[str] = "data_table"

class JSONDataRequest(BaseModel):
    query: str
    json_data: Dict[str, Any]
    data_description: Optional[str] = "JSON Data"

# Dependencies
def get_mcp_handler():
    return MCPHandler()

def get_fastmcp_handler():
    return FastMCPHandler()

def get_rag_handler():
    return RAGWithMCP()

def get_structured_data_handler():
    return StructuredDataHandler()

# Routes
@app.post("/api/generate", response_model=GenerateResponse)
async def generate_with_mcp(
    request: GenerateRequest,
    mcp_handler: MCPHandler = Depends(get_mcp_handler)
):
    """Generate text using MCP."""
    try:
        # Convert Pydantic models to MCP objects
        mcp_messages = []
        
        for msg in request.messages:
            contexts = []
            if msg.contexts:
                for ctx in msg.contexts:
                    contexts.append(mcp_handler.create_context(
                        context_type=ctx.type,
                        content=ctx.content,
                        metadata=ctx.metadata or {}
                    ))
            
            mcp_messages.append(mcp_handler.create_message(
                role=msg.role,
                content=msg.content,
                contexts=contexts
            ))
        
        # Generate response
        response_content = mcp_handler.generate_response(
            messages=mcp_messages,
            model=request.model,
            model_family=request.model_family
        )
        
        return GenerateResponse(content=response_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag", response_model=GenerateResponse)
async def query_rag(
    request: RAGRequest,
    rag_handler: RAGWithMCP = Depends(get_rag_handler)
):
    """Query the RAG system using MCP."""
    try:
        response_content = rag_handler.query(
            user_query=request.query,
            top_k=request.top_k
        )
        
        return GenerateResponse(content=response_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/table_data", response_model=GenerateResponse)
async def analyze_table_data(
    request: TableDataRequest,
    structured_handler: StructuredDataHandler = Depends(get_structured_data_handler)
):
    """Analyze table data using FastMCP."""
    try:
        # Convert dict to DataFrame
        df = pd.DataFrame(request.table_data)
        
        response_content = structured_handler.process_table_data(
            dataframe=df,
            user_query=request.query,
            table_name=request.table_name
        )
        
        return GenerateResponse(content=response_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/json_data", response_model=GenerateResponse)
async def analyze_json_data(
    request: JSONDataRequest,
    structured_handler: StructuredDataHandler = Depends(get_structured_data_handler)
):
    """Analyze JSON data using FastMCP."""
    try:
        response_content = structured_handler.process_json_data(
            json_data=request.json_data,
            user_query=request.query,
            data_description=request.data_description
        )
        
        return GenerateResponse(content=response_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)