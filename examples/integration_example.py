# examples/integration_example.py
import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_handler import ModelHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize ModelHandler as a singleton
model_handler = ModelHandler()

# Set up custom configurations for organization-specific use cases
model_handler.set_default_config(
    task_type="healthcare_query",
    config={
        "primary": {"model": "gpt4o", "model_family": "openai", "temperature": 0.3, "max_tokens": 1000},
        "fallbacks": [
            {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.3, "max_tokens": 1000},
        ]
    }
)

model_handler.set_default_config(
    task_type="medical_records_summary",
    config={
        "primary": {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.2, "max_tokens": 1500},
        "fallbacks": [
            {"model": "gpt4o", "model_family": "openai", "temperature": 0.2, "max_tokens": 1500},
        ]
    }
)

model_handler.set_default_config(
    task_type="claim_processing",
    config={
        "primary": {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.1, "max_tokens": 800},
        "fallbacks": [
            {"model": "gpt4o", "model_family": "openai", "temperature": 0.1, "max_tokens": 800},
            {"model": "llama3-70b", "model_family": "llama", "temperature": 0.1, "max_tokens": 800},
        ]
    }
)

# Initialize FastAPI app
app = FastAPI(title="Healthcare AI Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models
class MessageItem(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: List[MessageItem]
    task_type: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class GenerateResponse(BaseModel):
    content: str
    model_used: str
    model_family: str
    fallback_used: bool

class HealthResponse(BaseModel):
    status: str
    version: str
    metrics: Dict[str, Any]

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request path: {request.url.path}")
    response = await call_next(request)
    return response

# Dependency to get model handler
def get_model_handler():
    return model_handler

# API routes
@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest, handler: ModelHandler = Depends(get_model_handler)):
    """Generate text for healthcare-related applications."""
    try:
        # Convert Pydantic models to dictionaries
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Track metrics before the call
        total_calls_before = handler.performance_metrics["total_calls"]
        fallback_calls_before = handler.performance_metrics["fallback_calls"]
        
        # Generate text using the model handler
        response = handler.generate_text(
            messages=messages,
            task_type=request.task_type,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Check if fallback was used
        fallback_used = handler.performance_metrics["fallback_calls"] > fallback_calls_before
        
        # Extract content and model info
        if "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]
            model_used = response.get("model", "unknown")
            model_family = "openai"
        elif "content" in response:
            content = response["content"][0]["text"]
            model_used = response.get("model", "unknown")
            model_family = "anthropic"
        else:
            content = str(response)
            model_used = "unknown"
            model_family = "unknown"
        
        return GenerateResponse(
            content=content,
            model_used=model_used,
            model_family=model_family,
            fallback_used=fallback_used
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=HealthResponse)
async def health_check(handler: ModelHandler = Depends(get_model_handler)):
    """Health check and metrics endpoint."""
    metrics = handler.get_performance_metrics()
    
    # Calculate success rate
    success_rate = metrics["successful_calls"] / metrics["total_calls"] if metrics["total_calls"] > 0 else 0
    fallback_rate = metrics["fallback_calls"] / metrics["successful_calls"] if metrics["successful_calls"] > 0 else 0
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        metrics={
            "total_calls": metrics["total_calls"],
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
            "avg_latency": metrics["avg_latency"]
        }
    )

@app.post("/api/reset-metrics")
async def reset_metrics(handler: ModelHandler = Depends(get_model_handler)):
    """Reset performance metrics."""
    handler.reset_performance_metrics()
    return {"status": "success", "message": "Metrics reset successfully"}

# Main function to run the app
def main():
    """Run the FastAPI application."""
    uvicorn.run("integration_example:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()