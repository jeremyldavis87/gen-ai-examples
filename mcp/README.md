# Model Context Protocol (MCP)

This directory contains implementations and examples of the Model Context Protocol (MCP), a framework for structured interactions with AI models, following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Features](#features)
4. [Installation](#installation)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [Integration with AWS Services](#integration-with-aws-services)

## Overview

The Model Context Protocol (MCP) provides a structured way to interact with AI models, ensuring consistent inputs and outputs. It enables more reliable and predictable AI model behavior by defining clear context structures and response formats.

## Directory Structure

- `implementations/`: Contains different implementations of the MCP framework
- `examples/`: Contains example applications using MCP
- `mcp_implementation.py`: Base MCP implementation
- `fastmcp_implementation.py`: High-performance MCP implementation
- `api_with_mcp.py`: Example of integrating MCP with APIs
- `rag_with_mcp.py`: Example of using MCP with RAG systems
- `structured_data_with_fastmcp.py`: Example of extracting structured data with FastMCP

## Features

- Structured context management for AI model interactions
- Consistent input and output formats
- Support for various AI models through the AI Gateway
- High-performance implementation with FastMCP
- Integration with APIs, RAG systems, and other applications
- Structured data extraction capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/gen-ai-examples.git
cd gen-ai-examples

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.sample .env
# Edit .env with your AWS credentials and configuration
```

## Basic Usage

```python
from mcp.mcp_implementation import MCP
from ai_gateway.client import AIGatewayClient

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Initialize MCP
mcp = MCP(gateway_client)

# Define a context
context = {
    "system_message": "You are a helpful assistant that provides information about countries.",
    "examples": [
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."}
    ],
    "user_message": "What is the capital of Germany?"
}

# Get a response using MCP
response = mcp.get_response(context)
print(response)
```

## Advanced Usage

### Using FastMCP for High Performance

```python
from mcp.fastmcp_implementation import FastMCP
from ai_gateway.client import AIGatewayClient

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Initialize FastMCP
fast_mcp = FastMCP(gateway_client)

# Define a context
context = {
    "system_message": "You are a helpful assistant that provides information about countries.",
    "examples": [
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."}
    ],
    "user_message": "What is the capital of Germany?"
}

# Get a response using FastMCP
response = fast_mcp.get_response(context)
print(response)
```

### Extracting Structured Data

```python
from mcp.structured_data_with_fastmcp import extract_structured_data
from ai_gateway.client import AIGatewayClient

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Define the schema for extraction
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"},
        "interests": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age", "email"]
}

# Extract structured data from text
text = "My name is John Doe, I'm 30 years old. You can reach me at john.doe@example.com. I enjoy hiking, reading, and photography."
data = extract_structured_data(text, schema, gateway_client)
print(data)
```

### Integrating with APIs

```python
from mcp.api_with_mcp import MCPAPIHandler
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Initialize the MCP API handler
mcp_handler = MCPAPIHandler()

# Define request and response models
class MCPRequest(BaseModel):
    user_message: str

class MCPResponse(BaseModel):
    response: str

# Create an API endpoint
@app.post("/api/mcp", response_model=MCPResponse)
async def mcp_endpoint(request: MCPRequest):
    response = mcp_handler.handle_request(request.user_message)
    return {"response": response}
```

## Integration with AWS Services

The MCP framework integrates with various AWS services:

### Integration with Amazon S3

```python
from mcp.examples.s3_integration import S3MCPHandler
import os

# Initialize the S3 MCP handler
s3_handler = S3MCPHandler(
    bucket_name=os.getenv("S3_BUCKET"),
    region_name="us-east-1"
)

# Store MCP context in S3
context_id = s3_handler.store_context(context)

# Retrieve MCP context from S3
retrieved_context = s3_handler.retrieve_context(context_id)

# Process context and store response
response_id = s3_handler.process_and_store(context_id)

# Retrieve response from S3
response = s3_handler.retrieve_response(response_id)
```

### Integration with Amazon Cognito

```python
from mcp.examples.cognito_integration import CognitoMCPHandler
import os

# Initialize the Cognito MCP handler
cognito_handler = CognitoMCPHandler(
    user_pool_id=os.getenv("COGNITO_USER_POOL_ID"),
    app_client_id=os.getenv("COGNITO_APP_CLIENT_ID"),
    region_name="us-east-1"
)

# Authenticate and get user information
user_info = cognito_handler.authenticate(username="user@example.com", password="password")

# Get personalized MCP response
response = cognito_handler.get_personalized_response(
    user_id=user_info["sub"],
    user_message="What are my recent orders?"
)
```

For more detailed examples and use cases, see the files in the `examples/` directory.
