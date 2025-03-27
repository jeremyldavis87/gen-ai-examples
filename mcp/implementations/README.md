# MCP Implementations

This directory contains various implementations of the Model Context Protocol (MCP), following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Implementations](#implementations)
3. [Usage](#usage)
4. [Integration with AWS Services](#integration-with-aws-services)
5. [Best Practices](#best-practices)

## Overview

The implementations directory contains different versions and variations of the Model Context Protocol (MCP) framework. Each implementation is designed for specific use cases or performance requirements while maintaining a consistent interface.

## Implementations

### Base MCP

The base MCP implementation provides a standard framework for structured interactions with AI models. It includes:

- Context management for AI model interactions
- Consistent input and output formats
- Integration with the AI Gateway
- Basic error handling and retries

### FastMCP

The FastMCP implementation is optimized for high-performance applications. It includes:

- Asynchronous processing for improved throughput
- Optimized context handling for reduced latency
- Batched processing capabilities
- Advanced caching mechanisms

### RAGMCP

The RAGMCP implementation is specialized for Retrieval-Augmented Generation applications. It includes:

- Integration with vector databases (PostgreSQL with pgvector)
- Document retrieval and context augmentation
- Relevance scoring and filtering
- Source attribution in responses

## Usage

### Base MCP

```python
from mcp.implementations.base_mcp import BaseMCP
from ai_gateway.client import AIGatewayClient

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Initialize BaseMCP
mcp = BaseMCP(gateway_client)

# Define a context
context = {
    "system_message": "You are a helpful assistant that provides information about countries.",
    "examples": [
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."}
    ],
    "user_message": "What is the capital of Germany?"
}

# Get a response using BaseMCP
response = mcp.get_response(context)
print(response)
```

### FastMCP

```python
from mcp.implementations.fast_mcp import FastMCP
from ai_gateway.client import AIGatewayClient
import asyncio

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Initialize FastMCP
fast_mcp = FastMCP(gateway_client)

# Define multiple contexts for batch processing
contexts = [
    {
        "system_message": "You are a helpful assistant that provides information about countries.",
        "user_message": "What is the capital of Germany?"
    },
    {
        "system_message": "You are a helpful assistant that provides information about countries.",
        "user_message": "What is the capital of Japan?"
    }
]

# Process contexts asynchronously
async def process_contexts():
    tasks = [fast_mcp.get_response_async(context) for context in contexts]
    responses = await asyncio.gather(*tasks)
    return responses

responses = asyncio.run(process_contexts())
for response in responses:
    print(response)
```

### RAGMCP

```python
from mcp.implementations.rag_mcp import RAGMCP
from ai_gateway.client import AIGatewayClient
from vector_db.pgvector_client import PGVectorClient

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Initialize the PGVector client
pgvector_client = PGVectorClient()

# Initialize RAGMCP
rag_mcp = RAGMCP(gateway_client, pgvector_client)

# Define a context
context = {
    "system_message": "You are a helpful assistant that provides information based on the retrieved documents.",
    "user_message": "What are the key features of AWS Amplify Gen 2?",
    "collection_name": "aws_documentation",
    "num_results": 3
}

# Get a response using RAGMCP
response = rag_mcp.get_response(context)
print(response)
```

## Integration with AWS Services

The MCP implementations integrate with various AWS services:

### Amazon S3 Integration

```python
from mcp.implementations.s3_mcp import S3MCP
from ai_gateway.client import AIGatewayClient
import os

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Initialize S3MCP
s3_mcp = S3MCP(
    gateway_client,
    bucket_name=os.getenv("S3_BUCKET"),
    region_name="us-east-1"
)

# Define a context
context = {
    "system_message": "You are a helpful assistant that provides information about the documents in the S3 bucket.",
    "user_message": "Summarize the contents of the latest quarterly report.",
    "document_prefix": "reports/quarterly/"
}

# Get a response using S3MCP
response = s3_mcp.get_response(context)
print(response)
```

### Amazon Cognito Integration

```python
from mcp.implementations.cognito_mcp import CognitoMCP
from ai_gateway.client import AIGatewayClient
import os

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Initialize CognitoMCP
cognito_mcp = CognitoMCP(
    gateway_client,
    user_pool_id=os.getenv("COGNITO_USER_POOL_ID"),
    app_client_id=os.getenv("COGNITO_APP_CLIENT_ID"),
    region_name="us-east-1"
)

# Define a context with user information
context = {
    "system_message": "You are a helpful assistant that provides personalized information.",
    "user_message": "What are my recent orders?",
    "user_id": "user123"
}

# Get a personalized response using CognitoMCP
response = cognito_mcp.get_response(context)
print(response)
```

## Best Practices

1. **Context Management**: Keep contexts concise and relevant to improve model performance.

2. **Error Handling**: Implement proper error handling and retries for robustness.

3. **Security**: Never include sensitive information in the context.

4. **Performance Optimization**: Use FastMCP for high-throughput applications.

5. **Monitoring**: Log MCP interactions for debugging and improvement.

6. **Testing**: Validate MCP implementations with unit and integration tests.

7. **Documentation**: Document the expected format and behavior of each implementation.

Following these best practices ensures reliable and efficient use of the MCP implementations in your applications.
