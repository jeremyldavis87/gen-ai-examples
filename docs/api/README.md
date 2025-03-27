# API Documentation

This directory contains the OpenAPI specifications and API reference documentation for the gen-ai-examples project's APIs.

## Table of Contents

1. [Overview](#overview)
2. [API Specifications](#api-specifications)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
5. [Usage](#usage)

## Overview

The API documentation in this directory provides detailed specifications for the RESTful APIs exposed by the gen-ai-examples project. These APIs allow developers to interact with the project's AI capabilities programmatically.

## API Specifications

The API specifications follow the OpenAPI 3.0 standard and include:

- Detailed endpoint descriptions
- Request parameters and schemas
- Response formats and status codes
- Authentication requirements
- Error handling information

## Authentication

All APIs use Amazon Cognito for authentication. The authentication flow is as follows:

1. Users authenticate with Amazon Cognito to obtain a JWT token
2. The JWT token is included in the `Authorization` header of API requests
3. The API validates the token with Amazon Cognito
4. If the token is valid, the request is processed; otherwise, a 401 Unauthorized response is returned

Social logins via OAuth for Google and LinkedIn are also supported through Amazon Cognito.

## Endpoints

The API includes the following main endpoint groups:

### AI Gateway Endpoints

- `/api/v1/ai/generate-text`: Generate text using AI models
- `/api/v1/ai/generate-embeddings`: Generate embeddings for text
- `/api/v1/ai/analyze-image`: Analyze images using AI models

### Agent Endpoints

- `/api/v1/agents/code-assistant`: Interact with the code assistant agent
- `/api/v1/agents/data-analysis`: Perform data analysis tasks
- `/api/v1/agents/rag`: Use the RAG agent for knowledge retrieval
- `/api/v1/agents/task`: Execute tasks using the task agent

### Document Endpoints

- `/api/v1/documents/upload`: Upload documents to AWS S3
- `/api/v1/documents/query`: Query documents using natural language
- `/api/v1/documents/analyze`: Analyze document content

### Vector Database Endpoints

- `/api/v1/vectors/store`: Store vectors in the PostgreSQL database
- `/api/v1/vectors/search`: Search for similar vectors
- `/api/v1/vectors/delete`: Delete vectors from the database

## Usage

### Viewing the API Documentation

The API documentation can be viewed using Swagger UI or Redoc when the API service is running:

```bash
# Start the API service
python -m apps.api_service

# Access Swagger UI at http://localhost:8000/docs
# Access Redoc at http://localhost:8000/redoc
```

### Using the API with cURL

```bash
# Authenticate with Cognito (simplified example)
TOKEN=$(curl -X POST https://cognito-idp.us-east-1.amazonaws.com/ \
  -H 'Content-Type: application/x-amz-json-1.1' \
  -H 'X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth' \
  -d '{
    "AuthFlow": "USER_PASSWORD_AUTH",
    "ClientId": "your_client_id",
    "AuthParameters": {
      "USERNAME": "your_username",
      "PASSWORD": "your_password"
    }
  }' | jq -r '.AuthenticationResult.IdToken')

# Generate text
curl -X POST http://localhost:8000/api/v1/ai/generate-text \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "model": "gpt4o",
    "model_family": "openai",
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### Using the API with Python

```python
import requests
import boto3

# Get token from Cognito (simplified example)
client = boto3.client('cognito-idp', region_name='us-east-1')
response = client.initiate_auth(
    ClientId='your_client_id',
    AuthFlow='USER_PASSWORD_AUTH',
    AuthParameters={
        'USERNAME': 'your_username',
        'PASSWORD': 'your_password'
    }
)
token = response['AuthenticationResult']['IdToken']

# Generate text
response = requests.post(
    'http://localhost:8000/api/v1/ai/generate-text',
    headers={
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    },
    json={
        'messages': [{'role': 'user', 'content': 'Hello, how are you?'}],
        'model': 'gpt4o',
        'model_family': 'openai',
        'temperature': 0.7,
        'max_tokens': 1000
    }
)

print(response.json())
```

For more detailed information on using the API, refer to the guides in the `../guides/` directory.
