# AI Gateway Client

This package provides a client for interacting with AI services through a unified gateway, built on AWS managed services and following best practices for Python 3.12 development.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)

## Overview

The AI Gateway Client provides a unified interface for interacting with various AI models and services through a centralized gateway. It simplifies the process of making API calls to AI services and handles authentication, error handling, and response parsing.

## Features

- Unified interface for multiple AI models and services
- Integration with AWS Cognito for authentication
- Support for text generation, embeddings, and other AI capabilities
- Automatic error handling and retries
- Configurable timeouts and request parameters

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

## Usage

### Basic Usage

```python
from ai_gateway.client import AIGatewayClient

# Initialize the client
client = AIGatewayClient()

# Generate text
response = client.generate_text(
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    model="gpt4o",
    model_family="openai",
    temperature=0.7,
    max_tokens=1000
)

# Generate embeddings
embeddings = client.generate_embeddings(
    texts=["Embed this text"],
    model="text-embedding-3-large"
)
```

### Advanced Usage

```python
# Initialize with custom configuration
client = AIGatewayClient(
    api_url="https://custom-ai-gateway-url.com/api/v1/ai",
    project_name="custom_project",
    api_key="your_custom_api_key",
    timeout=30
)

# Make a custom API call
response = client.call_api(
    endpoint="/custom_endpoint",
    method="POST",
    data={
        "parameter1": "value1",
        "parameter2": "value2"
    },
    headers={
        "Custom-Header": "value"
    }
)
```

## Configuration

### Environment Variables

The client uses environment variables for configuration. Here's what you need to set in your `.env` file:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Amazon Cognito Configuration
COGNITO_USER_POOL_ID=your_user_pool_id
COGNITO_APP_CLIENT_ID=your_app_client_id

# AI Gateway Configuration
AI_GATEWAY_URL=https://your-ai-gateway-url.com/api/v1/ai
PROJECT_NAME=your_project_name
API_KEY=your_api_key
```

### Client Configuration

The `AIGatewayClient` can be configured through its constructor parameters:

- `api_url`: The URL of the AI Gateway API
- `project_name`: The name of your project in the AI Gateway
- `api_key`: Your API key for authentication
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum number of retries for failed requests
- `retry_delay`: Delay between retries in seconds

These parameters can be used to customize the client's behavior for specific use cases.
