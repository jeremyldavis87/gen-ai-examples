# MCP Examples

This directory contains example applications and use cases for the Model Context Protocol (MCP), following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Examples](#examples)
3. [Prerequisites](#prerequisites)
4. [Running the Examples](#running-the-examples)
5. [Integration with AWS Services](#integration-with-aws-services)

## Overview

The examples in this directory demonstrate how to use the Model Context Protocol (MCP) framework in various applications and scenarios. These examples showcase the flexibility and power of the MCP approach for structured interactions with AI models.

## Examples

### Basic MCP Example

Demonstrates the basic usage of the MCP framework with simple context structures and responses.

### FastMCP Example

Shows how to use the high-performance FastMCP implementation for improved throughput and reduced latency.

### RAG with MCP Example

Illustrates how to integrate MCP with Retrieval-Augmented Generation (RAG) systems using PostgreSQL with pgvector.

### API with MCP Example

Demonstrates how to build a FastAPI-based API service that uses MCP for handling requests and generating responses.

### Structured Data Extraction Example

Shows how to use FastMCP for extracting structured data from unstructured text according to a defined schema.

## Prerequisites

Before running the examples, ensure you have:

1. Installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   ```bash
   cp .env.sample .env
   # Edit .env with your AWS credentials and configuration
   ```

3. Set up the PostgreSQL database with pgvector (for RAG examples):
   ```bash
   python vector_db/pgvector_setup.py
   ```

## Running the Examples

### Basic MCP Example

```bash
python -m mcp.examples.basic_mcp_example
```

### FastMCP Example

```bash
python -m mcp.examples.fast_mcp_example
```

### RAG with MCP Example

```bash
python -m mcp.examples.rag_mcp_example
```

### API with MCP Example

```bash
python -m mcp.examples.api_mcp_example
```

### Structured Data Extraction Example

```bash
python -m mcp.examples.structured_data_example
```

## Integration with AWS Services

The examples demonstrate integration with various AWS services:

### Amazon S3 Integration

The S3 integration example shows how to use MCP with documents stored in Amazon S3:

```bash
python -m mcp.examples.s3_mcp_example
```

This example demonstrates:
- Loading documents from S3 buckets
- Processing document content with MCP
- Storing results back to S3

### Amazon Cognito Integration

The Cognito integration example shows how to use MCP with authenticated users:

```bash
python -m mcp.examples.cognito_mcp_example
```

This example demonstrates:
- Authenticating users with Amazon Cognito
- Personalizing MCP responses based on user information
- Handling authentication tokens and session management

### AWS Lambda Integration

The Lambda integration example shows how to deploy MCP as a serverless function:

```bash
python -m mcp.examples.lambda_mcp_example
```

This example demonstrates:
- Packaging MCP for AWS Lambda
- Handling Lambda events and responses
- Integrating with other AWS services from Lambda

### AWS Amplify Integration

The Amplify integration example shows how to use MCP with AWS Amplify Gen 2:

```bash
python -m mcp.examples.amplify_mcp_example
```

This example demonstrates:
- Building a React front-end with AWS Amplify Gen 2
- Connecting the front-end to MCP-powered APIs
- Handling authentication and authorization

These examples provide a solid foundation for building your own applications using the MCP framework and AWS services.
