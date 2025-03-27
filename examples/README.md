# Usage Examples

This directory contains example scripts and applications that demonstrate how to use the gen-ai-examples project, following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Examples](#examples)
3. [Prerequisites](#prerequisites)
4. [Running the Examples](#running-the-examples)
5. [Creating Your Own Examples](#creating-your-own-examples)

## Overview

The examples in this directory showcase various features and capabilities of the gen-ai-examples project. They are designed to be easy to understand and modify, providing a starting point for your own applications.

## Examples

This directory includes the following examples:

### AI Gateway Examples

Demonstrates how to use the AI Gateway client to interact with various AI models:

- Text generation with different models and parameters
- Generating embeddings for text
- Handling errors and retries
- Advanced API usage

### Agent Examples

Shows how to use the various agents provided by the project:

- Code assistant for generating and analyzing code
- Data analysis agent for extracting insights from data
- RAG agent for knowledge retrieval and question answering
- Task agent for executing complex workflows

### Application Examples

Provides examples of complete applications built with the project:

- API service with FastAPI
- Chat application with React frontend
- Document question-answering system
- RAG application with knowledge management

### Vector Database Examples

Demonstrates how to use the PostgreSQL vector database with pgvector:

- Storing and retrieving vectors
- Similarity search
- Managing vector collections
- Integration with LangChain and other frameworks

### MCP Examples

Shows how to use the Model Context Protocol (MCP) for structured interactions with AI models:

- Basic MCP usage
- FastMCP for high-performance applications
- Integrating MCP with APIs and RAG systems
- Structured data extraction with FastMCP

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

3. Set up the PostgreSQL database with pgvector (for vector database examples):
   ```bash
   python vector_db/pgvector_setup.py
   ```

## Running the Examples

Each example can be run directly using Python:

```bash
# Run an AI Gateway example
python -m examples.ai_gateway_example

# Run a RAG agent example
python -m examples.rag_agent_example

# Run a vector database example
python -m examples.vector_db_example
```

Some examples may require additional setup or configuration. Check the comments at the beginning of each example file for specific instructions.

## Creating Your Own Examples

You can use these examples as a starting point for your own applications. Here's a general approach:

1. Copy an existing example that's close to what you want to build
2. Modify the code to suit your needs
3. Update the configuration as necessary
4. Test and iterate

When creating your own examples, follow these best practices:

- Use environment variables for configuration
- Handle errors appropriately
- Include comments explaining key concepts
- Follow the AWS and Python best practices used in the existing examples

For more detailed information on using the project, refer to the guides in the `docs/guides/` directory.
