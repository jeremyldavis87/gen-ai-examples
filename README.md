# Gen AI Examples

A comprehensive framework for building production-ready generative AI applications with AWS managed services, featuring automatic fallback capabilities, extensive testing, and benchmarking tools.

## Overview

This package provides a robust solution for working with multiple AI models through a centralized AI Gateway. It follows AWS best practices and leverages managed services where applicable.

Key features include:

- **Model Handler**: A unified interface for model interaction with automatic fallback capabilities
- **Testing Framework**: Tools to evaluate model performance using metrics
- **Benchmarking Tools**: Compare different models across various tasks and use cases
- **Example Applications**: Ready-to-use applications and integration examples
- **AWS Integration**: Leverages AWS managed services including Cognito, S3, RDS, and Amplify
- **Docker Support**: Complete Docker configuration for local development and production
- **Infrastructure as Code**: Terraform and Terragrunt configurations for AWS resources
- **CI/CD Pipelines**: GitHub Actions workflows for testing and deployment
- **Database Migrations**: SQLAlchemy with Alembic for database schema management

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Core Components](#core-components)
- [Model Handler](#model-handler)
- [Testing Framework](#testing-framework)
- [Benchmarking](#benchmarking)
- [Example Applications](#example-applications)
- [Available Models](#available-models)
- [Vector Database Integration](#vector-database-integration)
- [AWS Integration](#aws-integration)
- [Docker Setup](#docker-setup)
- [Infrastructure as Code](#infrastructure-as-code)
- [CI/CD Pipelines](#cicd-pipelines)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Features

### Model Handler

- Easy model switching for different use cases
- Automatic fallback to alternative models on failure
- Default configurations for common tasks
- Performance tracking and metrics
- Built-in error handling and retries

### Testing Framework

- Comprehensive model evaluation
- Support for various evaluation metrics (factual consistency, relevancy, etc.)
- Test case management and persistence
- Model comparison with visualizations
- Export functionality for further analysis

### Benchmarking

- Task-specific performance measurement
- Multi-model comparison
- Parallel execution support
- Detailed reporting with visualizations
- Token usage and latency tracking

## Installation

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/gen-ai-examples.git
cd gen-ai-examples

# Copy and edit the environment file
cp .env.sample .env
# Edit .env with your credentials

# Start the services using Docker Compose
docker-compose -f docker/docker-compose.yml up -d
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-org/gen-ai-examples.git
cd gen-ai-examples

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit the environment file
cp .env.sample .env
# Edit .env with your credentials

# Run the setup script
python setup.py
```

## Core Components

The package consists of several key components:

```bash
gen-ai-examples/
├── ai_gateway/                   # AI Gateway client
│   ├── __init__.py
│   └── client.py                 # Gateway client implementation
├── vector_db/                    # Vector database utilities
│   ├── __init__.py
│   ├── pgvector_setup.py         # PostgreSQL setup script
│   ├── pgvector_client.py        # Vector DB client
│   └── models.py                 # SQLAlchemy models
├── models/                       # Model handling
│   ├── __init__.py
│   └── model_handler.py          # Main model handler
├── langchain_utils/              # LangChain integration
│   ├── __init__.py
│   └── gateway_integration.py    # LangChain + AI Gateway integration
├── agents/                       # Agentic frameworks
│   ├── __init__.py
│   ├── rag_agent.py              # RAG agent implementation
│   ├── task_agent.py             # Task planning agent
│   ├── data_analysis_agent.py    # Data analysis agent
│   └── code_assistant.py         # Code assistant agent
├── testing/                      # Testing framework
│   ├── __init__.py
│   ├── model_tester.py           # Model testing
│   └── benchmark.py              # Benchmarking utilities
├── apps/                         # Example applications
│   ├── __init__.py
│   ├── rag_app.py                # RAG application
│   ├── document_qa.py            # Document Q&A system
│   ├── chat_app.py               # Chat application
│   └── api_service.py            # FastAPI service
├── examples/                     # Usage examples
│   ├── __init__.py
│   ├── model_test_example.py     # Testing example
│   ├── production_handler_example.py  # Production usage 
│   ├── integration_example.py    # API integration
│   ├── client_example.py         # Client usage
│   └── comprehensive_integration.py   # Full integration
├── mcp/                          # Model Context Protocol
│   ├── __init__.py
│   ├── implementations/          # MCP implementations
│   │   ├── __init__.py
│   │   ├── mcp_implementation.py
│   │   └── fastmcp_implementation.py
│   └── examples/                 # MCP usage examples
│       ├── __init__.py
│       ├── api_with_mcp.py
│       ├── rag_with_mcp.py
│       └── structured_data_with_fastmcp.py
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   ├── unit/                     # Unit tests
│   │   ├── __init__.py
│   │   ├── test_model_handler.py
│   │   └── test_ai_gateway.py
│   └── integration/              # Integration tests
│       ├── __init__.py
│       └── test_end_to_end.py
├── migrations/                   # Database migrations
│   ├── __init__.py
│   ├── env.py
│   └── versions/                 # Migration versions
│       ├── __init__.py
│       └── 001_create_documents_table.py
├── docker/                       # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── infrastructure/               # Infrastructure as Code
│   ├── terraform/                # Terraform configuration
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── terragrunt/               # Terragrunt configuration
│       └── terragrunt.hcl
├── .github/                      # GitHub configuration
│   └── workflows/                # GitHub Actions workflows
│       ├── ci.yml                # Continuous Integration
│       └── cd.yml                # Continuous Deployment
├── docs/                         # Documentation
│   ├── api/                      # API documentation
│   │   └── openapi.json          # OpenAPI specification
│   └── guides/                   # User guides
│       ├── getting_started.md
│       └── advanced_usage.md
└── static/                       # Static files for web apps
    └── index.html                # Chat UI
```

## Model Handler

The ModelHandler class provides a unified interface for interacting with AI models through your AI Gateway.

### Basic Usage

```python
from models import ModelHandler

# Initialize handler
model_handler = ModelHandler()

# Generate text with automatic fallback
response = model_handler.generate_text(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    task_type="text_generation"  # Optional task type
)

# Generate embeddings with automatic fallback
embeddings = model_handler.generate_embeddings(
    texts=["Embed this text"]
)
```

### Customizing Default Models

```python
# Set custom default configuration for a task type
model_handler.set_default_config(
    task_type="customer_support",
    config={
        "primary": {"model": "gpt4o", "model_family": "openai", "temperature": 0.3, "max_tokens": 1000},
        "fallbacks": [
            {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.3, "max_tokens": 1000},
        ]
    }
)

# Get performance metrics
metrics = model_handler.get_performance_metrics()
print(metrics)
```

## Testing Framework

The testing framework evaluates model performance.

### Creating Test Datasets

```python
from testing import ModelTester

tester = ModelTester()

# Create a test dataset
test_cases = [
    {
        "input": "What is the capital of France?",
        "expected_output": "The capital of France is Paris.",
        "description": "Simple geography question",
        "tags": ["geography", "factual"]
    },
    # Add more test cases...
]

dataset_id = tester.create_test_dataset(
    task_type="text_generation",
    test_cases=test_cases,
    dataset_name="geography_questions"
)
```

### Running Tests

```python
# Define model configuration
model_config = {
    "model": "gpt4o",
    "model_family": "openai",
    "temperature": 0.3,
    "max_tokens": 1000
}

# Run the test
test_result_id = tester.run_model_test(
    dataset_id=dataset_id,
    model_config=model_config,
    metrics=["answer_relevancy", "factual_consistency", "faithfulness"]
)
```

### Comparing Models

```python
# Compare multiple model results
comparison = tester.compare_models([test_result_id_1, test_result_id_2, test_result_id_3])

# Export results to CSV
csv_path = tester.export_results_to_csv(test_result_id)
```

## Benchmarking

The benchmarking tools allow you to compare different models across various tasks.

### Creating Benchmark Files

Create JSON files that define the benchmark tasks:

```json
{
  "name": "healthcare_ai_benchmark",
  "description": "Benchmark for healthcare-specific AI tasks",
  "version": "1.0",
  "tasks": [
    {
      "id": "medical_summary_1",
      "task_type": "medical_records_summary",
      "description": "Summarize patient visit notes",
      "messages": [
        {"role": "system", "content": "You are a medical records summarization assistant..."},
        {"role": "user", "content": "Patient: Jane Smith, 58 y.o. female..."}
      ]
    },
    // More tasks...
  ]
}
```

### Running Benchmarks

```python
from testing import ModelBenchmark, run_benchmark

# Initialize benchmark
benchmark = ModelBenchmark()

# Define models to test
models_to_test = [
    {
        "model": "gpt4o",
        "model_family": "openai",
        "temperature": 0.2,
        "max_tokens": 1000
    },
    {
        "model": "sonnet-3.7",
        "model_family": "anthropic",
        "temperature": 0.2,
        "max_tokens": 1000
    }
]

# Run benchmark
benchmark_id = benchmark.run_benchmark(
    benchmark_file="benchmarks/healthcare_benchmark.json",
    models=models_to_test,
    parallel=True,
    max_workers=4
)

# Results will be available in benchmark_results/[benchmark_id]/
```

## Example Applications

### RAG Application

```python
from apps import rag_app

# Ingest documents
doc_ids = rag_app.ingest_documents(["data/doc1.txt", "data/doc2.pdf"])

# Query the system
answer = rag_app.query_documents("What is the company policy on remote work?")
```

### Document Q&A

```bash
# Ingest documents
python -m apps.document_qa ingest --directory /path/to/documents --recursive

# Ask a question
python -m apps.document_qa query --question "What is our approach to AI ethics?"
```

### API Service with FastAPI

```bash
# Run the API service
python -m apps.api_service
```

### Integrated Service Manager

```python
from examples import run_comprehensive_example
from examples.comprehensive_integration import AIServiceManager

# Initialize manager
manager = AIServiceManager()

# Process a query
result = manager.process_query(
    messages=[{"role": "user", "content": "How does our insurance cover telehealth?"}],
    task_type="healthcare_query"
)

# Run benchmark
benchmark_id = manager.run_benchmark("benchmarks/healthcare_benchmark.json")

# Get performance metrics
metrics = manager.get_performance_metrics()
```

## Available Models

### OpenAI Models

* GPT-4 (gpt4)
* GPT-4o (gpt4o)
* o1 (o1)
* o1-mini (o1-mini)
* o3-mini (o3-mini)
* Embedding model (text-embedding-3-large)

### Anthropic Models

* Claude Sonnet 3.7 (sonnet-3.7)
* Claude Sonnet 3.5 (sonnet-3.5)
* Claude Haiku 3.5 (haiku-3.5)

### Llama Models

* Llama 3 8B (llama3-8b)
* Llama 3 70B (llama3-70b)

### Mistral Models

* Mistral 7B (mistral-7b)
* Mistral 8x7B (mistral-8x7b)

## Vector Database Integration

This package includes utilities for working with PostgreSQL with pgvector for vector storage and retrieval.

### Setting Up PGVector

```python
from vector_db import setup_pgvector

# Set up PGVector database with tables and indices
setup_pgvector()
```

### Using the Vector Database

```python
from vector_db import PGVectorClient
from models import ModelHandler

# Initialize clients
model_handler = ModelHandler()
pgvector_client = PGVectorClient()

# Generate embeddings for documents
texts = ["Document 1 content", "Document 2 content"]
embeddings = model_handler.generate_embeddings(texts=texts)

# Store in vector database
doc_ids = pgvector_client.insert_embeddings(
    contents=texts,
    embeddings=embeddings,
    metadata=[{"source": "doc1.txt"}, {"source": "doc2.txt"}]
)

# Search for similar documents
query_embedding = model_handler.generate_embeddings(texts=["Search query"])[0]
results = pgvector_client.search_similar(
    query_embedding=query_embedding,
    limit=5
)
```

## AWS Integration

This package is designed to work seamlessly with AWS managed services:

### Amazon Cognito Authentication

User authentication is handled through Amazon Cognito, supporting both username/password and social logins via OAuth for Google and LinkedIn.

### AWS Amplify for Front-End

The front-end applications are built to deploy with AWS Amplify Gen 2, providing a streamlined deployment process.

### Amazon S3 for File Storage

Document storage and file management is handled through Amazon S3 buckets.

### Amazon RDS for PostgreSQL

Vector database is implemented using PostgreSQL with pgvector on Amazon Aurora Serverless.

## Docker Setup

The project includes a complete Docker configuration for both development and production environments.

### Development Environment

```bash
# Start the development environment
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop the environment
docker-compose -f docker/docker-compose.yml down
```

### Production Build

```bash
# Build the production image
docker build -f docker/Dockerfile -t gen-ai-examples:latest .

# Run the container
docker run -p 8000:8000 --env-file .env gen-ai-examples:latest
```

## Infrastructure as Code

The project includes Terraform and Terragrunt configurations for AWS infrastructure.

### Terraform Deployment

```bash
# Navigate to the Terraform directory
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Plan the deployment
terraform plan

# Apply the changes
terraform apply
```

### Terragrunt Deployment

```bash
# Navigate to the Terragrunt directory
cd infrastructure/terragrunt

# Plan all resources
terragrunt run-all plan

# Apply all resources
terragrunt run-all apply
```

## CI/CD Pipelines

The project includes GitHub Actions workflows for CI/CD:

### Continuous Integration

The CI pipeline runs on every pull request and push to main branches:

- Runs unit and integration tests
- Checks code formatting and linting
- Verifies type hints with mypy
- Tests with PostgreSQL and pgvector

### Continuous Deployment

The CD pipeline runs on pushes to the main branch:

- Deploys infrastructure with Terraform/Terragrunt
- Builds and pushes Docker images to ECR
- Updates ECS services

## Best Practices

### Working with Large Language Models

* Clear Instructions: Always provide clear and specific instructions to the models.
* Temperature Control: Use lower temperature (0.0-0.3) for factual responses and higher temperature (0.7-1.0) for creative content.
* Token Management: Be mindful of token usage, especially with context length.
* Error Handling: Always implement proper error handling for API calls.

### Vector Database Management

* Chunking Strategy: Choose an appropriate chunking strategy for documents (e.g., by paragraph, fixed size).
* Metadata: Store useful metadata with embeddings for better filtering.
* Index Optimization: Create appropriate indexes for your query patterns.
* Regular Maintenance: Implement regular vacuuming and index rebuilding.

### Security Considerations

* API Key Management: Never hardcode API keys in source code. Use .env files as configured in this project.
* Input Validation: Always validate user inputs before sending to the LLM.
* Output Filtering: Implement content filters for generated outputs.
* Rate Limiting: Implement rate limiting to prevent abuse.

## Troubleshooting

### Common Errors

#### Gateway Connection Issues:

* Check network connectivity
* Verify API key is valid
* Ensure project name is correct

#### Vector Database Issues:

* Check connection parameters
* Verify pgvector extension is installed
* Check if tables and indexes exist

#### Model Specific Errors:

* Check model availability in the AI Gateway
* Verify correct model family is specified
* Check token limits

### Support

For additional support, contact the AI Gateway team at aigateway-support@yourcompany.com

---

This developer enablement package provides a comprehensive foundation for building GenAI applications using Python 3.12, FastAPI, SQLAlchemy, PostgreSQL, and AWS managed services. The code is designed to be modular and extensible, allowing developers to quickly get started with common use cases while providing the flexibility to customize for specific needs.

The examples cover a wide range of applications including:
- Retrieval-Augmented Generation (RAG)
- Task planning agents
- Data analysis
- Code assistance
- Document Q&A
- Chat interfaces

Developers can use this package as a starting point and extend it to meet their specific requirements.