# Gen AI Code Examples
A comprehensive framework for managing AI model usage with automatic fallback, extensive testing, and benchmarking capabilities.
### Overview
This package provides a robust solution for working with multiple AI models through a centralized internal AI Gateway. It includes:

Model Handler: A unified interface for model interaction with automatic fallback capabilities
Testing Framework: Tools to evaluate model performance using deepeval metrics
Benchmarking Tools: Compare different models across various tasks and use cases
Example Applications: Ready-to-use applications and integration examples

### Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Core Components](#core-components)
- [Model Handler](#model-handler)
- [Testing Framework](#testing-framework)
- [Benchmarking](#benchmarking)
- [Example Applications](#example-applications)
- [Available Models](#available-models)
- [Vector Database Integration](#vector-database-integration)
- [Best Practices](#best-practices)

### Features

#### Model Handler

Easy model switching for different use cases
Automatic fallback to alternative models on failure
Default configurations for common tasks
Performance tracking and metrics
Built-in error handling and retries

#### Testing Framework

Comprehensive model evaluation using deepeval
Support for various evaluation metrics (factual consistency, relevancy, etc.)
Test case management and persistence
Model comparison with visualizations
Export functionality for further analysis

#### Benchmarking

Task-specific performance measurement
Multi-model comparison
Parallel execution support
Detailed reporting with visualizations
Token usage and latency tracking

#### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-model-handler.git
cd ai-model-handler

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.sample .env
# Edit .env with your credentials
```

### Core Components
The package consists of several key components:
```bash
ai-model-handler/
├── ai_gateway/                   # AI Gateway client
│   ├── __init__.py
│   └── client.py                 # Gateway client implementation
├── vector_db/                    # Vector database utilities
│   ├── __init__.py
│   ├── pgvector_setup.py         # DB setup script
│   └── pgvector_client.py        # Vector DB client
├── models/                       # Model handling
│   ├── __init__.py
│   └── model_handler.py          # Main model handler with fallbacks
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
│   ├── model_tester.py           # Model testing with deepeval
│   └── benchmark.py              # Benchmarking utilities
├── apps/                         # Example applications
│   ├── __init__.py
│   ├── rag_app.py                # RAG application
│   ├── document_qa.py            # Document Q&A system
│   ├── chat_app.py               # Chat application
│   └── api_service.py            # API service
├── examples/                     # Usage examples
│   ├── model_test_example.py     # Testing example
│   ├── production_handler_example.py  # Production usage 
│   ├── integration_example.py    # API integration
│   ├── client_example.py         # Client usage
│   └── comprehensive_integration.py   # Full integration
└── static/                       # Static files for web apps
    └── index.html                # Chat UI
```

### Model Handler
The ModelHandler class provides a unified interface for interacting with AI models through your internal AI Gateway.
#### Basic Usage
```python
from models.model_handler import ModelHandler

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

#### Customizing Default Models
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

### Testing Framework
The testing framework uses deepeval to evaluate model performance.
#### Creating Test Datasets
```python
from testing.model_tester import ModelTester

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
#### Running Tests
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
#### Comparing Models
```python
# Compare multiple model results
comparison = tester.compare_models([test_result_id_1, test_result_id_2, test_result_id_3])

# Export results to CSV
csv_path = tester.export_results_to_csv(test_result_id)
```

### Benchmarking
The benchmarking tools allow you to compare different models across various tasks.
#### Creating Benchmark Files
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
#### Running Benchmarks
```python
from testing.benchmark import ModelBenchmark

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

### Example Applications
#### RAG Application
```python
from apps.rag_app import ingest_documents, query_documents

# Ingest documents
doc_ids = ingest_documents(["data/doc1.txt", "data/doc2.pdf"])

# Query the system
answer = query_documents("What is the company policy on remote work?")
```

#### Document Q&A
```bash
# Ingest documents
python apps/document_qa.py ingest --directory /path/to/documents --recursive

# Ask a question
python apps/document_qa.py query --question "What is our approach to AI ethics?"
```

#### API Service
```bash
# Run the API service
python apps/api_service.py
```

#### Integrated Service Manager
```python
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

### Available Models
#### OpenAI Models

* GPT-4 (gpt4)
* GPT-4o (gpt4o)
* o1 (o1)
* o1-mini (o1-mini)
* o3-mini (o3-mini)
* Embedding model (text-embedding-3-large)

#### Anthropic Models

* Claude Sonnet 3.7 (sonnet-3.7)
* Claude Sonnet 3.5 (sonnet-3.5)
* Claude Haiku 3.5 (haiku-3.5)

#### Llama Models

* Llama 3 8B (llama3-8b)
* Llama 3 70B (llama3-70b)

#### Mistral Models

* Mistral 7B (mistral-7b)
* Mistral 8x7B (mistral-8x7b)

### Vector Database Integration
This package includes utilities for working with PGVector for vector storage and retrieval.
#### Setting Up PGVector
```python
from vector_db.pgvector_setup import setup_pgvector

# Set up PGVector database with tables and indices
setup_pgvector()
```
#### Using the Vector Database
```python
from vector_db.pgvector_client import PGVectorClient
from models.model_handler import ModelHandler

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
### Best Practices
#### Model Selection

Use gpt4o or sonnet-3.7 for complex reasoning tasks
Use haiku-3.5 for faster responses in simpler scenarios
Set appropriate fallbacks based on task importance

#### Fallback Configuration

Arrange fallbacks in order of preference
Consider different model families for diverse fallback options
Set appropriate retry attempts for critical tasks

#### Performance Monitoring

Regularly review get_performance_metrics() output
Track fallback usage to identify potential issues
Reset metrics periodically for fresh monitoring

#### Testing

Create comprehensive test datasets for your specific use cases
Test periodically as models and requirements evolve
Compare models using metrics relevant to your application

#### Production Integration

Implement proper error handling around model handler calls
Use custom task types for better organization and metrics
Consider parallel execution for benchmark tasks

### Configuration
#### Environment Variables
The package uses environment variables for configuration. Here's what you need to set in your .env file:
```bash
# AI Gateway Configuration
AI_GATEWAY_URL=https://aigateway-prod.apps-1.gp-1-prod.openshift.cignacloud.com/api/v1/ai
PROJECT_NAME=your_project_name
API_KEY=your_api_key

# Vector Database Configuration
PGVECTOR_HOST=your-pg-instance.rds.amazonaws.com
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password
PGVECTOR_DATABASE=vector_db

# Testing Configuration
DEEPEVAL_API_KEY=your_deepeval_api_key  # Optional, for additional features
```
This README provides a comprehensive overview of the AI Model Handler, Testing Framework, Vector Database Integration, Configuration, and Examples. Each component is designed to work together seamlessly, providing a robust solution for working with multiple AI models through the internal AI Gateway.




--------------------------

Let me continue with the comprehensive README file:
markdownCopy# GenAI Developer Enablement Package

This package provides everything developers need to build production-ready GenAI applications using the internal AI Gateway.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Core Components](#core-components)
4. [Available Models](#available-models)
5. [Vector Database Setup](#vector-database-setup)
6. [LangChain Integration](#langchain-integration)
7. [Agentic Frameworks](#agentic-frameworks)
8. [Example Applications](#example-applications)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Access to the internal AI Gateway

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-org/genai-developer-enablement.git
   cd genai-developer-enablement

Install dependencies:
bashCopypip install -r requirements.txt

Create your .env file by copying the template:
bashCopycp .env.sample .env

Update the .env file with your credentials and project details.
Run the setup script:
bashCopypython setup.py


Configuration
Environment Variables
The package uses environment variables for configuration. Here's what you need to set in your .env file:
Copy# AI Gateway Configuration
AI_GATEWAY_URL=https://aigateway-prod.apps-1.gp-1-prod.openshift.cignacloud.com/api/v1/ai
PROJECT_NAME=your_project_name
API_KEY=your_api_key

# Vector Database Configuration
PGVECTOR_HOST=your-pg-instance.rds.amazonaws.com
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password
PGVECTOR_DATABASE=vector_db
Core Components
AI Gateway Client
The AIGatewayClient class provides a unified interface to interact with the AI Gateway:
pythonCopyfrom ai_gateway.client import AIGatewayClient

# Initialize client
client = AIGatewayClient()

# Generate text
response = client.generate_text(
    model="gpt4o",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    model_family="openai"
)

# Generate embeddings
embeddings = client.generate_embeddings(
    texts=["Embed this text"],
    model="text-embedding-3-large"
)
PGVector Client
The PGVectorClient provides methods to interact with the PGVector database:
pythonCopyfrom vector_db.pgvector_client import PGVectorClient

# Initialize client
pgvector_client = PGVectorClient()

# Insert embeddings
doc_ids = pgvector_client.insert_embeddings(
    contents=["Document 1", "Document 2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadata=[{"source": "file1.txt"}, {"source": "file2.txt"}]
)

# Search for similar documents
results = pgvector_client.search_similar(
    query_embedding=[0.1, 0.2, ...],
    limit=5,
    similarity_threshold=0.7
)

# Close connection
pgvector_client.close()
Available Models
OpenAI Models

GPT-4 (gpt4)
GPT-4o (gpt4o)
o1 (o1)
o1-mini (o1-mini)
o3-mini (o3-mini)
Embedding model (text-embedding-3-large)

Anthropic Models

Claude Sonnet 3.7 (sonnet-3.7)
Claude Sonnet 3.5 (sonnet-3.5)
Claude Haiku 3.5 (haiku-3.5)

Llama Models

Llama 3 8B (llama3-8b)
Llama 3 70B (llama3-70b)

Mistral Models

Mistral 7B (mistral-7b)
Mistral 8x7B (mistral-8x7b)

Vector Database Setup
Setting up PGVector on AWS RDS

Create a parameter group for pgvector:
bashCopyaws rds create-db-parameter-group \
    --db-parameter-group-name pgvector-pg-13 \
    --db-parameter-group-family postgres13 \
    --description "Parameter group for pgvector extension"

Create the RDS instance:
bashCopyaws rds create-db-instance \
    --db-instance-identifier pgvector-instance \
    --db-parameter-group-name pgvector-pg-13 \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --engine-version 13.4 \
    --allocated-storage 20 \
    --master-username postgres \
    --master-user-password your_strong_password \
    --port 5432 \
    --no-publicly-accessible \
    --vpc-security-group-ids sg-your_security_group_id

Run the setup script to create the necessary tables and indices:
bashCopypython vector_db/pgvector_setup.py


LangChain Integration
Using LangChain with the AI Gateway
pythonCopyfrom langchain_utils.gateway_integration import get_langchain_llm, get_langchain_embeddings

# Get a LangChain LLM
llm = get_langchain_llm(model_name="gpt4o", model_family="openai")

# Use the LLM
result = llm.invoke("What is the capital of France?")
print(result.content)

# Get embeddings
embeddings = get_langchain_embeddings(model_name="text-embedding-3-large")
vectors = embeddings.embed_documents(["Embed this text"])
Agentic Frameworks
This package includes several pre-built agentic frameworks:
RAG Agent
pythonCopyfrom agents.rag_agent import query_rag_agent

# Query the RAG agent
answer = query_rag_agent("What is the company policy on remote work?")
print(answer)
Task Planning Agent
pythonCopyfrom agents.task_agent import solve_task

# Solve a complex task
result = solve_task("Find the top 5 competitors in our industry and summarize their strengths")
print(result["final_answer"])
Data Analysis Agent
pythonCopyfrom agents.data_analysis_agent import analyze_data

# Analyze data
result = analyze_data(
    question="What's the trend in our quarterly sales?",
    data_path="data/sales.csv"
)
print(result["final_answer"])
Code Assistant
pythonCopyfrom agents.code_assistant import ask_code_assistant

# Ask a coding question
answer = ask_code_assistant("Write a Python function to calculate the Fibonacci sequence")
print(answer)
Example Applications
Document Q&A System
bashCopy# Ingest documents
python apps/document_qa.py ingest --directory /path/to/documents --recursive

# Ask a question
python apps/document_qa.py query --question "What is the company's approach to AI ethics?"
Chat Application
bashCopy# Run the chat application
python apps/chat_app.py
Then open your browser to http://localhost:8000 to access the chat interface.
API Service
bashCopy# Run the API service
python apps/api_service.py
The API will be available at http://localhost:8000 with the following endpoints:

/generate - Generate text
/embed - Generate embeddings
/search - Search similar documents
/rag - Query the RAG system
/health - Health check

Best Practices
Working with Large Language Models

Clear Instructions: Always provide clear and specific instructions to the models.
Temperature Control: Use lower temperature (0.0-0.3) for factual responses and higher temperature (0.7-1.0) for creative content.
Token Management: Be mindful of token usage, especially with context length.
Error Handling: Always implement proper error handling for API calls.

Vector Database Management

Chunking Strategy: Choose an appropriate chunking strategy for documents (e.g., by paragraph, fixed size).
Metadata: Store useful metadata with embeddings for better filtering.
Index Optimization: Create appropriate indexes for your query patterns.
Regular Maintenance: Implement regular vacuuming and index rebuilding.

Security Considerations

API Key Management: Never hardcode API keys in source code.
Input Validation: Always validate user inputs before sending to the LLM.
Output Filtering: Implement content filters for generated outputs.
Rate Limiting: Implement rate limiting to prevent abuse.

Troubleshooting
Common Errors

Gateway Connection Issues:

Check network connectivity
Verify API key is valid
Ensure project name is correct


Vector Database Issues:

Check connection parameters
Verify pgvector extension is installed
Check if tables and indexes exist


Model Specific Errors:

Check model availability in the AI Gateway
Verify correct model family is specified
Check token limits



Support
For additional support, contact the AI Gateway team at aigateway-support@yourcompany.com
Copy
## Project Structure

Now let's provide a complete project structure as a reference for developers:
genai-developer-enablement/
├── .env.sample                    # Template for environment variables
├── README.md                      # Project documentation
├── setup.py                       # Setup script
├── requirements.txt               # Dependencies
├── ai_gateway/                    # AI Gateway client
│   ├── init.py
│   └── client.py                  # AIGatewayClient implementation
├── vector_db/                     # Vector database utilities
│   ├── init.py
│   ├── pgvector_setup.py          # Database setup script
│   └── pgvector_client.py         # PGVectorClient implementation
├── langchain_utils/               # LangChain integration
│   ├── init.py
│   └── gateway_integration.py     # LangChain integration with AI Gateway
├── agents/                        # Agentic frameworks
│   ├── init.py
│   ├── rag_agent.py               # RAG agent implementation
│   ├── task_agent.py              # Task planning agent
│   ├── data_analysis_agent.py     # Data analysis agent
│   └── code_assistant.py          # Code assistant agent
├── apps/                          # Example applications
│   ├── init.py
│   ├── rag_app.py                 # RAG application
│   ├── document_qa.py             # Document Q&A system
│   ├── chat_app.py                # Chat application
│   └── api_service.py             # API service
└── static/                        # Static files for web applications
└── index.html                 # Chat UI
Copy
This developer enablement package provides a comprehensive foundation for building GenAI applications using your internal AI Gateway. The code is designed to be modular and extensible, allowing developers to quickly get started with common use cases while providing the flexibility to customize for specific needs.

The examples cover a wide range of applications including:
- Retrieval-Augmented Generation (RAG)
- Task planning agents
- Data analysis
- Code assistance
- Document Q&A
- Chat interfaces

Developers can use this package as a starting point and extend it to meet their specific requirements.  