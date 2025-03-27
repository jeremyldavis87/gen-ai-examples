# AI Agents

This package provides a collection of specialized AI agents for different tasks, built on AWS managed services and following best practices for Python 3.12 development.

## Table of Contents

1. [Overview](#overview)
2. [Agents](#agents)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)

## Overview

The agents package contains specialized AI agents that leverage AWS services and modern AI models to perform various tasks. Each agent is designed to handle specific use cases while maintaining a consistent interface.

## Agents

### Code Assistant

The `code_assistant.py` module provides an AI-powered coding assistant that can:

- Generate code based on natural language descriptions
- Debug and fix issues in existing code
- Refactor code for better performance or readability
- Explain code functionality and suggest improvements

### Data Analysis Agent

The `data_analysis_agent.py` module offers capabilities for:

- Analyzing datasets and extracting insights
- Generating visualizations and reports
- Performing statistical analysis
- Identifying patterns and anomalies in data

### RAG Agent

The `rag_agent.py` module implements a Retrieval-Augmented Generation agent that:

- Retrieves relevant information from a knowledge base
- Augments AI responses with retrieved information
- Provides accurate and contextual answers to queries
- Integrates with AWS S3 for document storage

### Task Agent

The `task_agent.py` module offers a general-purpose task execution agent that:

- Breaks down complex tasks into manageable steps
- Executes tasks using appropriate tools and APIs
- Monitors progress and handles errors
- Reports results in a structured format

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
from agents.code_assistant import CodeAssistant
from agents.data_analysis_agent import DataAnalysisAgent
from agents.rag_agent import RAGAgent
from agents.task_agent import TaskAgent

# Initialize a code assistant
code_assistant = CodeAssistant()

# Generate code
code = code_assistant.generate_code(
    description="Create a function that calculates the factorial of a number",
    language="python"
)

# Initialize a RAG agent
rag_agent = RAGAgent(
    knowledge_base_path="s3://your-bucket/knowledge-base"
)

# Get an answer with context
response = rag_agent.get_answer(
    query="What is the capital of France?"
)
```

### Advanced Usage

```python
# Initialize a data analysis agent with custom configuration
data_agent = DataAnalysisAgent(
    model="gpt4o",
    aws_region="us-east-1",
    s3_bucket="your-data-bucket"
)

# Analyze a dataset
insights = data_agent.analyze_dataset(
    dataset_path="s3://your-data-bucket/dataset.csv",
    analysis_type="exploratory",
    visualization=True
)

# Initialize a task agent
task_agent = TaskAgent()

# Execute a complex task
result = task_agent.execute_task(
    task_description="Collect data from an API, clean it, analyze it, and generate a report",
    tools=["api_client", "data_cleaner", "analyzer", "report_generator"],
    output_format="pdf"
)
```

## Configuration

### Environment Variables

The agents use environment variables for configuration. Here's what you need to set in your `.env` file:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Amazon Cognito Configuration
COGNITO_USER_POOL_ID=your_user_pool_id
COGNITO_APP_CLIENT_ID=your_app_client_id

# S3 Configuration
S3_BUCKET=your_bucket_name

# AI Gateway Configuration
AI_GATEWAY_URL=https://your-ai-gateway-url.com/api/v1/ai
PROJECT_NAME=your_project_name
API_KEY=your_api_key
```

Each agent can be further configured through its constructor parameters to customize behavior for specific use cases.
