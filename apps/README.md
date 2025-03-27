# AI-Powered Applications

This package provides a collection of ready-to-use AI-powered applications built on AWS managed services, using Python 3.12, FastAPI, and React.

## Table of Contents

1. [Overview](#overview)
2. [Applications](#applications)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Deployment](#deployment)

## Overview

The apps package contains complete AI-powered applications that demonstrate the capabilities of the gen-ai-examples library. Each application is designed to solve specific use cases while following AWS best practices and modern development standards.

## Applications

### API Service

The `api_service.py` module provides a FastAPI-based API service that:

- Exposes AI capabilities through RESTful endpoints
- Handles authentication using Amazon Cognito
- Manages rate limiting and request validation
- Provides comprehensive API documentation

### Chat App

The `chat_app.py` module implements a full-stack chat application that:

- Offers a modern React-based UI
- Supports real-time messaging with AI assistants
- Integrates with Amazon Cognito for user authentication
- Allows social logins via OAuth for Google and LinkedIn

### Document QA

The `document_qa.py` module provides a document question-answering application that:

- Allows users to upload documents to AWS S3
- Extracts information from documents using AI
- Answers questions based on document content
- Supports multiple document formats

### RAG App

The `rag_app.py` module implements a Retrieval-Augmented Generation application that:

- Maintains a knowledge base in PostgreSQL with pgvector
- Retrieves relevant information for user queries
- Generates accurate responses based on retrieved information
- Provides a user-friendly interface for interacting with the knowledge base

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

# Start the application using Docker
docker-compose up -d
```

## Usage

### API Service

```bash
# Start the API service
python -m apps.api_service

# Access the API documentation at http://localhost:8000/docs
```

### Chat App

```bash
# Start the chat application
python -m apps.chat_app

# Access the chat app at http://localhost:8000
```

### Document QA

```bash
# Start the document QA application
python -m apps.document_qa

# Access the document QA app at http://localhost:8000
```

### RAG App

```bash
# Start the RAG application
python -m apps.rag_app

# Access the RAG app at http://localhost:8000
```

## Configuration

### Environment Variables

The applications use environment variables for configuration. Here's what you need to set in your `.env` file:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Amazon Cognito Configuration
COGNITO_USER_POOL_ID=your_user_pool_id
COGNITO_APP_CLIENT_ID=your_app_client_id
COGNITO_DOMAIN=your_cognito_domain

# S3 Configuration
S3_BUCKET=your_bucket_name

# PostgreSQL Configuration
PGVECTOR_HOST=your_rds_instance.region.rds.amazonaws.com
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password
PGVECTOR_DATABASE=vector_db

# AI Gateway Configuration
AI_GATEWAY_URL=https://your-ai-gateway-url.com/api/v1/ai
PROJECT_NAME=your_project_name
API_KEY=your_api_key
```

## Deployment

### AWS Amplify Deployment

The applications can be deployed using AWS Amplify Gen 2:

```bash
# Install the Amplify CLI
npm install -g @aws-amplify/cli

# Configure Amplify
amplify configure

# Initialize Amplify in your project
amplify init

# Add hosting
amplify add hosting

# Publish the application
amplify publish
```

### Docker Deployment

The applications can also be deployed using Docker:

```bash
# Build the Docker image
docker build -t gen-ai-app -f docker/Dockerfile .

# Run the Docker container
docker run -p 8000:8000 --env-file .env gen-ai-app
```

### Terraform Deployment

For production deployments, use the Terraform configurations in the `infrastructure/terraform` directory:

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan the deployment
terraform plan

# Apply the deployment
terraform apply
```

This will deploy the application to AWS using best practices for security, scalability, and reliability.
