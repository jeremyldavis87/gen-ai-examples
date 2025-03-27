# LangChain Utilities

This directory contains utilities for integrating LangChain with the gen-ai-examples project, following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Integration with AWS Services](#integration-with-aws-services)

## Overview

The LangChain utilities provide a bridge between the gen-ai-examples project and the LangChain framework, enabling the use of LangChain's powerful capabilities with AWS managed services and the project's AI Gateway.

## Features

- Custom LangChain language model classes that use the AI Gateway
- Custom LangChain embedding classes for vector operations
- Integration with AWS services like Amazon S3 and Amazon RDS
- Utilities for creating LangChain chains and agents
- Adapters for using LangChain with the project's RAG capabilities

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
from langchain_utils.gateway_integration import get_langchain_llm, get_langchain_embeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Get a LangChain LLM that uses the AI Gateway
llm = get_langchain_llm(model_name="gpt4o", model_family="openai")

# Create a simple chain
prompt = PromptTemplate(template="{question}", input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run(question="What is the capital of France?")
print(response)

# Get LangChain embeddings that use the AI Gateway
embeddings = get_langchain_embeddings(model_name="text-embedding-3-large")

# Generate embeddings
embedding = embeddings.embed_query("This is a test sentence.")
print(f"Embedding dimension: {len(embedding)}")
```

### Advanced Usage

```python
from langchain_utils.gateway_integration import GatewayChatModel, GatewayEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import PGVector
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import os
from dotenv import load_dotenv

load_dotenv()

# Create a custom Gateway chat model
chat_model = GatewayChatModel(
    model="gpt4o",
    model_family="openai",
    temperature=0.3,
    max_tokens=2000
)

# Create custom Gateway embeddings
embeddings = GatewayEmbeddings(model="text-embedding-3-large")

# Connect to PostgreSQL with pgvector
connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("PGVECTOR_USER"),
    password=os.getenv("PGVECTOR_PASSWORD"),
    host=os.getenv("PGVECTOR_HOST"),
    port=os.getenv("PGVECTOR_PORT"),
    database=os.getenv("PGVECTOR_DATABASE")
)

# Create a vector store
vector_store = PGVector(
    connection_string=str(connection_url),
    embedding_function=embeddings,
    collection_name="documents"
)

# Create a retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Run the QA chain
response = qa_chain.run("What is the capital of France?")
print(response)
```

## Integration with AWS Services

The LangChain utilities integrate with various AWS services:

### Amazon S3 Integration

```python
from langchain_utils.s3_document_loader import S3DocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents from S3
loader = S3DocumentLoader(
    bucket="your-s3-bucket",
    prefix="documents/",
    region_name="us-east-1"
)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Store document chunks in vector database
vector_store = PGVector.from_documents(
    documents=splits,
    embedding=embeddings,
    connection_string=str(connection_url),
    collection_name="s3_documents"
)
```

### Amazon Cognito Integration

```python
from langchain_utils.cognito_auth import CognitoAuthenticator

# Authenticate with Cognito
authenticator = CognitoAuthenticator(
    user_pool_id=os.getenv("COGNITO_USER_POOL_ID"),
    app_client_id=os.getenv("COGNITO_APP_CLIENT_ID"),
    region_name="us-east-1"
)

# Get tokens
tokens = authenticator.authenticate(username="user@example.com", password="password")

# Use the tokens for authenticated API calls
headers = {"Authorization": f"Bearer {tokens['id_token']}"}
```

These utilities make it easy to integrate LangChain with the gen-ai-examples project and AWS services, enabling powerful AI applications with minimal boilerplate code.
