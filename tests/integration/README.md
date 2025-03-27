# Integration Tests

This directory contains integration tests for the gen-ai-examples project, following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Writing Integration Tests](#writing-integration-tests)
5. [AWS Service Integration](#aws-service-integration)
6. [Docker-Based Testing](#docker-based-testing)

## Overview

Integration tests verify that different components of the gen-ai-examples project work correctly together. These tests focus on testing the interactions between modules, services, and external dependencies to ensure they integrate properly.

## Test Organization

The integration tests are organized by functional area:

```
integration/
├── test_api/              # Tests for API endpoints and services
├── test_database/         # Tests for database interactions
├── test_aws_services/     # Tests for AWS service integrations
├── test_ai_gateway/       # Tests for AI Gateway interactions
├── test_vector_search/    # Tests for vector search functionality
└── test_end_to_end/       # End-to-end application tests
```

Each test file follows the naming convention `test_*.py` to ensure it's automatically discovered by pytest.

## Running Tests

### Prerequisites

Before running integration tests, ensure you have:

1. A PostgreSQL database with pgvector extension for testing
2. Docker installed (for containerized tests)
3. Test environment variables configured in `.env.test`

### Running All Integration Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run with coverage report
pytest tests/integration/ --cov=. --cov-report=term-missing
```

### Running Specific Tests

```bash
# Run tests for a specific area
pytest tests/integration/test_database/

# Run a specific test file
pytest tests/integration/test_database/test_pgvector.py

# Run a specific test function
pytest tests/integration/test_database/test_pgvector.py::test_vector_similarity_search
```

### Test Environment

Integration tests can use different environments:

```bash
# Run tests with local services
ENV=local pytest tests/integration/

# Run tests with AWS services
ENV=aws pytest tests/integration/

# Run tests with mock services
ENV=mock pytest tests/integration/
```

## Writing Integration Tests

### Test Structure

Integration tests should follow these principles:

1. **Setup**: Initialize the components being tested and their dependencies
2. **Exercise**: Perform the operations being tested
3. **Verify**: Check that the results are as expected
4. **Cleanup**: Clean up any resources created during the test

### Example Integration Test

```python
# tests/integration/test_database/test_pgvector.py
import pytest
import os
import numpy as np
from vector_db.pgvector_client import PGVectorClient
from dotenv import load_dotenv

load_dotenv(".env.test")

@pytest.fixture
def vector_client():
    """Fixture that provides a PGVectorClient connected to the test database."""
    # Connect to test database
    client = PGVectorClient(
        host=os.getenv("TEST_PGVECTOR_HOST", "localhost"),
        port=int(os.getenv("TEST_PGVECTOR_PORT", "5432")),
        user=os.getenv("TEST_PGVECTOR_USER", "postgres"),
        password=os.getenv("TEST_PGVECTOR_PASSWORD", "postgres"),
        database=os.getenv("TEST_PGVECTOR_DATABASE", "test_vector_db")
    )
    
    # Create test collection
    client.create_collection("test_collection", drop_existing=True)
    
    yield client
    
    # Cleanup
    client.delete_collection("test_collection")

def test_add_and_query_vectors(vector_client):
    """Test adding vectors to the database and querying them."""
    # Setup - Create test vectors
    vectors = [
        np.random.rand(1536).astype(np.float32).tolist(),
        np.random.rand(1536).astype(np.float32).tolist(),
        np.random.rand(1536).astype(np.float32).tolist()
    ]
    metadata = [
        {"text": "Document 1", "source": "test"},
        {"text": "Document 2", "source": "test"},
        {"text": "Document 3", "source": "test"}
    ]
    
    # Exercise - Add vectors to the database
    ids = vector_client.add_vectors(
        collection_name="test_collection",
        vectors=vectors,
        metadata=metadata
    )
    
    # Verify - Check that vectors were added
    assert len(ids) == 3
    
    # Exercise - Query vectors
    query_vector = np.random.rand(1536).astype(np.float32).tolist()
    results = vector_client.query_vectors(
        collection_name="test_collection",
        query_vector=query_vector,
        limit=2
    )
    
    # Verify - Check query results
    assert len(results) == 2
    assert "text" in results[0]["metadata"]
    assert "source" in results[0]["metadata"]
    assert "distance" in results[0]
    
    # Exercise - Get vector by ID
    vector = vector_client.get_vector(
        collection_name="test_collection",
        vector_id=ids[0]
    )
    
    # Verify - Check retrieved vector
    assert vector is not None
    assert vector["id"] == ids[0]
    assert vector["metadata"]["text"] == "Document 1"
    
    # Exercise - Delete vector
    vector_client.delete_vector(
        collection_name="test_collection",
        vector_id=ids[0]
    )
    
    # Verify - Check that vector was deleted
    deleted_vector = vector_client.get_vector(
        collection_name="test_collection",
        vector_id=ids[0]
    )
    assert deleted_vector is None
```

## AWS Service Integration

Integration tests can interact with actual AWS services for comprehensive testing:

### Testing with AWS Services

```python
# tests/integration/test_aws_services/test_s3_integration.py
import pytest
import os
import boto3
from vector_db.s3_integration import S3DocumentLoader
import uuid

@pytest.fixture
def s3_test_bucket():
    """Fixture that creates a temporary S3 bucket for testing."""
    # Create a unique bucket name
    bucket_name = f"test-bucket-{uuid.uuid4()}"
    region = "us-east-1"
    
    # Create the bucket
    s3 = boto3.client('s3', region_name=region)
    s3.create_bucket(Bucket=bucket_name)
    
    # Upload test documents
    s3.put_object(Bucket=bucket_name, Key='documents/doc1.txt', Body=b'Test document 1 content')
    s3.put_object(Bucket=bucket_name, Key='documents/doc2.txt', Body=b'Test document 2 content')
    
    yield bucket_name
    
    # Cleanup - Delete objects and bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in objects:
        for obj in objects['Contents']:
            s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
    s3.delete_bucket(Bucket=bucket_name)

def test_s3_document_loader(s3_test_bucket):
    """Test loading documents from S3."""
    # Arrange
    loader = S3DocumentLoader(
        bucket=s3_test_bucket,
        prefix="documents/",
        region_name="us-east-1"
    )
    
    # Act
    documents = loader.load()
    
    # Assert
    assert len(documents) == 2
    assert any("Test document 1 content" in doc.page_content for doc in documents)
    assert any("Test document 2 content" in doc.page_content for doc in documents)
    assert all(doc.metadata["source"].startswith(f"s3://{s3_test_bucket}/documents/") for doc in documents)
```

## Docker-Based Testing

Docker can be used to create isolated environments for integration testing:

### Using Docker Compose for Testing

```python
# tests/integration/conftest.py
import pytest
import os
import subprocess
import time
import psycopg2

@pytest.fixture(scope="session")
def docker_compose_up():
    """Fixture that starts the test environment using Docker Compose."""
    # Start Docker Compose
    subprocess.run(
        ["docker-compose", "-f", "tests/integration/docker-compose.test.yml", "up", "-d"],
        check=True
    )
    
    # Wait for services to be ready
    time.sleep(10)  # Simple wait, could be replaced with more robust health checks
    
    yield
    
    # Stop Docker Compose
    subprocess.run(
        ["docker-compose", "-f", "tests/integration/docker-compose.test.yml", "down", "-v"],
        check=True
    )

@pytest.fixture
def pg_connection(docker_compose_up):
    """Fixture that provides a connection to the PostgreSQL database in Docker."""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="test_vector_db"
    )
    
    yield conn
    
    conn.close()

# Example test using Docker-based database
def test_database_connection(pg_connection):
    """Test connecting to the PostgreSQL database."""
    cursor = pg_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1
```

The corresponding `docker-compose.test.yml` file:

```yaml
version: '3.8'

services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: test_vector_db
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

For more information on integration testing best practices, refer to the documentation in the `docs/guides/` directory.
