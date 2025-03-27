# Tests

This directory contains unit and integration tests for the gen-ai-examples project, following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Test Types](#test-types)
4. [Running Tests](#running-tests)
5. [Writing Tests](#writing-tests)
6. [CI/CD Integration](#ci-cd-integration)

## Overview

The tests directory contains automated tests that verify the functionality and reliability of the gen-ai-examples project. These tests ensure that the code works as expected and help prevent regressions when making changes to the codebase.

## Directory Structure

- `unit/`: Contains unit tests for individual components
- `integration/`: Contains integration tests that verify interactions between components
- `conftest.py`: Contains pytest fixtures and configuration
- `requirements-test.txt`: Contains test-specific dependencies

## Test Types

### Unit Tests

Unit tests focus on testing individual components in isolation. They verify that each function, class, or module behaves as expected. Unit tests are located in the `unit/` directory and are organized to mirror the structure of the main codebase.

Key unit test areas include:

- AI Gateway client tests
- Model handler tests
- Agent functionality tests
- Vector database utility tests
- MCP implementation tests

### Integration Tests

Integration tests verify that different components work together correctly. They test the interactions between modules and external services. Integration tests are located in the `integration/` directory.

Key integration test areas include:

- End-to-end API tests
- Database integration tests
- AWS service integration tests
- Full application workflow tests

## Running Tests

### Prerequisites

Before running tests, ensure you have:

1. Installed test dependencies:
   ```bash
   pip install -r tests/requirements-test.txt
   ```

2. Set up environment variables for testing:
   ```bash
   cp .env.sample .env.test
   # Edit .env.test with test-specific configuration
   ```

### Running All Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=. --cov-report=term-missing
```

### Running Specific Tests

```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run tests for a specific module
pytest tests/unit/test_ai_gateway.py

# Run a specific test
pytest tests/unit/test_ai_gateway.py::test_generate_text
```

### Test Environment

Tests can be run in different environments:

```bash
# Run tests with test environment
ENV=test pytest

# Run tests with mock AWS services
MOCK_AWS=true pytest
```

## Writing Tests

### Unit Test Example

```python
# tests/unit/test_ai_gateway.py
import pytest
from unittest.mock import patch, MagicMock
from ai_gateway.client import AIGatewayClient

@pytest.fixture
def mock_gateway_client():
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        client = AIGatewayClient(api_url="https://test-url.com", api_key="test-key")
        yield client

def test_generate_text(mock_gateway_client):
    # Arrange
    messages = [{"role": "user", "content": "Hello"}]
    
    # Act
    response = mock_gateway_client.generate_text(
        messages=messages,
        model="gpt4o",
        model_family="openai"
    )
    
    # Assert
    assert response["choices"][0]["message"]["content"] == "Test response"
```

### Integration Test Example

```python
# tests/integration/test_vector_db.py
import pytest
import os
from vector_db.pgvector_client import PGVectorClient
from ai_gateway.client import AIGatewayClient

@pytest.fixture
def vector_client():
    # Connect to test database
    client = PGVectorClient(
        host=os.getenv("TEST_PGVECTOR_HOST"),
        port=os.getenv("TEST_PGVECTOR_PORT"),
        user=os.getenv("TEST_PGVECTOR_USER"),
        password=os.getenv("TEST_PGVECTOR_PASSWORD"),
        database=os.getenv("TEST_PGVECTOR_DATABASE")
    )
    
    # Clear test collection
    client.delete_collection("test_collection")
    
    yield client
    
    # Cleanup
    client.delete_collection("test_collection")

@pytest.fixture
def gateway_client():
    return AIGatewayClient()

def test_store_and_query_vectors(vector_client, gateway_client):
    # Arrange
    texts = ["This is a test document", "Another test document"]
    embeddings = gateway_client.generate_embeddings(texts=texts, model="text-embedding-3-small")
    
    # Act
    # Store vectors
    vector_client.create_collection("test_collection")
    ids = vector_client.add_vectors(
        collection_name="test_collection",
        vectors=embeddings,
        metadata=[{"text": text} for text in texts]
    )
    
    # Query vectors
    query_text = "test document"
    query_embedding = gateway_client.generate_embeddings(texts=[query_text], model="text-embedding-3-small")[0]
    results = vector_client.query_vectors(
        collection_name="test_collection",
        query_vector=query_embedding,
        limit=2
    )
    
    # Assert
    assert len(results) == 2
    assert results[0]["metadata"]["text"] in texts
    assert results[1]["metadata"]["text"] in texts
```

## CI/CD Integration

The tests are integrated with GitHub Actions for continuous integration and deployment:

### GitHub Actions Workflow

The CI workflow runs tests automatically on pull requests and pushes to the main branch:

```yaml
name: Run Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: ankane/pgvector:latest
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_vector_db
        ports:
          - 5432:5432
        options: --health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.12
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r tests/requirements-test.txt
    - name: Run tests
      env:
        TEST_PGVECTOR_HOST: localhost
        TEST_PGVECTOR_PORT: 5432
        TEST_PGVECTOR_USER: postgres
        TEST_PGVECTOR_PASSWORD: postgres
        TEST_PGVECTOR_DATABASE: test_vector_db
        MOCK_AWS: true
      run: |
        pytest --cov=. --cov-report=xml
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

For more information on the testing approach and best practices, refer to the documentation in the `docs/guides/` directory.
