# Unit Tests

This directory contains unit tests for the gen-ai-examples project, following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Writing Unit Tests](#writing-unit-tests)
5. [Mocking AWS Services](#mocking-aws-services)

## Overview

Unit tests verify that individual components of the gen-ai-examples project work correctly in isolation. These tests focus on testing specific functions, classes, or modules without dependencies on external services or components.

## Test Organization

The unit tests are organized to mirror the structure of the main codebase:

```
unit/
u251cu2500u2500 test_ai_gateway/        # Tests for the ai_gateway package
u251cu2500u2500 test_agents/           # Tests for the agents package
u251cu2500u2500 test_apps/             # Tests for the apps package
u251cu2500u2500 test_langchain_utils/  # Tests for the langchain_utils package
u251cu2500u2500 test_mcp/              # Tests for the mcp package
u251cu2500u2500 test_models/           # Tests for the models package
u251cu2500u2500 test_testing/          # Tests for the testing package
u2514u2500u2500 test_vector_db/        # Tests for the vector_db package
```

Each test file follows the naming convention `test_*.py` to ensure it's automatically discovered by pytest.

## Running Tests

### Running All Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run with coverage report
pytest tests/unit/ --cov=. --cov-report=term-missing
```

### Running Specific Tests

```bash
# Run tests for a specific package
pytest tests/unit/test_ai_gateway/

# Run a specific test file
pytest tests/unit/test_ai_gateway/test_client.py

# Run a specific test function
pytest tests/unit/test_ai_gateway/test_client.py::test_generate_text
```

### Test Environment

Unit tests can be run with different configurations:

```bash
# Run tests with mocked AWS services
MOCK_AWS=true pytest tests/unit/

# Run tests with specific log level
LOG_LEVEL=DEBUG pytest tests/unit/
```

## Writing Unit Tests

### Test Structure

Each unit test should follow the Arrange-Act-Assert (AAA) pattern:

1. **Arrange**: Set up the test conditions and inputs
2. **Act**: Call the function or method being tested
3. **Assert**: Verify the expected outcomes

### Example Unit Test

```python
# tests/unit/test_ai_gateway/test_client.py
import pytest
from unittest.mock import patch, MagicMock
from ai_gateway.client import AIGatewayClient

@pytest.fixture
def mock_gateway_client():
    """Fixture that provides a mocked AIGatewayClient."""
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        client = AIGatewayClient(api_url="https://test-url.com", api_key="test-key")
        yield client

def test_generate_text(mock_gateway_client):
    """Test that generate_text returns the expected response."""
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

def test_generate_text_with_parameters(mock_gateway_client):
    """Test that generate_text passes parameters correctly."""
    # Arrange
    messages = [{"role": "user", "content": "Hello"}]
    
    # Act
    with patch.object(mock_gateway_client, '_make_request') as mock_request:
        mock_request.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_gateway_client.generate_text(
            messages=messages,
            model="gpt4o",
            model_family="openai",
            temperature=0.5,
            max_tokens=100
        )
    
    # Assert
    mock_request.assert_called_once()
    args, kwargs = mock_request.call_args
    assert kwargs["data"]["temperature"] == 0.5
    assert kwargs["data"]["max_tokens"] == 100

def test_generate_text_error_handling(mock_gateway_client):
    """Test that generate_text handles errors correctly."""
    # Arrange
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_post.return_value = mock_response
        
        # Act & Assert
        with pytest.raises(Exception) as excinfo:
            mock_gateway_client.generate_text(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt4o",
                model_family="openai"
            )
        
        assert "Bad request" in str(excinfo.value)
```

### Using Fixtures

Pytest fixtures are used to set up test preconditions and provide test dependencies:

```python
# tests/unit/conftest.py
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_s3_client():
    """Fixture that provides a mocked boto3 S3 client."""
    mock_client = MagicMock()
    mock_client.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'test/document1.pdf', 'Size': 1024},
            {'Key': 'test/document2.pdf', 'Size': 2048}
        ]
    }
    return mock_client

@pytest.fixture
def mock_cognito_client():
    """Fixture that provides a mocked boto3 Cognito client."""
    mock_client = MagicMock()
    mock_client.initiate_auth.return_value = {
        'AuthenticationResult': {
            'IdToken': 'test-id-token',
            'AccessToken': 'test-access-token',
            'RefreshToken': 'test-refresh-token'
        }
    }
    return mock_client
```

## Mocking AWS Services

Unit tests should not depend on actual AWS services. Instead, use mocking to simulate AWS service responses:

### Mocking with boto3

```python
# tests/unit/test_vector_db/test_s3_integration.py
import pytest
from unittest.mock import patch, MagicMock
from vector_db.s3_integration import S3DocumentLoader

@pytest.fixture
def mock_s3_loader():
    with patch('boto3.client') as mock_boto3_client:
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'documents/doc1.pdf', 'Size': 1024},
                {'Key': 'documents/doc2.pdf', 'Size': 2048}
            ]
        }
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: b'Test document content')
        }
        mock_boto3_client.return_value = mock_s3
        
        loader = S3DocumentLoader(bucket="test-bucket", prefix="documents/")
        yield loader

def test_load_documents(mock_s3_loader):
    # Act
    documents = mock_s3_loader.load()
    
    # Assert
    assert len(documents) == 2
    assert documents[0].page_content == "Test document content"
    assert documents[0].metadata["source"].startswith("s3://test-bucket/documents/doc")
```

### Mocking with moto

For more complex AWS interactions, use the moto library:

```python
# tests/unit/test_vector_db/test_s3_integration_with_moto.py
import pytest
import boto3
from moto import mock_s3
from vector_db.s3_integration import S3DocumentLoader

@pytest.fixture
def s3_bucket():
    with mock_s3():
        # Create mock S3 bucket and objects
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-bucket'
        s3.create_bucket(Bucket=bucket_name)
        
        # Upload test documents
        s3.put_object(Bucket=bucket_name, Key='documents/doc1.pdf', Body=b'Test document 1 content')
        s3.put_object(Bucket=bucket_name, Key='documents/doc2.pdf', Body=b'Test document 2 content')
        
        yield bucket_name

def test_load_documents_with_moto(s3_bucket):
    # Arrange
    loader = S3DocumentLoader(bucket=s3_bucket, prefix="documents/")
    
    # Act
    documents = loader.load()
    
    # Assert
    assert len(documents) == 2
    assert any(doc.page_content == "Test document 1 content" for doc in documents)
    assert any(doc.page_content == "Test document 2 content" for doc in documents)
```

For more information on unit testing best practices, refer to the documentation in the `docs/guides/` directory.
