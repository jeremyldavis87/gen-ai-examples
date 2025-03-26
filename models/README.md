# AI Model Handler and Testing Framework

This package provides a robust model handler that makes it easy to swap between different AI models with automatic fallback capabilities, along with a comprehensive testing framework that uses deepeval to evaluate model performance.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Model Handler](#model-handler)
4. [Testing Framework](#testing-framework)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)

## Features

### Model Handler
- Easy model switching for different use cases
- Automatic fallback to alternative models on failure
- Default configurations for common tasks
- Performance tracking and metrics
- Simplified error handling

### Testing Framework
- Comprehensive model evaluation using deepeval
- Support for various evaluation metrics
- Model comparison and visualization
- Test case management and persistence
- Export results to various formats

## Installation

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

## Model Handler
The ModelHandler class provides a unified interface to interact with different AI models, with automatic fallback capabilities.

### Basic Usage
```python
from models.model_handler import ModelHandler

# Initialize handler
model_handler = ModelHandler()

# Generate text using default models and fallbacks
response = model_handler.generate_text(
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

# Generate embeddings using default models and fallbacks
embeddings = model_handler.generate_embeddings(
    texts=["Embed this text"]
)
```

### Advanced Usage
```python
# Specify model and parameters
response = model_handler.generate_text(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    task_type="text_generation",
    model="gpt4o",
    model_family="openai",
    temperature=0.7,
    max_tokens=1000
)

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
The ModelTester class provides a comprehensive framework for evaluating and comparing models.

### Creating Test Datasets
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
# Compare multiple models
comparison = tester.compare_models([test_result_id_1, test_result_id_2, test_result_id_3])

# Export results to CSV
csv_path = tester.export_results_to_csv(test_result_id)
```

## Usage Examples
Complete examples are available in the examples directory:

model_test_example.py: Demonstrates how to test and compare different models
production_handler_example.py: Shows how to use the model handler in production scenarios

### Running Examples
```bash
# Run the testing example
python examples/model_test_example.py

# Run the production handler example
python examples/production_handler_example.py
```

### Configuration

#### Environment Variables
The package uses environment variables for configuration. Here's what you need to set in your .env file:

```bash
# AI Gateway Configuration
AI_GATEWAY_URL=https://aigateway-prod.apps-1.gp-1-prod.openshift.cignacloud.com/api/v1/ai
PROJECT_NAME=your_project_name
API_KEY=your_api_key

# Testing Configuration
DEEPEVAL_API_KEY=your_deepeval_api_key  # Optional, for additional features
Default Model Configurations
Default model configurations are defined in the ModelHandler class:
pythonCopyself.default_configs = {
    "text_generation": {
        "primary": {"model": "gpt4o", "model_family": "openai", "temperature": 0.7, "max_tokens": 1000},
        "fallbacks": [
            {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.7, "max_tokens": 1000},
            {"model": "llama3-70b", "model_family": "llama", "temperature": 0.7, "max_tokens": 1000},
        ]
    },
    "embeddings": {
        "primary": {"model": "text-embedding-3-large", "model_family": "openai"},
        "fallbacks": [
            {"model": "text-embedding-3-small", "model_family": "openai"},
        ]
    },
    # Additional task types...
}
```

This model handler provides a robust solution for easily swapping between different AI models with fallback capabilities, while the testing framework allows you to evaluate and compare the performance of these models for your specific use cases. The package is designed to be flexible and extensible, allowing you to customize it for your needs.