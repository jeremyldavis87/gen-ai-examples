# Testing Framework

This directory contains a comprehensive testing framework for AI models in the gen-ai-examples project, following AWS best practices and modern Python 3.12 development standards.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Test Types](#test-types)
6. [AWS Integration](#aws-integration)

## Overview

The testing directory provides a robust framework for evaluating and comparing AI models. It includes tools for creating test datasets, running model evaluations, comparing model performance, and generating reports. The framework is designed to work seamlessly with AWS services and follows best practices for AI model testing.

## Features

- Comprehensive model evaluation using various metrics
- Support for creating and managing test datasets
- Model comparison and visualization tools
- Test case management and persistence
- Integration with AWS services for scalable testing
- Export results to various formats

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
from testing.model_tester import ModelTester

# Initialize the model tester
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

# Get the test results
results = tester.get_test_results(test_result_id)
print(results)
```

### Advanced Usage

```python
# Compare multiple models
model_configs = [
    {
        "name": "GPT-4o",
        "model": "gpt4o",
        "model_family": "openai",
        "temperature": 0.3,
        "max_tokens": 1000
    },
    {
        "name": "Claude 3 Sonnet",
        "model": "sonnet-3.7",
        "model_family": "anthropic",
        "temperature": 0.3,
        "max_tokens": 1000
    },
    {
        "name": "Llama 3 70B",
        "model": "llama3-70b",
        "model_family": "llama",
        "temperature": 0.3,
        "max_tokens": 1000
    }
]

# Run tests for all models
test_result_ids = []
for config in model_configs:
    result_id = tester.run_model_test(
        dataset_id=dataset_id,
        model_config=config,
        metrics=["answer_relevancy", "factual_consistency", "faithfulness"]
    )
    test_result_ids.append(result_id)

# Compare the models
comparison = tester.compare_models(test_result_ids)

# Generate a comparison report
report_path = tester.generate_comparison_report(
    comparison=comparison,
    output_format="html",
    title="Model Comparison: Geography Questions"
)

# Export results to CSV
csv_path = tester.export_results_to_csv(test_result_ids)
```

## Test Types

The testing framework supports various types of tests:

### Factual Accuracy Tests

Evaluates the model's ability to provide factually correct information:

```python
factual_test_cases = [
    {
        "input": "What is the capital of France?",
        "expected_output": "Paris",
        "description": "Capital city question"
    },
    # Add more factual test cases...
]

factual_dataset_id = tester.create_test_dataset(
    task_type="factual_qa",
    test_cases=factual_test_cases,
    dataset_name="factual_questions"
)
```

### Reasoning Tests

Evaluates the model's reasoning capabilities:

```python
reasoning_test_cases = [
    {
        "input": "If John is taller than Mary, and Mary is taller than Sue, who is the tallest?",
        "expected_output": "John is the tallest.",
        "description": "Transitive reasoning"
    },
    # Add more reasoning test cases...
]

reasoning_dataset_id = tester.create_test_dataset(
    task_type="reasoning",
    test_cases=reasoning_test_cases,
    dataset_name="reasoning_questions"
)
```

### Code Generation Tests

Evaluates the model's code generation capabilities:

```python
code_test_cases = [
    {
        "input": "Write a Python function to calculate the factorial of a number.",
        "expected_output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
        "description": "Factorial function",
        "evaluation_criteria": {"functionality": True, "efficiency": True, "style": False}
    },
    # Add more code test cases...
]

code_dataset_id = tester.create_test_dataset(
    task_type="code_generation",
    test_cases=code_test_cases,
    dataset_name="code_generation_tasks"
)
```

## AWS Integration

The testing framework integrates with various AWS services:

### Amazon S3 Integration

Test datasets and results can be stored in Amazon S3:

```python
# Initialize the model tester with S3 storage
tester = ModelTester(
    storage_type="s3",
    s3_bucket="your-test-data-bucket",
    s3_prefix="model-tests/"
)

# Create and run tests as usual
# Results will be stored in S3
```

### AWS Lambda Integration

Tests can be run as AWS Lambda functions for scalability:

```python
# Initialize the model tester with Lambda execution
tester = ModelTester(
    execution_type="lambda",
    lambda_function_name="model-test-executor",
    lambda_region="us-east-1"
)

# Create and run tests as usual
# Tests will be executed as Lambda functions
```

### Amazon CloudWatch Integration

Test results can be monitored and visualized using Amazon CloudWatch:

```python
# Initialize the model tester with CloudWatch monitoring
tester = ModelTester(
    monitoring_type="cloudwatch",
    cloudwatch_namespace="ModelTests",
    cloudwatch_region="us-east-1"
)

# Create and run tests as usual
# Metrics will be published to CloudWatch
```

For more detailed information on using the testing framework, refer to the examples in the `examples/` directory.
