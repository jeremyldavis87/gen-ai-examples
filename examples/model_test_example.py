# examples/model_test_example.py
import os
import json
from dotenv import load_dotenv
from models.model_handler import ModelHandler
from testing.model_tester import ModelTester

load_dotenv()

def main():
    """Run model testing example."""
    print("Initializing model handler and tester...")
    model_handler = ModelHandler()
    tester = ModelTester(model_handler)
    
    # Define test cases for text classification
    classification_test_cases = [
        {
            "input": "I'm having trouble accessing my account. I keep getting an error message saying 'Invalid credentials'.",
            "expected_output": "This is a technical support issue related to account access.",
            "description": "Account access issue",
            "tags": ["technical", "account", "access"]
        },
        {
            "input": "I'd like to upgrade my plan to the premium tier. How much will that cost?",
            "expected_output": "This is a billing inquiry about plan upgrades.",
            "description": "Plan upgrade inquiry",
            "tags": ["billing", "upgrade", "pricing"]
        },
        {
            "input": "Your service has been down for 3 hours now! This is unacceptable for a business that depends on your platform.",
            "expected_output": "This is a complaint about service outage.",
            "description": "Service outage complaint",
            "tags": ["outage", "complaint", "urgent"]
        },
        {
            "input": "What's the difference between the basic and premium plans?",
            "expected_output": "This is a general inquiry about plan differences.",
            "description": "Plan comparison inquiry",
            "tags": ["plans", "comparison", "information"]
        },
        {
            "input": "I need to cancel my subscription immediately and get a refund for this month.",
            "expected_output": "This is a cancellation and refund request.",
            "description": "Cancellation and refund request",
            "tags": ["cancellation", "refund", "urgent"]
        },
        {
            "input": "I need to cancel my subscription immediately and get a refund for this month.",
            "expected_output": "This is a cancellation and refund request.",
            "description": "Cancellation and refund request",
            "tags": ["cancellation", "refund", "urgent"]
        },
        {
            "input": "Can you help me understand how to use your API for integrating with my application?",
            "expected_output": "This is a technical inquiry about API integration.",
            "description": "API integration inquiry",
            "tags": ["technical", "api", "integration"]
        },
        {
            "input": "I just wanted to say that your customer service team was amazing! They resolved my issue so quickly.",
            "expected_output": "This is a positive feedback about customer service.",
            "description": "Positive feedback",
            "tags": ["feedback", "positive", "customer service"]
        }
    ]
    
    # Define test cases for RAG
    rag_test_cases = [
        {
            "input": "What is the current refund policy for premium accounts?",
            "context": [
                "Premium account refunds are processed within 14 days of cancellation, provided the cancellation occurs within 30 days of subscription renewal.",
                "Standard accounts can be refunded within 7 days of subscription renewal.",
                "Enterprise accounts have customized refund policies based on contract terms."
            ],
            "expected_output": "Premium accounts can be refunded within 30 days of renewal, with processing taking up to 14 days.",
            "description": "Refund policy inquiry",
            "tags": ["policy", "refund", "premium"]
        },
        {
            "input": "How do I enable two-factor authentication?",
            "context": [
                "To enable two-factor authentication, go to Account Settings > Security > Two-Factor Authentication and click 'Enable'.",
                "You can use an authenticator app like Google Authenticator or receive codes via SMS.",
                "Two-factor authentication provides an additional layer of security for your account."
            ],
            "expected_output": "To enable two-factor authentication, navigate to Account Settings, then Security, and select Two-Factor Authentication where you can enable it using either an authenticator app or SMS.",
            "description": "Two-factor authentication setup",
            "tags": ["security", "2fa", "account"]
        },
        {
            "input": "What integrations are available for the marketing analytics dashboard?",
            "context": [
                "The marketing analytics dashboard supports integrations with Google Analytics, Facebook Ads, Twitter Ads, and LinkedIn Marketing.",
                "Custom API integrations are available for Enterprise tier customers.",
                "Data from integrations is refreshed every 4 hours by default, but can be configured for real-time updates."
            ],
            "expected_output": "The marketing analytics dashboard integrates with Google Analytics, Facebook Ads, Twitter Ads, and LinkedIn Marketing, with Enterprise customers having access to custom API integrations. Data refreshes every 4 hours by default.",
            "description": "Marketing dashboard integrations",
            "tags": ["integrations", "marketing", "analytics"]
        }
    ]
    
    # Define test cases for code generation
    code_test_cases = [
        {
            "input": "Write a Python function to calculate the Fibonacci sequence up to n terms.",
            "expected_output": "def fibonacci(n):\n    fib_sequence = [0, 1]\n    while len(fib_sequence) < n:\n        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])\n    return fib_sequence[:n]",
            "description": "Fibonacci sequence function",
            "tags": ["python", "algorithm", "recursion"],
            "task_type": "code_generation"
        },
        {
            "input": "Create a function to validate an email address in JavaScript.",
            "expected_output": "function validateEmail(email) {\n    const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n    return regex.test(email);\n}",
            "description": "Email validation function",
            "tags": ["javascript", "validation", "regex"],
            "task_type": "code_generation"
        }
    ]
    
    # Create test datasets
    print("Creating test datasets...")
    classification_dataset_id = tester.create_test_dataset(
        task_type="text_generation",
        test_cases=classification_test_cases,
        dataset_name="text_classification"
    )
    
    rag_dataset_id = tester.create_test_dataset(
        task_type="text_generation",
        test_cases=rag_test_cases,
        dataset_name="rag_queries"
    )
    
    code_dataset_id = tester.create_test_dataset(
        task_type="code_generation",
        test_cases=code_test_cases,
        dataset_name="code_generation"
    )
    
    print(f"Created test datasets: {classification_dataset_id}, {rag_dataset_id}, {code_dataset_id}")
    
    # Test different models
    print("Testing models...")
    
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
        },
        {
            "model": "llama3-70b",
            "model_family": "llama",
            "temperature": 0.2,
            "max_tokens": 1000
        }
    ]
    
    # Run tests for classification
    classification_results = []
    for model_config in models_to_test:
        print(f"Testing {model_config['model']} on classification tasks...")
        result_id = tester.run_model_test(
            dataset_id=classification_dataset_id,
            model_config=model_config,
            metrics=["answer_relevancy", "factual_consistency"]
        )
        classification_results.append(result_id)
    
    # Run tests for RAG
    rag_results = []
    for model_config in models_to_test:
        print(f"Testing {model_config['model']} on RAG tasks...")
        result_id = tester.run_model_test(
            dataset_id=rag_dataset_id,
            model_config=model_config,
            metrics=["answer_relevancy", "factual_consistency", "faithfulness", "contextual_relevancy"]
        )
        rag_results.append(result_id)
    
    # Run tests for code generation
    code_results = []
    for model_config in models_to_test:
        print(f"Testing {model_config['model']} on code generation tasks...")
        result_id = tester.run_model_test(
            dataset_id=code_dataset_id,
            model_config=model_config,
            metrics=["answer_relevancy", "factual_consistency"]
        )
        code_results.append(result_id)
    
    # Compare results
    print("Comparing model results...")
    classification_comparison = tester.compare_models(classification_results)
    rag_comparison = tester.compare_models(rag_results)
    code_comparison = tester.compare_models(code_results)
    
    # Export results to CSV
    for result_id in classification_results + rag_results + code_results:
        csv_path = tester.export_results_to_csv(result_id)
        print(f"Exported results to {csv_path}")
    
    # Print summary
    print("\n=== TESTING SUMMARY ===")
    print("\nClassification Tasks:")
    for i, model in enumerate(classification_comparison["models"]):
        print(f"  {model}:")
        for metric in classification_comparison["metrics"]:
            score = classification_comparison["metrics"][metric][i]
            pass_rate = classification_comparison["pass_rates"][metric][i]
            print(f"    {metric}: Score={score:.2f}, Pass Rate={pass_rate:.2%}")
    
    print("\nRAG Tasks:")
    for i, model in enumerate(rag_comparison["models"]):
        print(f"  {model}:")
        for metric in rag_comparison["metrics"]:
            score = rag_comparison["metrics"][metric][i]
            pass_rate = rag_comparison["pass_rates"][metric][i]
            print(f"    {metric}: Score={score:.2f}, Pass Rate={pass_rate:.2%}")
    
    print("\nCode Generation Tasks:")
    for i, model in enumerate(code_comparison["models"]):
        print(f"  {model}:")
        for metric in code_comparison["metrics"]:
            score = code_comparison["metrics"][metric][i]
            pass_rate = code_comparison["pass_rates"][metric][i]
            print(f"    {metric}: Score={score:.2f}, Pass Rate={pass_rate:.2%}")
    
    print("\nResults saved in the test_results directory.")

if __name__ == "__main__":
    main()