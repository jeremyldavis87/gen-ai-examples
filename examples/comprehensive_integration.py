# examples/comprehensive_integration.py
import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_handler import ModelHandler
from testing.model_tester import ModelTester
from testing.benchmark import ModelBenchmark

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class AIServiceManager:
    """
    Manager for AI services integration in an organization.
    
    This class demonstrates how to integrate the model handler, testing framework,
    and benchmarking tools into an existing application.
    """
    
    def __init__(self):
        """Initialize the AI service manager."""
        # Create model handler with custom configurations
        self.model_handler = self._configure_model_handler()
        
        # Initialize testing and benchmarking tools
        self.model_tester = ModelTester(self.model_handler)
        self.benchmark = ModelBenchmark(self.model_handler)
        
        # Track production usage
        self.usage_stats = {
            "total_requests": 0,
            "requests_by_task": {},
            "requests_by_model": {}
        }
    
    def _configure_model_handler(self) -> ModelHandler:
        """Configure the model handler with custom defaults."""
        handler = ModelHandler()
        
        # Configure for healthcare use cases
        handler.set_default_config(
            task_type="healthcare_query",
            config={
                "primary": {"model": "gpt4o", "model_family": "openai", "temperature": 0.3, "max_tokens": 1000},
                "fallbacks": [
                    {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.3, "max_tokens": 1000},
                ]
            }
        )
        
        handler.set_default_config(
            task_type="medical_records_summary",
            config={
                "primary": {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.2, "max_tokens": 1500},
                "fallbacks": [
                    {"model": "gpt4o", "model_family": "openai", "temperature": 0.2, "max_tokens": 1500},
                ]
            }
        )
        
        # Configure for customer service use cases
        handler.set_default_config(
            task_type="customer_support",
            config={
                "primary": {"model": "haiku-3.5", "model_family": "anthropic", "temperature": 0.4, "max_tokens": 800},
                "fallbacks": [
                    {"model": "gpt4o", "model_family": "openai", "temperature": 0.4, "max_tokens": 800},
                ]
            }
        )
        
        return handler
    
    def process_query(self, 
                     messages: List[Dict[str, str]],
                     task_type: str,
                     custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using the model handler.
        
        Args:
            messages: List of message objects
            task_type: Type of task
            custom_config: Optional custom configuration
            
        Returns:
            Response data
        """
        # Update usage stats
        self.usage_stats["total_requests"] += 1
        
        if task_type in self.usage_stats["requests_by_task"]:
            self.usage_stats["requests_by_task"][task_type] += 1
        else:
            self.usage_stats["requests_by_task"][task_type] = 1
        
        # Process the query
        try:
            if custom_config:
                # Use custom configuration
                model = custom_config.get("model")
                model_family = custom_config.get("model_family")
                temperature = custom_config.get("temperature")
                max_tokens = custom_config.get("max_tokens")
                
                response = self.model_handler.generate_text(
                    messages=messages,
                    task_type=task_type,
                    model=model,
                    model_family=model_family,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use default configuration for the task type
                response = self.model_handler.generate_text(
                    messages=messages,
                    task_type=task_type
                )
            
            # Extract response content and model info
            content, model_used, model_family = self._extract_response_data(response)
            
            # Update model usage stats
            if model_used in self.usage_stats["requests_by_model"]:
                self.usage_stats["requests_by_model"][model_used] += 1
            else:
                self.usage_stats["requests_by_model"][model_used] = 1
            
            # Prepare response data
            return {
                "success": True,
                "content": content,
                "model_used": model_used,
                "model_family": model_family,
                "task_type": task_type
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_type": task_type
            }
    
    def _extract_response_data(self, response: Dict[str, Any]) -> tuple:
        """Extract content and model info from response."""
        if "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]
            model_used = response.get("model", "unknown")
            model_family = "openai"
        elif "content" in response:
            content = response["content"][0]["text"]
            model_used = response.get("model", "unknown")
            model_family = "anthropic"
        else:
            content = str(response)
            model_used = "unknown"
            model_family = "unknown"
        
        return content, model_used, model_family
    
    def run_model_test(self, test_dataset_path: str) -> str:
        """
        Run a model test using the testing framework.
        
        Args:
            test_dataset_path: Path to the test dataset
            
        Returns:
            Test result ID
        """
        logger.info(f"Running model test using dataset: {test_dataset_path}")
        
        # Load test dataset
        with open(test_dataset_path, 'r') as f:
            test_data = json.load(f)
        
        task_type = test_data.get("task_type", "text_generation")
        test_cases = test_data.get("test_cases", [])
        
        if not test_cases:
            raise ValueError("No test cases found in the dataset")
        
        # Create test dataset
        dataset_id = self.model_tester.create_test_dataset(
            task_type=task_type,
            test_cases=test_cases,
            dataset_name=test_data.get("name", "unnamed_dataset")
        )
        
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
                "model": "haiku-3.5",
                "model_family": "anthropic",
                "temperature": 0.2,
                "max_tokens": 1000
            }
        ]
        
        # Run tests for each model
        test_results = []
        for model_config in models_to_test:
            logger.info(f"Testing model: {model_config['model']}")
            result_id = self.model_tester.run_model_test(
                dataset_id=dataset_id,
                model_config=model_config,
                metrics=["answer_relevancy", "factual_consistency", "faithfulness"]
            )
            test_results.append(result_id)
        
        # Compare results
        comparison = self.model_tester.compare_models(test_results)
        
        logger.info(f"Model test completed with {len(test_results)} model configurations")
        return dataset_id
    
    def run_benchmark(self, benchmark_file: str) -> str:
        """
        Run a benchmark using the benchmarking tool.
        
        Args:
            benchmark_file: Path to the benchmark file
            
        Returns:
            Benchmark result ID
        """
        logger.info(f"Running benchmark: {benchmark_file}")
        
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
                "model": "haiku-3.5",
                "model_family": "anthropic",
                "temperature": 0.2,
                "max_tokens": 1000
            }
        ]
        
        # Run benchmark
        benchmark_id = self.benchmark.run_benchmark(
            benchmark_file=benchmark_file,
            models=models_to_test,
            parallel=True,
            max_workers=4
        )
        
        logger.info(f"Benchmark completed. ID: {benchmark_id}")
        return benchmark_id
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the model handler."""
        return {
            "model_metrics": self.model_handler.get_performance_metrics(),
            "usage_stats": self.usage_stats
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.model_handler.reset_performance_metrics()
        self.usage_stats = {
            "total_requests": 0,
            "requests_by_task": {},
            "requests_by_model": {}
        }
        logger.info("All metrics reset")


def main():
    """Main function to demonstrate usage."""
    parser = argparse.ArgumentParser(description="AI Service Manager Example")
    parser.add_argument("--mode", choices=["query", "test", "benchmark"], required=True, help="Operation mode")
    
    # Query mode arguments
    parser.add_argument("--task-type", help="Task type for query mode")
    parser.add_argument("--message", help="User message for query mode")
    
    # Test mode arguments
    parser.add_argument("--test-dataset", help="Path to test dataset for test mode")
    
    # Benchmark mode arguments
    parser.add_argument("--benchmark-file", help="Path to benchmark file for benchmark mode")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = AIServiceManager()
    
    if args.mode == "query":
        if not args.task_type or not args.message:
            parser.error("--task-type and --message are required for query mode")
        
        # Process query
        messages = [
            {"role": "user", "content": args.message}
        ]
        
        result = manager.process_query(
            messages=messages,
            task_type=args.task_type
        )
        
        if result["success"]:
            print(f"Response: {result['content']}")
            print(f"Model used: {result['model_used']} ({result['model_family']})")
        else:
            print(f"Error: {result['error']}")
        
    elif args.mode == "test":
        if not args.test_dataset:
            parser.error("--test-dataset is required for test mode")
        
        # Run test
        dataset_id = manager.run_model_test(args.test_dataset)
        print(f"Test completed. Dataset ID: {dataset_id}")
        
    elif args.mode == "benchmark":
        if not args.benchmark_file:
            parser.error("--benchmark-file is required for benchmark mode")
        
        # Run benchmark
        benchmark_id = manager.run_benchmark(args.benchmark_file)
        print(f"Benchmark completed. ID: {benchmark_id}")
    
    # Print performance metrics
    metrics = manager.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"Total Calls: {metrics['model_metrics']['total_calls']}")
    print(f"Successful Calls: {metrics['model_metrics']['successful_calls']}")
    print(f"Fallback Calls: {metrics['model_metrics']['fallback_calls']}")
    print(f"Failed Calls: {metrics['model_metrics']['failed_calls']}")
    
    print("\nUsage Statistics:")
    print(f"Total Requests: {metrics['usage_stats']['total_requests']}")
    print("Requests by Task Type:")
    for task, count in metrics['usage_stats']['requests_by_task'].items():
        print(f"  {task}: {count}")
    
    print("Requests by Model:")
    for model, count in metrics['usage_stats']['requests_by_model'].items():
        print(f"  {model}: {count}")

if __name__ == "__main__":
    main()