# testing/model_tester.py
import os
import json
import time
import csv
from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FactualConsistencyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    BiasMetric,
    ToxicityMetric,
    ConversationalMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset

from models.model_handler import ModelHandler

load_dotenv()

class ModelTester:
    """
    Testing framework for evaluating and comparing different models.
    """
    
    def __init__(self, model_handler: Optional[ModelHandler] = None):
        """Initialize the model tester."""
        self.model_handler = model_handler or ModelHandler()
        self.test_results = {}
        
        # Create results directory if it doesn't exist
        os.makedirs("test_results", exist_ok=True)
    
    def create_test_dataset(self, 
                          task_type: str, 
                          test_cases: List[Dict[str, Any]],
                          dataset_name: Optional[str] = None) -> str:
        """
        Create a test dataset from a list of test cases.
        
        Args:
            task_type: Type of task (text_generation, code_generation, summarization)
            test_cases: List of test case dictionaries
            dataset_name: Optional name for the dataset
            
        Returns:
            Dataset ID
        """
        # Generate dataset ID
        dataset_id = dataset_name or f"{task_type}_{int(time.time())}"
        
        # Convert to deepeval test cases
        deepeval_test_cases = []
        
        for i, test_case in enumerate(test_cases):
            # Extract required parameters
            input_text = test_case.get("input", "")
            context = test_case.get("context", [])
            expected_output = test_case.get("expected_output", "")
            
            # Create test case
            deepeval_test_case = LLMTestCase(
                input=input_text,
                actual_output="",  # Will be filled during testing
                expected_output=expected_output,
                context=context,
                retrieval_context=context,
                metadata={
                    "id": f"{dataset_id}_{i}",
                    "task_type": task_type,
                    "description": test_case.get("description", ""),
                    "tags": test_case.get("tags", [])
                }
            )
            
            deepeval_test_cases.append(deepeval_test_case)
        
        # Create and save dataset
        dataset = EvaluationDataset(
            name=dataset_id,
            test_cases=deepeval_test_cases
        )
        
        # Save dataset to file
        dataset_path = os.path.join("test_results", f"{dataset_id}.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2)
        
        return dataset_id
    
    def load_test_dataset(self, dataset_id: str) -> EvaluationDataset:
        """
        Load a test dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Evaluation dataset
        """
        dataset_path = os.path.join("test_results", f"{dataset_id}.json")
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        with open(dataset_path, "r") as f:
            dataset_dict = json.load(f)
        
        return EvaluationDataset.from_dict(dataset_dict)
    
    def run_model_test(self, 
                      dataset_id: str, 
                      model_config: Dict[str, Any],
                      metrics: Optional[List[str]] = None) -> str:
        """
        Run a test using a specific model configuration.
        
        Args:
            dataset_id: Dataset ID
            model_config: Model configuration (model, model_family, etc.)
            metrics: List of metrics to evaluate
            
        Returns:
            Test result ID
        """
        # Load dataset
        dataset = self.load_test_dataset(dataset_id)
        test_cases = dataset.test_cases
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ["answer_relevancy", "factual_consistency", "faithfulness"]
        
        # Initialize metrics
        metric_instances = []
        for metric_name in metrics:
            if metric_name == "answer_relevancy":
                metric_instances.append(AnswerRelevancyMetric(threshold=0.7))
            elif metric_name == "factual_consistency":
                metric_instances.append(FactualConsistencyMetric(threshold=0.7))
            elif metric_name == "faithfulness":
                metric_instances.append(FaithfulnessMetric(threshold=0.7))
            elif metric_name == "contextual_relevancy":
                metric_instances.append(ContextualRelevancyMetric(threshold=0.7))
            elif metric_name == "contextual_recall":
                metric_instances.append(ContextualRecallMetric(threshold=0.7))
            elif metric_name == "bias":
                metric_instances.append(BiasMetric(threshold=0.7))
            elif metric_name == "toxicity":
                metric_instances.append(ToxicityMetric(threshold=0.05))
            elif metric_name == "conversational":
                metric_instances.append(ConversationalMetric(threshold=0.7))
        
        # Generate test result ID
        model_name = model_config.get("model", "unknown")
        model_family = model_config.get("model_family", "unknown")
        test_result_id = f"{dataset_id}_{model_name}_{int(time.time())}"
        
        # Run tests
        test_results = []
        
        for test_case in tqdm(test_cases, desc=f"Testing model: {model_name}"):
            # Extract test case data
            input_text = test_case.input
            context = test_case.context
            task_type = test_case.metadata.get("task_type", "text_generation")
            
            # Format input as message
            messages = [{"role": "user", "content": input_text}]
            
            # Add context if available
            if context:
                context_text = "\n".join(context)
                messages = [{"role": "system", "content": f"Context: {context_text}"}] + messages
            
            try:
                # Generate response
                response = self.model_handler.generate_text(
                    messages=messages,
                    task_type=task_type,
                    model=model_config.get("model"),
                    model_family=model_config.get("model_family"),
                    temperature=model_config.get("temperature"),
                    max_tokens=model_config.get("max_tokens")
                )
                
                # Extract actual output
                if model_config.get("model_family") == "openai":
                    actual_output = response["choices"][0]["message"]["content"]
                elif model_config.get("model_family") == "anthropic":
                    actual_output = response["content"][0]["text"]
                else:
                    actual_output = str(response)
                
                # Update test case with actual output
                test_case.actual_output = actual_output
                
                # Evaluate metrics
                test_case_results = {"id": test_case.metadata.get("id", "")}
                
                for metric in metric_instances:
                    try:
                        metric.measure(test_case)
                        test_case_results[metric.name] = {
                            "score": metric.score,
                            "passed": metric.passed(),
                            "reason": metric.reason
                        }
                    except Exception as e:
                        test_case_results[metric.name] = {
                            "score": 0,
                            "passed": False,
                            "reason": f"Error: {str(e)}"
                        }
                
                test_results.append(test_case_results)
                
            except Exception as e:
                # Log error and continue with next test case
                test_results.append({
                    "id": test_case.metadata.get("id", ""),
                    "error": str(e)
                })
        
        # Calculate aggregate results
        aggregate_results = {
            "dataset_id": dataset_id,
            "model": model_name,
            "model_family": model_family,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for metric in metric_instances:
            metric_name = metric.name
            scores = [result[metric_name]["score"] for result in test_results if metric_name in result]
            passed = [result[metric_name]["passed"] for result in test_results if metric_name in result]
            
            if scores:
                aggregate_results["metrics"][metric_name] = {
                    "avg_score": sum(scores) / len(scores),
                    "pass_rate": sum(passed) / len(passed) if passed else 0
                }
        
        # Save results
        result_data = {
            "config": model_config,
            "test_results": test_results,
            "aggregate_results": aggregate_results
        }
        
        result_path = os.path.join("test_results", f"{test_result_id}.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Store in instance
        self.test_results[test_result_id] = result_data
        
        return test_result_id
    
    def compare_models(self, test_result_ids: List[str]) -> Dict[str, Any]:
        """
        Compare models based on test results.
        
        Args:
            test_result_ids: List of test result IDs
            
        Returns:
            Comparison results
        """
        # Load test results if not already in memory
        results = []
        for result_id in test_result_ids:
            if result_id in self.test_results:
                results.append(self.test_results[result_id])
            else:
                result_path = os.path.join("test_results", f"{result_id}.json")
                if os.path.exists(result_path):
                    with open(result_path, "r") as f:
                        result_data = json.load(f)
                        self.test_results[result_id] = result_data
                        results.append(result_data)
                else:
                    raise ValueError(f"Test result not found: {result_id}")
        
        # Extract model names and metrics
        models = [result["aggregate_results"]["model"] for result in results]
        all_metrics = set()
        for result in results:
            all_metrics.update(result["aggregate_results"]["metrics"].keys())
        
        # Prepare comparison data
        comparison = {
            "models": models,
            "metrics": {metric: [] for metric in all_metrics},
            "pass_rates": {metric: [] for metric in all_metrics}
        }
        
        for result in results:
            model_metrics = result["aggregate_results"]["metrics"]
            for metric in all_metrics:
                if metric in model_metrics:
                    comparison["metrics"][metric].append(model_metrics[metric]["avg_score"])
                    comparison["pass_rates"][metric].append(model_metrics[metric]["pass_rate"])
                else:
                    comparison["metrics"][metric].append(0)
                    comparison["pass_rates"][metric].append(0)
        
        # Generate visualizations
        self._generate_comparison_charts(comparison, test_result_ids)
        
        return comparison
    
    def _generate_comparison_charts(self, comparison: Dict[str, Any], test_result_ids: List[str]) -> None:
        """Generate comparison charts."""
        models = comparison["models"]
        metrics = comparison["metrics"]
        pass_rates = comparison["pass_rates"]
        
        # Create figure for metrics
        plt.figure(figsize=(12, 8))
        x = range(len(models))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            plt.bar([pos + width * i for pos in x], metrics[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Comparison - Metric Scores')
        plt.xticks([pos + width * (len(metrics) - 1) / 2 for pos in x], models)
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join("test_results", f"comparison_{'_'.join(test_result_ids)}_metrics.png")
        plt.savefig(chart_path)
        
        # Create figure for pass rates
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(pass_rates):
            plt.bar([pos + width * i for pos in x], pass_rates[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Pass Rate')
        plt.title('Model Comparison - Pass Rates')
        plt.xticks([pos + width * (len(metrics) - 1) / 2 for pos in x], models)
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join("test_results", f"comparison_{'_'.join(test_result_ids)}_pass_rates.png")
        plt.savefig(chart_path)
    
    def export_results_to_csv(self, test_result_id: str) -> str:
        """
        Export test results to CSV.
        
        Args:
            test_result_id: Test result ID
            
        Returns:
            Path to the CSV file
        """
        # Load test results
        if test_result_id in self.test_results:
            result_data = self.test_results[test_result_id]
        else:
            result_path = os.path.join("test_results", f"{test_result_id}.json")
            if not os.path.exists(result_path):
                raise ValueError(f"Test result not found: {test_result_id}")
            
            with open(result_path, "r") as f:
                result_data = json.load(f)
        
        # Prepare CSV data
        csv_rows = []
        
        for test_case in result_data["test_results"]:
            row = {
                "test_case_id": test_case.get("id", ""),
                "model": result_data["config"].get("model", ""),
                "model_family": result_data["config"].get("model_family", "")
            }
            
            # Add metric scores
            for metric_name, metric_data in test_case.items():
                if isinstance(metric_data, dict) and "score" in metric_data:
                    row[f"{metric_name}_score"] = metric_data["score"]
                    row[f"{metric_name}_passed"] = metric_data["passed"]
            
            csv_rows.append(row)
        
        # Write to CSV
        csv_path = os.path.join("test_results", f"{test_result_id}.csv")
        
        with open(csv_path, "w", newline="") as f:
            if csv_rows:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
        
        return csv_path