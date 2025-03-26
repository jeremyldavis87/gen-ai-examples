# testing/benchmark.py
import os
import json
import time
import csv
import argparse
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project directory to the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_handler import ModelHandler

class ModelBenchmark:
    """Benchmark different models across various tasks."""
    
    def __init__(self, model_handler: Optional[ModelHandler] = None):
        """Initialize the benchmark."""
        self.model_handler = model_handler or ModelHandler()
        
        # Create results directory if it doesn't exist
        os.makedirs("benchmark_results", exist_ok=True)
    
    def load_benchmark_tasks(self, benchmark_file: str) -> Dict[str, Any]:
        """
        Load benchmark tasks from a JSON file.  
        
        Args:
            benchmark_file: Path to the benchmark file
            
        Returns:
            Benchmark configuration
        """
        with open(benchmark_file, 'r') as f:
            return json.load(f)
    
    def run_single_task(self, 
                       model_config: Dict[str, Any], 
                       task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single benchmark task.
        
        Args:
            model_config: Model configuration
            task: Task configuration
            
        Returns:
            Task result
        """
        task_type = task.get("task_type", "text_generation")
        messages = task.get("messages", [])
        
        start_time = time.time()
        
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
            
            # Extract content
            if model_config.get("model_family") == "openai":
                content = response["choices"][0]["message"]["content"]
            elif model_config.get("model_family") == "anthropic":
                content = response["content"][0]["text"]
            else:
                content = str(response)
            
            execution_time = time.time() - start_time
            
            # Compute token counts
            input_tokens = self._estimate_tokens(str(messages))
            output_tokens = self._estimate_tokens(content)
            
            return {
                "task_id": task.get("id", "unknown"),
                "model": model_config.get("model", "unknown"),
                "model_family": model_config.get("model_family", "unknown"),
                "task_type": task_type,
                "success": True,
                "content": content,
                "execution_time": execution_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                "task_id": task.get("id", "unknown"),
                "model": model_config.get("model", "unknown"),
                "model_family": model_config.get("model_family", "unknown"),
                "task_type": task_type,
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "input_tokens": self._estimate_tokens(str(messages)),
                "output_tokens": 0,
                "total_tokens": self._estimate_tokens(str(messages))
            }
    
    def run_benchmark(self, 
                    benchmark_file: str, 
                    models: List[Dict[str, Any]],
                    parallel: bool = False,
                    max_workers: int = 4) -> str:
        """
        Run a benchmark across multiple models.
        
        Args:
            benchmark_file: Path to the benchmark file
            models: List of model configurations
            parallel: Whether to run tasks in parallel
            max_workers: Maximum number of worker threads
            
        Returns:
            Benchmark result ID
        """
        # Load benchmark tasks
        benchmark = self.load_benchmark_tasks(benchmark_file)
        
        benchmark_name = benchmark.get("name", "unnamed_benchmark")
        tasks = benchmark.get("tasks", [])
        
        if not tasks:
            raise ValueError("No tasks found in the benchmark file")
        
        # Generate benchmark ID
        benchmark_id = f"{benchmark_name}_{int(time.time())}"
        
        # Run the benchmark for each model
        all_results = []
        
        for model_config in models:
            model_name = model_config.get("model", "unknown")
            model_family = model_config.get("model_family", "unknown")
            
            logger.info(f"Running benchmark for {model_name} ({model_family})")
            
            if parallel:
                # Run tasks in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {
                        executor.submit(self.run_single_task, model_config, task): task
                        for task in tasks
                    }
                    
                    task_results = []
                    for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Tasks for {model_name}"):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            task_results.append(result)
                        except Exception as e:
                            logger.error(f"Task {task.get('id', 'unknown')} generated an exception: {str(e)}")
                            task_results.append({
                                "task_id": task.get("id", "unknown"),
                                "model": model_name,
                                "model_family": model_family,
                                "task_type": task.get("task_type", "text_generation"),
                                "success": False,
                                "error": str(e),
                                "execution_time": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0
                            })
            else:
                # Run tasks sequentially
                task_results = []
                for task in tqdm(tasks, desc=f"Tasks for {model_name}"):
                    result = self.run_single_task(model_config, task)
                    task_results.append(result)
            
            all_results.extend(task_results)
        
        # Save results
        result_path = os.path.join("benchmark_results", f"{benchmark_id}.json")
        with open(result_path, 'w') as f:
            json.dump({
                "benchmark_id": benchmark_id,
                "benchmark_name": benchmark_name,
                "models": models,
                "results": all_results
            }, f, indent=2)
        
        # Generate report
        self.generate_report(benchmark_id)
        
        return benchmark_id
    
    def generate_report(self, benchmark_id: str) -> None:
        """
        Generate a report for a benchmark.
        
        Args:
            benchmark_id: Benchmark ID
        """
        # Load benchmark results
        result_path = os.path.join("benchmark_results", f"{benchmark_id}.json")
        
        if not os.path.exists(result_path):
            raise ValueError(f"Benchmark results not found: {benchmark_id}")
        
        with open(result_path, 'r') as f:
            benchmark_data = json.load(f)
        
        benchmark_name = benchmark_data.get("benchmark_name", "unnamed_benchmark")
        models = benchmark_data.get("models", [])
        results = benchmark_data.get("results", [])
        
        if not results:
            logger.warning("No results found in the benchmark data")
            return
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Calculate aggregate metrics
        model_metrics = {}
        for model_config in models:
            model_name = model_config.get("model", "unknown")
            model_results = df[df["model"] == model_name]
            
            success_rate = model_results["success"].mean() if not model_results.empty else 0
            avg_execution_time = model_results["execution_time"].mean() if not model_results.empty else 0
            avg_input_tokens = model_results["input_tokens"].mean() if not model_results.empty else 0
            avg_output_tokens = model_results["output_tokens"].mean() if not model_results.empty else 0
            avg_total_tokens = model_results["total_tokens"].mean() if not model_results.empty else 0
            
            model_metrics[model_name] = {
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "avg_total_tokens": avg_total_tokens
            }
        
        # Generate aggregates by task type
        task_type_metrics = {}
        task_types = df["task_type"].unique()
        
        for task_type in task_types:
            task_results = df[df["task_type"] == task_type]
            
            model_task_metrics = {}
            for model_name in df["model"].unique():
                model_task_results = task_results[task_results["model"] == model_name]
                
                success_rate = model_task_results["success"].mean() if not model_task_results.empty else 0
                avg_execution_time = model_task_results["execution_time"].mean() if not model_task_results.empty else 0
                
                model_task_metrics[model_name] = {
                    "success_rate": success_rate,
                    "avg_execution_time": avg_execution_time
                }
            
            task_type_metrics[task_type] = model_task_metrics
        
        # Create report directory
        report_dir = os.path.join("benchmark_results", benchmark_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # Export results to CSV
        df.to_csv(os.path.join(report_dir, "raw_results.csv"), index=False)
        
        # Generate model comparison table
        model_comparison = []
        for model_name, metrics in model_metrics.items():
            model_comparison.append({
                "Model": model_name,
                "Success Rate": f"{metrics['success_rate']:.2%}",
                "Avg. Execution Time (s)": f"{metrics['avg_execution_time']:.3f}",
                "Avg. Input Tokens": f"{metrics['avg_input_tokens']:.0f}",
                "Avg. Output Tokens": f"{metrics['avg_output_tokens']:.0f}",
                "Avg. Total Tokens": f"{metrics['avg_total_tokens']:.0f}"
            })
        
        # Save model comparison
        with open(os.path.join(report_dir, "model_comparison.txt"), 'w') as f:
            f.write(f"Benchmark: {benchmark_name}\n")
            f.write(f"ID: {benchmark_id}\n\n")
            f.write("Model Comparison\n")
            f.write("=" * 80 + "\n\n")
            f.write(tabulate(model_comparison, headers="keys", tablefmt="grid"))
            f.write("\n\n")
            
            # Write task type metrics
            f.write("Performance by Task Type\n")
            f.write("=" * 80 + "\n\n")
            
            for task_type, task_metrics in task_type_metrics.items():
                f.write(f"Task Type: {task_type}\n")
                f.write("-" * 40 + "\n")
                
                task_comparison = []
                for model_name, metrics in task_metrics.items():
                    task_comparison.append({
                        "Model": model_name,
                        "Success Rate": f"{metrics['success_rate']:.2%}",
                        "Avg. Execution Time (s)": f"{metrics['avg_execution_time']:.3f}"
                    })
                
                f.write(tabulate(task_comparison, headers="keys", tablefmt="grid"))
                f.write("\n\n")
        
        # Generate charts
        self._generate_benchmark_charts(benchmark_id, df, model_metrics, task_type_metrics, report_dir)
        
        logger.info(f"Report generated at {report_dir}")
    
    def _generate_benchmark_charts(self, 
                                 benchmark_id: str, 
                                 df: pd.DataFrame,
                                 model_metrics: Dict[str, Dict[str, float]],
                                 task_type_metrics: Dict[str, Dict[str, Dict[str, float]]],
                                 report_dir: str) -> None:
        """Generate benchmark charts."""
        # Success rate comparison
        plt.figure(figsize=(10, 6))
        model_names = list(model_metrics.keys())
        success_rates = [metrics["success_rate"] for metrics in model_metrics.values()]
        
        plt.bar(model_names, success_rates)
        plt.xlabel('Models')
        plt.ylabel('Success Rate')
        plt.title('Model Comparison - Success Rate')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        for i, rate in enumerate(success_rates):
            plt.text(i, rate + 0.02, f"{rate:.2%}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "success_rate.png"))
        plt.close()
        
        # Execution time comparison
        plt.figure(figsize=(10, 6))
        execution_times = [metrics["avg_execution_time"] for metrics in model_metrics.values()]
        
        plt.bar(model_names, execution_times)
        plt.xlabel('Models')
        plt.ylabel('Average Execution Time (s)')
        plt.title('Model Comparison - Execution Time')
        plt.grid(axis='y', alpha=0.3)
        
        for i, time_val in enumerate(execution_times):
            plt.text(i, time_val + 0.1, f"{time_val:.3f}s", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "execution_time.png"))
        plt.close()
        
        # Token comparison
        plt.figure(figsize=(12, 8))
        input_tokens = [metrics["avg_input_tokens"] for metrics in model_metrics.values()]
        output_tokens = [metrics["avg_output_tokens"] for metrics in model_metrics.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, input_tokens, width, label='Input Tokens')
        plt.bar(x + width/2, output_tokens, width, label='Output Tokens')
        
        plt.xlabel('Models')
        plt.ylabel('Average Tokens')
        plt.title('Model Comparison - Token Usage')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "token_usage.png"))
        plt.close()
        
        # Task type success rate comparison
        for task_type, task_metrics in task_type_metrics.items():
            plt.figure(figsize=(10, 6))
            model_names = list(task_metrics.keys())
            success_rates = [metrics["success_rate"] for metrics in task_metrics.values()]
            
            plt.bar(model_names, success_rates)
            plt.xlabel('Models')
            plt.ylabel('Success Rate')
            plt.title(f'Task Type: {task_type} - Success Rate')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            for i, rate in enumerate(success_rates):
                plt.text(i, rate + 0.02, f"{rate:.2%}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f"task_{task_type}_success_rate.png"))
            plt.close()
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        This is a rough approximation using the common rule of thumb.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4


def main():
    """Run benchmark from command line."""
    parser = argparse.ArgumentParser(description="Benchmark AI models")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark configuration file")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for parallel execution")
    
    args = parser.parse_args()
    
    # Initialize model handler and benchmark
    model_handler = ModelHandler()
    benchmark = ModelBenchmark(model_handler)
    
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
    
    # Run benchmark
    benchmark_id = benchmark.run_benchmark(
        benchmark_file=args.benchmark,
        models=models_to_test,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    print(f"Benchmark completed. ID: {benchmark_id}")
    print(f"Report available at: benchmark_results/{benchmark_id}/")

if __name__ == "__main__":
    main()