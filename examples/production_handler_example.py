# examples/production_handler_example.py
import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
import logging

from models.model_handler import ModelHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def configure_custom_model_defaults():
    """Configure custom model defaults for specific use cases."""
    model_handler = ModelHandler()
    
    # Configure for customer support tasks
    model_handler.set_default_config(
        task_type="customer_support",
        config={
            "primary": {"model": "gpt4o", "model_family": "openai", "temperature": 0.3, "max_tokens": 1000},
            "fallbacks": [
                {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.3, "max_tokens": 1000},
                {"model": "llama3-70b", "model_family": "llama", "temperature": 0.3, "max_tokens": 1000},
            ]
        }
    )
    
    # Configure for data analysis
    model_handler.set_default_config(
        task_type="data_analysis",
        config={
            "primary": {"model": "gpt4o", "model_family": "openai", "temperature": 0.1, "max_tokens": 2000},
            "fallbacks": [
                {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.1, "max_tokens": 2000},
            ]
        }
    )
    
    # Configure for creative writing
    model_handler.set_default_config(
        task_type="creative_writing",
        config={
            "primary": {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.8, "max_tokens": 3000},
            "fallbacks": [
                {"model": "gpt4o", "model_family": "openai", "temperature": 0.8, "max_tokens": 3000},
            ]
        }
    )
    
    return model_handler

def process_customer_support_request(query: str, model_handler: ModelHandler) -> str:
    """Process a customer support request."""
    logger.info(f"Processing support request: {query}")
    
    # Format as a conversation
    messages = [
        {"role": "system", "content": "You are a helpful customer support agent. Provide concise, accurate responses."},
        {"role": "user", "content": query}
    ]
    
    try:
        # Use the model handler with the custom task type
        response = model_handler.generate_text(
            messages=messages,
            task_type="customer_support"
        )
        
        # Extract the response content
        if "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]
        elif "content" in response:
            content = response["content"][0]["text"]
        else:
            content = str(response)
        
        logger.info("Successfully generated response")
        return content
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return "I apologize, but I'm having trouble processing your request at the moment. Please try again later."

def analyze_data_query(query: str, context: str, model_handler: ModelHandler) -> str:
    """Process a data analysis query."""
    logger.info(f"Processing data analysis query: {query}")
    
    # Format as a conversation with context
    messages = [
        {"role": "system", "content": f"You are a data analysis assistant. Use the following data context to answer questions:\n\n{context}"},
        {"role": "user", "content": query}
    ]
    
    try:
        # Use the model handler with the custom task type
        response = model_handler.generate_text(
            messages=messages,
            task_type="data_analysis"
        )
        
        # Extract the response content
        if "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]
        elif "content" in response:
            content = response["content"][0]["text"]
        else:
            content = str(response)
        
        logger.info("Successfully generated data analysis")
        return content
        
    except Exception as e:
        logger.error(f"Error processing data analysis: {str(e)}")
        return "I apologize, but I'm having trouble analyzing this data at the moment. Please try again later."

def main():
    """Run the production handler example."""
    # Configure model handler with custom defaults
    model_handler = configure_custom_model_defaults()
    
    # Customer support example
    print("\n=== CUSTOMER SUPPORT EXAMPLE ===")
    support_query = "I'm having trouble resetting my password. The reset email never arrives."
    support_response = process_customer_support_request(support_query, model_handler)
    print(f"Query: {support_query}")
    print(f"Response: {support_response}")
    
    # Data analysis example
    print("\n=== DATA ANALYSIS EXAMPLE ===")
    data_context = """
    Monthly Sales Data (2023):
    January: $120,000
    February: $135,000
    March: $142,000
    April: $138,000
    May: $156,000
    June: $178,000
    July: $182,000
    August: $176,000
    September: $195,000
    October: $210,000
    November: $225,000
    December: $240,000
    """
    data_query = "What's the trend in sales over the year, and what might be the forecast for Q1 2024?"
    data_response = analyze_data_query(data_query, data_context, model_handler)
    print(f"Query: {data_query}")
    print(f"Response: {data_response}")
    
    # Show performance metrics
    print("\n=== PERFORMANCE METRICS ===")
    metrics = model_handler.get_performance_metrics()
    print(f"Total Calls: {metrics['total_calls']}")
    print(f"Successful Calls: {metrics['successful_calls']}")
    print(f"Fallback Calls: {metrics['fallback_calls']}")
    print(f"Failed Calls: {metrics['failed_calls']}")
    print(f"Average Latency: {metrics['avg_latency']:.2f} seconds")
    
    print("\nModel-Specific Performance:")
    for model, model_metrics in metrics["model_performance"].items():
        success_rate = model_metrics["successful_calls"] / model_metrics["calls"] if model_metrics["calls"] > 0 else 0
        print(f"  {model}:")
        print(f"    Calls: {model_metrics['calls']}")
        print(f"    Success Rate: {success_rate:.2%}")
        print(f"    Average Latency: {model_metrics['avg_latency']:.2f} seconds")

if __name__ == "__main__":
    main()