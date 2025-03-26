# models/model_handler.py
import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class ModelHandler:
    """
    Handler for AI models with fallback capabilities and performance tracking.
    """
    
    def __init__(self, ai_gateway_url: Optional[str] = None, api_key: Optional[str] = None, project_name: Optional[str] = None):
        """Initialize the model handler."""
        self.ai_gateway_url = ai_gateway_url or os.getenv("AI_GATEWAY_URL")
        self.api_key = api_key or os.getenv("API_KEY")
        self.project_name = project_name or os.getenv("PROJECT_NAME")
        
        if not self.ai_gateway_url or not self.api_key or not self.project_name:
            raise ValueError("Missing required environment variables for AI Gateway")
            
        # Default models configuration
        self.default_configs = {
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
            "code_generation": {
                "primary": {"model": "gpt4o", "model_family": "openai", "temperature": 0.3, "max_tokens": 2000},
                "fallbacks": [
                    {"model": "sonnet-3.7", "model_family": "anthropic", "temperature": 0.3, "max_tokens": 2000},
                ]
            },
            "summarization": {
                "primary": {"model": "haiku-3.5", "model_family": "anthropic", "temperature": 0.3, "max_tokens": 500},
                "fallbacks": [
                    {"model": "o1-mini", "model_family": "openai", "temperature": 0.3, "max_tokens": 500},
                ]
            }
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_calls": 0,
            "failed_calls": 0,
            "avg_latency": 0,
            "model_performance": {}
        }
        
        # Import here to avoid circular imports
        from ai_gateway.client import AIGatewayClient
        self.client = AIGatewayClient(project_name=self.project_name, api_key=self.api_key)
    
    def _update_metrics(self, model: str, success: bool, latency: float, used_fallback: bool = False) -> None:
        """Update performance metrics."""
        self.performance_metrics["total_calls"] += 1
        
        if success:
            self.performance_metrics["successful_calls"] += 1
            if used_fallback:
                self.performance_metrics["fallback_calls"] += 1
        else:
            self.performance_metrics["failed_calls"] += 1
        
        # Update average latency
        total_successful = self.performance_metrics["successful_calls"]
        current_avg = self.performance_metrics["avg_latency"]
        new_avg = ((current_avg * (total_successful - 1)) + latency) / total_successful if total_successful > 0 else latency
        self.performance_metrics["avg_latency"] = new_avg
        
        # Update model-specific metrics
        if model not in self.performance_metrics["model_performance"]:
            self.performance_metrics["model_performance"][model] = {
                "calls": 0,
                "successful_calls": 0,
                "avg_latency": 0
            }
        
        model_metrics = self.performance_metrics["model_performance"][model]
        model_metrics["calls"] += 1
        if success:
            model_metrics["successful_calls"] += 1
            model_calls = model_metrics["successful_calls"]
            current_model_avg = model_metrics["avg_latency"]
            new_model_avg = ((current_model_avg * (model_calls - 1)) + latency) / model_calls if model_calls > 0 else latency
            model_metrics["avg_latency"] = new_model_avg
    
    def generate_text(self, 
                     messages: List[Dict[str, str]],
                     task_type: str = "text_generation",
                     model: Optional[str] = None,
                     model_family: Optional[str] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     retry_attempts: int = 2) -> Dict[str, Any]:
        """
        Generate text with automatic fallback to alternative models if the primary fails.
        
        Args:
            messages: List of message objects in the format [{"role": "user", "content": "Hello"}]
            task_type: Type of task (text_generation, code_generation, summarization)
            model: Optional model override
            model_family: Optional model family override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            retry_attempts: Number of retry attempts before using fallbacks
            
        Returns:
            The response from the AI Gateway
        """
        # Get configs for this task type
        if task_type not in self.default_configs:
            logger.warning(f"Unknown task type: {task_type}. Using text_generation defaults.")
            task_type = "text_generation"
            
        configs = self.default_configs[task_type]
        
        # Start with primary model (or overridden model)
        current_config = configs["primary"].copy()
        
        # Apply overrides if provided
        if model:
            current_config["model"] = model
        if model_family:
            current_config["model_family"] = model_family
        if temperature is not None:
            current_config["temperature"] = temperature
        if max_tokens is not None:
            current_config["max_tokens"] = max_tokens
        
        # Try primary model with retries
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                response = self.client.generate_text(
                    model=current_config["model"],
                    messages=messages,
                    temperature=current_config["temperature"],
                    max_tokens=current_config["max_tokens"],
                    model_family=current_config["model_family"]
                )
                
                latency = time.time() - start_time
                self._update_metrics(
                    model=current_config["model"],
                    success=True,
                    latency=latency
                )
                
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed with model {current_config['model']}: {str(e)}")
                if attempt < retry_attempts - 1:
                    # Wait before retrying
                    time.sleep(1)
                else:
                    # Out of retry attempts, update metrics before trying fallbacks
                    self._update_metrics(
                        model=current_config["model"],
                        success=False,
                        latency=0
                    )
        
        # If we're here, primary model failed after all retries
        # Try fallback models
        for fallback_config in configs["fallbacks"]:
            try:
                logger.info(f"Trying fallback model: {fallback_config['model']}")
                
                # Apply relevant overrides to fallback config too
                if temperature is not None:
                    fallback_config["temperature"] = temperature
                if max_tokens is not None:
                    fallback_config["max_tokens"] = max_tokens
                
                start_time = time.time()
                
                response = self.client.generate_text(
                    model=fallback_config["model"],
                    messages=messages,
                    temperature=fallback_config["temperature"],
                    max_tokens=fallback_config["max_tokens"],
                    model_family=fallback_config["model_family"]
                )
                
                latency = time.time() - start_time
                self._update_metrics(
                    model=fallback_config["model"],
                    success=True,
                    latency=latency,
                    used_fallback=True
                )
                
                return response
                
            except Exception as e:
                logger.warning(f"Fallback model {fallback_config['model']} failed: {str(e)}")
                self._update_metrics(
                    model=fallback_config["model"],
                    success=False,
                    latency=0
                )
        
        # If we're here, all models failed
        raise Exception(f"All models failed for task type: {task_type}")
    
    def generate_embeddings(self, 
                           texts: List[str],
                           model: Optional[str] = None,
                           model_family: Optional[str] = None,
                           retry_attempts: int = 2) -> List[List[float]]:
        """
        Generate embeddings with automatic fallback to alternative models if the primary fails.
        
        Args:
            texts: List of text strings to embed
            model: Optional model override
            model_family: Optional model family override
            retry_attempts: Number of retry attempts before using fallbacks
            
        Returns:
            List of embeddings (as float vectors)
        """
        # Get configs for embeddings
        configs = self.default_configs["embeddings"]
        
        # Start with primary model (or overridden model)
        current_config = configs["primary"].copy()
        
        # Apply overrides if provided
        if model:
            current_config["model"] = model
        if model_family:
            current_config["model_family"] = model_family
        
        # Try primary model with retries
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                embeddings = self.client.generate_embeddings(
                    texts=texts,
                    model=current_config["model"],
                    model_family=current_config["model_family"]
                )
                
                latency = time.time() - start_time
                self._update_metrics(
                    model=current_config["model"],
                    success=True,
                    latency=latency
                )
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed with model {current_config['model']}: {str(e)}")
                if attempt < retry_attempts - 1:
                    # Wait before retrying
                    time.sleep(1)
                else:
                    # Out of retry attempts, update metrics before trying fallbacks
                    self._update_metrics(
                        model=current_config["model"],
                        success=False,
                        latency=0
                    )
        
        # If we're here, primary model failed after all retries
        # Try fallback models
        for fallback_config in configs["fallbacks"]:
            try:
                logger.info(f"Trying fallback embedding model: {fallback_config['model']}")
                
                start_time = time.time()
                
                embeddings = self.client.generate_embeddings(
                    texts=texts,
                    model=fallback_config["model"],
                    model_family=fallback_config["model_family"]
                )
                
                latency = time.time() - start_time
                self._update_metrics(
                    model=fallback_config["model"],
                    success=True,
                    latency=latency,
                    used_fallback=True
                )
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Fallback model {fallback_config['model']} failed: {str(e)}")
                self._update_metrics(
                    model=fallback_config["model"],
                    success=False,
                    latency=0
                )
        
        # If we're here, all models failed
        raise Exception("All embedding models failed")
    
    def set_default_config(self, task_type: str, config: Dict[str, Any]) -> None:
        """
        Set a custom default configuration for a task type.
        
        Args:
            task_type: Type of task
            config: Configuration including primary and fallback models
        """
        if not ("primary" in config and "fallbacks" in config):
            raise ValueError("Config must contain 'primary' and 'fallbacks' keys")
            
        self.default_configs[task_type] = config
        logger.info(f"Updated default config for task type: {task_type}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get the current performance metrics."""
        return self.performance_metrics
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_calls": 0,
            "failed_calls": 0,
            "avg_latency": 0,
            "model_performance": {}
        }
        logger.info("Performance metrics reset")