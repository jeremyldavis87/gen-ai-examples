# ai_gateway/client.py
import os
import json
import requests
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

load_dotenv()

class AIGatewayClient:
    def __init__(self, project_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the AI Gateway client.
        
        Args:
            project_name: The project name to use with the AI Gateway.
            api_key: The API key for authentication.
        """
        self.base_url = os.getenv("AI_GATEWAY_URL")
        self.project_name = project_name or os.getenv("PROJECT_NAME")
        self.api_key = api_key or os.getenv("API_KEY")
        
        if not self.base_url or not self.project_name or not self.api_key:
            raise ValueError("Missing required environment variables for AI Gateway")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _get_endpoint_url(self, model_family: str) -> str:
        """Get the full endpoint URL for a model family."""
        return f"{self.base_url}/{self.project_name}/{model_family}"
    
    def generate_text(self, 
                     model: str, 
                     messages: List[Dict[str, str]],
                     temperature: float = 0.7,
                     max_tokens: int = 1000,
                     model_family: str = "openai") -> Dict[str, Any]:
        """
        Generate text using a specified model.
        
        Args:
            model: The model to use (e.g., "gpt4", "sonnet-3.7")
            messages: List of message objects in the format [{"role": "user", "content": "Hello"}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model_family: The model family (openai, anthropic, etc.)
            
        Returns:
            The response from the AI Gateway
        """
        endpoint_url = self._get_endpoint_url(model_family)
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if model_family == "anthropic":
            # Convert OpenAI format to Anthropic format
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        
        response = requests.post(endpoint_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from AI Gateway: {response.status_code}, {response.text}")
        
        return response.json()
    
    def generate_embeddings(self, 
                           texts: List[str],
                           model: str = "text-embedding-3-large",
                           model_family: str = "openai") -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: The embedding model to use
            model_family: The model family (typically "openai" for embeddings)
            
        Returns:
            List of embeddings (as float vectors)
        """
        endpoint_url = self._get_endpoint_url(model_family)
        
        payload = {
            "model": model,
            "input": texts
        }
        
        response = requests.post(f"{endpoint_url}/embeddings", headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from AI Gateway: {response.status_code}, {response.text}")
        
        result = response.json()
        return [item["embedding"] for item in result["data"]]