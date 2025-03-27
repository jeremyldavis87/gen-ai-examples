# fastmcp_implementation.py
import os
from typing import List, Dict, Any, Optional
from fastmcp import FastModelContextProtocol, Context, Message
from dotenv import load_dotenv

load_dotenv()

class FastMCPHandler:
    """Handler for Fast Model Context Protocol (FastMCP)."""
    
    def __init__(self, ai_gateway_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the FastMCP handler."""
        self.ai_gateway_url = ai_gateway_url or os.getenv("AI_GATEWAY_URL")
        self.api_key = api_key or os.getenv("API_KEY")
        
        if not self.ai_gateway_url or not self.api_key:
            raise ValueError("Missing required environment variables for AI Gateway")
        
        # Initialize FastMCP
        self.mcp = FastModelContextProtocol()
    
    def create_context(self, 
                      context_type: str, 
                      content: Any, 
                      metadata: Optional[Dict[str, Any]] = None) -> Context:
        """
        Create a context object.
        
        Args:
            context_type: Type of context (e.g., "document", "database_results")
            content: Content of the context
            metadata: Optional metadata for the context
            
        Returns:
            Context object
        """
        return self.mcp.create_context(
            type=context_type,
            content=content,
            metadata=metadata or {}
        )
    
    def create_message(self, 
                      role: str, 
                      content: str, 
                      contexts: Optional[List[Context]] = None) -> Message:
        """
        Create a message with optional contexts.
        
        Args:
            role: Role of the message sender (e.g., "user", "assistant")
            content: Content of the message
            contexts: Optional list of context objects
            
        Returns:
            Message object
        """
        return self.mcp.create_message(
            role=role,
            content=content,
            contexts=contexts or []
        )
    
    def format_for_model(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Format FastMCP messages for model input.
        
        Args:
            messages: List of FastMCP message objects
            
        Returns:
            List of model-compatible message dictionaries
        """
        return self.mcp.to_model_format(messages)
    
    def generate_response(self, 
                        messages: List[Message], 
                        model: str = "gpt4o",
                        model_family: str = "openai") -> str:
        """
        Generate a response using the model with FastMCP-formatted messages.
        
        Args:
            messages: List of FastMCP message objects
            model: Name of the model to use
            model_family: Model family
            
        Returns:
            Response content
        """
        # Format messages for the model
        formatted_messages = self.format_for_model(messages)
        
        # Here you would call your AI Gateway
        # This is a placeholder for your actual model call
        from ai_gateway.client import AIGatewayClient
        client = AIGatewayClient()
        
        response = client.generate_text(
            model=model,
            messages=formatted_messages,
            model_family=model_family
        )
        
        # Extract content from response
        if model_family == "openai":
            content = response["choices"][0]["message"]["content"]
        elif model_family == "anthropic":
            content = response["content"][0]["text"]
        else:
            content = str(response)
        
        return content
    
    def batch_process_contexts(self, contexts: List[Dict[str, Any]]) -> List[Context]:
        """
        Batch process multiple contexts for better performance.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            List of processed Context objects
        """
        processed_contexts = []
        
        # Use FastMCP's batch processing capabilities
        for ctx in contexts:
            processed_contexts.append(self.create_context(
                context_type=ctx.get("type", "document"),
                content=ctx.get("content", ""),
                metadata=ctx.get("metadata", {})
            ))
        
        return processed_contexts