# mcp_implementation.py
import os
from typing import List, Dict, Any, Optional
from mcp import ModelContextProtocol, Context, Message
from dotenv import load_dotenv

load_dotenv()

class MCPHandler:
    """Handler for Model Context Protocol (MCP)."""
    
    def __init__(self, ai_gateway_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the MCP handler."""
        self.ai_gateway_url = ai_gateway_url or os.getenv("AI_GATEWAY_URL")
        self.api_key = api_key or os.getenv("API_KEY")
        
        if not self.ai_gateway_url or not self.api_key:
            raise ValueError("Missing required environment variables for AI Gateway")
        
        # Initialize MCP
        self.mcp = ModelContextProtocol()
    
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
        Format MCP messages for model input.
        
        Args:
            messages: List of MCP message objects
            
        Returns:
            List of model-compatible message dictionaries
        """
        return self.mcp.to_model_format(messages)
    
    def generate_response(self, 
                        messages: List[Message], 
                        model: str = "gpt4o",
                        model_family: str = "openai") -> str:
        """
        Generate a response using the model with MCP-formatted messages.
        
        Args:
            messages: List of MCP message objects
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