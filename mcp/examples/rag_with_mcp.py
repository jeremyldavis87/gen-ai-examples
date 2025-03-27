# rag_with_mcp.py
import os
from typing import List, Dict, Any, Optional
from mcp import ModelContextProtocol, Context, Message
from dotenv import load_dotenv
import json

# Import your vector database client
from vector_db.pgvector_client import PGVectorClient

# Import your AI Gateway client
from ai_gateway.client import AIGatewayClient

# Import MCP Handler
from mcp_implementation import MCPHandler

load_dotenv()

class RAGWithMCP:
    """RAG system using Model Context Protocol."""
    
    def __init__(self):
        """Initialize the RAG with MCP system."""
        self.vector_db = PGVectorClient()
        self.gateway_client = AIGatewayClient()
        self.mcp_handler = MCPHandler()
    
    def query(self, user_query: str, top_k: int = 5) -> str:
        """
        Process a query using RAG with MCP.
        
        Args:
            user_query: User's query text
            top_k: Number of top documents to retrieve
            
        Returns:
            Response from the model
        """
        # Generate embedding for the query
        query_embedding = self.gateway_client.generate_embeddings(
            texts=[user_query],
            model="text-embedding-3-large"
        )[0]
        
        # Retrieve relevant documents
        search_results = self.vector_db.search_similar(
            query_embedding=query_embedding,
            limit=top_k,
            similarity_threshold=0.7
        )
        
        # Convert search results to MCP contexts
        contexts = []
        for result in search_results:
            # Create context from search result
            context = self.mcp_handler.create_context(
                context_type="document",
                content=result["content"],
                metadata={
                    "id": result["id"],
                    "source": result.get("metadata", {}).get("source", "unknown"),
                    "similarity": result["similarity"]
                }
            )
            contexts.append(context)
        
        # Create system message with instructions
        system_message = self.mcp_handler.create_message(
            role="system",
            content="You are a helpful assistant that answers questions based on the provided context. "
                    "If the context doesn't contain relevant information, just say that you don't know."
        )
        
        # Create user message with query and contexts
        user_message = self.mcp_handler.create_message(
            role="user",
            content=user_query,
            contexts=contexts
        )
        
        # Generate response
        response_content = self.mcp_handler.generate_response(
            messages=[system_message, user_message],
            model="gpt4o",
            model_family="openai"
        )
        
        return response_content