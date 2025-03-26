# example_usage.py
import pandas as pd
import json
from mcp_implementation import MCPHandler
from fastmcp_implementation import FastMCPHandler
from rag_with_mcp import RAGWithMCP
from structured_data_with_fastmcp import StructuredDataHandler

def mcp_basic_example():
    """Basic MCP example."""
    print("=== Basic MCP Example ===")
    
    # Initialize handler
    mcp_handler = MCPHandler()
    
    # Create contexts
    document_context = mcp_handler.create_context(
        context_type="document",
        content="The Model Context Protocol (MCP) is a standardized format for providing context to language models.",
        metadata={"source": "documentation", "id": "doc123"}
    )
    
    code_context = mcp_handler.create_context(
        context_type="code",
        content="def hello_world():\n    print('Hello, world!')",
        metadata={"language": "python", "id": "code456"}
    )
    
    # Create messages
    system_message = mcp_handler.create_message(
        role="system",
        content="You are a helpful assistant."
    )
    
    user_message = mcp_handler.create_message(
        role="user",
        content="Can you explain this code and document to me?",
        contexts=[document_context, code_context]
    )
    
    # Generate response
    response = mcp_handler.generate_response(
        messages=[system_message, user_message]
    )
    
    print(f"Response: {response}\n")

def fastmcp_example():
    """FastMCP example with batch processing."""
    print("=== FastMCP Example ===")
    
    # Initialize handler
    fastmcp_handler = FastMCPHandler()
    
    # Create multiple contexts using batch processing
    contexts_data = [
        {
            "type": "document",
            "content": "FastMCP is an optimized implementation of the Model Context Protocol.",
            "metadata": {"source": "documentation", "id": "doc789"}
        },
        {
            "type": "image_description",
            "content": "A diagram showing the architecture of the Model Context Protocol.",
            "metadata": {"source": "diagram", "id": "img101"}
        },
        {
            "type": "code",
            "content": "from fastmcp import FastModelContextProtocol\nmcp = FastModelContextProtocol()",
            "metadata": {"language": "python", "id": "code202"}
        }
    ]
    
    # Batch process contexts
    contexts = fastmcp_handler.batch_process_contexts(contexts_data)
    
    # Create messages
    system_message = fastmcp_handler.create_message(
        role="system",
        content="You are a helpful assistant that explains technical concepts."
    )
    
    user_message = fastmcp_handler.create_message(
        role="user",
        content="Can you help me understand FastMCP and how to implement it?",
        contexts=contexts
    )
    
    # Generate response
    response = fastmcp_handler.generate_response(
        messages=[system_message, user_message]
    )
    
    print(f"Response: {response}\n")

def rag_example():
    """RAG with MCP example."""
    print("=== RAG with MCP Example ===")
    
    # Initialize RAG with MCP
    rag = RAGWithMCP()
    
    # Query the system
    query = "What are the benefits of using Model Context Protocol?"
    response = rag.query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}\n")

def structured_data_example():
    """Structured data with FastMCP example."""
    print("=== Structured Data with FastMCP Example ===")
    
    # Initialize handler
    structured_handler = StructuredDataHandler()
    
    # Create sample DataFrame
    data = {
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 34, 45, 29, 38],
        "subscription": ["Basic", "Premium", "Premium", "Basic", "Enterprise"],
        "monthly_spend": [29.99, 49.99, 49.99, 29.99, 99.99],
        "signup_date": ["2022-01-15", "2021-10-05", "2023-02-20", "2022-11-30", "2023-01-10"]
    }
    
    df = pd.DataFrame(data)
    
    # Process table data
    table_query = "Can you analyze the customer base and their spending patterns?"
    table_response = structured_handler.process_table_data(
        dataframe=df,
        user_query=table_query,
        table_name="customer_data"
    )
    
    print(f"Table Query: {table_query}")
    print(f"Table Response: {table_response}\n")
    
    # Sample JSON data
    json_data = {
        "organization": "Acme Corp",
        "departments": [
            {
                "name": "Engineering",
                "employees": [
                    {"id": 101, "name": "John Smith", "role": "Software Engineer", "projects": ["Project A", "Project B"]},
                    {"id": 102, "name": "Emily Jones", "role": "Data Scientist", "projects": ["Project C"]}
                ],
                "budget": 500000
            },
            {
                "name": "Marketing",
                "employees": [
                    {"id": 201, "name": "Michael Brown", "role": "Marketing Manager", "campaigns": ["Campaign X", "Campaign Y"]},
                    {"id": 202, "name": "Sarah Davis", "role": "Content Specialist", "campaigns": ["Campaign Z"]}
                ],
                "budget": 350000
            }
        ]
    }
    
    # Process JSON data
    json_query = "Who works in the Engineering department and what projects are they on?"
    json_response = structured_handler.process_json_data(
        json_data=json_data,
        user_query=json_query,
        data_description="Company Organization Data"
    )
    
    print(f"JSON Query: {json_query}")
    print(f"JSON Response: {json_response}")

if __name__ == "__main__":
    mcp_basic_example()
    fastmcp_example()
    rag_example()
    structured_data_example()