# structured_data_with_fastmcp.py
import os
from typing import List, Dict, Any, Optional
from fastmcp import FastModelContextProtocol, Context, Message
from dotenv import load_dotenv
import json
import pandas as pd

# Import FastMCP Handler
from fastmcp_implementation import FastMCPHandler

load_dotenv()

class StructuredDataHandler:
    """Handler for structured data using FastMCP."""
    
    def __init__(self):
        """Initialize the structured data handler."""
        self.mcp_handler = FastMCPHandler()
    
    def process_table_data(self, 
                         dataframe: pd.DataFrame, 
                         user_query: str,
                         table_name: str = "data_table") -> str:
        """
        Process tabular data using FastMCP.
        
        Args:
            dataframe: Pandas DataFrame with tabular data
            user_query: User's query about the data
            table_name: Name of the table for reference
            
        Returns:
            Response from the model
        """
        # Convert DataFrame to dictionary for context
        table_dict = {
            "columns": list(dataframe.columns),
            "data": dataframe.head(100).to_dict(orient="records"),  # Limit to first 100 rows
            "summary": {
                "shape": dataframe.shape,
                "dtypes": {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                "missing_values": dataframe.isna().sum().to_dict()
            }
        }
        
        # Create table context
        table_context = self.mcp_handler.create_context(
            context_type="tabular_data",
            content=table_dict,
            metadata={
                "table_name": table_name,
                "row_count": dataframe.shape[0],
                "column_count": dataframe.shape[1]
            }
        )
        
        # Create system message with instructions
        system_message = self.mcp_handler.create_message(
            role="system",
            content="You are a data analysis assistant. Answer questions about the provided table data. "
                    "Provide insights and analysis based on the data."
        )
        
        # Create schema context with column descriptions
        schema_data = {}
        for column in dataframe.columns:
            col_type = str(dataframe[column].dtype)
            unique_values = dataframe[column].nunique()
            sample_values = dataframe[column].dropna().sample(min(5, len(dataframe))).tolist()
            
            schema_data[column] = {
                "type": col_type,
                "unique_values": unique_values,
                "sample_values": sample_values
            }
        
        schema_context = self.mcp_handler.create_context(
            context_type="schema",
            content=schema_data,
            metadata={"table_name": table_name}
        )
        
        # Create user message with query and contexts
        user_message = self.mcp_handler.create_message(
            role="user",
            content=user_query,
            contexts=[table_context, schema_context]
        )
        
        # Generate response
        response_content = self.mcp_handler.generate_response(
            messages=[system_message, user_message],
            model="gpt4o",
            model_family="openai"
        )
        
        return response_content
    
    def process_json_data(self, 
                        json_data: Dict[str, Any], 
                        user_query: str,
                        data_description: str = "JSON Data") -> str:
        """
        Process JSON data using FastMCP.
        
        Args:
            json_data: JSON data structure
            user_query: User's query about the data
            data_description: Description of the JSON data
            
        Returns:
            Response from the model
        """
        # Create JSON context
        json_context = self.mcp_handler.create_context(
            context_type="json_data",
            content=json_data,
            metadata={
                "description": data_description,
                "structure": self._analyze_json_structure(json_data)
            }
        )
        
        # Create system message with instructions
        system_message = self.mcp_handler.create_message(
            role="system",
            content="You are a JSON data analysis assistant. Answer questions about the provided JSON data. "
                    "Provide insights and help extract relevant information from the data."
        )
        
        # Create user message with query and contexts
        user_message = self.mcp_handler.create_message(
            role="user",
            content=user_query,
            contexts=[json_context]
        )
        
        # Generate response
        response_content = self.mcp_handler.generate_response(
            messages=[system_message, user_message],
            model="gpt4o",
            model_family="openai"
        )
        
        return response_content
    
    def _analyze_json_structure(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of JSON data."""
        if isinstance(json_data, dict):
            return {
                "type": "object",
                "keys": list(json_data.keys()),
                "sample_keys": list(json_data.keys())[:5]
            }
        elif isinstance(json_data, list):
            return {
                "type": "array",
                "length": len(json_data),
                "sample_items": json_data[:3] if json_data else []
            }
        else:
            return {
                "type": str(type(json_data).__name__)
            }