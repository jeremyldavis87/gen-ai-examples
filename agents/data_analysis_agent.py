# agents/data_analysis_agent.py
from typing import Dict, List, Any, Optional, Tuple, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import io
import base64

from langchain_utils.gateway_integration import get_langchain_llm

# Define agent state
class DataAnalysisState(dict):
    """State for the data analysis agent."""
    question: str
    data_path: str
    dataframe: Optional[pd.DataFrame]
    analysis_results: Dict[str, Any]
    plots: List[str]  # Base64 encoded plots
    final_answer: Optional[str]
    
    def __init__(self, question: str, data_path: str):
        self.question = question
        self.data_path = data_path
        self.dataframe = None
        self.analysis_results = {}
        self.plots = []
        self.final_answer = None
        dict.__init__(
            self, 
            question=question,
            data_path=data_path,
            dataframe=None,
            analysis_results={},
            plots=[],
            final_answer=None
        )

# Define node functions
def load_data(state: DataAnalysisState) -> DataAnalysisState:
    """Load data from the provided path."""
    data_path = state["data_path"]
    
    # Determine file type from extension
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(data_path)
    elif data_path.endswith(".json"):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Simple data preprocessing
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == np.float64 or df[col].dtype == np.int64:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("unknown")
    
    # Basic data summary
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isna().sum().to_dict(),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object']).columns)
    }
    
    # Update state
    state["dataframe"] = df
    state["analysis_results"]["data_summary"] = summary
    
    return state

def determine_analysis(state: DataAnalysisState) -> DataAnalysisState:
    """Determine what analysis to perform based on the question."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    # Get data summary
    data_summary = state["analysis_results"]["data_summary"]
    
    system_message = SystemMessage(content=f"""You are a data analysis expert. 
Based on the user's question and the available data, determine what analysis to perform.

The data has the following structure:
- Shape: {data_summary["shape"]}
- Columns: {data_summary["columns"]}
- Data types: {data_summary["dtypes"]}
- Numeric columns: {data_summary["numeric_columns"]}
- Categorical columns: {data_summary["categorical_columns"]}

Output a JSON object with the following structure:
{{
  "analysis_type": "descriptive|inferential|predictive|exploratory",
  "methods": ["method1", "method2", ...],
  "columns_to_use": ["col1", "col2", ...],
  "visualizations": ["viz1", "viz2", ...],
  "reasoning": "Your reasoning for this analysis plan"
}}""")
    
    human_message = HumanMessage(content=f"Question: {state['question']}")
    
    result = llm.invoke([system_message, human_message])
    
    # Parse the JSON response
    try:
        # Extract JSON from the response
        json_match = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_match:
            analysis_plan = json.loads(json_match.group(1))
        else:
            # Try to find json without code blocks
            json_text = result.content
            analysis_plan = json.loads(json_text)
    except Exception as e:
        # If parsing fails, create a default plan
        analysis_plan = {
            "analysis_type": "exploratory",
            "methods": ["summary_statistics", "correlation_analysis"],
            "columns_to_use": data_summary["columns"][:5],  # Use first 5 columns
            "visualizations": ["histogram", "scatter_plot"],
            "reasoning": "Default analysis plan due to parsing error"
        }
    
    # Update state
    state["analysis_results"]["analysis_plan"] = analysis_plan
    
    return state

def execute_analysis(state: DataAnalysisState) -> DataAnalysisState:
    """Execute the determined analysis."""
    df = state["dataframe"]
    analysis_plan = state["analysis_results"]["analysis_plan"]
    
    analysis_type = analysis_plan["analysis_type"]
    methods = analysis_plan["methods"]
    columns_to_use = analysis_plan["columns_to_use"]
    visualizations = analysis_plan["visualizations"]
    
    results = {}
    
    # Filter dataframe to use only specified columns
    available_columns = [col for col in columns_to_use if col in df.columns]
    if not available_columns:
        available_columns = df.columns[:5]  # Fallback to first 5 columns
    
    df_subset = df[available_columns]
    
    # Execute methods
    if "summary_statistics" in methods:
        results["summary_statistics"] = df_subset.describe().to_dict()
    
    if "correlation_analysis" in methods:
        numeric_cols = df_subset.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            results["correlation"] = df_subset[numeric_cols].corr().to_dict()
    
    if "group_by_analysis" in methods:
        if len(df_subset.columns) >= 2:
            cat_col = df_subset.select_dtypes(include=['object']).columns
            num_col = df_subset.select_dtypes(include=['number']).columns
            
            if len(cat_col) > 0 and len(num_col) > 0:
                group_col = cat_col[0]
                agg_col = num_col[0]
                results["groupby"] = df_subset.groupby(group_col)[agg_col].agg(['mean', 'count']).to_dict()
    
    # Create visualizations
    plots = []
    
    if "histogram" in visualizations and df_subset.select_dtypes(include=['number']).columns.any():
        for col in df_subset.select_dtypes(include=['number']).columns[:3]:  # Limit to 3 columns
            plt.figure(figsize=(10, 6))
            plt.hist(df_subset[col].dropna(), bins=20)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            
            # Save plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plots.append({"type": "histogram", "column": col, "data": img_str})
            plt.close()
    
    if "scatter_plot" in visualizations and len(df_subset.select_dtypes(include=['number']).columns) >= 2:
        num_cols = df_subset.select_dtypes(include=['number']).columns
        for i in range(min(len(num_cols), 2)):
            for j in range(i+1, min(len(num_cols), 3)):
                col1, col2 = num_cols[i], num_cols[j]
                plt.figure(figsize=(10, 6))
                plt.scatter(df_subset[col1], df_subset[col2])
                plt.title(f"Scatter Plot: {col1} vs {col2}")
                plt.xlabel(col1)
                plt.ylabel(col2)
                
                # Save plot to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode()
                plots.append({"type": "scatter", "columns": [col1, col2], "data": img_str})
                plt.close()
    
    if "bar_chart" in visualizations and df_subset.select_dtypes(include=['object']).columns.any():
        cat_col = df_subset.select_dtypes(include=['object']).columns[0]
        plt.figure(figsize=(12, 6))
        df_subset[cat_col].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
        plt.title(f"Bar Chart of {cat_col}")
        plt.xlabel(cat_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plots.append({"type": "bar", "column": cat_col, "data": img_str})
        plt.close()
    
    # Update state
    state["analysis_results"]["analysis_output"] = results
    state["plots"] = plots
    
    return state

def generate_insights(state: DataAnalysisState) -> DataAnalysisState:
    """Generate insights from the analysis."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    # Format analysis results and plots
    analysis_plan = state["analysis_results"]["analysis_plan"]
    analysis_output = state["analysis_results"]["analysis_output"]
    
    # Count plots by type
    plot_summary = {}
    for plot in state["plots"]:
        plot_type = plot["type"]
        if plot_type in plot_summary:
            plot_summary[plot_type] += 1
        else:
            plot_summary[plot_type] = 1
    
    system_message = SystemMessage(content=f"""You are a data analysis expert.
Based on the analysis results, generate insights that address the user's question.

Question: {state['question']}

Analysis Plan:
{json.dumps(analysis_plan, indent=2)}

Analysis Results:
{json.dumps(analysis_output, indent=2)}

Visualizations Created:
{json.dumps(plot_summary, indent=2)}

Provide a comprehensive answer to the user's question based on the data analysis.
Focus on insights that directly answer the question and provide actionable recommendations if appropriate.""")
    
    human_message = HumanMessage(content="Generate insights from the analysis.")
    
    result = llm.invoke([system_message, human_message])
    
    # Update state
    state["final_answer"] = result.content
    
    return state

# Build the graph
def build_data_analysis_agent() -> StateGraph:
    """Build a data analysis agent using LangGraph."""
    # Define the graph
    graph = StateGraph(DataAnalysisState)
    
    # Add nodes
    graph.add_node("load_data", load_data)
    graph.add_node("determine_analysis", determine_analysis)
    graph.add_node("execute_analysis", execute_analysis)
    graph.add_node("generate_insights", generate_insights)
    
    # Add edges
    graph.add_edge("load_data", "determine_analysis")
    graph.add_edge("determine_analysis", "execute_analysis")
    graph.add_edge("execute_analysis", "generate_insights")
    graph.add_edge("generate_insights", END)
    
    # Set the entry point
    graph.set_entry_point("load_data")
    
    # Compile the graph
    return graph.compile()

# Example usage
def analyze_data(question: str, data_path: str) -> Dict[str, Any]:
    """Analyze data using the data analysis agent."""
    # Build the agent
    agent = build_data_analysis_agent()
    
    # Run the agent
    result = agent.invoke({"question": question, "data_path": data_path})
    
    return result