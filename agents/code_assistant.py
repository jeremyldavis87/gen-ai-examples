# agents/code_assistant.py
from typing import Dict, List, Any, Optional, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import json
import re
import subprocess
import tempfile
import os

from langchain_utils.gateway_integration import get_langchain_llm

# Define tool functions
def execute_python_code(code: str) -> str:
    """Execute Python code and return the result."""
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
        temp_name = temp.name
        temp.write(code.encode('utf-8'))
    
    try:
        result = subprocess.run(
            ['python', temp_name], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        output = result.stdout
        error = result.stderr
        
        if error:
            return f"Error: {error}"
        else:
            return f"Output: {output}"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        os.unlink(temp_name)

def search_documentation(query: str, language: str = "python") -> str:
    """Search for documentation."""
    # In a real implementation, this would query a documentation API or database
    return f"Documentation results for {language}: {query}"

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Execute Python code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documentation",
            "description": "Search for documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "language": {
                        "type": "string",
                        "description": "The programming language",
                        "default": "python"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Create tool executor
tool_executor = ToolExecutor({
    "execute_python_code": execute_python_code,
    "search_documentation": search_documentation
})

# Define agent state
class CodeAssistantState(dict):
    """State for the code assistant agent."""
    query: str
    messages: List[Dict[str, str]]
    code_snippets: List[str]
    execution_results: List[str]
    documentation: List[str]
    final_response: Optional[str]
    
    def __init__(self, query: str):
        self.query = query
        self.messages = [{"role": "user", "content": query}]
        self.code_snippets = []
        self.execution_results = []
        self.documentation = []
        self.final_response = None
        dict.__init__(
            self, 
            query=query,
            messages=[{"role": "user", "content": query}],
            code_snippets=[],
            execution_results=[],
            documentation=[],
            final_response=None
        )

# Define node functions
def understand_query(state: CodeAssistantState) -> CodeAssistantState:
    """Understand the user's query and decide on an approach."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    system_message = SystemMessage(content="""You are a helpful code assistant.
Analyze the user's query to understand what they're asking for. 
Think about what programming concepts are involved and what approach to take.
Do they need code generation, debugging, explanation, or documentation?""")
    
    human_message = HumanMessage(content=state["query"])
    
    result = llm.invoke([system_message, human_message])
    
    # Add to messages
    state["messages"].append({"role": "assistant", "content": result.content})
    
    return state

def decide_next_action(state: CodeAssistantState) -> str:
    """Decide the next action to take."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    system_message = SystemMessage(content="""Based on the current state, decide what to do next:
1. "search_docs" if you need to look up documentation
2. "generate_code" if you need to write code
3. "test_code" if you have code that needs testing
4. "finalize" if you have all the information needed to provide a final response""")
    
    # Format the current state for the LLM
    state_info = f"""
Query: {state['query']}
Code Snippets Generated: {len(state['code_snippets'])}
Execution Results: {len(state['execution_results'])}
Documentation Searched: {len(state['documentation'])}
"""
    
    human_message = HumanMessage(content=f"Current state: {state_info}\nWhat should we do next?")
    
    result = llm.invoke([system_message, human_message])
    
    # Simple parsing of the result
    result_lower = result.content.lower()
    if "search_docs" in result_lower:
        return "search_docs"
    elif "generate_code" in result_lower:
        return "generate_code"
    elif "test_code" in result_lower:
        return "test_code"
    else:
        return "finalize"

def search_docs(state: CodeAssistantState) -> CodeAssistantState:
    """Search documentation for relevant information."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    system_message = SystemMessage(content="""You are a code assistant that needs to search for documentation.
Based on the user's query, decide what to search for. Be specific in your search terms.""")
    
    # Create a message with the current context
    context = f"User's query: {state['query']}"
    if state["code_snippets"]:
        context += f"\n\nCode generated so far:\n```python\n{state['code_snippets'][-1]}\n```"
    
    human_message = HumanMessage(content=f"{context}\n\nWhat documentation should we search for?")
    
    result = llm.invoke([system_message, human_message])
    
    # Extract the search query
    search_query = result.content.strip()
    
    # Execute the search tool
    tool_invocation = ToolInvocation(
        name="search_documentation",
        arguments=json.dumps({"query": search_query})
    )
    
    search_result = tool_executor.invoke(tool_invocation)
    
    # Add to documentation list
    state["documentation"].append(search_result)
    
    # Add to messages
    state["messages"].append({
        "role": "assistant", 
        "content": f"I searched for documentation on: {search_query}\nResults: {search_result}"
    })
    
    return state

def generate_code(state: CodeAssistantState) -> CodeAssistantState:
    """Generate code based on the user's query and collected information."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    system_message = SystemMessage(content="""You are a code assistant that generates Python code.
Based on the user's query and any documentation or context collected, write clean, efficient, and well-commented code.
Output only the code without additional explanation.""")
    
    # Compile context from the state
    context = f"User's query: {state['query']}"
    
    if state["documentation"]:
        context += "\n\nRelevant documentation:"
        for i, doc in enumerate(state["documentation"]):
            context += f"\n{i+1}. {doc}"
    
    if state["code_snippets"]:
        context += "\n\nPrevious code attempt:"
        context += f"\n```python\n{state['code_snippets'][-1]}\n```"
        
    if state["execution_results"]:
        context += "\n\nExecution results:"
        context += f"\n{state['execution_results'][-1]}"
    
    human_message = HumanMessage(content=f"{context}\n\nGenerate Python code for this task:")
    
    result = llm.invoke([system_message, human_message])
    
    # Extract code from the response
    code_match = re.search(r'```python\n(.*?)\n```', result.content, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # Try to extract code without markdown
        code = result.content
    
    # Add to code snippets
    state["code_snippets"].append(code)
    
    # Add to messages
    state["messages"].append({
        "role": "assistant", 
        "content": f"Here's the code I've generated:\n```python\n{code}\n```"
    })
    
    return state

def test_code(state: CodeAssistantState) -> CodeAssistantState:
    """Test the generated code."""
    if not state["code_snippets"]:
        # No code to test
        state["messages"].append({
            "role": "assistant", 
            "content": "No code available to test."
        })
        return state
    
    # Get the latest code snippet
    latest_code = state["code_snippets"][-1]
    
    # Execute the code
    tool_invocation = ToolInvocation(
        name="execute_python_code",
        arguments=json.dumps({"code": latest_code})
    )
    
    execution_result = tool_executor.invoke(tool_invocation)
    
    # Add to execution results
    state["execution_results"].append(execution_result)
    
    # Add to messages
    state["messages"].append({
        "role": "assistant", 
        "content": f"I tested the code. Result:\n{execution_result}"
    })
    
    return state

def finalize_response(state: CodeAssistantState) -> CodeAssistantState:
    """Finalize the response to the user."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    system_message = SystemMessage(content="""You are a code assistant providing a final response to the user.
Synthesize all the information collected and provide a comprehensive, helpful response.
Include the final code, explain how it works, and address any issues encountered during testing.
Make your response clear, concise, and educational.""")
    
    # Compile context from the state
    context = f"User's query: {state['query']}"
    
    if state["documentation"]:
        context += "\n\nDocumentation consulted:"
        for i, doc in enumerate(state["documentation"]):
            context += f"\n{i+1}. {doc}"
    
    if state["code_snippets"]:
        context += "\n\nFinal code:"
        context += f"\n```python\n{state['code_snippets'][-1]}\n```"
        
    if state["execution_results"]:
        context += "\n\nExecution results:"
        context += f"\n{state['execution_results'][-1]}"
    
    human_message = HumanMessage(content=f"{context}\n\nProvide a final comprehensive response:")
    
    result = llm.invoke([system_message, human_message])
    
    # Set final response
    state["final_response"] = result.content
    
    # Add to messages
    state["messages"].append({
        "role": "assistant", 
        "content": result.content
    })
    
    return state

# Build the graph
def build_code_assistant() -> StateGraph:
    """Build a code assistant agent using LangGraph."""
    # Define the graph
    graph = StateGraph(CodeAssistantState)
    
    # Add nodes
    graph.add_node("understand_query", understand_query)
    graph.add_node("search_docs", search_docs)
    graph.add_node("generate_code", generate_code)
    graph.add_node("test_code", test_code)
    graph.add_node("finalize_response", finalize_response)
    
    # Add edges
    graph.add_edge("understand_query", decide_next_action)
    graph.add_conditional_edges(
        decide_next_action,
        {
            "search_docs": "search_docs",
            "generate_code": "generate_code",
            "test_code": "test_code",
            "finalize": "finalize_response"
        }
    )
    
    # Connect all action nodes back to the decision node
    graph.add_edge("search_docs", decide_next_action)
    graph.add_edge("generate_code", decide_next_action)
    graph.add_edge("test_code", decide_next_action)
    
    # End the graph after finalizing
    graph.add_edge("finalize_response", END)
    
    # Set the entry point
    graph.set_entry_point("understand_query")
    
    # Compile the graph
    return graph.compile()

# Example usage
def ask_code_assistant(query: str) -> str:
    """Ask the code assistant a question."""
    # Build the assistant
    assistant = build_code_assistant()
    
    # Run the assistant
    result = assistant.invoke({"query": query})
    
    # Return the final response
    return result["final_response"]