# agents/task_agent.py
from typing import Dict, List, Any, Optional, Tuple, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import json
import re
from enum import Enum

from langchain_utils.gateway_integration import get_langchain_llm
from vector_db.pgvector_client import PGVectorClient

# Define tool functions
def search_web(query: str) -> str:
    """Search the web for information."""
    # In production, replace with actual web search API
    return f"Web search results for: {query}"

def run_code(code: str, language: str = "python") -> str:
    """Run code and return the result."""
    # In production, use a secure code execution environment
    # This is a simplified example
    if language.lower() == "python":
        try:
            # WARNING: Never use exec in production without proper sandboxing
            # This is for demonstration only
            result = {}
            exec(code, {}, result)
            return f"Code executed successfully. Result: {result}"
        except Exception as e:
            return f"Error executing code: {str(e)}"
    else:
        return f"Language {language} not supported for execution."

def query_database(query_text: str) -> str:
    """Query vector database for information."""
    # Initialize vector database client
    pgvector_client = PGVectorClient()
    
    # Get embeddings for the query
    llm = get_langchain_llm()
    
    # For this example, we'll just return a placeholder
    # In a real implementation, you would:
    # 1. Generate embeddings for the query
    # 2. Search the vector database
    # 3. Format and return the results
    return f"Database results for: {query_text}"

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Run code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to run"
                    },
                    "language": {
                        "type": "string",
                        "description": "The programming language",
                        "default": "python"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query vector database for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The text to search for in the database"
                    }
                },
                "required": ["query_text"]
            }
        }
    }
]

# Create tool executor
tool_executor = ToolExecutor({
    "search_web": search_web,
    "run_code": run_code,
    "query_database": query_database
})

# Define agent state
class AgentState(dict):
    """State for the agent."""
    task: str
    plan: Optional[List[str]]
    current_step: Optional[int]
    working_memory: Dict[str, Any]
    tools_results: Dict[str, Any]
    messages: List[Dict[str, str]]
    final_answer: Optional[str]
    
    def __init__(self, task: str):
        self.task = task
        self.plan = None
        self.current_step = None
        self.working_memory = {}
        self.tools_results = {}
        self.messages = []
        self.final_answer = None
        dict.__init__(
            self, 
            task=task,
            plan=None,
            current_step=None,
            working_memory={},
            tools_results={},
            messages=[],
            final_answer=None
        )

# Define node functions
def create_plan(state: AgentState) -> AgentState:
    """Create a plan to solve the task."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    system_message = SystemMessage(content="""You are an AI agent that creates a plan to solve tasks.
Break down the task into 3-5 clear steps. Each step should be achievable using the available tools.
Output the plan as a numbered list. Be specific about what each step needs to accomplish.""")
    
    human_message = HumanMessage(content=f"Create a plan to solve the following task: {state['task']}")
    
    result = llm.invoke([system_message, human_message])
    
    # Extract plan steps using regex
    pattern = r"\d+\.\s+(.*)"
    matches = re.findall(pattern, result.content)
    
    plan = matches if matches else [result.content]
    
    # Add to messages
    state["messages"].append({"role": "system", "content": system_message.content})
    state["messages"].append({"role": "user", "content": human_message.content})
    state["messages"].append({"role": "assistant", "content": result.content})
    
    return {**state, "plan": plan, "current_step": 0}

def decide_next_action(state: AgentState) -> str:
    """Decide the next action to take."""
    if state["current_step"] >= len(state["plan"]):
        return "finish"
    else:
        return "execute_step"

def execute_step(state: AgentState) -> AgentState:
    """Execute the current step of the plan."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    # Get current step
    current_step_idx = state["current_step"]
    current_step = state["plan"][current_step_idx]
    
    # Format working memory and tool results for context
    working_memory_str = json.dumps(state["working_memory"], indent=2)
    tools_results_str = json.dumps(state["tools_results"], indent=2)
    
    system_message = SystemMessage(content=f"""You are an AI agent that executes steps in a plan.
Your task is: {state['task']}
Your current step is: {current_step}

You have access to the following tools:
- search_web: Search the web for information
- run_code: Run code and return the result
- query_database: Query vector database for information

Your working memory: {working_memory_str}
Your previous tool results: {tools_results_str}

Decide which tool to use to execute this step. You must use one of the available tools.
Format your response as JSON with the following structure:
{{
  "tool": "tool_name",
  "parameters": {{
    "param1": "value1",
    ...
  }},
  "reasoning": "Your reasoning for using this tool"
}}""")
    
    human_message = HumanMessage(content=f"Execute step {current_step_idx + 1}: {current_step}")
    
    result = llm.invoke([system_message, human_message])
    
    # Parse the JSON response
    try:
        # Extract JSON from the response
        json_match = re.search(r'```json\n(.*?)\n```', result.content, re.DOTALL)
        if json_match:
            action_json = json.loads(json_match.group(1))
        else:
            # Try to find json without code blocks
            json_text = result.content
            action_json = json.loads(json_text)
    except Exception as e:
        # If parsing fails, return a default action
        action_json = {
            "tool": "search_web",
            "parameters": {"query": current_step},
            "reasoning": "Fallback due to parsing error"
        }
    
    # Execute the tool
    tool_name = action_json.get("tool")
    parameters = action_json.get("parameters", {})
    reasoning = action_json.get("reasoning", "")
    
    # Create tool invocation
    tool_invocation = ToolInvocation(
        name=tool_name,
        arguments=json.dumps(parameters)
    )
    
    # Execute the tool
    tool_result = tool_executor.invoke(tool_invocation)
    
    # Update working memory with the result
    state["tools_results"][f"step_{current_step_idx}_{tool_name}"] = tool_result
    
    # Add to messages
    state["messages"].append({"role": "system", "content": system_message.content})
    state["messages"].append({"role": "user", "content": human_message.content})
    state["messages"].append({"role": "assistant", "content": result.content})
    state["messages"].append({"role": "system", "content": f"Tool result: {tool_result}"})
    
    # Move to the next step
    return {**state, "current_step": current_step_idx + 1}

def finish_task(state: AgentState) -> AgentState:
    """Finish the task and provide a final answer."""
    llm = get_langchain_llm(model_name="gpt4o")
    
    system_message = SystemMessage(content=f"""You are an AI agent that has completed a task.
Your task was: {state['task']}

Your plan was:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(state['plan'])])}

Tool results:
{json.dumps(state['tools_results'], indent=2)}

Provide a comprehensive final answer that addresses the original task.""")
    
    human_message = HumanMessage(content="Provide your final answer.")
    
    result = llm.invoke([system_message, human_message])
    
    # Add to messages
    state["messages"].append({"role": "system", "content": system_message.content})
    state["messages"].append({"role": "user", "content": human_message.content})
    state["messages"].append({"role": "assistant", "content": result.content})
    
    return {**state, "final_answer": result.content}

# Build the graph
def build_task_agent() -> StateGraph:
    """Build a task planning agent using LangGraph."""
    # Define the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("create_plan", create_plan)
    graph.add_node("execute_step", execute_step)
    graph.add_node("finish_task", finish_task)
    
    # Add edges
    graph.add_edge("create_plan", "execute_step")
    graph.add_conditional_edges(
        "execute_step",
        decide_next_action,
        {
            "execute_step": "execute_step",
            "finish": "finish_task"
        }
    )
    graph.add_edge("finish_task", END)
    
    # Set the entry point
    graph.set_entry_point("create_plan")
    
    # Compile the graph
    return graph.compile()

# Example usage
def solve_task(task: str) -> Dict[str, Any]:
    """Solve a task using the task planning agent."""
    # Build the agent
    agent = build_task_agent()
    
    # Run the agent
    result = agent.invoke({"task": task})
    
    return result