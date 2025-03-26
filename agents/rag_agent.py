# agents/rag_agent.py
from typing import Dict, List, Tuple, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import json

from langchain_utils.gateway_integration import get_langchain_llm, get_langchain_embeddings
from vector_db.pgvector_client import PGVectorClient

# Initialize clients
pgvector_client = PGVectorClient()
llm = get_langchain_llm(model_name="gpt4o")
embeddings = get_langchain_embeddings()

# Define tools
def search_documents(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for relevant documents based on the query.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of relevant documents
    """
    query_embedding = embeddings.embed_query(query)
    results = pgvector_client.search_similar(query_embedding, limit=limit)
    return results

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search for relevant documents based on the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Create tool executor
tool_executor = ToolExecutor({"search_documents": search_documents})

# Define state type
class AgentState(dict):
    """State for the agent."""
    question: str
    context: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    
    def __init__(self, question: str):
        self.question = question
        self.context = None
        self.answer = None
        dict.__init__(self, question=question, context=None, answer=None)


def should_search(state: AgentState) -> str:
    """Determine if the agent should search for context."""
    if state["context"] is None:
        return "search"
    else:
        return "answer"


def search(state: AgentState) -> AgentState:
    """Search for relevant documents."""
    # Create a tool invocation to search for documents
    search_tool = ToolInvocation(
        name="search_documents",
        arguments=json.dumps({"query": state["question"]})
    )
    
    # Execute the tool
    results = tool_executor.invoke(search_tool)
    
    # Update state with search results
    return {**state, "context": results}


def generate_answer(state: AgentState) -> AgentState:
    """Generate an answer based on the context and question."""
    # Create system message with context
    context_str = ""
    for doc in state["context"]:
        context_str += f"Document ID: {doc['id']}\nContent: {doc['content']}\n\n"
    
    # RAG prompt
    system_template = """You are a helpful AI assistant. Answer the user's question based on the following context. 
If the context doesn't contain relevant information, just say that you don't know.

Context:
{context}"""
    
    messages = [
        SystemMessage(content=system_template.format(context=context_str)),
        HumanMessage(content=state["question"])
    ]
    
    # Generate response
    answer = llm.invoke(messages).content
    
    # Update state with the answer
    return {**state, "answer": answer}


# Build the graph
def build_rag_agent() -> StateGraph:
    """Build a RAG agent using LangGraph."""
    # Define the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("search", search)
    graph.add_node("answer", generate_answer)
    
    # Add edges
    graph.add_conditional_edges("", should_search, {
        "search": "search",
        "answer": "answer"
    })
    
    graph.add_edge("search", "answer")
    graph.add_edge("answer", END)
    
    # Compile the graph
    return graph.compile()


# Example usage
def query_rag_agent(question: str) -> str:
    """Query the RAG agent with a question."""
    # Build the agent
    agent = build_rag_agent()
    
    # Run the agent
    result = agent.invoke({"question": question})
    
    # Return the answer
    return result["answer"]