# apps/chat_app.py
import os
import time
import json
from typing import Dict, List, Any, Generator
from flask import Flask, request, jsonify, Response, stream_with_context
import threading
import queue

from ai_gateway.client import AIGatewayClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize the AI Gateway client
gateway_client = AIGatewayClient()

# Store conversation history
conversations = {}

def stream_response(model: str, messages: List[Dict[str, str]], model_family: str = "openai") -> Generator[str, None, None]:
    """
    Stream a response from the model.
    
    Args:
        model: The model to use
        messages: The conversation history
        model_family: The model family
        
    Yields:
        Chunks of the response
    """
    # This is a simplified implementation since the actual implementation
    # would depend on how your AI Gateway handles streaming
    
    # For demonstration purposes, we'll simulate streaming by breaking up the response
    response = gateway_client.generate_text(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        model_family=model_family
    )
    
    # Extract the response text
    if model_family == "openai":
        full_response = response["choices"][0]["message"]["content"]
    elif model_family == "anthropic":
        full_response = response["content"][0]["text"]
    else:
        full_response = str(response)
    
    # Split the response into words for simulated streaming
    words = full_response.split()
    
    # Yield words with a slight delay to simulate streaming
    for i in range(0, len(words), 3):
        chunk = " ".join(words[i:i+3])
        yield json.dumps({"chunk": chunk}) + "\n"
        time.sleep(0.05)  # Simulate network delay

@app.route("/api/chat", methods=["POST"])
def chat():
    """API endpoint for chat."""
    data = request.json
    
    # Extract parameters
    conversation_id = data.get("conversation_id", str(time.time()))
    message = data.get("message", "")
    model = data.get("model", "gpt4o")
    model_family = data.get("model_family", "openai")
    stream = data.get("stream", False)
    
    # Initialize conversation if it doesn't exist
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to history
    conversations[conversation_id].append({"role": "user", "content": message})
    
    if stream:
        # Stream the response
        def generate():
            for chunk in stream_response(model, conversations[conversation_id], model_family):
                yield chunk
                
            # Once done, capture the full response and add it to history
            response = gateway_client.generate_text(
                model=model,
                messages=conversations[conversation_id],
                temperature=0.7,
                max_tokens=1000,
                model_family=model_family
            )
            
            if model_family == "openai":
                assistant_message = response["choices"][0]["message"]["content"]
            elif model_family == "anthropic":
                assistant_message = response["content"][0]["text"]
            else:
                assistant_message = str(response)
                
            conversations[conversation_id].append({"role": "assistant", "content": assistant_message})
            
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        # Get a complete response
        response = gateway_client.generate_text(
            model=model,
            messages=conversations[conversation_id],
            temperature=0.7,
            max_tokens=1000,
            model_family=model_family
        )
        
        # Extract and add assistant message to history
        if model_family == "openai":
            assistant_message = response["choices"][0]["message"]["content"]
        elif model_family == "anthropic":
            assistant_message = response["content"][0]["text"]
        else:
            assistant_message = str(response)
            
        conversations[conversation_id].append({"role": "assistant", "content": assistant_message})
        
        return jsonify({
            "conversation_id": conversation_id,
            "message": assistant_message
        })

@app.route("/api/history/<conversation_id>", methods=["GET"])
def get_history(conversation_id):
    """Get conversation history."""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify({
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    })

@app.route("/api/clear/<conversation_id>", methods=["POST"])
def clear_history(conversation_id):
    """Clear conversation history."""
    if conversation_id in conversations:
        conversations.pop(conversation_id)
    
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)