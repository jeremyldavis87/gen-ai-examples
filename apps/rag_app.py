# apps/rag_app.py
import os
import argparse
from typing import Dict, List, Any, Optional

from ai_gateway.client import AIGatewayClient
from vector_db.pgvector_client import PGVectorClient
from agents.rag_agent import query_rag_agent

from dotenv import load_dotenv
load_dotenv()

def ingest_documents(file_paths: List[str], chunk_size: int = 1000) -> List[int]:
    """
    Ingest documents into the vector database.
    
    Args:
        file_paths: List of file paths to ingest
        chunk_size: Size of text chunks for embedding
        
    Returns:
        List of ingested document IDs
    """
    # Initialize clients
    gateway_client = AIGatewayClient()
    pgvector_client = PGVectorClient()
    
    all_chunks = []
    all_metadata = []
    
    # Process each file
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple chunking by splitting text (you can use a more sophisticated chunker)
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        # Create metadata for each chunk
        metadata = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
        
        all_chunks.extend(chunks)
        all_metadata.extend(metadata)
    
    # Generate embeddings
    embeddings = gateway_client.generate_embeddings(texts=all_chunks)
    
    # Insert into vector database
    doc_ids = pgvector_client.insert_embeddings(
        contents=all_chunks,
        embeddings=embeddings,
        metadata=all_metadata
    )
    
    pgvector_client.close()
    
    return doc_ids


def query_documents(question: str) -> str:
    """
    Query the RAG system with a question.
    
    Args:
        question: The question to ask
        
    Returns:
        The generated answer
    """
    return query_rag_agent(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Application")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--files", nargs="+", required=True, help="Files to ingest")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--question", required=True, help="Question to ask")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        doc_ids = ingest_documents(args.files, args.chunk_size)
        print(f"Ingested {len(doc_ids)} document chunks.")
    elif args.command == "query":
        answer = query_documents(args.question)
        print(f"Q: {args.question}")
        print(f"A: {answer}")
    else:
        parser.print_help()