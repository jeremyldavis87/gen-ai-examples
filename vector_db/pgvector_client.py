# vector_db/pgvector_client.py
import os
import json
import psycopg2
import psycopg2.extras
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

class PGVectorClient:
    def __init__(self):
        """Initialize the PGVector client."""
        self.conn = psycopg2.connect(
            host=os.getenv("PGVECTOR_HOST"),
            port=os.getenv("PGVECTOR_PORT"),
            user=os.getenv("PGVECTOR_USER"),
            password=os.getenv("PGVECTOR_PASSWORD"),
            database=os.getenv("PGVECTOR_DATABASE")
        )
    
    def insert_embeddings(self, 
                         contents: List[str], 
                         embeddings: List[List[float]], 
                         metadata: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Insert documents and their embeddings into the database.
        
        Args:
            contents: List of document contents
            embeddings: List of embedding vectors
            metadata: Optional list of metadata for each document
            
        Returns:
            List of inserted document IDs
        """
        cursor = self.conn.cursor()
        
        if metadata is None:
            metadata = [{}] * len(contents)
        
        inserted_ids = []
        for content, embedding, meta in zip(contents, embeddings, metadata):
            cursor.execute(
                "INSERT INTO document_embeddings (content, embedding, metadata) VALUES (%s, %s, %s) RETURNING id",
                (content, embedding, json.dumps(meta))
            )
            inserted_ids.append(cursor.fetchone()[0])
        
        self.conn.commit()
        cursor.close()
        return inserted_ids
    
    def search_similar(self, 
                      query_embedding: List[float], 
                      limit: int = 5, 
                      similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.
        
        Args:
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar documents with their metadata and similarity scores
        """
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute(
            """
            SELECT id, content, metadata, 
                   1 - (embedding <=> %s) as similarity
            FROM document_embeddings
            WHERE 1 - (embedding <=> %s) > %s
            ORDER BY similarity DESC
            LIMIT %s
            """,
            (query_embedding, query_embedding, similarity_threshold, limit)
        )
        
        results = cursor.fetchall()
        cursor.close()
        
        # Convert results to list of dicts
        return [dict(result) for result in results]
    
    def close(self):
        """Close the database connection."""
        self.conn.close()