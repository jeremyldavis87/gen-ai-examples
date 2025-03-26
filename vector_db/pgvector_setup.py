# vector_db/pgvector_setup.py
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

def setup_pgvector():
    """Set up a PostgreSQL database with the pgvector extension."""
    
    # Connection parameters from environment variables
    host = os.getenv("PGVECTOR_HOST")
    port = os.getenv("PGVECTOR_PORT")
    user = os.getenv("PGVECTOR_USER")
    password = os.getenv("PGVECTOR_PASSWORD")
    database = os.getenv("PGVECTOR_DATABASE")
    
    # First connect to PostgreSQL server
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create database if it doesn't exist
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{database}'")
    exists = cursor.fetchone()
    if not exists:
        print(f"Creating database {database}...")
        cursor.execute(f"CREATE DATABASE {database}")
    
    cursor.close()
    conn.close()
    
    # Connect to the target database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create pgvector extension if it doesn't exist
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create example vector table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_embeddings (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding VECTOR(1536) NOT NULL,
        metadata JSONB
    )
    """)
    
    # Create index for vector similarity search
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS document_embeddings_vector_idx 
    ON document_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100)
    """)
    
    print("PGVector setup complete!")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    setup_pgvector()