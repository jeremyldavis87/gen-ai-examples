from sqlalchemy import Column, Integer, String, Text, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create SQLAlchemy base class
Base = declarative_base()

# Define document model with pgvector extension
class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    # The embedding column is created with a special type in the migration
    # since SQLAlchemy doesn't natively support the vector type
    metadata = Column(JSON, nullable=True)
    source = Column(String(255), nullable=True)
    chunk_id = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, source={self.source})>"

# Database connection function
def get_db_engine():
    """Create a SQLAlchemy engine for PostgreSQL with pgvector"""
    host = os.getenv("PGVECTOR_HOST", "localhost")
    port = os.getenv("PGVECTOR_PORT", "5432")
    user = os.getenv("PGVECTOR_USER", "postgres")
    password = os.getenv("PGVECTOR_PASSWORD", "postgres")
    database = os.getenv("PGVECTOR_DATABASE", "vector_db")
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(connection_string)

# Session factory
def get_db_session():
    """Create a SQLAlchemy session"""
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()
