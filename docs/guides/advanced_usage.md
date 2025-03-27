# Advanced Usage Guide

This guide covers advanced usage scenarios for the Gen AI Examples framework with AWS managed services.

## Custom Model Configurations

You can customize model configurations for different task types:

```python
from models import ModelHandler

# Initialize the handler
model_handler = ModelHandler()

# Set custom default configuration for a task type
model_handler.set_default_config(
    task_type="customer_support",
    config={
        "primary": {
            "model": "gpt4o", 
            "model_family": "openai", 
            "temperature": 0.3, 
            "max_tokens": 1000
        },
        "fallbacks": [
            {
                "model": "sonnet-3.7", 
                "model_family": "anthropic", 
                "temperature": 0.3, 
                "max_tokens": 1000
            },
            {
                "model": "llama3-70b", 
                "model_family": "llama", 
                "temperature": 0.3, 
                "max_tokens": 1000
            }
        ]
    }
)

# Use the custom configuration
response = model_handler.generate_text(
    messages=[{"role": "user", "content": "I need help with my order"}],
    task_type="customer_support"
)
```

## Building RAG Applications

### Document Processing Pipeline

Create a comprehensive RAG application with document processing:

```python
import os
from typing import List, Dict, Any
import boto3
from models import ModelHandler
from vector_db import PGVectorClient

class DocumentProcessor:
    def __init__(self):
        self.model_handler = ModelHandler()
        self.vector_client = PGVectorClient()
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
    
    def upload_to_s3(self, file_path: str) -> str:
        """Upload a file to S3 and return the S3 URI"""
        file_name = os.path.basename(file_path)
        s3_key = f"documents/{file_name}"
        
        self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def chunk_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into overlapping chunks"""
        chunks = []
        start = 0
        content_len = len(content)
        
        while start < content_len:
            end = min(start + chunk_size, content_len)
            if end < content_len and content[end] != ' ':
                # Find the next space to avoid cutting words
                while end < content_len and content[end] != ' ':
                    end += 1
            
            chunks.append(content[start:end])
            start = end - overlap if end - overlap > 0 else 0
            
            if start >= content_len:
                break
        
        return chunks
    
    def process_document(self, file_path: str) -> List[str]:
        """Process a document and store in vector database"""
        # Upload to S3
        s3_uri = self.upload_to_s3(file_path)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk the document
        chunks = self.chunk_document(content)
        
        # Generate embeddings
        embeddings = self.model_handler.generate_embeddings(texts=chunks)
        
        # Store in vector database
        doc_ids = self.vector_client.insert_embeddings(
            contents=chunks,
            embeddings=embeddings,
            metadata=[{
                "source": file_path,
                "s3_uri": s3_uri,
                "chunk_index": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]
        )
        
        return doc_ids
    
    def query_documents(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Query documents using RAG"""
        # Generate query embedding
        query_embedding = self.model_handler.generate_embeddings(texts=[query])[0]
        
        # Search for similar documents
        similar_docs = self.vector_client.search_similar(
            query_embedding=query_embedding,
            limit=top_k
        )
        
        # Prepare context for the model
        context = "\n\n".join([doc["content"] for doc in similar_docs])
        
        # Generate answer using RAG
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the answer is not in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self.model_handler.generate_text(
            messages=messages,
            task_type="rag_qa"
        )
        
        return {
            "answer": response["response"],
            "sources": similar_docs
        }
```

## Integration with Amazon Cognito

### User Authentication

```python
import boto3
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt.algorithms import RSAAlgorithm
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Cognito configuration
USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
APP_CLIENT_ID = os.getenv("COGNITO_APP_CLIENT_ID")
REGION = os.getenv("COGNITO_REGION", "us-east-1")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Cache for JWKs
jwks_client = None
jwks = None

def get_jwks():
    global jwks_client, jwks
    if jwks is None:
        jwks_url = f"https://cognito-idp.{REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json"
        jwks_client = requests.get(jwks_url)
        jwks = jwks_client.json()["keys"]
    return jwks

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Get the key id from the token header
        header = jwt.get_unverified_header(token)
        kid = header["kid"]
        
        # Find the matching key in JWKs
        key = None
        for jwk in get_jwks():
            if jwk["kid"] == kid:
                key = jwk
                break
        
        if key is None:
            raise credentials_exception
        
        # Convert JWK to PEM format
        public_key = RSAAlgorithm.from_jwk(key)
        
        # Verify and decode the token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=APP_CLIENT_ID,
            options={"verify_exp": True}
        )
        
        # Extract user information
        username = payload.get("username") or payload.get("cognito:username")
        if username is None:
            raise credentials_exception
            
        return {"username": username, "payload": payload}
    
    except jwt.PyJWTError:
        raise credentials_exception

@app.get("/api/protected")
async def protected_route(user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {user['username']}!", "user_info": user["payload"]}

@app.get("/api/public")
async def public_route():
    return {"message": "This is a public endpoint"}
```

## Advanced Vector Database Usage

### Custom Similarity Search

```python
from vector_db import PGVectorClient
import json

class AdvancedVectorSearch:
    def __init__(self):
        self.vector_client = PGVectorClient()
    
    def metadata_filtered_search(self, query_embedding, metadata_filter, limit=5):
        """Search with metadata filtering"""
        filter_json = json.dumps(metadata_filter)
        
        query = """
        SELECT content, metadata, 1 - (embedding <=> %s) AS similarity 
        FROM documents 
        WHERE metadata @> %s::jsonb 
        ORDER BY similarity DESC 
        LIMIT %s
        """
        
        results = self.vector_client.execute_query(
            query, 
            (query_embedding, filter_json, limit),
            fetch=True
        )
        
        return [{
            "content": row[0],
            "metadata": row[1],
            "similarity": row[2]
        } for row in results]
    
    def hybrid_search(self, query_text, query_embedding, limit=5, text_weight=0.3):
        """Hybrid search combining vector similarity and text search"""
        query = """
        SELECT 
            content, 
            metadata, 
            ((%s * (1 - (embedding <=> %s))) + ((1 - %s) * ts_rank(to_tsvector('english', content), to_tsquery('english', %s)))) AS score 
        FROM documents 
        WHERE to_tsvector('english', content) @@ to_tsquery('english', %s)
        ORDER BY score DESC 
        LIMIT %s
        """
        
        # Convert query to tsquery format
        ts_query = " & ".join(query_text.split())
        
        results = self.vector_client.execute_query(
            query, 
            (text_weight, query_embedding, text_weight, ts_query, ts_query, limit),
            fetch=True
        )
        
        return [{
            "content": row[0],
            "metadata": row[1],
            "score": row[2]
        } for row in results]
```

## AWS S3 Integration for Document Storage

```python
import boto3
import os
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional

class S3DocumentStorage:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
    
    def upload_document(self, file_path: str, prefix: str = "documents/") -> str:
        """Upload a document to S3 and return its S3 URI"""
        file_name = os.path.basename(file_path)
        s3_key = f"{prefix}{file_name}"
        
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except ClientError as e:
            print(f"Error uploading file to S3: {e}")
            raise
    
    def download_document(self, s3_uri: str, local_path: Optional[str] = None) -> str:
        """Download a document from S3"""
        if not s3_uri.startswith("s3://"):
            raise ValueError("Invalid S3 URI format")
        
        # Parse S3 URI
        s3_path = s3_uri[5:]  # Remove "s3://"
        parts = s3_path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        if not local_path:
            local_path = os.path.basename(key)
        
        try:
            self.s3_client.download_file(bucket, key, local_path)
            return local_path
        except ClientError as e:
            print(f"Error downloading file from S3: {e}")
            raise
    
    def list_documents(self, prefix: str = "documents/") -> List[Dict[str, Any]]:
        """List documents in the S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            return [{
                "key": item["Key"],
                "size": item["Size"],
                "last_modified": item["LastModified"],
                "uri": f"s3://{self.bucket_name}/{item['Key']}"
            } for item in response["Contents"]]
        except ClientError as e:
            print(f"Error listing objects in S3: {e}")
            raise
    
    def delete_document(self, s3_uri: str) -> bool:
        """Delete a document from S3"""
        if not s3_uri.startswith("s3://"):
            raise ValueError("Invalid S3 URI format")
        
        # Parse S3 URI
        s3_path = s3_uri[5:]  # Remove "s3://"
        parts = s3_path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            print(f"Error deleting object from S3: {e}")
            return False
```

## Performance Optimization

### Caching with Redis

```python
import redis
import json
import hashlib
from typing import Any, Optional, Dict, List
from functools import wraps

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0, password=None, ttl=3600):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        self.default_ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache"""
        value = self.redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        serialized = json.dumps(value)
        return self.redis_client.setex(key, ttl, serialized)
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache"""
        return bool(self.redis_client.delete(key))
    
    def hash_key(self, *args, **kwargs) -> str:
        """Generate a hash key from arguments"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

# Cache decorator
def cached(ttl=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Skip cache if explicitly requested
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return func(self, *args, **kwargs)
            
            # Get cache instance from self
            if not hasattr(self, 'cache') or not isinstance(self.cache, RedisCache):
                return func(self, *args, **kwargs)
            
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__qualname__}:{self.cache.hash_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            self.cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

# Example usage
class EnhancedModelHandler:
    def __init__(self):
        from models import ModelHandler
        self.model_handler = ModelHandler()
        self.cache = RedisCache(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD')
        )
    
    @cached(ttl=3600)
    def generate_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings with caching"""
        return self.model_handler.generate_embeddings(texts, **kwargs)
    
    @cached(ttl=300)
    def generate_text(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text with caching"""
        return self.model_handler.generate_text(messages, **kwargs)
```

## CI/CD Pipeline with GitHub Actions

The repository includes GitHub Actions workflows for CI/CD. Here's how to customize them for your needs:

### Customizing the CI Workflow

Edit the `.github/workflows/ci.yml` file to adjust the testing parameters:

```yaml
# Change Python version
python-version: [3.12]  # Add more versions if needed

# Modify test database settings
PGVECTOR_USER: your_test_user
PGVECTOR_PASSWORD: your_test_password
PGVECTOR_DATABASE: your_test_db
```

### Customizing the CD Workflow

Edit the `.github/workflows/cd.yml` file to adjust the deployment parameters:

```yaml
# Change AWS region
aws-region: us-west-2

# Modify ECR repository name
ECR_REPOSITORY: your-ecr-repo-name

# Change ECS cluster and service names
aws ecs update-service --cluster your-cluster-name --service your-service-name --force-new-deployment
```

## Monitoring and Logging

### Setting Up CloudWatch Logging

```python
import boto3
import logging
from botocore.exceptions import ClientError
import os
from datetime import datetime

class CloudWatchLogger:
    def __init__(self, log_group_name=None, log_stream_prefix=None):
        self.log_group_name = log_group_name or os.getenv('LOG_GROUP_NAME', 'gen-ai-examples')
        self.log_stream_prefix = log_stream_prefix or os.getenv('LOG_STREAM_PREFIX', 'api-')
        self.log_stream_name = f"{self.log_stream_prefix}{datetime.now().strftime('%Y-%m-%d')}"
        
        self.logs_client = boto3.client('logs')
        self.setup_log_group()
        self.setup_log_stream()
    
    def setup_log_group(self):
        """Create log group if it doesn't exist"""
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group_name)
        except ClientError as e:
            # Ignore if log group already exists
            if 'ResourceAlreadyExistsException' not in str(e):
                raise
    
    def setup_log_stream(self):
        """Create log stream if it doesn't exist"""
        try:
            self.logs_client.create_log_stream(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name
            )
        except ClientError as e:
            # Ignore if log stream already exists
            if 'ResourceAlreadyExistsException' not in str(e):
                raise
    
    def log_event(self, message, level='INFO', **kwargs):
        """Log an event to CloudWatch Logs"""
        timestamp = int(datetime.now().timestamp() * 1000)
        
        event = {
            'timestamp': timestamp,
            'message': f"[{level}] {message}",
        }
        
        if kwargs:
            event_data = {'message': message, 'level': level, **kwargs}
            event['message'] = f"[{level}] {event_data}"
        
        try:
            self.logs_client.put_log_events(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name,
                logEvents=[event]
            )
        except ClientError as e:
            # Handle sequence token issues
            if 'InvalidSequenceTokenException' in str(e):
                # Get the correct sequence token
                response = self.logs_client.describe_log_streams(
                    logGroupName=self.log_group_name,
                    logStreamNamePrefix=self.log_stream_name
                )
                
                sequence_token = response['logStreams'][0].get('uploadSequenceToken')
                
                # Retry with the correct sequence token
                self.logs_client.put_log_events(
                    logGroupName=self.log_group_name,
                    logStreamName=self.log_stream_name,
                    logEvents=[event],
                    sequenceToken=sequence_token
                )
            else:
                raise

# Example usage
logger = CloudWatchLogger()
logger.log_event(
    "API request processed",
    level="INFO",
    user_id="user123",
    endpoint="/generate",
    latency_ms=150
)
```

This advanced guide covers many of the sophisticated features and integrations available in the Gen AI Examples framework. For specific use cases or further customization, refer to the code examples and documentation in the repository.
