AWS RDS PGVector Setup Guide
Let's provide instructions for setting up PGVector on AWS RDS:
markdownCopy# Setting up PGVector on AWS RDS

## Prerequisites
- AWS Account with RDS access
- AWS CLI configured
- psql client installed locally

## Steps

### 1. Create a Parameter Group for pgvector
```bash
aws rds create-db-parameter-group \
    --db-parameter-group-name pgvector-pg-13 \
    --db-parameter-group-family postgres13 \
    --description "Parameter group for pgvector extension"
```

### 2. Create the RDS Instance
```bash
aws rds create-db-instance \
    --db-instance-identifier pgvector-instance \
    --db-parameter-group-name pgvector-pg-13 \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --engine-version 13.4 \
    --allocated-storage 20 \
    --master-username postgres \
    --master-user-password your_strong_password \
    --port 5432 \
    --no-publicly-accessible \
    --vpc-security-group-ids sg-your_security_group_id
```
### 3. Install the pgvector Extension
Connect to your RDS instance:
```bash
psql -h your-pgvector-instance.region.rds.amazonaws.com -U postgres -p 5432
Create a database:
sqlCopyCREATE DATABASE vector_db;
\c vector_db

-- Install the pgvector extension
CREATE EXTENSION vector;

-- Verify installation
\dx
```
### 4. Update Your .env File
```bash
PGVECTOR_HOST=your-pgvector-instance.region.rds.amazonaws.com
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_strong_password
PGVECTOR_DATABASE=vector_db
```
### 5. Run the Setup Script
```bash
python vector_db/pgvector_setup.py
```
## 6. LangChain Integration

Create a utility for using LangChain with your AI Gateway:

```python
# langchain_utils/gateway_integration.py
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from typing import Dict, List, Optional, Any, Union
import os
from dotenv import load_dotenv
from ai_gateway.client import AIGatewayClient

load_dotenv()

class GatewayChatModel(BaseLanguageModel):
    """Custom LangChain chat model that uses the AI Gateway."""
    
    def __init__(self, 
                model: str = "gpt4o", 
                model_family: str = "openai",
                temperature: float = 0.7,
                max_tokens: int = 1000,
                gateway_client: Optional[AIGatewayClient] = None):
        """
        Initialize the Gateway chat model.
        
        Args:
            model: The model to use
            model_family: The model family (openai, anthropic)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            gateway_client: Optional existing Gateway client
        """
        self.model = model
        self.model_family = model_family
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.gateway_client = gateway_client or AIGatewayClient()
    
    def _generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text from the model."""
        response = self.gateway_client.generate_text(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_family=self.model_family
        )
        
        # Extract the actual response based on model family
        if self.model_family == "openai":
            return response["choices"][0]["message"]["content"]
        elif self.model_family == "anthropic":
            return response["content"][0]["text"]
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")


class GatewayEmbeddings(Embeddings):
    """Custom LangChain embeddings that use the AI Gateway."""
    
    def __init__(self, 
                model: str = "text-embedding-3-large", 
                gateway_client: Optional[AIGatewayClient] = None):
        """
        Initialize the Gateway embeddings.
        
        Args:
            model: The embedding model to use
            gateway_client: Optional existing Gateway client
        """
        self.model = model
        self.gateway_client = gateway_client or AIGatewayClient()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        return self.gateway_client.generate_embeddings(texts=texts, model=self.model)
    
    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query."""
        embeddings = self.gateway_client.generate_embeddings(texts=[text], model=self.model)
        return embeddings[0]


def get_langchain_llm(model_name: str = "gpt4o", model_family: str = "openai") -> Union[BaseLanguageModel, ChatOpenAI]:
    """
    Get a LangChain LLM based on the model name.
    
    Args:
        model_name: The name of the model to use
        model_family: The model family
        
    Returns:
        A LangChain LLM instance
    """
    if model_family == "openai":
        # For OpenAI-compatible models, we can use ChatOpenAI with custom base URL
        return ChatOpenAI(
            openai_api_base=f"{os.getenv('AI_GATEWAY_URL')}/{os.getenv('PROJECT_NAME')}/openai",
            openai_api_key=os.getenv("API_KEY"),
            model_name=model_name
        )
    else:
        # For other models, use our custom Gateway integration
        return GatewayChatModel(model=model_name, model_family=model_family)


def get_langchain_embeddings(model_name: str = "text-embedding-3-large") -> Union[Embeddings, OpenAIEmbeddings]:
    """
    Get LangChain embeddings based on the model name.
    
    Args:
        model_name: The name of the embedding model to use
        
    Returns:
        A LangChain Embeddings instance
    """
    # For OpenAI-compatible embeddings
    return OpenAIEmbeddings(
        openai_api_base=f"{os.getenv('AI_GATEWAY_URL')}/{os.getenv('PROJECT_NAME')}/openai",
        openai_api_key=os.getenv("API_KEY"),
        model=model_name
    )
```