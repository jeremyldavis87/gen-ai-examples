# Getting Started with Gen AI Examples

This guide will help you set up and start using the Gen AI Examples framework with AWS managed services.

## Prerequisites

- Python 3.12
- Docker and Docker Compose (for local development)
- AWS account with appropriate permissions
- Access to AI Gateway

## Installation

### Option 1: Using Docker (Recommended)

The easiest way to get started is using Docker:

```bash
# Clone the repository
git clone https://github.com/your-org/gen-ai-examples.git
cd gen-ai-examples

# Copy and edit the environment file
cp .env.sample .env
# Edit .env with your credentials

# Start the services using Docker Compose
docker-compose -f docker/docker-compose.yml up -d
```

### Option 2: Manual Installation

If you prefer to install directly on your system:

```bash
# Clone the repository
git clone https://github.com/your-org/gen-ai-examples.git
cd gen-ai-examples

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit the environment file
cp .env.sample .env
# Edit .env with your credentials

# Run the setup script
python setup.py
```

## Configuration

The framework uses environment variables for configuration. Edit the `.env` file with your specific settings:

```
# AI Gateway Configuration
AI_GATEWAY_URL=https://your-ai-gateway-url.com/api/v1/ai
PROJECT_NAME=your_project_name
API_KEY=your_api_key

# Vector Database Configuration
PGVECTOR_HOST=your-pg-instance.rds.amazonaws.com
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password
PGVECTOR_DATABASE=vector_db

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key

# S3 Configuration
S3_BUCKET_NAME=your-s3-bucket-name

# Cognito Configuration
COGNITO_USER_POOL_ID=your_user_pool_id
COGNITO_APP_CLIENT_ID=your_app_client_id
COGNITO_REGION=us-east-1
```

## Setting Up AWS Resources

The framework is designed to work with AWS managed services. You can set up the required resources using Terraform:

```bash
# Navigate to the Terraform directory
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Create a terraform.tfvars file with your variables
cat > terraform.tfvars << EOL
project_name = "gen-ai-examples"
environment = "dev"
cognito_callback_urls = ["http://localhost:3000/callback"]
cognito_logout_urls = ["http://localhost:3000"]
database_subnet_ids = ["subnet-12345", "subnet-67890"]
database_access_cidr_blocks = ["10.0.0.0/16"]
vpc_id = "vpc-12345"
repository_url = "https://github.com/your-org/gen-ai-examples"
api_url = "https://api.example.com"
EOL

# Plan the deployment
terraform plan -var-file=terraform.tfvars

# Apply the changes
terraform apply -var-file=terraform.tfvars
```

Alternatively, you can use Terragrunt for more advanced infrastructure management:

```bash
# Navigate to the Terragrunt directory
cd infrastructure/terragrunt

# Plan all resources
terragrunt run-all plan

# Apply all resources
terragrunt run-all apply
```

## Basic Usage

### Using the Model Handler

The ModelHandler provides a unified interface for interacting with AI models:

```python
from models import ModelHandler

# Initialize the handler
model_handler = ModelHandler()

# Generate text with automatic fallback
response = model_handler.generate_text(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    task_type="text_generation"  # Optional task type
)

print(response["response"])
```

### Using the Vector Database

```python
from vector_db import PGVectorClient
from models import ModelHandler

# Initialize clients
model_handler = ModelHandler()
pgvector_client = PGVectorClient()

# Generate embeddings for documents
texts = ["Document 1 content", "Document 2 content"]
embeddings = model_handler.generate_embeddings(texts=texts)

# Store in vector database
doc_ids = pgvector_client.insert_embeddings(
    contents=texts,
    embeddings=embeddings,
    metadata=[{"source": "doc1.txt"}, {"source": "doc2.txt"}]
)

# Search for similar documents
query_embedding = model_handler.generate_embeddings(texts=["Search query"])[0]
results = pgvector_client.search_similar(
    query_embedding=query_embedding,
    limit=5
)
```

### Running the API Service

```bash
# Start the FastAPI service
python -m apps.api_service
```

The API will be available at http://localhost:8000 with Swagger documentation at http://localhost:8000/docs.

## Next Steps

- Check out the [Advanced Usage Guide](./advanced_usage.md) for more complex scenarios
- Explore the [Example Applications](./example_applications.md) for ready-to-use implementations
- Learn about [AWS Integration](./aws_integration.md) for production deployments
