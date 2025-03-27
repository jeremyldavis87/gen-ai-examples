# Docker Configuration

This directory contains Docker configuration files for containerizing the gen-ai-examples project, using Python 3.12 and PostgreSQL with pgvector.

## Table of Contents

1. [Overview](#overview)
2. [Files](#files)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Best Practices](#best-practices)

## Overview

The Docker configuration in this directory enables containerized development and deployment of the gen-ai-examples project. It provides a consistent environment across development, testing, and production, ensuring that the application behaves the same way regardless of where it runs.

## Files

### Dockerfile

The `Dockerfile` defines the container image for the gen-ai-examples application. It:

- Uses Python 3.12 as the base image
- Installs all required dependencies
- Sets up the application environment
- Configures the entry point for the application

### docker-compose.yml

The `docker-compose.yml` file defines a multi-container setup that includes:

- The gen-ai-examples application container
- A PostgreSQL database container with pgvector extension
- Network configuration for container communication
- Volume mounts for persistent data storage

## Usage

### Development Environment

```bash
# Start the development environment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the environment
docker-compose down

# Rebuild containers after changes
docker-compose build
docker-compose up -d
```

### Production Deployment

```bash
# Build the production image
docker build -t gen-ai-examples:latest .

# Run the container with production settings
docker run -d \
  --name gen-ai-app \
  -p 8000:8000 \
  --env-file .env \
  gen-ai-examples:latest
```

### AWS Deployment

For AWS deployment, you can use Amazon ECR and ECS:

```bash
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com

# Create a repository if it doesn't exist
aws ecr create-repository --repository-name gen-ai-examples --region us-east-1

# Tag the image
docker tag gen-ai-examples:latest your-account-id.dkr.ecr.us-east-1.amazonaws.com/gen-ai-examples:latest

# Push the image
docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/gen-ai-examples:latest
```

Then deploy to ECS using the Terraform configurations in the `infrastructure/terraform` directory.

## Configuration

### Environment Variables

The Docker containers use environment variables for configuration. These can be provided through the `.env` file or directly in the `docker-compose.yml` file.

Key environment variables include:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# PostgreSQL Configuration
PGVECTOR_HOST=postgres
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password
PGVECTOR_DATABASE=vector_db

# Application Configuration
APP_PORT=8000
DEBUG=false
```

### Volume Mounts

The `docker-compose.yml` file defines volume mounts for persistent data storage:

- `postgres-data`: Stores PostgreSQL data
- `app-data`: Stores application data that needs to persist between container restarts

## Best Practices

1. **Security**: Never store sensitive information like API keys or passwords in the Dockerfile or docker-compose.yml. Use environment variables or secrets management.

2. **Image Size**: Keep the Docker image as small as possible by using multi-stage builds and removing unnecessary files.

3. **Caching**: Organize Dockerfile commands to take advantage of Docker's layer caching mechanism.

4. **Health Checks**: Include health checks in your docker-compose.yml to ensure services are running correctly.

5. **Logging**: Configure proper logging to make debugging easier.

6. **Resource Limits**: Set memory and CPU limits to prevent containers from consuming too many resources.

7. **Non-Root User**: Run containers as a non-root user for better security.

Following these best practices ensures a secure, efficient, and maintainable containerized environment for the gen-ai-examples project.
