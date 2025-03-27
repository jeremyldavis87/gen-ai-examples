# Infrastructure as Code

This directory contains Infrastructure as Code (IaC) configurations for deploying the gen-ai-examples project to AWS, using Terraform and Terragrunt.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Usage](#usage)
5. [AWS Resources](#aws-resources)
6. [Best Practices](#best-practices)

## Overview

The infrastructure directory contains Terraform and Terragrunt configurations for deploying the gen-ai-examples project to AWS. These configurations follow AWS best practices and provide a repeatable, version-controlled approach to infrastructure management.

## Directory Structure

- `terraform/`: Contains Terraform modules and configurations
- `terragrunt/`: Contains Terragrunt configurations for advanced infrastructure management

## Prerequisites

Before using these configurations, ensure you have:

1. An AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Terraform (version 1.0.0 or later) installed
4. Terragrunt (latest version) installed

## Usage

### Basic Terraform Deployment

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan the deployment
terraform plan

# Apply the deployment
terraform apply

# Destroy resources when no longer needed
terraform destroy
```

### Advanced Terragrunt Deployment

```bash
# Initialize and apply all configurations
cd infrastructure/terragrunt
terragrunt run-all init
terragrunt run-all apply

# Apply specific module
cd infrastructure/terragrunt/cognito
terragrunt apply

# Destroy resources when no longer needed
cd infrastructure/terragrunt
terragrunt run-all destroy
```

## AWS Resources

The infrastructure configurations deploy the following AWS resources:

### Amazon Cognito

- User Pool for authentication
- App Client for application integration
- Identity Pool for AWS service access
- OAuth configuration for social logins (Google and LinkedIn)

### Amazon S3

- Bucket for file storage
- Bucket policies for secure access
- CORS configuration for web access

### Amazon RDS

- Aurora PostgreSQL Serverless v2 cluster
- Parameter group for pgvector extension
- Security group for database access
- Subnet group for VPC placement

### AWS Amplify

- Amplify Gen 2 application for front-end deployment
- Build and deployment configuration
- Domain and hosting configuration

### Additional Resources

- VPC and networking components
- IAM roles and policies
- CloudWatch logs and metrics
- Lambda functions for custom logic

## Best Practices

The infrastructure configurations follow these best practices:

1. **Modularity**: Resources are organized into reusable modules
2. **Security**: Least privilege principles are applied to IAM roles
3. **Encryption**: Data is encrypted at rest and in transit
4. **High Availability**: Resources are deployed across multiple availability zones
5. **Cost Optimization**: Serverless and auto-scaling resources are used where appropriate
6. **Tagging**: Resources are tagged for cost allocation and management
7. **State Management**: Terraform state is stored in S3 with locking via DynamoDB

For more detailed information on the infrastructure, refer to the README files in the `terraform/` and `terragrunt/` directories.
