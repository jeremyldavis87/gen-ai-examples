# Terraform Configuration

This directory contains Terraform modules and configurations for deploying the gen-ai-examples project to AWS, following AWS best practices.

## Table of Contents

1. [Overview](#overview)
2. [Modules](#modules)
3. [Prerequisites](#prerequisites)
4. [Usage](#usage)
5. [Variables](#variables)
6. [Outputs](#outputs)
7. [State Management](#state-management)

## Overview

The Terraform configurations in this directory define the AWS infrastructure required to run the gen-ai-examples project. These configurations follow AWS best practices and provide a repeatable, version-controlled approach to infrastructure management.

## Modules

The Terraform configuration is organized into the following modules:

### Cognito Module

Defines Amazon Cognito resources for authentication:

- User Pool with appropriate password policies and MFA settings
- App Client with OAuth configuration
- Identity Pool for AWS service access
- IAM roles for authenticated and unauthenticated users

### S3 Module

Defines Amazon S3 resources for file storage:

- S3 bucket with appropriate permissions
- Bucket policies for secure access
- CORS configuration for web access
- Lifecycle policies for cost optimization

### RDS Module

Defines Amazon RDS resources for the PostgreSQL database:

- Aurora PostgreSQL Serverless v2 cluster
- Parameter group for pgvector extension
- Security group for database access
- Subnet group for VPC placement

### Amplify Module

Defines AWS Amplify resources for front-end deployment:

- Amplify Gen 2 application
- Build and deployment configuration
- Domain and hosting configuration
- Branch-specific settings

### VPC Module

Defines VPC and networking components:

- VPC with public and private subnets
- Internet Gateway and NAT Gateway
- Route tables and security groups
- Network ACLs for additional security

## Prerequisites

Before using these configurations, ensure you have:

1. An AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Terraform (version 1.0.0 or later) installed
4. A backend configuration for storing Terraform state

## Usage

### Basic Deployment

```bash
# Initialize Terraform
terraform init

# Plan the deployment
terraform plan -var-file=environments/dev.tfvars

# Apply the deployment
terraform apply -var-file=environments/dev.tfvars

# Destroy resources when no longer needed
terraform destroy -var-file=environments/dev.tfvars
```

### Environment-Specific Deployment

The configuration supports multiple environments (dev, staging, prod) through variable files:

```bash
# Deploy to development environment
terraform apply -var-file=environments/dev.tfvars

# Deploy to staging environment
terraform apply -var-file=environments/staging.tfvars

# Deploy to production environment
terraform apply -var-file=environments/prod.tfvars
```

## Variables

The configuration uses the following key variables:

- `environment`: The deployment environment (dev, staging, prod)
- `aws_region`: The AWS region to deploy to
- `project_name`: The name of the project
- `vpc_cidr`: The CIDR block for the VPC
- `database_name`: The name of the PostgreSQL database
- `database_username`: The username for the PostgreSQL database
- `database_password`: The password for the PostgreSQL database (should be provided through AWS Secrets Manager in production)

For a complete list of variables, see the `variables.tf` file.

## Outputs

The configuration provides the following key outputs:

- `cognito_user_pool_id`: The ID of the Cognito User Pool
- `cognito_app_client_id`: The ID of the Cognito App Client
- `s3_bucket_name`: The name of the S3 bucket
- `rds_endpoint`: The endpoint of the RDS instance
- `amplify_app_id`: The ID of the Amplify application
- `amplify_app_url`: The URL of the deployed Amplify application

For a complete list of outputs, see the `outputs.tf` file.

## State Management

The Terraform state is managed using an S3 backend with DynamoDB locking. This ensures that the state is stored securely and that multiple users can work with the configuration without conflicts.

The backend configuration is defined in the `backend.tf` file and should be customized for your environment.

```hcl
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "gen-ai-examples/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}
```

For more detailed information on using Terraform with AWS, refer to the [Terraform AWS Provider documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs).
