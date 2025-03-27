# Terragrunt Configuration

This directory contains Terragrunt configurations for advanced infrastructure management of the gen-ai-examples project on AWS, following AWS best practices.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Usage](#usage)
5. [Environment Management](#environment-management)
6. [Best Practices](#best-practices)

## Overview

Terragrunt is a thin wrapper around Terraform that provides extra tools for working with multiple Terraform modules. The configurations in this directory use Terragrunt to manage the deployment of the gen-ai-examples project to AWS, with support for multiple environments and component-level management.

## Directory Structure

The Terragrunt configuration is organized as follows:

```
terragrunt/
├── terragrunt.hcl                # Root Terragrunt configuration
├── environments/                 # Environment-specific configurations
│   ├── dev/                      # Development environment
│   │   ├── terragrunt.hcl        # Environment-level configuration
│   │   ├── cognito/              # Cognito module for dev
│   │   ├── s3/                   # S3 module for dev
│   │   ├── rds/                  # RDS module for dev
│   │   └── amplify/              # Amplify module for dev
│   ├── staging/                  # Staging environment
│   └── prod/                     # Production environment
└── modules/                      # Reusable Terragrunt module configurations
    ├── cognito/                  # Cognito module configuration
    ├── s3/                       # S3 module configuration
    ├── rds/                      # RDS module configuration
    └── amplify/                  # Amplify module configuration
```

## Prerequisites

Before using these configurations, ensure you have:

1. An AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Terraform (version 1.0.0 or later) installed
4. Terragrunt (latest version) installed

## Usage

### Deploying All Components

```bash
# Initialize and apply all configurations for development environment
cd infrastructure/terragrunt/environments/dev
terragrunt run-all init
terragrunt run-all plan
terragrunt run-all apply

# Destroy all resources when no longer needed
terragrunt run-all destroy
```

### Deploying Specific Components

```bash
# Deploy only the Cognito resources
cd infrastructure/terragrunt/environments/dev/cognito
terragrunt init
terragrunt plan
terragrunt apply

# Deploy only the RDS resources
cd infrastructure/terragrunt/environments/dev/rds
terragrunt init
terragrunt plan
terragrunt apply
```

### Deploying to Different Environments

```bash
# Deploy to staging environment
cd infrastructure/terragrunt/environments/staging
terragrunt run-all apply

# Deploy to production environment
cd infrastructure/terragrunt/environments/prod
terragrunt run-all apply
```

## Environment Management

Terragrunt makes it easy to manage multiple environments (dev, staging, prod) with different configurations. Each environment has its own directory with environment-specific variables.

### Environment-Specific Variables

Environment-specific variables are defined in the `terragrunt.hcl` file in each environment directory:

```hcl
# environments/dev/terragrunt.hcl
inputs = {
  environment     = "dev"
  aws_region      = "us-east-1"
  project_name    = "gen-ai-examples-dev"
  vpc_cidr        = "10.0.0.0/16"
  database_name   = "vector_db_dev"
  instance_type   = "db.t3.medium"
  # Other environment-specific variables
}
```

### Inheriting Common Configuration

Common configuration is defined in the root `terragrunt.hcl` file and inherited by all modules:

```hcl
# terragrunt.hcl
remote_state {
  backend = "s3"
  config = {
    bucket         = "${get_env("TG_BUCKET_PREFIX", "")}terraform-state"
    key            = "${path_relative_to_include()}/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

## Best Practices

The Terragrunt configurations follow these best practices:

1. **DRY (Don't Repeat Yourself)**: Common configurations are defined once and inherited
2. **Modularity**: Resources are organized into reusable modules
3. **Environment Isolation**: Each environment has its own isolated configuration
4. **Remote State Management**: State is stored in S3 with locking via DynamoDB
5. **Dependency Management**: Dependencies between modules are explicitly defined
6. **Variable Management**: Variables are organized by environment and module
7. **Consistent Naming**: Resources follow a consistent naming convention

For more detailed information on using Terragrunt with AWS, refer to the [Terragrunt documentation](https://terragrunt.gruntwork.io/docs/).
