# Terragrunt configuration

remote_state {
  backend = "s3"
  config = {
    bucket         = "${get_env("TG_BUCKET_PREFIX", "")}terraform-state-${get_aws_account_id()}"
    key            = "${path_relative_to_include()}/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "aws" {
  region = "${local.aws_region}"
}
EOF
}

# Load common variables
locals {
  # Load environment-specific variables
  env_vars = read_terragrunt_config(find_in_parent_folders("env.hcl"))
  
  # Extract commonly used variables
  aws_region   = local.env_vars.locals.aws_region
  environment  = local.env_vars.locals.environment
  project_name = local.env_vars.locals.project_name
  
  # Common tags for all resources
  common_tags = {
    Environment = local.environment
    Project     = local.project_name
    ManagedBy   = "Terragrunt"
  }
}

# Default inputs for all Terraform configurations
inputs = {
  aws_region   = local.aws_region
  environment  = local.environment
  project_name = local.project_name
  common_tags  = local.common_tags
}

# Include all settings from the root terragrunt.hcl file
include {
  path = find_in_parent_folders()
}
