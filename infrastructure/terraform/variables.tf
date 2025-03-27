# Variables for AWS infrastructure

variable "aws_region" {
  description = "The AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Cognito variables
variable "cognito_callback_urls" {
  description = "List of allowed callback URLs for Cognito"
  type        = list(string)
}

variable "cognito_logout_urls" {
  description = "List of allowed logout URLs for Cognito"
  type        = list(string)
}

variable "google_client_id" {
  description = "Google OAuth client ID"
  type        = string
  sensitive   = true
}

variable "google_client_secret" {
  description = "Google OAuth client secret"
  type        = string
  sensitive   = true
}

variable "linkedin_client_id" {
  description = "LinkedIn OAuth client ID"
  type        = string
  sensitive   = true
}

variable "linkedin_client_secret" {
  description = "LinkedIn OAuth client secret"
  type        = string
  sensitive   = true
}

# Database variables
variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "vector_db"
}

variable "database_username" {
  description = "Username for database access"
  type        = string
  default     = "postgres"
}

variable "database_password" {
  description = "Password for database access"
  type        = string
  sensitive   = true
}

variable "database_subnet_ids" {
  description = "List of subnet IDs for the database"
  type        = list(string)
}

variable "database_access_cidr_blocks" {
  description = "CIDR blocks that can access the database"
  type        = list(string)
}

variable "vpc_id" {
  description = "ID of the VPC"
  type        = string
}

# Amplify variables
variable "repository_url" {
  description = "URL of the GitHub repository"
  type        = string
}

variable "github_access_token" {
  description = "GitHub access token for Amplify"
  type        = string
  sensitive   = true
}

variable "api_url" {
  description = "URL of the backend API"
  type        = string
}
