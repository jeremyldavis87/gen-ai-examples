# Main Terraform configuration for AWS resources

provider "aws" {
  region = var.aws_region
}

# Cognito User Pool for authentication
resource "aws_cognito_user_pool" "main" {
  name = "${var.project_name}-user-pool"
  
  # Username attributes
  username_attributes = ["email"]
  auto_verify_attributes = ["email"]
  
  # Password policy
  password_policy {
    minimum_length = 8
    require_lowercase = true
    require_numbers = true
    require_symbols = true
    require_uppercase = true
  }
  
  # Email configuration
  email_configuration {
    email_sending_account = "COGNITO_DEFAULT"
  }
  
  # Schema attributes
  schema {
    name = "email"
    attribute_data_type = "String"
    mutable = true
    required = true
  }
  
  schema {
    name = "name"
    attribute_data_type = "String"
    mutable = true
    required = true
  }
  
  tags = var.common_tags
}

# Cognito User Pool Client
resource "aws_cognito_user_pool_client" "web_client" {
  name = "${var.project_name}-web-client"
  user_pool_id = aws_cognito_user_pool.main.id
  
  # OAuth configuration
  allowed_oauth_flows = ["code", "implicit"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes = ["email", "openid", "profile"]
  callback_urls = var.cognito_callback_urls
  logout_urls = var.cognito_logout_urls
  supported_identity_providers = ["COGNITO", "Google", "LinkedIn"]
  
  # Token configuration
  refresh_token_validity = 30
  access_token_validity = 1
  id_token_validity = 1
  token_validity_units {
    access_token = "hours"
    id_token = "hours"
    refresh_token = "days"
  }
  
  # Don't generate a client secret for web apps
  generate_secret = false
}

# Identity providers for social login
resource "aws_cognito_identity_provider" "google" {
  user_pool_id = aws_cognito_user_pool.main.id
  provider_name = "Google"
  provider_type = "Google"
  
  provider_details = {
    client_id = var.google_client_id
    client_secret = var.google_client_secret
    authorize_scopes = "email profile openid"
  }
  
  attribute_mapping = {
    email = "email"
    name = "name"
    username = "sub"
  }
}

resource "aws_cognito_identity_provider" "linkedin" {
  user_pool_id = aws_cognito_user_pool.main.id
  provider_name = "LinkedIn"
  provider_type = "LinkedIn"
  
  provider_details = {
    client_id = var.linkedin_client_id
    client_secret = var.linkedin_client_secret
    authorize_scopes = "r_emailaddress r_liteprofile"
  }
  
  attribute_mapping = {
    email = "emailAddress"
    name = "formattedName"
    username = "id"
  }
}

# S3 bucket for file storage
resource "aws_s3_bucket" "storage" {
  bucket = "${var.project_name}-storage-${var.environment}"
  
  tags = var.common_tags
}

resource "aws_s3_bucket_ownership_controls" "storage" {
  bucket = aws_s3_bucket.storage.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "storage" {
  depends_on = [aws_s3_bucket_ownership_controls.storage]
  bucket = aws_s3_bucket.storage.id
  acl    = "private"
}

resource "aws_s3_bucket_versioning" "storage" {
  bucket = aws_s3_bucket.storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

# RDS Aurora PostgreSQL Serverless v2 with pgvector
resource "aws_rds_cluster" "postgres" {
  cluster_identifier = "${var.project_name}-postgres-${var.environment}"
  engine = "aurora-postgresql"
  engine_version = "15.5"
  engine_mode = "provisioned"
  database_name = var.database_name
  master_username = var.database_username
  master_password = var.database_password
  backup_retention_period = 7
  preferred_backup_window = "03:00-04:00"
  preferred_maintenance_window = "sun:04:30-sun:05:30"
  db_subnet_group_name = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  storage_encrypted = true
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.project_name}-postgres-final-snapshot"
  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 16
  }
  
  tags = var.common_tags
}

resource "aws_rds_cluster_instance" "postgres" {
  count = 2
  identifier = "${var.project_name}-postgres-${var.environment}-${count.index}"
  cluster_identifier = aws_rds_cluster.postgres.id
  instance_class = "db.serverless"
  engine = aws_rds_cluster.postgres.engine
  engine_version = aws_rds_cluster.postgres.engine_version
  db_parameter_group_name = aws_db_parameter_group.postgres.name
  
  tags = var.common_tags
}

resource "aws_db_parameter_group" "postgres" {
  name = "${var.project_name}-postgres-${var.environment}"
  family = "aurora-postgresql15"
  description = "Parameter group for PostgreSQL with pgvector"
  
  parameter {
    name = "shared_preload_libraries"
    value = "pg_stat_statements,pgvector"
    apply_method = "pending-reboot"
  }
  
  tags = var.common_tags
}

resource "aws_db_subnet_group" "postgres" {
  name = "${var.project_name}-postgres-${var.environment}"
  subnet_ids = var.database_subnet_ids
  
  tags = var.common_tags
}

resource "aws_security_group" "postgres" {
  name = "${var.project_name}-postgres-${var.environment}"
  description = "Security group for PostgreSQL RDS"
  vpc_id = var.vpc_id
  
  ingress {
    from_port = 5432
    to_port = 5432
    protocol = "tcp"
    cidr_blocks = var.database_access_cidr_blocks
  }
  
  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = var.common_tags
}

# Amplify App for frontend deployment
resource "aws_amplify_app" "frontend" {
  name = "${var.project_name}-frontend"
  repository = var.repository_url
  access_token = var.github_access_token
  build_spec = <<-EOT
    version: 1
    frontend:
      phases:
        preBuild:
          commands:
            - npm ci
        build:
          commands:
            - npm run build
      artifacts:
        baseDirectory: build
        files:
          - '**/*'
      cache:
        paths:
          - node_modules/**/*
  EOT
  
  # Environment variables
  environment_variables = {
    AMPLIFY_DIFF_DEPLOY = "false"
    AMPLIFY_MONOREPO_APP_ROOT = "frontend"
    API_URL = var.api_url
    USER_POOL_ID = aws_cognito_user_pool.main.id
    USER_POOL_CLIENT_ID = aws_cognito_user_pool_client.web_client.id
  }
  
  tags = var.common_tags
}

# Amplify Branch for main branch
resource "aws_amplify_branch" "main" {
  app_id = aws_amplify_app.frontend.id
  branch_name = "main"
  framework = "React"
  stage = "PRODUCTION"
  
  environment_variables = {
    NODE_ENV = "production"
  }
  
  tags = var.common_tags
}
