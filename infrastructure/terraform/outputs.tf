# Output values from Terraform resources

output "cognito_user_pool_id" {
  description = "ID of the Cognito User Pool"
  value       = aws_cognito_user_pool.main.id
}

output "cognito_user_pool_client_id" {
  description = "ID of the Cognito User Pool Client"
  value       = aws_cognito_user_pool_client.web_client.id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for file storage"
  value       = aws_s3_bucket.storage.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for file storage"
  value       = aws_s3_bucket.storage.arn
}

output "rds_cluster_endpoint" {
  description = "Endpoint of the RDS Aurora PostgreSQL cluster"
  value       = aws_rds_cluster.postgres.endpoint
}

output "rds_cluster_reader_endpoint" {
  description = "Reader endpoint of the RDS Aurora PostgreSQL cluster"
  value       = aws_rds_cluster.postgres.reader_endpoint
}

output "amplify_app_id" {
  description = "ID of the Amplify App"
  value       = aws_amplify_app.frontend.id
}

output "amplify_default_domain" {
  description = "Default domain of the Amplify App"
  value       = aws_amplify_app.frontend.default_domain
}
