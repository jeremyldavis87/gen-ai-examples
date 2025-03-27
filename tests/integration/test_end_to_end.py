import unittest
import os
import tempfile
import json
from unittest.mock import patch

from models import ModelHandler
from vector_db import PGVectorClient
from apps import rag_app

# Skip these tests if environment variables are not set
@unittest.skipIf(not os.environ.get('INTEGRATION_TEST'), 'Skipping integration tests')
class TestEndToEnd(unittest.TestCase):
    
    def setUp(self):
        # Create test documents
        self.test_docs = [
            "Amazon Web Services (AWS) is a cloud computing platform that provides a wide range of services.",
            "AWS Cognito is a user authentication and authorization service provided by Amazon Web Services.",
            "Amazon S3 (Simple Storage Service) is an object storage service offering industry-leading scalability."
        ]
        
        # Create temporary files with test content
        self.temp_files = []
        for i, content in enumerate(self.test_docs):
            temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False)
            temp_file.write(content)
            temp_file.close()
            self.temp_files.append(temp_file.name)
        
        # Initialize model handler and vector DB client
        self.model_handler = ModelHandler()
        self.vector_client = PGVectorClient()
        
        # Set up test environment
        self.setup_test_environment()
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clean up test data from vector DB
        self.cleanup_test_data()
    
    def setup_test_environment(self):
        """Set up the test environment including database tables"""
        try:
            # Create test collection in vector DB
            self.vector_client.execute_query(
                "CREATE TABLE IF NOT EXISTS test_documents (id SERIAL PRIMARY KEY, content TEXT, embedding vector(1536), metadata JSONB)"
            )
            self.vector_client.execute_query(
                "CREATE INDEX IF NOT EXISTS test_documents_embedding_idx ON test_documents USING ivfflat (embedding vector_cosine_ops)"
            )
        except Exception as e:
            self.skipTest(f"Failed to set up test environment: {str(e)}")
    
    def cleanup_test_data(self):
        """Clean up test data from the database"""
        try:
            self.vector_client.execute_query("DROP TABLE IF EXISTS test_documents")
        except Exception:
            pass  # Ignore cleanup errors
    
    @patch('apps.rag_app.ModelHandler')
    @patch('apps.rag_app.PGVectorClient')
    def test_rag_workflow(self, mock_vector_client, mock_model_handler):
        # Configure mocks
        mock_model_handler.return_value = self.model_handler
        mock_vector_client.return_value = self.vector_client
        
        # Test document ingestion
        doc_ids = rag_app.ingest_documents(self.temp_files)
        self.assertEqual(len(doc_ids), len(self.temp_files))
        
        # Test querying
        query = "What is AWS Cognito?"
        response = rag_app.query_documents(query)
        
        # Verify response contains relevant information
        self.assertIsNotNone(response)
        self.assertIsInstance(response, dict)
        self.assertIn('answer', response)
    
    def test_model_handler_with_real_api(self):
        # Skip if API key not available
        if not os.environ.get('API_KEY'):
            self.skipTest("API_KEY not available")
        
        # Test text generation with real API
        messages = [{"role": "user", "content": "What are AWS managed services?"}]
        response = self.model_handler.generate_text(messages)
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIsInstance(response, dict)
        
        # Test embeddings with real API
        texts = ["AWS managed services provide scalable cloud infrastructure."]
        embeddings = self.model_handler.generate_embeddings(texts)
        
        # Verify embeddings
        self.assertIsNotNone(embeddings)
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 1)
        self.assertIsInstance(embeddings[0], list)

if __name__ == '__main__':
    unittest.main()
