import unittest
from unittest.mock import patch, MagicMock
import json
import os
from ai_gateway import AIGatewayClient

class TestAIGatewayClient(unittest.TestCase):
    
    def setUp(self):
        # Create a mock for requests
        self.requests_patcher = patch('ai_gateway.client.requests')
        self.mock_requests = self.requests_patcher.start()
        
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'AI_GATEWAY_URL': 'https://test-gateway.com/api',
            'PROJECT_NAME': 'test-project',
            'API_KEY': 'test-api-key'
        })
        self.env_patcher.start()
        
        # Initialize the client
        self.client = AIGatewayClient()
    
    def tearDown(self):
        self.requests_patcher.stop()
        self.env_patcher.stop()
    
    def test_generate_text(self):
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt4o"
        model_family = "openai"
        expected_response = {"response": "Hi there!"}
        
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        self.mock_requests.post.return_value = mock_response
        
        # Act
        response = self.client.generate_text(model, messages, model_family)
        
        # Assert
        self.assertEqual(response, expected_response)
        self.mock_requests.post.assert_called_once()
        
        # Verify the request payload
        call_args = self.mock_requests.post.call_args
        url = call_args[0][0]
        payload = json.loads(call_args[1]['data'])
        
        self.assertTrue(url.endswith('/generate'))
        self.assertEqual(payload['model'], model)
        self.assertEqual(payload['model_family'], model_family)
        self.assertEqual(payload['messages'], messages)
    
    def test_generate_embeddings(self):
        # Arrange
        texts = ["This is a test"]
        model = "text-embedding-3-large"
        expected_embeddings = [[0.1, 0.2, 0.3]]
        
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": expected_embeddings}
        self.mock_requests.post.return_value = mock_response
        
        # Act
        embeddings = self.client.generate_embeddings(texts, model)
        
        # Assert
        self.assertEqual(embeddings, expected_embeddings)
        self.mock_requests.post.assert_called_once()
        
        # Verify the request payload
        call_args = self.mock_requests.post.call_args
        url = call_args[0][0]
        payload = json.loads(call_args[1]['data'])
        
        self.assertTrue(url.endswith('/embeddings'))
        self.assertEqual(payload['model'], model)
        self.assertEqual(payload['texts'], texts)
    
    def test_error_handling(self):
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt4o"
        
        # Configure mock response for error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_response.text = json.dumps({"error": "Internal server error"})
        self.mock_requests.post.return_value = mock_response
        
        # Act & Assert
        with self.assertRaises(Exception):
            self.client.generate_text(model, messages)

if __name__ == '__main__':
    unittest.main()
