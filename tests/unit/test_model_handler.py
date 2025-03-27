import unittest
from unittest.mock import patch, MagicMock
from models import ModelHandler

class TestModelHandler(unittest.TestCase):
    
    def setUp(self):
        # Create a mock for the AI Gateway client
        self.ai_gateway_patcher = patch('models.model_handler.AIGatewayClient')
        self.mock_ai_gateway = self.ai_gateway_patcher.start()
        self.mock_client_instance = MagicMock()
        self.mock_ai_gateway.return_value = self.mock_client_instance
        
        # Initialize the model handler with the mocked client
        self.model_handler = ModelHandler()
    
    def tearDown(self):
        self.ai_gateway_patcher.stop()
    
    def test_generate_text_success(self):
        # Arrange
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        expected_response = {"response": "I'm doing well, thank you!"}
        self.mock_client_instance.generate_text.return_value = expected_response
        
        # Act
        response = self.model_handler.generate_text(messages)
        
        # Assert
        self.assertEqual(response, expected_response)
        self.mock_client_instance.generate_text.assert_called_once()
    
    def test_generate_text_with_fallback(self):
        # Arrange
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        expected_response = {"response": "I'm doing well, thank you!"}
        
        # Configure the mock to fail on first call and succeed on second call
        self.mock_client_instance.generate_text.side_effect = [
            Exception("Model unavailable"),  # Primary model fails
            expected_response                # Fallback model succeeds
        ]
        
        # Act
        response = self.model_handler.generate_text(messages)
        
        # Assert
        self.assertEqual(response, expected_response)
        self.assertEqual(self.mock_client_instance.generate_text.call_count, 2)
    
    def test_generate_embeddings_success(self):
        # Arrange
        texts = ["This is a test"]
        expected_embeddings = [[0.1, 0.2, 0.3]]
        self.mock_client_instance.generate_embeddings.return_value = expected_embeddings
        
        # Act
        embeddings = self.model_handler.generate_embeddings(texts)
        
        # Assert
        self.assertEqual(embeddings, expected_embeddings)
        self.mock_client_instance.generate_embeddings.assert_called_once()
    
    def test_set_default_config(self):
        # Arrange
        task_type = "customer_support"
        config = {
            "primary": {"model": "gpt4o", "model_family": "openai"},
            "fallbacks": [
                {"model": "sonnet-3.7", "model_family": "anthropic"}
            ]
        }
        
        # Act
        self.model_handler.set_default_config(task_type, config)
        
        # Assert
        self.assertIn(task_type, self.model_handler.default_configs)
        self.assertEqual(self.model_handler.default_configs[task_type], config)
    
    def test_get_performance_metrics(self):
        # Act
        metrics = self.model_handler.get_performance_metrics()
        
        # Assert
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_requests", metrics)
        self.assertIn("fallback_count", metrics)
        self.assertIn("error_count", metrics)

if __name__ == '__main__':
    unittest.main()
