# examples/client_example.py
import os
import requests
import json
import logging
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthcareAIClient:
    """Client for the Healthcare AI Service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client."""
        self.base_url = base_url
    
    def generate_text(self, 
                     messages: List[Dict[str, str]],
                     task_type: str,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate text using the AI service.
        
        Args:
            messages: List of message objects in the format [{"role": "user", "content": "Hello"}]
            task_type: Type of task (healthcare_query, medical_records_summary, claim_processing)
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            The response from the AI service
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "messages": messages,
            "task_type": task_type
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error: {e}")
            raise
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health and metrics information."""
        url = f"{self.base_url}/api/health"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting health metrics: {e}")
            raise
    
    def reset_metrics(self) -> Dict[str, Any]:
        """Reset performance metrics."""
        url = f"{self.base_url}/api/reset-metrics"
        
        try:
            response = requests.post(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error resetting metrics: {e}")
            raise

def main():
    """Example usage of the Healthcare AI Client."""
    client = HealthcareAIClient()
    
    # Example 1: Healthcare query
    print("\n=== HEALTHCARE QUERY EXAMPLE ===")
    healthcare_query = [
        {"role": "system", "content": "You are a healthcare assistant providing general health information. Do not provide medical advice or diagnoses."},
        {"role": "user", "content": "What are some common symptoms of seasonal allergies?"}
    ]
    
    try:
        healthcare_response = client.generate_text(
            messages=healthcare_query,
            task_type="healthcare_query"
        )
        
        print(f"Content: {healthcare_response['content']}")
        print(f"Model used: {healthcare_response['model_used']}")
        print(f"Fallback used: {healthcare_response['fallback_used']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 2: Medical records summary
    print("\n=== MEDICAL RECORDS SUMMARY EXAMPLE ===")
    medical_records = [
        {"role": "system", "content": "You are a medical records summarization assistant. Summarize the provided medical information concisely."},
        {"role": "user", "content": """
        Patient: John Doe
        Age: 45
        Visit Date: 2023-03-15
        
        Chief Complaint: Persistent cough for 2 weeks, mild fever, fatigue
        
        Vital Signs:
        - Blood Pressure: 128/82
        - Heart Rate: 88 bpm
        - Temperature: 99.8°F
        - Respiration Rate: 18/min
        - O2 Saturation: 97%
        
        Assessment: Patient presents with symptoms consistent with acute bronchitis. No signs of pneumonia on chest X-ray. Recommend rest, increased fluid intake, and over-the-counter cough suppressants.
        
        Plan: Follow-up in 1 week if symptoms persist or worsen. Prescribed amoxicillin 500mg TID for 7 days.
        """}
    ]
    
    try:
        medical_records_response = client.generate_text(
            messages=medical_records,
            task_type="medical_records_summary"
        )
        
        print(f"Content: {medical_records_response['content']}")
        print(f"Model used: {medical_records_response['model_used']}")
        print(f"Fallback used: {medical_records_response['fallback_used']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 3: Claim processing
    print("\n=== CLAIM PROCESSING EXAMPLE ===")
    claim_data = [
        {"role": "system", "content": "You are a claims processing assistant. Extract key information from claims and provide a structured summary."},
        {"role": "user", "content": """
        Claim #: CL-2023-78945
        Date of Service: 2023-02-10
        Provider: Dr. Sarah Johnson, Internal Medicine
        Facility: Greenview Medical Center
        
        Services:
        1. Office visit, established patient, level 3 (CPT: 99213) - $120
        2. Comprehensive metabolic panel (CPT: 80053) - $45
        3. Lipid panel (CPT: 80061) - $35
        
        Diagnosis Codes:
        - E11.9 (Type 2 diabetes without complications)
        - I10 (Essential hypertension)
        
        Patient Responsibility:
        - Copay: $25
        - Coinsurance: $35
        - Deductible: $0
        
        Insurance Payment: $140
        """}
    ]
    
    try:
        claim_response = client.generate_text(
            messages=claim_data,
            task_type="claim_processing"
        )
        
        print(f"Content: {claim_response['content']}")
        print(f"Model used: {claim_response['model_used']}")
        print(f"Fallback used: {claim_response['fallback_used']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get health metrics
    print("\n=== HEALTH METRICS ===")
    try:
        health_metrics = client.get_health_metrics()
        print(f"Status: {health_metrics['status']}")
        print(f"Version: {health_metrics['version']}")
        print(f"Total calls: {health_metrics['metrics']['total_calls']}")
        print(f"Success rate: {health_metrics['metrics']['success_rate']:.2%}")
        print(f"Fallback rate: {health_metrics['metrics']['fallback_rate']:.2%}")
        print(f"Average latency: {health_metrics['metrics']['avg_latency']:.2f} seconds")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()