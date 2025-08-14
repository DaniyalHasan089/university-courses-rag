"""Configuration management for University AI Assistant"""

import os
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the University AI Assistant"""
    
    # API Configuration
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Default Model Settings
    DEFAULT_MODEL = "microsoft/phi-3-mini-128k-instruct:free"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1000
    
    # RAG Configuration
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_RETRIEVAL_K = 3
    
    # Database Configuration
    DB_PERSIST_DIRECTORY = "./db"
    PDF_FOLDER = "./pdfs"
    
    # Embedding Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Streamlit Configuration
    PAGE_TITLE = "University AI Assistant"
    PAGE_ICON = "🎓"
    
    # Free Models (confirmed available on OpenRouter)
    FALLBACK_MODELS = {
        "Microsoft Phi-3 Mini": "microsoft/phi-3-mini-128k-instruct:free",
        "Meta Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct:free", 
        "Google Gemma 2 9B": "google/gemma-2-9b-it:free",
        "Mistral 7B": "mistralai/mistral-7b-instruct:free"
    }
    
    # Model Categories
    FREE_MODELS = [
        "microsoft/phi-3-mini-128k-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free"
    ]
    
    @classmethod
    def get_model_info(cls, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        is_free = model_id in cls.FREE_MODELS
        
        # Model-specific information
        model_info = {
            "is_free": is_free,
            "provider": model_id.split('/')[0] if '/' in model_id else "unknown",
            "requires_credits": not is_free
        }
        
        # Add specific model details
        if "gpt-4o" in model_id:
            model_info.update({
                "context_length": "128k tokens",
                "strengths": ["Reasoning", "Code", "Math"],
                "cost_per_1k_tokens": "$0.005" if "mini" in model_id else "$0.03"
            })
        elif "claude" in model_id:
            model_info.update({
                "context_length": "200k tokens",
                "strengths": ["Writing", "Analysis", "Reasoning"],
                "cost_per_1k_tokens": "$0.015"
            })
        elif "phi-3" in model_id:
            model_info.update({
                "context_length": "128k tokens",
                "strengths": ["General purpose", "Efficiency"],
                "cost_per_1k_tokens": "Free"
            })
        elif "llama" in model_id:
            model_info.update({
                "context_length": "128k tokens",
                "strengths": ["Open source", "General purpose"],
                "cost_per_1k_tokens": "Free"
            })
        
        return model_info
    
    @classmethod
    def validate_api_key(cls, api_key: str) -> bool:
        """Validate OpenRouter API key format"""
        if not api_key:
            return False
        
        # Basic validation - OpenRouter keys typically start with 'sk-or-'
        return len(api_key) > 10 and (api_key.startswith('sk-or-') or api_key.startswith('sk-'))
    
    @classmethod
    def fetch_available_models(cls, api_key: str) -> Dict[str, str]:
        """Fetch available models from OpenRouter API"""
        if not api_key:
            return cls.FALLBACK_MODELS
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "University AI Assistant"
            }
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = {}
                
                # Free models available on OpenRouter
                priority_models = [
                    "microsoft/phi-3-mini-128k-instruct:free",
                    "meta-llama/llama-3.1-8b-instruct:free",
                    "google/gemma-2-9b-it:free", 
                    "mistralai/mistral-7b-instruct:free"
                ]
                
                # Process models from API response
                for model in models_data.get("data", []):
                    model_id = model.get("id", "")
                    model_name = model.get("name", model_id)
                    
                    # Skip models that are not in our priority list
                    if model_id not in priority_models:
                        continue
                    
                    # Create a friendly display name (all are free)
                    display_name = model_name
                    
                    available_models[display_name] = model_id
                
                # If we got models, return them; otherwise fallback
                return available_models if available_models else cls.FALLBACK_MODELS
                
        except Exception as e:
            print(f"Error fetching models from OpenRouter: {e}")
        
        # Return fallback models if API call fails
        return cls.FALLBACK_MODELS
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for the LLM"""
        return """You are a helpful university course assistant with access to course materials. 

Your responsibilities:
1. Provide accurate, detailed answers based on the provided context from course materials
2. If the context doesn't contain enough information, clearly state this limitation
3. Format your responses clearly with proper structure and bullet points when appropriate
4. Cite specific sources when possible
5. Be encouraging and supportive to students
6. If asked about topics outside the course materials, politely redirect to course-related content

Always maintain a professional, educational tone while being approachable and helpful."""
