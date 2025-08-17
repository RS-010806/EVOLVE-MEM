"""
EVOLVE-MEM: LLM Backend Interface

Simple LLM backend for hierarchical memory operations.
- Provides interface for LLM operations
- Handles Gemini API integration
- Supports summarization and abstraction tasks

"""
import os
import json
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class BaseLLM:
    """Base class for LLM backends."""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logging.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt."""
        raise NotImplementedError()

class GeminiLLM(BaseLLM):
    """Gemini LLM backend implementation."""
    
    def __init__(self):
        super().__init__()
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("Gemini LLM backend initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini: {e}", exc_info=True)
            raise
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using Gemini API."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

def get_llm_backend(backend_type: str = "gemini") -> BaseLLM:
    """Get LLM backend instance."""
    if backend_type.lower() == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"Unsupported LLM backend: {backend_type}") 
