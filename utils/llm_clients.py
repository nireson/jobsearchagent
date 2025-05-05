"""
LLM client implementations for various model providers.
This module provides standardized interfaces for different LLM providers.
"""

import os
import requests
import json
import time
from typing import Dict, Any, List, Optional, Union

class BaseLLMClient:
    """Base class for LLM clients with common functionality."""
    
    def __init__(self):
        """Initialize the base LLM client."""
        pass
        
    def chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate a chat completion using the LLM.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            str: The generated completion text
        """
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        super().__init__()
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set")
    
    def chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate a chat completion using the OpenAI API.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            str: The generated completion text
        """
        if not self.api_key:
            return "Error: OpenAI API key not set"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return f"Error: {str(e)}"

class AnthropicClient(BaseLLMClient):
    """Anthropic API client implementation."""
    
    def __init__(self):
        """Initialize the Anthropic client."""
        super().__init__()
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        
        if not self.api_key:
            print("Warning: ANTHROPIC_API_KEY not set")
    
    def chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate a chat completion using the Anthropic API.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            str: The generated completion text
        """
        if not self.api_key:
            return "Error: Anthropic API key not set"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"].strip()
            else:
                print(f"Anthropic API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            return f"Error: {str(e)}"

class OllamaClient(BaseLLMClient):
    """Ollama API client implementation."""
    
    def __init__(self):
        """Initialize the Ollama client."""
        super().__init__()
        self.api_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3")
        
        # Ensure the URL has a scheme
        if not self.api_url.startswith(("http://", "https://")):
            self.api_url = f"http://{self.api_url}"
        
        # Remove trailing slash if present
        self.api_url = self.api_url.rstrip('/')
    
    def chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate a chat completion using the Ollama API.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            str: The generated completion text
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        payload = {
            "model": self.model,
            "prompt": combined_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            return f"Error: {str(e)}" 