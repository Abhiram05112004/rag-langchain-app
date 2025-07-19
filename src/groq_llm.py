import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - only API key comes from .env, rest are hardcoded defaults
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Correct Groq endpoint
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key")  # Only this from .env
QWEN_MODEL = "qwen/qwen3-32b"  # Updated to available Qwen model
DEFAULT_MAX_TOKENS = 1024  # Hardcoded default
DEFAULT_TEMPERATURE = 0.1  # Hardcoded default

class GroqLLM:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or QWEN_MODEL

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": temperature or DEFAULT_TEMPERATURE
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Remove thinking tags from reasoning models
        content = self._remove_thinking_tags(content)
        
        return content
    
    def _remove_thinking_tags(self, content):
        """
        Remove <think>...</think> tags and their content from the response.
        This is needed for reasoning models that include their thinking process.
        """
        # Remove <think>...</think> blocks (including multiline)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Clean up any extra whitespace left behind
        content = content.strip()
        
        return content
