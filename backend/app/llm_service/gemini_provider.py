import os
import google.generativeai as genai
from .base import BaseLLMProvider

class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> str:
        config = genai.types.GenerationConfig(
            temperature=temperature,
        )
        
        if system_prompt:
            temp_model = genai.GenerativeModel(
                self.model.model_name, 
                system_instruction=system_prompt
            )
            response = temp_model.generate_content(prompt, generation_config=config)
        else:
            response = self.model.generate_content(prompt, generation_config=config)
            
        return response.text