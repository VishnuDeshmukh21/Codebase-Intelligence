import os
from huggingface_hub import InferenceClient
from .base import BaseLLMProvider

class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN environment variable is not set.")
        
        self.client = InferenceClient(model=model_name, token=api_key)

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})

        # HF Inference API chokes on absolute 0.0 temperature sometimes, so we clamp it safely
        safe_temp = max(temperature, 0.01)

        response = self.client.chat_completion(
            messages=messages,
            temperature=safe_temp,
            max_tokens=1024
        )
        
        return response.choices[0].message.content