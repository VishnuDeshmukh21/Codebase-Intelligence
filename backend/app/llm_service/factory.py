import os
from .gemini_provider import GeminiProvider
from .hf_provider import HuggingFaceProvider
from .base import BaseLLMProvider

class LLMFactory:
    _instance: BaseLLMProvider | None = None

    @classmethod
    def get_client(cls) -> BaseLLMProvider:
        """Returns a singleton instance of the configured LLM provider."""
        if cls._instance is not None:
            return cls._instance

        provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        
        if provider == "gemini":
            model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
            cls._instance = GeminiProvider(model_name=model)
        elif provider == "huggingface":
            model = os.environ.get("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
            cls._instance = HuggingFaceProvider(model_name=model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return cls._instance