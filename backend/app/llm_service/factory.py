from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from .base import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .hf_provider import HuggingFaceProvider

# Ensure .env is loaded
load_dotenv()
_local_env = Path(__file__).resolve().parent.parent / ".env"
if _local_env.exists():
    load_dotenv(dotenv_path=_local_env)


class LLMFactory:
    _instance: BaseLLMProvider | None = None

    @classmethod
    def get_client(cls) -> BaseLLMProvider:
        """Returns a singleton instance of the configured LLM provider."""
        if cls._instance is not None:
            return cls._instance

        provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

        if provider == "gemini":
            model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
            cls._instance = GeminiProvider(model_name=model)
        elif provider == "huggingface":
            model = os.environ.get("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
            cls._instance = HuggingFaceProvider(model_name=model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Resets the singleton instance (useful for testing or config switching)."""
        cls._instance = None