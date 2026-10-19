from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import structlog

from .base import BaseLLMProvider

# Ensure .env is loaded across execution contexts
load_dotenv()
_local_env = Path(__file__).resolve().parent.parent / ".env"
if _local_env.exists():
    load_dotenv(dotenv_path=_local_env)

log = structlog.get_logger()


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider using the official Google GenAI SDK (google-genai).
    Compatible with standard Gemini API keys and GCP service environments.
    """

    def __init__(self, model_name: str | None = None):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please configure GEMINI_API_KEY in your environment or app/.env file."
            )

        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.client = genai.Client(api_key=api_key)
        log.info("gemini_provider.initialized", model=self.model_name)

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0, max_retries: int = 4) -> str:
        """
        Generates content using the official Google GenAI SDK models client,
        with automated exponential backoff retry for transient API spikes (503/429).
        """
        import time

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as exc:
                err_str = str(exc)
                is_transient = any(code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand"))
                if is_transient and attempt < max_retries - 1:
                    wait_seconds = 2 ** attempt
                    log.warning(
                        "gemini_provider.transient_error_retrying",
                        attempt=attempt + 1,
                        wait_seconds=wait_seconds,
                        error=err_str[:120],
                    )
                    time.sleep(wait_seconds)
                    continue

                log.error("gemini_provider.generate_failed", error=err_str, model=self.model_name)
                raise