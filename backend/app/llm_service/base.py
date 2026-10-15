from abc import ABC, abstractmethod
from typing import Optional

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        """
        Generates text from the LLM. 
        Temperature defaults to 0.0 for deterministic graph query generation.
        """
        pass