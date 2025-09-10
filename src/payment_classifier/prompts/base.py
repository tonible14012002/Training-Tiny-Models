
from abc import ABC, abstractmethod
from typing import Optional


class BasePromptManager(ABC):
    """Base class for prompt manager implementations."""
    
    @abstractmethod
    def get_prompt(self, prompt_key: str) -> str:
        """
        Retrieve a prompt string by its key.
        
        Args:
            prompt_key: The key identifying the prompt to retrieve
            
        Returns:
            The prompt string associated with the given key
            
        Raises:
            KeyError: If the prompt_key is not found
        """
        pass