from typing import Dict
from .base import BasePromptManager
from .constant import payment_intent_prompt, payment_related_message, non_payment_related_message, payment_related_message_no_seed

class InmemoryPromptManager(BasePromptManager):
    """In-memory implementation of PromptManager that stores prompts as variables."""

    def __init__(self):
        self._prompts: Dict[str, str] = {
            "payment_intent": payment_intent_prompt,
            "payment_related_message": payment_related_message,
            "payment_related_no_seed": payment_related_message_no_seed,
            "non_payment_related_message": non_payment_related_message

        }
    
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
        if prompt_key not in self._prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found")
        
        return self._prompts[prompt_key]