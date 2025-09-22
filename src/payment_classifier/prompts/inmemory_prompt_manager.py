from typing import Dict
from pathlib import Path
from .base import BasePromptManager
from .constant import (
    payment_intent_prompt,
    payment_related_message,
    non_payment_related_message,
    payment_related_message_no_seed,
    non_payment_related_message_v2,
    transform_text_to_payment_related,
    exclude_payment_intent,
    transform_text_to_non_payment_relate,
    label_payment_intent
)

class InmemoryPromptManager(BasePromptManager):
    """In-memory implementation of PromptManager that stores prompts as variables."""

    def __init__(self):
        self._prompts: Dict[str, str] = {
            "payment_intent": payment_intent_prompt,
            "payment_related_message": payment_related_message,
            "payment_related_no_seed": payment_related_message_no_seed,
            "non_payment_related_message": non_payment_related_message,
            "non_payment_related_message_v2": non_payment_related_message_v2,
            "transform_text_to_payment_related": transform_text_to_payment_related,
            "exclude_payment_intent": exclude_payment_intent,
            "transform_text_to_non_payment_relate": transform_text_to_non_payment_relate,
            "label_payment_intent": label_payment_intent
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
        if prompt_key in self._prompts:
            return self._prompts[prompt_key]
        # Find files inside folder app/core/prompts with given prompt_key
        prompt_file = "app/core/prompts/" + f"{prompt_key}.txt"
        if Path(prompt_file).exists():
            with open(prompt_file, "r") as f:
                return f.read()
        
        raise KeyError(f"Prompt key '{prompt_key}' not found")