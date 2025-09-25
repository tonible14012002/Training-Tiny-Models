from typing import Dict
from pathlib import Path
import glob
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
            prompt_key: The key identifying the prompt to retrieve.
                       Can be just filename (e.g., "seed.txt") or path (e.g., "eval/seed.txt")

        Returns:
            The prompt string associated with the given key

        Raises:
            KeyError: If the prompt_key is not found
        """
        if prompt_key in self._prompts:
            return self._prompts[prompt_key]

        # Auto-detect the full path structure
        prompt_file = self._find_prompt_file(prompt_key)
        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file, "r") as f:
                return f.read()

        raise KeyError(f"Prompt key '{prompt_key}' not found")

    def _find_prompt_file(self, prompt_key: str) -> str:
        """
        Find the full path to a prompt file by searching the prompts directory.

        Args:
            prompt_key: The prompt key, which can be:
                       - Just a filename: "seed.txt"
                       - A path with folder: "eval/seed.txt"

        Returns:
            The full path to the prompt file, or None if not found
        """
        base_dir = "app/core/prompts"

        # If prompt_key already contains a path separator, try it directly
        if "/" in prompt_key:
            direct_path = f"{base_dir}/{prompt_key}"
            if Path(direct_path).exists():
                return direct_path

        # If prompt_key doesn't end with .txt, add it
        if not prompt_key.endswith('.txt'):
            prompt_key = f"{prompt_key}.txt"

        # Search recursively for the file
        pattern = f"{base_dir}/**/{prompt_key}"
        matches = glob.glob(pattern, recursive=True)

        if matches:
            # Return the first match
            return matches[0]

        # Fallback: try direct path in base directory
        fallback_path = f"{base_dir}/{prompt_key}"
        if Path(fallback_path).exists():
            return fallback_path

        return None