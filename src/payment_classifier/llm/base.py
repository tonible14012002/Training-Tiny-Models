from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

from .settings import LLMSettings


class BaseLLM(ABC):
    """
    Abstract Base Class for Language Learning Models (LLMs).

    The `BaseLLM` class defines a standardized interface for interacting with various
    Language Learning Models. It provides methods for generating both plain text
    and structured outputs based on provided prompts. Subclasses must implement the
    abstract methods to define the specific behavior of the LLM.

    Attributes:
        settings (LLMSettings): Configuration settings for the LLM, including API keys,
                                 model names, and other parameters.

    Methods:
        generate_output(prompts: List[Dict[str, str]]) -> str:
            Generates plain text output based on the provided prompts.

        _generate_output(prompts: List[Dict[str, str]]) -> str:
            Abstract method to generate plain text output. Must be implemented by subclasses.

        generate_structured_output(prompts: List[Dict[str, str]]) -> Dict[str, Any]:
            Generates structured output (e.g., JSON) based on the provided prompts.

        _generate_structured_output(prompts: List[Dict[str, str]]) -> Dict[str, Any]:
            Abstract method to generate structured output. Must be implemented by subclasses.
    """

    def __init__(self, settings: LLMSettings) -> None:
        """
        Initializes the BaseLLM with the provided settings.

        Args:
            settings (LLMSettings): Configuration settings for the LLM.
        """
        self.settings: LLMSettings = settings

    def generate_output(self, prompts: List[Dict[str, str]]) -> str:
        """
        Generates plain text output based on the provided prompts.

        This method serves as a public interface that delegates the generation
        process to the `_generate_output` method, which must be implemented by
        subclasses.

        Args:
            prompts (List[Dict[str, str]]): A list of prompts, where each prompt
                                            is represented as a dictionary with string keys and values.

        Returns:
            str: The generated plain text output from the LLM.
        """
        return self._generate_output(prompts=prompts)

    @abstractmethod
    def _generate_output(self, prompts: List[Dict[str, str]]) -> str:
        """
        Abstract method to generate plain text output based on prompts.

        Subclasses must provide an implementation for this method to define
        how the LLM generates plain text output.

        Args:
            prompts (List[Dict[str, str]]): A list of prompts, where each prompt
                                            is represented as a dictionary with string keys and values.

        Returns:
            str: The generated plain text output from the LLM.
        """
        pass

    def generate_structured_output(self, prompts: List[Dict[str, str]], output_format: BaseModel, **options) -> Dict[str, Any]:
        """
        Generates structured output based on the provided prompts.

        This method serves as a public interface that delegates the generation
        process to the `_generate_structured_output` method, which must be
        implemented by subclasses.

        Args:
            prompts (List[Dict[str, str]]): A list of prompts, where each prompt
                                            is represented as a dictionary with string keys and values.

        Returns:
            Dict[str, Any]: The generated structured output (e.g., JSON) from the LLM.
        """
        return self._generate_structured_output(prompts=prompts, output_format=output_format, **options)

    @abstractmethod
    def _generate_structured_output(self, prompts: List[Dict[str, str]], output_format: BaseModel, **options) -> Dict[str, Any]:
        """
        Abstract method to generate structured output based on prompts.

        Subclasses must provide an implementation for this method to define
        how the LLM generates structured output.

        Args:
            prompts (List[Dict[str, str]]): A list of prompts, where each prompt
                                            is represented as a dictionary with string keys and values.

        Returns:
            Dict[str, Any]: The generated structured output (e.g., JSON) from the LLM.
        """
        pass
