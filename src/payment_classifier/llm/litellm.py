import asyncio
import json
from typing import Any, Dict, List

import instructor
import litellm
from pydantic import BaseModel

from .base import BaseLLM
from .settings import LLMSettings


class LiteLLMProvider(BaseLLM):
    """LiteLLM implementation of the BaseLLM interface.

    This class provides integration with various LLM providers through the LiteLLM library,
    supporting standard chat completions, structured outputs, and batch processing capabilities.

    Args:
        settings (LLMSettings): Configuration settings for the LLM, including API keys,
            model names, and other parameters.
    """

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings=settings)
        litellm.api_key = self.settings.api_key

    async def _generate_output(self, prompts: List[Dict[str, str]]) -> str:
        """Generate a text completion using LiteLLM.

        Args:
            prompts (List[Dict[str, str]]): List of message dictionaries containing
                the conversation history. Each dictionary should have 'role' and 'content' keys.

        Returns:
            str: The generated text response from the model.
        """

        response = await litellm.acompletion(
            model=self.settings.llm_model_name,
            messages=prompts,
            num_retries=self.settings.num_retries,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            **self.settings.additional_params,
        )

        return response.choices[0].message.content
    
    async def _generate_structured_output_with_json_object(self, prompts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a structured output using LiteLLM with Instructor integration.
        """
        response = await litellm.acompletion(
            model=self.settings.llm_model_name,
            messages=prompts,
            response_format={"type": "json_object"},
            **self.settings.additional_params,
        )

        return json.loads(response.choices[0].message.content)

    async def _generate_structured_output(
        self, prompts: List[Dict[str, str]], output_format: BaseModel, **options
    ) -> Dict[str, Any]:
        """Generate a structured output using LiteLLM with Instructor integration.

        Args:
            prompts (List[Dict[str, str]]): List of message dictionaries containing
                the conversation history. Each dictionary should have 'role' and 'content' keys.
            output_format (BaseModel): A Pydantic model defining the expected response structure.

        Returns:
            Dict[str, Any]: The structured response matching the provided output_format.
        """
        valid_options = ["num_retries", "temperature", "max_tokens"]
        for op in options.keys():
            assert op in valid_options, "Invalid option provided"

        for op in valid_options:
            options.setdefault(op, getattr(self.settings, op))
        
        print("USE TEmp", options["temperature"])

        if output_format is None:
            return await self._generate_structured_output_with_json_object(prompts)

        instructor_client = instructor.from_litellm(completion=litellm.acompletion)

        response = await instructor_client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=prompts,
            **options,
            **self.settings.additional_params,
            response_model=output_format,
        )

        return response
    
    async def generate_batch_output(
        self, 
        prompt_list: List[List[Dict[str, str]]], 
        output_format: BaseModel
    ) -> List[Dict[str, Any]]:
        """Process multiple prompts in batch using async gather with structured output.

        Args:
            prompt_list (List[List[Dict[str, str]]]): A list of prompt lists, where each inner
                list contains message dictionaries for a separate conversation.
            output_format (BaseModel): A Pydantic model defining the expected response structure.

        Returns:
            List[Dict[str, Any]]: List of structured responses matching the provided output_format.

        Note:
            This method processes prompts in batches of 8 using asyncio.gather for
            parallel processing to improve throughput while managing resource usage.
        """
        
        # Split prompt_list into batches of 8
        batch_size = 8
        batches = [prompt_list[i:i + batch_size] for i in range(0, len(prompt_list), batch_size)]
        
        all_results = []
        
        # Process each batch concurrently
        for batch in batches:
            # Create tasks for current batch
            tasks = [
                self._generate_structured_output(prompts, output_format) 
                for prompts in batch
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
        
        return all_results
