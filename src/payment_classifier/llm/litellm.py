import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import instructor
import litellm
from litellm.exceptions import RateLimitError
from pydantic import BaseModel

from .base import BaseLLM
from .settings import LLMSettings

logger = logging.getLogger(__name__)


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
        self._rate_limit_retry_delay = settings.rate_limit_retry_delay
        self._max_rate_limit_retries = settings.max_rate_limit_retries
        self._batch_delay = settings.batch_delay

    async def _retry_with_rate_limit_handling(self, func, *args, **kwargs):
        """Wrapper to handle rate limit errors with exponential backoff.

        Args:
            func: The async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result from the function call

        Raises:
            RateLimitError: If max retries exceeded
        """
        for attempt in range(self._max_rate_limit_retries):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == self._max_rate_limit_retries - 1:
                    logger.error(f"Rate limit retry exhausted after {self._max_rate_limit_retries} attempts")
                    raise

                # Exponential backoff: 5s, 10s, 20s, 40s, 80s
                wait_time = self._rate_limit_retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit: {str(e)}")
                logger.info(f"Waiting {wait_time}s before retry (attempt {attempt + 1}/{self._max_rate_limit_retries})...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                # Check if the error message contains rate limit information (some APIs wrap it differently)
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "ratelimit" in error_msg:
                    if attempt == self._max_rate_limit_retries - 1:
                        logger.error(f"Rate limit retry exhausted after {self._max_rate_limit_retries} attempts")
                        raise

                    # Exponential backoff: 5s, 10s, 20s, 40s, 80s
                    wait_time = self._rate_limit_retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit detected in error: {str(e)}")
                    logger.info(f"Waiting {wait_time}s before retry (attempt {attempt + 1}/{self._max_rate_limit_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    # Re-raise other exceptions immediately
                    raise

    async def _generate_output(self, prompts: List[Dict[str, str]]) -> str:
        """Generate a text completion using LiteLLM.

        Args:
            prompts (List[Dict[str, str]]): List of message dictionaries containing
                the conversation history. Each dictionary should have 'role' and 'content' keys.

        Returns:
            str: The generated text response from the model.
        """
        async def _call_api():
            return await litellm.acompletion(
                model=self.settings.llm_model_name,
                messages=prompts,
                num_retries=0,  # Disable LiteLLM's internal retry - we handle it ourselves
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                **self.settings.additional_params,
            )

        response = await self._retry_with_rate_limit_handling(_call_api)
        return response.choices[0].message.content
    
    async def _generate_structured_output_with_json_object(self, prompts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a structured output using LiteLLM with Instructor integration.
        """
        async def _call_api():
            return await litellm.acompletion(
                model=self.settings.llm_model_name,
                messages=prompts,
                response_format={"type": "json_object"},
                **self.settings.additional_params,
            )

        response = await self._retry_with_rate_limit_handling(_call_api)
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

        if output_format is None:
            return await self._generate_structured_output_with_json_object(prompts)

        instructor_client = instructor.from_litellm(completion=litellm.acompletion)

        async def _call_api():
            return await instructor_client.chat.completions.create(
                model=self.settings.llm_model_name,
                messages=prompts,
                **options,
                **self.settings.additional_params,
                response_model=output_format,
            )

        response = await self._retry_with_rate_limit_handling(_call_api)
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
            Adds delay between batches to avoid rate limits.
        """

        # Split prompt_list into batches of 8
        batch_size = 8
        batches = [prompt_list[i:i + batch_size] for i in range(0, len(prompt_list), batch_size)]

        all_results = []

        # Process each batch concurrently
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} items...")

            # Create tasks for current batch
            tasks = [
                self._generate_structured_output(prompts, output_format)
                for prompts in batch
            ]

            # Execute batch concurrently with rate limit handling
            # Note: _generate_structured_output already has retry logic via _retry_with_rate_limit_handling
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for any exceptions in the results
            errors = [r for r in batch_results if isinstance(r, Exception)]
            if errors:
                # Check if any are rate limit errors
                rate_limit_errors = [e for e in errors if isinstance(e, RateLimitError) or "rate limit" in str(e).lower()]
                if rate_limit_errors:
                    logger.warning(f"Rate limit errors in batch {batch_idx + 1}: {len(rate_limit_errors)}/{len(batch_results)}")
                    # Re-raise the first rate limit error (retry logic already exhausted)
                    raise rate_limit_errors[0]
                else:
                    # Re-raise the first non-rate-limit error
                    raise errors[0]

            all_results.extend(batch_results)

            # Add delay between batches to avoid hitting rate limits
            if batch_idx < len(batches) - 1:
                logger.debug(f"Waiting {self._batch_delay}s before next batch...")
                await asyncio.sleep(self._batch_delay)

        return all_results
