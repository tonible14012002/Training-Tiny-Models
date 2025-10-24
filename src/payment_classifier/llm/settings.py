from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

__all__ = ["LLMSettings"]


class LLMSettings(BaseModel):
    """
    Configuration settings for a Language Learning Model (LLM).

    Attributes:
        api_key (str): API Key for accessing the LLM. Default is an empty string.
        llm_model_name (str): The name of the LLM model to be used. Default is an empty string.
        temperature (float): The temperature value for controlling the randomness of the LLM's output.
                             A higher value results in more random outputs. Default is None.
        max_tokens (int): The maximum number of tokens to generate in the LLM's response. Default is None.
        additional_params (Optional[Dict[str, Any]]): A dictionary of additional parameters that can be
                                                      passed to the LLM. Default is an empty dictionary.
    """

    api_key: str = Field(default="", description="API Key for LLM")
    llm_model_name: str = Field(default="", description="LLM's model name")
    temperature: float = Field(default=0.15, description="Temperature value of LLM")
    max_tokens: int = Field(default=4096, description="Maximum number of tokens for generation")
    num_retries: int = Field(default=5, description="Number of retries")
    rate_limit_retry_delay: int = Field(default=5, description="Base seconds to wait when rate limit is hit (exponential backoff: 5s, 10s, 20s, 40s, 80s)")
    max_rate_limit_retries: int = Field(default=5, description="Maximum number of rate limit retries")
    batch_delay: int = Field(default=10, description="Seconds to wait between batches")
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
