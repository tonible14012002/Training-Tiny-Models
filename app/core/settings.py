import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    API_SECRET_KEY: str = Field(default="your-secret-key-here", description="API secret key for authentication")
    ENVIRONMENT: str = Field(default="dev", description="Environment (dev/staging/prod)")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, description="HuggingFace API token (only needed for training)")
    # TEACHER_MODEL: str = Field(default="gpt-4", description="Teacher model for data generation")
    # STUDENT_MODEL_PATH: str = Field(default="./models/student", description="Path to student model checkpoint")
    # VERIFIER_MODEL: str = Field(default="gpt-3.5-turbo", description="Verifier model for data validation")

    # Data Generation Parameters
    GENERATE_OPEN_INTENT_EVAL: bool = Field(default=True, description="Generate open intent samples during evaluation data generation")

    # Training Configuration

    # # Data Storage
    # DATA_POOL_PATH: str = Field(default="./data/data_pool", description="Path to store curated data")
    # DEV_SET_PATH: str = Field(default="./data/dev_set.json", description="Path to frozen dev set")
    # TEST_SET_PATH: str = Field(default="./data/test_set.json", description="Path to frozen test set")

    # # Monitoring & Logging
    # ENABLE_LOGGING: bool = Field(default=True, description="Enable detailed logging")
    # METRICS_OUTPUT_PATH: str = Field(default="./metrics", description="Path to store metrics")

    # # Human-in-the-Loop
    # HUMAN_REVIEW_FREQUENCY: int = Field(default=1000, description="Review every N examples")
    # ALERT_F1_DROP_THRESHOLD: float = Field(default=0.05, description="Alert if F1 drops by this much")
    # SCHEMA_ERROR_THRESHOLD: float = Field(default=0.05, description="Alert if schema errors exceed this rate")
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./test.db", description="Database connection URL")
    BASE_DIR: str = "."

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
os.environ["HF_HOME"] = "./.cache"