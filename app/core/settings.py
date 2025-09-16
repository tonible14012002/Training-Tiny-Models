import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    API_SECRET_KEY: str = Field(default="your-secret-key-here", description="API secret key for authentication")
    ENVIRONMENT: str = Field(default="dev", description="Environment (dev/staging/prod)")

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, description="HuggingFace API token (only needed for training)")
    TEACHER_MODEL: str = Field(default="gpt-4", description="Teacher model for data generation")
    STUDENT_MODEL_PATH: str = Field(default="./models/student", description="Path to student model checkpoint")
    VERIFIER_MODEL: str = Field(default="gpt-3.5-turbo", description="Verifier model for data validation")

    # Data Generation Parameters
    GENERATION_TEMPERATURE: float = Field(default=0.7, description="Temperature for data generation")
    GENERATION_TOP_P: float = Field(default=1.0, description="Top-p for data generation")
    MAX_EXAMPLES_PER_LOOP: int = Field(default=1000, description="Maximum examples to generate per loop")
    DEDUPLICATION_THRESHOLD: float = Field(default=0.85, description="Cosine similarity threshold for deduplication")

    # Training Configuration
    MAX_TRAINING_LOOPS: int = Field(default=10, description="Maximum number of training loops")
    EARLY_STOP_PATIENCE: int = Field(default=3, description="Early stopping patience")
    MIN_F1_IMPROVEMENT: float = Field(default=0.001, description="Minimum F1 improvement threshold")
    BATCH_SIZE: int = Field(default=16, description="Training batch size")
    LEARNING_RATE: float = Field(default=2e-5, description="Learning rate")

    # Data Storage
    DATA_POOL_PATH: str = Field(default="./data/data_pool", description="Path to store curated data")
    DEV_SET_PATH: str = Field(default="./data/dev_set.json", description="Path to frozen dev set")
    TEST_SET_PATH: str = Field(default="./data/test_set.json", description="Path to frozen test set")

    # Monitoring & Logging
    ENABLE_LOGGING: bool = Field(default=True, description="Enable detailed logging")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    METRICS_OUTPUT_PATH: str = Field(default="./metrics", description="Path to store metrics")

    # Human-in-the-Loop
    HUMAN_REVIEW_FREQUENCY: int = Field(default=1000, description="Review every N examples")
    ALERT_F1_DROP_THRESHOLD: float = Field(default=0.05, description="Alert if F1 drops by this much")
    SCHEMA_ERROR_THRESHOLD: float = Field(default=0.05, description="Alert if schema errors exceed this rate")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()