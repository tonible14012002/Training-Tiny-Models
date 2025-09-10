
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = None

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")