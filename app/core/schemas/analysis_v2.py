from pydantic import BaseModel, Field
from .workflow import Sample
import typing as t

class ErrorBucket(BaseModel):
    name: str
    description: str
    count: int
    examples: list[Sample]
    data_generation_strategy: t.Optional[str] = None
    priority: float = Field(description="Priority weight based on error frequency", default=1.0)

class LLmErrorAnalysis(BaseModel):
    error_buckets: t.List[str] = Field(description="List of error bucket names", default=[])
