from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional

class ThresholdConfig(BaseModel):
    """Per-label probability thresholds for fallback logic"""
    thresholds: Dict[str, float] = Field(
        description="Label to minimum probability threshold mapping"
    )
    fallback_label: str = Field(
        default="Unknown",
        description="Label to use when probability is below threshold"
    )

class InferenceRequest(BaseModel):
    text: List[str]