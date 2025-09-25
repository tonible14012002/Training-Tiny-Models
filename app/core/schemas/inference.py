from pydantic import BaseModel
from typing import List, Literal

class InferenceRequest(BaseModel):
    text: List[str]