from pydantic import BaseModel
from typing import List, Literal
from enum import Enum

class Sample(BaseModel):
    msg: str
    label: Literal[
        "payment_intent",
        "payment_request",
        "smart_payment_system_command"
    ]


class Result(BaseModel):
    messages: List[Sample]


class PAYMENT_LABEL:
    PAYMENT_INTENT = "payment_intent"
    PAYMENT_REQUEST = "payment_request"
    PAYMENT_COMMAND = "smart_payment_system_command"