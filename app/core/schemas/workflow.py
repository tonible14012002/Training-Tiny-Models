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
    PAYMENT_INTENT = 0
    PAYMENT_REQUEST = 1
    PAYMENT_COMMAND = 2

    @staticmethod
    def to_dict():
        return {
            "payment_intent": PAYMENT_LABEL.PAYMENT_INTENT,
            "payment_request": PAYMENT_LABEL.PAYMENT_REQUEST,
            "smart_payment_system_command": PAYMENT_LABEL.PAYMENT_COMMAND,
        }

    @staticmethod
    def to_id2label():
        return {
            PAYMENT_LABEL.PAYMENT_INTENT: "payment_intent",
            PAYMENT_LABEL.PAYMENT_REQUEST: "payment_request",
            PAYMENT_LABEL.PAYMENT_COMMAND: "smart_payment_system_command",
        }

    @staticmethod
    def from_str(label: str) -> int:
        if label == "payment_intent":
            return PAYMENT_LABEL.PAYMENT_INTENT
        elif label == "payment_request":
            return PAYMENT_LABEL.PAYMENT_REQUEST
        elif label == "smart_payment_system_command":
            return PAYMENT_LABEL.PAYMENT_COMMAND
        else:
            raise ValueError(f"Unknown label: {label}")
    
    @staticmethod
    def to_str(label: int) -> str:
        if label == PAYMENT_LABEL.PAYMENT_INTENT:
            return "payment_intent"
        elif label == PAYMENT_LABEL.PAYMENT_REQUEST:
            return "payment_request"
        elif label == PAYMENT_LABEL.PAYMENT_COMMAND:
            return "smart_payment_system_command"
        else:
            raise ValueError(f"Unknown label: {label}")