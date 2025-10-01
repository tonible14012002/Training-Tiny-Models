from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class Sample(BaseModel):
    msg: str
    label: Union[str, int]  # Made flexible to support different label types

class Result(BaseModel):
    messages: List[Sample]

class BaseLabelConfig:
    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def to_dict():
        raise NotImplementedError

    @staticmethod
    def to_id2label():
        raise NotImplementedError

    @staticmethod
    def from_str(label: str) -> int:
        raise NotImplementedError

    @staticmethod
    def to_str(label: int) -> str:
        raise NotImplementedError

    @staticmethod
    def get_label_explanation() -> dict:
        raise NotImplementedError

class PAYMENT_LABEL(BaseLabelConfig):
    PAYMENT_INTENT = 0
    PAYMENT_REQUEST = 1
    PAYMENT_COMMAND = 2

    @staticmethod
    def name() -> str:
        return "Payment Classification v1"

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

    @staticmethod
    def get_label_explanation() -> dict:
        return {
            "payment_intent": "User intends to send/pay money to someone",
            "payment_request": "User asking someone to send them money",
            "smart_payment_system_command": "User instructing a system to make a payment"
        }

class PAYMENT_LABEL_V2(BaseLabelConfig):
    PAYMENT_INTENT = 1
    PAYMENT_REQUEST = 0
    OPEN_INTENT = 2

    @staticmethod
    def name() -> str:
        return "Payment Classification v2"

    @staticmethod
    def to_dict():
        return {
            "payment_intent": PAYMENT_LABEL_V2.PAYMENT_INTENT,
            "payment_request": PAYMENT_LABEL_V2.PAYMENT_REQUEST,
            "open_intent": PAYMENT_LABEL_V2.OPEN_INTENT,
        }

    @staticmethod
    def to_id2label():
        return {
            PAYMENT_LABEL_V2.PAYMENT_INTENT: "payment_intent",
            PAYMENT_LABEL_V2.PAYMENT_REQUEST: "payment_request",
            PAYMENT_LABEL_V2.OPEN_INTENT: "open_intent",
        }

    @staticmethod
    def from_str(label: str) -> int:
        if label == "payment_intent":
            return PAYMENT_LABEL_V2.PAYMENT_INTENT
        elif label == "payment_request":
            return PAYMENT_LABEL_V2.PAYMENT_REQUEST
        elif label == "open_intent":
            return PAYMENT_LABEL_V2.OPEN_INTENT
        else:
            raise ValueError(f"Unknown label: {label}")

    @staticmethod
    def to_str(label: int) -> str:
        if label == PAYMENT_LABEL_V2.PAYMENT_INTENT:
            return "payment_intent"
        elif label == PAYMENT_LABEL_V2.PAYMENT_REQUEST:
            return "payment_request"
        elif label == PAYMENT_LABEL_V2.OPEN_INTENT:
            return "open_intent"
        else:
            raise ValueError(f"Unknown label: {label}")

    @staticmethod
    def get_label_explanation() -> dict:
        return {
            "payment_intent": "The user is declaring they will send money right now or in near future OR The user gives an imperative instruction to a system to execute a payment",
            "payment_request": "The user request to receive money (can be request, inform, force, remind, ...)",
            "open_intent": "All arbitrary chat messages that are not related to any payment intent"
        }

class EvaluationRequest(BaseModel):
    iteration_number: Optional[int] = None
    include_test_cases: bool = False
    include_open_intent: bool = True
    checkpoint_id: Optional[str] = None  # e.g., "1", "2", "1.1", "2.3"

class EvaluationResponse(BaseModel):
    message: str
    status: str
    checkpoint_path: str
    checkpoint_id: Optional[str] = None  # e.g., "1", "2", "1.1", "2.3"
    evaluation_data_info: Dict[str, Any]
    results: Dict[str, Any]

class FixGenRequest(BaseModel):
    prompt: str
    amount: Optional[int] = None

class AnalyzeErrorPatternsRequest(BaseModel):
    checkpoint_id: Optional[str] = None  # e.g., "1", "2", "1.1", "2.3"
    iteration_number: Optional[int] = None