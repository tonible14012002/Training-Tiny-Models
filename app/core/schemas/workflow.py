from pydantic import BaseModel, Field
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
    threshold_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Threshold configuration for predictions",
        json_schema_extra={
            "example": {
                "thresholds": {
                    "payment_intent": 0.75,
                    "payment_request": 0.70,
                    "open_intent": 0.60
                },
                "fallback_label": "Unknown"
            }
        }
    )

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

class OrchestratorRunRequest(BaseModel):
    """Request schema for orchestrator run endpoint"""
    initial_checkpoint_id: str = Field(
        default="11.7",
        description="Starting checkpoint ID (e.g., '11.7', '11', '12.5')"
    )
    max_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of training iterations to prevent infinite loops"
    )
    target_f1_per_label: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Target F1 score that all labels must achieve for convergence"
    )
    samples_per_action: int = Field(
        default=500,
        ge=10,
        le=2000,
        description="Number of samples to generate per data generation action"
    )
    iteration_number: Optional[int] = Field(
        default=None,
        description="Evaluation dataset iteration number to use. If None, uses latest."
    )

class LabelConfigRequest(BaseModel):
    """Request schema for label configuration"""
    name: str = Field(
        ...,
        description="Name of the label configuration",
        examples=["Payment Classification v2"]
    )
    id2label: Dict[str, str] = Field(
        ...,
        description="Mapping from ID to label string",
        examples=[{
            "0": "payment_request",
            "1": "payment_intent",
            "2": "open_intent"
        }]
    )
    label2id: Dict[str, int] = Field(
        ...,
        description="Mapping from label string to ID",
        examples=[{
            "payment_request": 0,
            "payment_intent": 1,
            "open_intent": 2
        }]
    )
    label_explanation: Optional[Dict[str, str]] = Field(
        default=None,
        description="Explanation for each label",
        examples=[{
            "payment_intent": "The user is declaring they will send money right now or in near future OR The user gives an imperative instruction to a system to execute a payment",
            "payment_request": "The user request to receive money (can be request, inform, force, remind, ...)",
            "open_intent": "All arbitrary chat messages that are not related to any payment intent"
        }]
    )

class CreatePipelineRequest(BaseModel):
    """Request schema for creating a new pipeline"""
    name: str = Field(
        ...,
        description="Name of the pipeline",
        examples=["Payment Classification Pipeline"]
    )
    label_config: LabelConfigRequest = Field(
        ...,
        description="Label configuration for the pipeline"
    )

class LabelConfigResponse(BaseModel):
    """Response schema for label configuration"""
    id: str
    name: str
    id2label: Dict[str, str]
    label2id: Dict[str, int]
    label_explanation: Optional[Dict[str, str]]
    created_at: str

class PipelineResponse(BaseModel):
    """Response schema for pipeline"""
    id: str
    name: str
    created_at: str
    updated_at: str
    label_config: Optional[LabelConfigResponse] = None

class ListPipelinesResponse(BaseModel):
    """Response schema for listing pipelines"""
    message: str
    data: List[PipelineResponse]


class ClassifyErrorRequest(BaseModel):
    phase_id: Optional[str] = Field(
        None,
        description="The ID of the phase within the pipeline"
    )

class StartPipelineRequest(BaseModel):
    pipeline_id: str = Field(
        ...,
        description="The ID of the pipeline to start a new phase for"
    )
class TestTrainPhaseRequest(BaseModel):
    phase_id: str = Field(
        ...,
        description="The ID of the phase to test training"
    )
    ds_file_path: str = Field(
        ...,
        description="Path to the dataset file for training"
    )
    checkpoint_path: str = Field(
        default=".checkpoints",
        description="Base path for saving model checkpoints"
    )
    cache_path: str = Field(
        default=".cache",
        description="Base path for caching data"
    )

class StartTrainPhase(BaseModel):
    phase_id: str = Field(
        ...,
        description="The ID of the phase to train"
    )

class StartEvaluationPhase(BaseModel):
    phase_id: str = Field(
        ...,
        description="The ID of the phase to evaluate"
    )
    confidence_thresholds: float = Field(
        0.5,
        description="Confidence threshold for filter low confidence predictions"
    )

class StartErrBucketPhaese(BaseModel):
    phase_id: str = Field(
        ...,
        description="The ID of the phase to classify error buckets"
    )

class TestEvaluationRequest(BaseModel):
    model_path: str = Field(
        ...,
        description="Path to the trained model checkpoint"
    )
    pipeline_id: str = Field(
        ...,
        description="The ID of the pipeline (used to get label configuration)"
    )
    cache_path: str = Field(
        default=".cache",
        description="Base path for caching data"
    )

class TestFirstGenRequest(BaseModel):
    pipeline_id: str = Field(
        ...,
        description="The ID of the pipeline (used to get label configuration)"
    )
    cache_path: str = Field(
        default=".cache",
        description="Base path for caching data and output"
    )

class PHASE_STATUS:
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class DATASET_FILE_STATUS:
    GENERATING = "generating"
    DONE = "done"