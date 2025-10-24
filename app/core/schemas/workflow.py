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
    training_argument_profile_id: Optional[str] = Field(
        default=None,
        description="Optional ID of the training argument profile to use for training"
    )

class StartContinualTrainPhase(BaseModel):
    trained_model_id: str = Field(
        ...,
        description="The ID of the previous trained model to continue training from"
    )
    training_argument_profile_id: Optional[str] = Field(
        default=None,
        description="Optional ID of the training argument profile to use for training"
    )

class StartEvaluationPhase(BaseModel):
    trained_model_id: str = Field(
        ...,
        description="The ID of the trained model to evaluate"
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

class TrainingConfig(BaseModel):
    """Schema for training arguments configuration"""
    learning_rate: Optional[float] = Field(
        default=2e-5,
        description="Learning rate for training"
    )
    per_device_train_batch_size: Optional[int] = Field(
        default=8,
        description="Batch size per device during training"
    )
    per_device_eval_batch_size: Optional[int] = Field(
        default=16,
        description="Batch size per device during evaluation"
    )
    gradient_accumulation_steps: Optional[int] = Field(
        default=4,
        description="Number of gradient accumulation steps"
    )
    num_train_epochs: Optional[int] = Field(
        default=3,
        description="Total number of training epochs"
    )
    warmup_ratio: Optional[float] = Field(
        default=0.15,
        description="Ratio of total training steps for warmup"
    )
    weight_decay: Optional[float] = Field(
        default=0.01,
        description="Weight decay for regularization"
    )
    max_grad_norm: Optional[float] = Field(
        default=1.0,
        description="Maximum gradient norm for clipping"
    )
    logging_steps: Optional[int] = Field(
        default=10,
        description="Number of steps between logging"
    )
    save_steps: Optional[int] = Field(
        default=500,
        description="Number of steps between checkpoints"
    )
    eval_steps: Optional[int] = Field(
        default=100,
        description="Number of steps between evaluations"
    )
    save_strategy: Optional[str] = Field(
        default="steps",
        description="Checkpoint save strategy (steps, epoch, no)"
    )
    eval_strategy: Optional[str] = Field(
        default="no",
        description="Evaluation strategy (steps, epoch, no)"
    )
    load_best_model_at_end: Optional[bool] = Field(
        default=False,
        description="Whether to load the best model at the end of training"
    )
    metric_for_best_model: Optional[str] = Field(
        default=None,
        description="Metric to use for determining best model"
    )
    greater_is_better: Optional[bool] = Field(
        default=True,
        description="Whether higher metric value is better"
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    class Config:
        extra = "allow"  # Allow additional fields for flexibility

class LoRAConfig(BaseModel):
    """Schema for LoRA configuration

    Must match the structure in TrainerService.lora_config
    """
    r: int = Field(
        default=16,
        description="LoRA rank",
        ge=1,
        le=256
    )
    lora_alpha: int = Field(
        default=32,
        description="LoRA alpha parameter",
        ge=1,
        le=256
    )
    lora_dropout: float = Field(
        default=0.1,
        description="LoRA dropout rate",
        ge=0.0,
        le=1.0
    )
    bias: str = Field(
        default="none",
        description="Bias configuration (none, all, lora_only)"
    )
    target_modules: List[str] = Field(
        default=["query", "value", "dense"],
        description="Target modules for LoRA adaptation"
    )

    @classmethod
    def model_validate(cls, value):
        """Validate LoRA config matches TrainerService expectations"""
        if isinstance(value, dict):
            # Validate bias field
            if "bias" in value and value["bias"] not in ["none", "all", "lora_only"]:
                raise ValueError(f"Invalid bias value: {value['bias']}. Must be one of: none, all, lora_only")

            # Validate target_modules is a list
            if "target_modules" in value and not isinstance(value["target_modules"], list):
                raise ValueError("target_modules must be a list of strings")

        return super().model_validate(value)

class CreateTrainingArgumentProfileRequest(BaseModel):
    """Request schema for creating a training argument profile"""
    name: str = Field(
        ...,
        description="Unique name for the profile",
        examples=["default", "fast-training", "high-accuracy"]
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the profile"
    )
    training_config: TrainingConfig = Field(
        ...,
        description="Training configuration parameters"
    )
    lora_config: LoRAConfig = Field(
        ...,
        description="LoRA configuration parameters"
    )

class UpdateTrainingArgumentProfileRequest(BaseModel):
    """Request schema for updating a training argument profile"""
    name: Optional[str] = Field(
        default=None,
        description="New name for the profile"
    )
    description: Optional[str] = Field(
        default=None,
        description="New description"
    )
    training_config: Optional[TrainingConfig] = Field(
        default=None,
        description="New training configuration"
    )
    lora_config: Optional[LoRAConfig] = Field(
        default=None,
        description="New LoRA configuration"
    )

class TrainingArgumentProfileResponse(BaseModel):
    """Response schema for training argument profile"""
    id: str
    name: str
    description: Optional[str]
    training_config: Dict[str, Any]
    lora_config: Dict[str, Any]
    created_at: str
    updated_at: str

class ModelInferenceRequest(BaseModel):
    """Request schema for model inference"""
    model_path: str = Field(
        ...,
        description="Path to the trained model checkpoint"
    )
    texts: List[str] = Field(
        ...,
        description="List of texts to classify",
        min_length=1
    )
    pipeline_id: str = Field(
        ...,
        description="Pipeline ID to get label configuration"
    )

class PredictionResult(BaseModel):
    """Single prediction result"""
    text: str
    label: str
    probability: float
    all_probabilities: Dict[str, float]

class ModelInferenceResponse(BaseModel):
    """Response schema for model inference"""
    message: str
    predictions: List[PredictionResult]
    model_info: Dict[str, Any]