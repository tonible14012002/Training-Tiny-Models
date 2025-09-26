from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class IterationMetrics(BaseModel):
    """Metrics for a single training iteration"""
    iteration: int
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy (0-1)")
    macro_f1: float = Field(..., ge=0.0, le=1.0, description="Macro F1-score (0-1)")
    coverage: float = Field(..., ge=0.0, le=1.0, description="Coverage rate (0-1)")
    unknown_rate: float = Field(..., ge=0.0, le=1.0, description="Unknown prediction rate (0-1)")
    total_samples: int = Field(..., gt=0, description="Total number of evaluation samples")
    checkpoint_path: str = Field(..., description="Path to the model checkpoint")
    timestamp: float = Field(..., description="Unix timestamp when iteration completed")
    training_time: float = Field(..., ge=0.0, description="Training time in seconds")
    evaluation_time: float = Field(..., ge=0.0, description="Evaluation time in seconds")

    class Config:
        json_encoders = {
            datetime: lambda v: v.timestamp()
        }


class PipelineConfig(BaseModel):
    """Configuration for the auto-training pipeline"""
    max_iterations: int = Field(default=10, gt=0, le=100, description="Maximum number of training iterations")
    target_accuracy: float = Field(default=0.85, ge=0.0, le=1.0, description="Target accuracy to achieve")
    target_macro_f1: float = Field(default=0.80, ge=0.0, le=1.0, description="Target macro F1-score to achieve")
    early_termination_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Minimum improvement threshold for early termination"
    )
    min_improvement_iterations: int = Field(
        default=2,
        gt=0,
        le=10,
        description="Number of iterations to check for improvement"
    )
    data_generation_batch_size: int = Field(
        default=15,
        gt=0,
        le=100,
        description="Number of samples to generate per iteration"
    )


class PipelineStatus(BaseModel):
    """Current status of the training pipeline"""
    is_running: bool = Field(default=False, description="Whether pipeline is currently running")
    current_iteration: int = Field(default=0, ge=0, description="Current iteration number")
    total_iterations: int = Field(default=0, ge=0, description="Total planned iterations")
    last_metrics: Optional[IterationMetrics] = Field(default=None, description="Metrics from last completed iteration")
    best_metrics: Optional[IterationMetrics] = Field(default=None, description="Best metrics achieved so far")
    termination_reason: Optional[str] = Field(default=None, description="Reason why pipeline terminated")
    start_time: Optional[float] = Field(default=None, description="Unix timestamp when pipeline started")
    metrics_history: List[IterationMetrics] = Field(default_factory=list, description="History of all iteration metrics")

    class Config:
        json_encoders = {
            datetime: lambda v: v.timestamp()
        }


class PipelineResult(BaseModel):
    """Result of a completed training pipeline"""
    success: bool = Field(..., description="Whether pipeline completed successfully")
    status: str = Field(..., description="Pipeline completion status")
    termination_reason: str = Field(..., description="Reason for pipeline termination")
    total_iterations: int = Field(..., ge=0, description="Total iterations completed")
    total_time: float = Field(..., ge=0.0, description="Total pipeline execution time in seconds")
    best_metrics: Optional[dict] = Field(default=None, description="Best metrics achieved during pipeline")
    final_metrics: Optional[dict] = Field(default=None, description="Final metrics from last iteration")
    metrics_history: List[dict] = Field(default_factory=list, description="Complete history of iteration metrics")
    error: Optional[str] = Field(default=None, description="Error message if pipeline failed")


class PipelineStatusResponse(BaseModel):
    """Response for pipeline status endpoint"""
    message: str = Field(..., description="Status message")
    status: str = Field(..., description="Response status")
    is_running: bool = Field(..., description="Whether pipeline is currently running")
    current_iteration: int = Field(..., ge=0, description="Current iteration number")
    total_iterations: int = Field(..., ge=0, description="Total planned iterations")
    termination_reason: Optional[str] = Field(default=None, description="Termination reason if stopped")
    last_metrics: Optional[dict] = Field(default=None, description="Last iteration metrics")
    best_metrics: Optional[dict] = Field(default=None, description="Best metrics achieved")
    metrics_history: List[dict] = Field(default_factory=list, description="Metrics history")


class PipelineActionResponse(BaseModel):
    """Response for pipeline control actions (start, stop, reset)"""
    success: bool = Field(..., description="Whether action was successful")
    message: str = Field(..., description="Action result message")
    iterations_completed: Optional[int] = Field(default=None, ge=0, description="Number of iterations completed")
    error: Optional[str] = Field(default=None, description="Error message if action failed")