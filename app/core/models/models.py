from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime, timezone
from uuid import uuid4
import json


def utc_now() -> datetime:
    """Get current UTC time with timezone awareness"""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate a UUID string for SQLite"""
    return str(uuid4())


class Pipeline(SQLModel, table=True):
    """Pipeline table for managing training pipelines"""
    __tablename__ = "pipeline"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    name: str = Field(max_length=255)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    label_configs: List["LabelConfig"] = Relationship(back_populates="pipeline")
    phases: List["PipelinePhase"] = Relationship(back_populates="pipeline")
    datasets: List["ComposalDataset"] = Relationship(back_populates="pipeline")
    error_buckets: List["ErrorBucket"] = Relationship(back_populates="pipeline")


class LabelConfig(SQLModel, table=True):
    """Label configuration table for storing label mappings"""
    __tablename__ = "label_config"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    pipeline_id: str = Field(foreign_key="pipeline.id")
    name: str = Field(max_length=255)
    id2label: str = Field()  # JSON string: {"0": "payment_request", "1": "payment_intent", "2": "open_intent"}
    label2id: str = Field()  # JSON string: {"payment_request": 0, "payment_intent": 1, "open_intent": 2}
    label_explanation: Optional[str] = Field(default=None)  # JSON string
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    pipeline: Pipeline = Relationship(back_populates="label_configs")

    def get_id2label(self) -> dict:
        """Parse id2label JSON string to dict"""
        return json.loads(self.id2label)

    def get_label2id(self) -> dict:
        """Parse label2id JSON string to dict"""
        return json.loads(self.label2id)

    def get_label_explanation(self) -> Optional[dict]:
        """Parse label_explanation JSON string to dict"""
        return json.loads(self.label_explanation) if self.label_explanation else None


class PipelinePhase(SQLModel, table=True):
    """Pipeline phase table for tracking training phases

    The phase_path column stores the hierarchical path of parent phases:
    - Empty string '' for base phases (phase_number = 0)
    - 'parent_id' for first generation children
    - 'parent1_id/parent2_id' for deeper nested phases

    This allows grouping phases by their root phase (first segment) and
    tracking order without storing a separate order column.
    """
    __tablename__ = "pipeline_phase"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    pipeline_id: str = Field(foreign_key="pipeline.id")
    phase_path: str = Field(default="")  # Hierarchical path: "" for root, "parent_id" or "p1_id/p2_id" for children
    phase_number: int
    checkpoint_id: Optional[str] = Field(default=None, max_length=100)
    checkpoint_path: Optional[str] = Field(default=None)
    status: str = Field(default="pending", max_length=50)  # pending, in_progress, completed, failed
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = Field(default=None)

    # Relationships
    pipeline: Pipeline = Relationship(back_populates="phases")
    composal_datasets: List["ComposalDataset"] = Relationship(back_populates="phase")
    dataset_files: List["DatasetFile"] = Relationship(back_populates="phase")
    phase_error_buckets: List["PhaseErrorBucket"] = Relationship(back_populates="phase")

    def get_parent_phase_id(self) -> Optional[str]:
        """Get the immediate parent phase ID from phase_path"""
        if not self.phase_path:
            return None
        path_parts = self.phase_path.split('/')
        return path_parts[-1] if path_parts else None

    def get_root_phase_id(self) -> Optional[str]:
        """Get the root (first) phase ID from phase_path"""
        if not self.phase_path:
            return None
        path_parts = self.phase_path.split('/')
        return path_parts[0] if path_parts else None

    def get_depth(self) -> int:
        """Get the depth of this phase in the hierarchy (0 for root phases)"""
        if not self.phase_path:
            return 0
        return len(self.phase_path.split('/'))

    def build_child_path(self) -> str:
        """Build the phase_path that a child of this phase should have"""
        if not self.phase_path:
            return self.id
        return f"{self.phase_path}/{self.id}"


class ComposalDataset(SQLModel, table=True):
    """Composal dataset table for managing dataset collections"""
    __tablename__ = "composal_dataset"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    pipeline_id: str = Field(foreign_key="pipeline.id")
    phase_id: Optional[str] = Field(default=None, foreign_key="pipeline_phase.id")  # Phase that generated this composal dataset
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None)
    file_path: Optional[str] = Field(default=None)  # Path to the combined dataset file
    total_samples: int = Field(default=0)
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    pipeline: Pipeline = Relationship(back_populates="datasets")
    phase: Optional["PipelinePhase"] = Relationship(back_populates="composal_datasets")
    dataset_files: List["DatasetFile"] = Relationship(back_populates="parent_dataset")


class DatasetFile(SQLModel, table=True):
    """Dataset file table for tracking individual dataset files"""
    __tablename__ = "dataset_file"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    parent_dataset_id: str = Field(foreign_key="composal_dataset.id")
    file_path: str
    phase_id: str = Field(foreign_key="pipeline_phase.id")
    file_type: Optional[str] = Field(default=None, max_length=50)  # train, validation, test, generated
    sample_count: int = Field(default=0)
    status: str = Field(default="generating", max_length=50)  # generating, done
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    parent_dataset: ComposalDataset = Relationship(back_populates="dataset_files")
    phase: PipelinePhase = Relationship(back_populates="dataset_files")
    batch_files: List["BatchGeneratedDatasetFile"] = Relationship(back_populates="parent_dataset_file")


class BatchGeneratedDatasetFile(SQLModel, table=True):
    """Batch generated dataset file table for tracking incremental generation batches"""
    __tablename__ = "batch_generated_dataset_file"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    parent_dataset_file_id: str = Field(foreign_key="dataset_file.id")
    file_path: str  # Path to the batch temp file
    batch_number: int  # Sequential batch number (1, 2, 3, ...)
    sample_count: int = Field(default=0)  # Number of samples in this batch
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    parent_dataset_file: DatasetFile = Relationship(back_populates="batch_files")


class ErrorBucket(SQLModel, table=True):
    """Error bucket table for categorizing model errors"""
    __tablename__ = "error_bucket"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    pipeline_id: str = Field(foreign_key="pipeline.id")
    name: str = Field(max_length=255)
    description: str
    examples: str = Field()  # JSON serialized list of Sample objects
    data_generation_strategy: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    pipeline: Pipeline = Relationship(back_populates="error_buckets")
    phase_error_buckets: List["PhaseErrorBucket"] = Relationship(back_populates="bucket")

    def get_examples(self) -> List[dict]:
        """Parse examples JSON string to list of dicts"""
        return json.loads(self.examples)

class PhaseErrorBucket(SQLModel, table=True):
    """Phase error bucket table for tracking errors per phase"""
    __tablename__ = "phase_error_bucket"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    phase_id: str = Field(foreign_key="pipeline_phase.id")
    bucket_id: str = Field(foreign_key="error_bucket.id")
    error_count: int = Field(default=0)
    generation_count: int = Field(default=0)
    examples: str = Field()  # JSON serialized list of Sample objects
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    phase: PipelinePhase = Relationship(back_populates="phase_error_buckets")
    bucket: ErrorBucket = Relationship(back_populates="phase_error_buckets")

    def get_examples(self) -> List[dict]:
        """Parse examples JSON string to list of dicts"""
        return json.loads(self.examples)


class ErrorBucketPhase(SQLModel, table=True):
    """Error bucket phase table - cloned from base ErrorBucket with phase-specific error categorization"""
    __tablename__ = "error_bucket_phase"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    phase_id: str = Field(foreign_key="pipeline_phase.id")
    base_error_bucket_id: str = Field(foreign_key="error_bucket.id")  # The base bucket this was cloned from
    name: str = Field(max_length=255)  # Cloned from base bucket
    description: str  # Cloned from base bucket
    examples: str = Field()  # JSON list of examples with labels: [{"text": "...", "label": "...", "prediction": "..."}, ...]
    count: int = Field(default=0)  # Number of errors in this category for this phase
    data_generation_strategy: Optional[str] = Field(default=None)  # Cloned from base bucket
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def get_examples(self) -> List[dict]:
        """Parse examples JSON string to list of dicts"""
        return json.loads(self.examples)


class HumanTestSet(SQLModel, table=True):
    """Human test set table for storing custom human dataset files"""
    __tablename__ = "human_test_set"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    pipeline_id: Optional[str] = Field(default=None, foreign_key="pipeline.id")
    name: str = Field(max_length=255)
    file_path: str
    description: Optional[str] = Field(default=None)
    sample_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    evaluation_results: List["EvaluationResult"] = Relationship(back_populates="human_test_set")


class TrainedModel(SQLModel, table=True):
    """Trained model table for storing trained model information"""
    __tablename__ = "trained_model"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    phase_id: str = Field(foreign_key="pipeline_phase.id")
    model_name: str = Field(max_length=255)
    model_save_path: str
    training_time: Optional[float] = Field(default=None)  # Training time in seconds
    dataset_file_id: Optional[str] = Field(default=None, foreign_key="dataset_file.id")
    training_params: Optional[str] = Field(default=None)  # JSON string with training parameters
    status: str = Field(default="training", max_length=50)  # training, completed, failed
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = Field(default=None)

    # Relationships
    evaluation_results: List["EvaluationResult"] = Relationship(back_populates="trained_model")

    def get_training_params(self) -> Optional[dict]:
        """Parse training_params JSON string to dict"""
        return json.loads(self.training_params) if self.training_params else None


class EvaluationResult(SQLModel, table=True):
    """Evaluation result table for storing model evaluation results"""
    __tablename__ = "evaluation_result"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    trained_model_id: str = Field(foreign_key="trained_model.id")
    human_test_set_id: Optional[str] = Field(default=None, foreign_key="human_test_set.id")
    dataset_file_id: Optional[str] = Field(default=None, foreign_key="dataset_file.id")
    accuracy: Optional[float] = Field(default=None)
    precision: Optional[float] = Field(default=None)
    recall: Optional[float] = Field(default=None)
    f1_score: Optional[float] = Field(default=None)
    label_metrics: Optional[str] = Field(default=None)  # JSON string for per-label metrics: {"label_name": {"precision": 0.95, "recall": 0.92, "f1_score": 0.93, "support": 100}}
    metrics: Optional[str] = Field(default=None)  # JSON string for additional metrics
    evaluated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    trained_model: TrainedModel = Relationship(back_populates="evaluation_results")
    human_test_set: Optional[HumanTestSet] = Relationship(back_populates="evaluation_results")

    def get_metrics(self) -> Optional[dict]:
        """Parse metrics JSON string to dict"""
        return json.loads(self.metrics) if self.metrics else None

    def get_label_metrics(self) -> Optional[dict]:
        """Parse label_metrics JSON string to dict"""
        return json.loads(self.label_metrics) if self.label_metrics else None
