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
    """Pipeline phase table for tracking training phases"""
    __tablename__ = "pipeline_phase"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    pipeline_id: str = Field(foreign_key="pipeline.id")
    previous_phase_id: Optional[str] = Field(default=None, foreign_key="pipeline_phase.id")
    phase_number: int
    checkpoint_id: Optional[str] = Field(default=None, max_length=100)
    checkpoint_path: Optional[str] = Field(default=None)
    status: str = Field(default="pending", max_length=50)  # pending, in_progress, completed, failed
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = Field(default=None)

    # Relationships
    pipeline: Pipeline = Relationship(back_populates="phases")
    dataset_files: List["DatasetFile"] = Relationship(back_populates="phase")
    phase_error_buckets: List["PhaseErrorBucket"] = Relationship(back_populates="phase")


class ComposalDataset(SQLModel, table=True):
    """Composal dataset table for managing dataset collections"""
    __tablename__ = "composal_dataset"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    pipeline_id: str = Field(foreign_key="pipeline.id")
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None)
    total_samples: int = Field(default=0)
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    pipeline: Pipeline = Relationship(back_populates="datasets")
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
    created_at: datetime = Field(default_factory=utc_now)

    # Relationships
    parent_dataset: ComposalDataset = Relationship(back_populates="dataset_files")
    phase: PipelinePhase = Relationship(back_populates="dataset_files")


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
