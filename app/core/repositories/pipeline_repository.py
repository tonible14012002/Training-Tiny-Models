from typing import Optional, List
from sqlmodel import Session, select

from app.core.models.models import (
    Pipeline,
    LabelConfig,
    PipelinePhase,
    utc_now
)


class PipelineRepository:
    """Repository for Pipeline operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str) -> Pipeline:
        """Create a new pipeline"""
        pipeline = Pipeline(name=name)
        self.session.add(pipeline)
        self.session.commit()
        self.session.refresh(pipeline)
        return pipeline

    def get_by_id(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID"""
        return self.session.get(Pipeline, pipeline_id)

    def get_all(self) -> List[Pipeline]:
        """Get all pipelines"""
        statement = select(Pipeline)
        return list(self.session.exec(statement).all())

    def update(self, pipeline_id: str, name: Optional[str] = None) -> Optional[Pipeline]:
        """Update pipeline"""
        pipeline = self.get_by_id(pipeline_id)
        if not pipeline:
            return None

        if name:
            pipeline.name = name
        pipeline.updated_at = utc_now()

        self.session.add(pipeline)
        self.session.commit()
        self.session.refresh(pipeline)
        return pipeline

    def delete(self, pipeline_id: str) -> bool:
        """Delete pipeline"""
        pipeline = self.get_by_id(pipeline_id)
        if not pipeline:
            return False

        self.session.delete(pipeline)
        self.session.commit()
        return True

    def get_with_phases(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline with all phases"""
        pipeline = self.get_by_id(pipeline_id)
        if pipeline:
            # Force load relationships
            _ = pipeline.phases
        return pipeline

    def get_with_all_relations(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline with all relationships loaded"""
        pipeline = self.get_by_id(pipeline_id)
        if pipeline:
            _ = pipeline.phases
            _ = pipeline.label_configs
            _ = pipeline.datasets
            _ = pipeline.error_buckets
        return pipeline

    def get_active_pipelines(self) -> List[Pipeline]:
        """Get pipelines with in-progress phases"""
        statement = select(Pipeline).join(PipelinePhase).where(
            PipelinePhase.status == "in_progress"
        )
        return list(self.session.exec(statement).all())

    def create_label_config(
        self,
        pipeline_id: str,
        name: str,
        id2label: str,
        label2id: str,
        label_explanation: Optional[str] = None
    ) -> Optional[LabelConfig]:
        """Create label config for pipeline"""
        pipeline = self.get_by_id(pipeline_id)
        if not pipeline:
            return None

        label_config = LabelConfig(
            pipeline_id=pipeline_id,
            name=name,
            id2label=id2label,
            label2id=label2id,
            label_explanation=label_explanation
        )
        self.session.add(label_config)
        self.session.commit()
        self.session.refresh(label_config)
        return label_config

    def get_label_config(self, pipeline_id: str) -> Optional[LabelConfig]:
        """Get label config for pipeline"""
        statement = select(LabelConfig).where(LabelConfig.pipeline_id == pipeline_id)
        return self.session.exec(statement).first()
