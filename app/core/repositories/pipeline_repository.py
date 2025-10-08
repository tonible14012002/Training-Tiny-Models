from typing import Optional, List
from sqlmodel import select
from sqlalchemy.orm import *
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.models import (
    Pipeline,
    LabelConfig,
    PipelinePhase,
    utc_now
)


class PipelineRepository:
    """Repository for Pipeline operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str):
        """Create a new pipeline"""
        pipeline = Pipeline(name=name)
        self.session.add(pipeline)
        await self.session.commit()
        await self.session.refresh(pipeline)
        return pipeline

    async def get_by_id(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID"""
        statement = select(Pipeline).where(Pipeline.id == pipeline_id).options(
            selectinload(Pipeline.label_configs)
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_all(self) -> List[Pipeline]:
        """Get all pipelines"""
        statement = select(Pipeline).options(selectinload(Pipeline.label_configs))
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update(self, pipeline_id: str, name: Optional[str] = None) -> Optional[Pipeline]:
        """Update pipeline"""
        pipeline = await self.get_by_id(pipeline_id)
        if not pipeline:
            return None

        if name:
            pipeline.name = name
        pipeline.updated_at = utc_now()

        self.session.add(pipeline)
        await self.session.commit()
        await self.session.refresh(pipeline)
        return pipeline

    async def delete(self, pipeline_id: str) -> bool:
        """Delete pipeline"""
        pipeline = await self.get_by_id(pipeline_id)
        if not pipeline:
            return False

        self.session.delete(pipeline)
        await self.session.commit()
        return True

    async def get_with_phases(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline with all phases"""
        statement = select(Pipeline).where(Pipeline.id == pipeline_id).options(
            selectinload(Pipeline.phases),
            selectinload(Pipeline.label_configs)
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_with_all_relations(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline with all relationships loaded"""
        statement = select(Pipeline).where(Pipeline.id == pipeline_id).options(
            selectinload(Pipeline.phases),
            selectinload(Pipeline.label_configs),
            selectinload(Pipeline.datasets),
            selectinload(Pipeline.error_buckets)
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_active_pipelines(self) -> List[Pipeline]:
        """Get pipelines with in-progress phases"""
        statement = select(Pipeline).join(PipelinePhase).where(
            PipelinePhase.status == "in_progress"
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def create_label_config(
        self,
        pipeline_id: str,
        name: str,
        id2label: str,
        label2id: str,
        label_explanation: Optional[str] = None
    ) -> Optional[LabelConfig]:
        """Create label config for pipeline"""
        pipeline = await self.get_by_id(pipeline_id)
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
        await self.session.commit()
        await self.session.refresh(label_config)
        return label_config

    async def get_label_config(self, pipeline_id: str) -> Optional[LabelConfig]:
        """Get label config for pipeline"""
        statement = select(LabelConfig).where(LabelConfig.pipeline_id == pipeline_id)
        result = await self.session.execute(statement)
        return result.scalars().first()
