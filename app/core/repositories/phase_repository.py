from typing import Optional, List
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.models.models import PipelinePhase, utc_now


class PhaseRepository:
    """Repository for PipelinePhase operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        pipeline_id: str,
        phase_number: int,
        checkpoint_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        previous_phase_id: Optional[str] = None,
        status: str = "pending"
    ) -> PipelinePhase:
        """Create a new phase"""
        phase = PipelinePhase(
            pipeline_id=pipeline_id,
            phase_number=phase_number,
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
            previous_phase_id=previous_phase_id,
            status=status
        )
        self.session.add(phase)
        await self.session.commit()
        await self.session.refresh(phase)
        return phase

    async def get_by_id(self, phase_id: str) -> Optional[PipelinePhase]:
        """Get phase by ID"""
        return await self.session.get(PipelinePhase, phase_id)

    async def get_by_pipeline(self, pipeline_id: str) -> List[PipelinePhase]:
        """Get all phases for a pipeline"""
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id
        ).order_by(PipelinePhase.phase_number)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_phase_number(self, pipeline_id: str, phase_number: int) -> Optional[PipelinePhase]:
        """Get phase by pipeline ID and phase number"""
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id,
            PipelinePhase.phase_number == phase_number
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_latest_phase(self, pipeline_id: str) -> Optional[PipelinePhase]:
        """Get the latest phase for a pipeline"""
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id
        ).order_by(PipelinePhase.phase_number.desc())
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_by_checkpoint_id(self, checkpoint_id: str) -> Optional[PipelinePhase]:
        """Get phase by checkpoint ID"""
        statement = select(PipelinePhase).where(
            PipelinePhase.checkpoint_id == checkpoint_id
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def update_status(
        self,
        phase_id: str,
        status: str,
        completed_at: Optional[datetime] = None
    ):
        """Update phase status"""
        phase = await self.get_by_id(phase_id)
        if not phase:
            return None

        phase.status = status
        if completed_at:
            phase.completed_at = completed_at
        elif status == "completed":
            phase.completed_at = utc_now()

        self.session.add(phase)
        await self.session.commit()
        await self.session.refresh(phase)
        return phase

    async def update_checkpoint(
        self,
        phase_id: str,
        checkpoint_id: str,
        checkpoint_path: str
    ) -> Optional[PipelinePhase]:
        """Update phase checkpoint information"""
        phase = await self.get_by_id(phase_id)
        if not phase:
            return None

        phase.checkpoint_id = checkpoint_id
        phase.checkpoint_path = checkpoint_path

        self.session.add(phase)
        await self.session.commit()
        await self.session.refresh(phase)
        return phase

    async def get_completed_phases(self, pipeline_id: str) -> List[PipelinePhase]:
        """Get all completed phases for a pipeline"""
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id,
            PipelinePhase.status == "completed"
        ).order_by(PipelinePhase.phase_number)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_in_progress_phase(self, pipeline_id: str) -> Optional[PipelinePhase]:
        """Get the in-progress phase for a pipeline"""
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id,
            PipelinePhase.status == "in_progress"
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_pending_phases(self, pipeline_id: str) -> List[PipelinePhase]:
        """Get all pending phases for a pipeline"""
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id,
            PipelinePhase.status == "pending"
        ).order_by(PipelinePhase.phase_number)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def delete(self, phase_id: str) -> bool:
        """Delete phase"""
        phase = await self.get_by_id(phase_id)
        if not phase:
            return False

        self.session.delete(phase)
        await self.session.commit()
        return True

    async def get_phase_chain(self, phase_id: str) -> List[PipelinePhase]:
        """Get the chain of phases leading to this phase"""
        phases = []
        current_phase = await self.get_by_id(phase_id)

        while current_phase:
            phases.insert(0, current_phase)
            if current_phase.previous_phase_id:
                current_phase = await self.get_by_id(current_phase.previous_phase_id)
            else:
                break

        return phases

    async def get_with_error_buckets(self, phase_id: str) -> Optional[PipelinePhase]:
        """Get phase with error buckets loaded"""
        phase = await self.get_by_id(phase_id)
        if phase:
            _ = phase.phase_error_buckets
        return phase
