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
        """Create a new phase

        Args:
            pipeline_id: ID of the pipeline
            phase_number: Phase number
            checkpoint_id: Optional checkpoint ID
            checkpoint_path: Optional checkpoint path
            previous_phase_id: Optional previous phase ID (will be used to build phase_path)
                - If None: creates a base phase with phase_path=""
                - If provided: appends new phase ID to previous phase's path
            status: Phase status (default: "pending")

        Returns:
            The created PipelinePhase

        Phase path structure:
            - Base phase: phase_path=""
            - 1st child: phase_path="<base_phase_id>"
            - 2nd child: phase_path="<base_phase_id>/<1st_child_id>"
            - nth child: phase_path="<base_phase_id>/.../<n-1_phase_id>"
        """
        # Build phase_path from previous phase
        phase_path = ""
        if previous_phase_id:
            previous_phase = await self.get_by_id(previous_phase_id)
            if previous_phase:
                # Append previous phase ID to its path
                if previous_phase.phase_path == "":
                    # Previous is base phase, so path is just its ID
                    phase_path = previous_phase.id
                else:
                    # Previous has a path, append its ID
                    phase_path = f"{previous_phase.phase_path}/{previous_phase.id}"

        phase = PipelinePhase(
            pipeline_id=pipeline_id,
            phase_number=phase_number,
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
            phase_path=phase_path,
            status=status
        )
        self.session.add(phase)
        await self.session.commit()
        await self.session.refresh(phase)
        return phase

    async def get_by_id(
        self,
        phase_id: str,
        include_composal_datasets: bool = False,
        include_dataset_files: bool = False,
        include_phase_error_buckets: bool = False,
        include_trained_models: bool = False
    ) -> Optional[PipelinePhase]:
        """
        Get phase by ID with optional relationship prefetching

        Args:
            phase_id: The phase ID to fetch
            include_composal_datasets: Whether to prefetch composal datasets
            include_dataset_files: Whether to prefetch dataset files
            include_phase_error_buckets: Whether to prefetch phase error buckets
            include_trained_models: Whether to prefetch trained models (not implemented yet)

        Returns:
            PipelinePhase or None if not found
        """
        # If no relationships requested, use simple get
        if not any([include_composal_datasets, include_dataset_files, include_phase_error_buckets, include_trained_models]):
            return await self.session.get(PipelinePhase, phase_id)

        # Build query with selectinload for requested relationships
        from sqlalchemy.orm import selectinload
        options = []

        if include_composal_datasets:
            options.append(selectinload(PipelinePhase.composal_datasets))
        if include_dataset_files:
            options.append(selectinload(PipelinePhase.dataset_files))
        if include_phase_error_buckets:
            options.append(selectinload(PipelinePhase.phase_error_buckets))
        # Note: trained_models relationship not in model yet, would need to be added

        statement = select(PipelinePhase).where(PipelinePhase.id == phase_id).options(*options)
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_by_pipeline(
        self,
        pipeline_id: str,
        include_composal_datasets: bool = False,
        include_dataset_files: bool = False,
        include_phase_error_buckets: bool = False,
        include_trained_models: bool = False
    ) -> List[PipelinePhase]:
        """
        Get all phases for a pipeline with optional relationship prefetching

        Args:
            pipeline_id: The pipeline ID
            include_composal_datasets: Whether to prefetch composal datasets
            include_dataset_files: Whether to prefetch dataset files
            include_phase_error_buckets: Whether to prefetch phase error buckets
            include_trained_models: Whether to prefetch trained models

        Returns:
            List of PipelinePhase objects
        """
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id
        ).order_by(PipelinePhase.phase_number)

        # Add relationship prefetching if requested
        if any([include_composal_datasets, include_dataset_files, include_phase_error_buckets, include_trained_models]):
            from sqlalchemy.orm import selectinload
            options = []

            if include_composal_datasets:
                options.append(selectinload(PipelinePhase.composal_datasets))
            if include_dataset_files:
                options.append(selectinload(PipelinePhase.dataset_files))
            if include_phase_error_buckets:
                options.append(selectinload(PipelinePhase.phase_error_buckets))

            statement = statement.options(*options)

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
        """Get the chain of phases leading to this phase (from root to current)

        Returns phases in order from root (phase_path='') to the current phase.
        """
        current_phase = await self.get_by_id(phase_id)
        if not current_phase:
            return []

        phases = []

        # If phase has no path, it's a root phase
        if not current_phase.phase_path:
            return [current_phase]

        # Get all parent phase IDs from the path
        parent_ids = current_phase.phase_path.split('/')

        # Fetch all parent phases
        for parent_id in parent_ids:
            parent_phase = await self.get_by_id(parent_id)
            if parent_phase:
                phases.append(parent_phase)

        # Add the current phase at the end
        phases.append(current_phase)

        return phases

    async def get_with_error_buckets(self, phase_id: str) -> Optional[PipelinePhase]:
        """Get phase with error buckets loaded"""
        phase = await self.get_by_id(phase_id)
        if phase:
            _ = phase.phase_error_buckets
        return phase

    async def get_with_all_relations(self, phase_id: str) -> Optional[PipelinePhase]:
        """Get phase with all relationships loaded"""
        from sqlalchemy.orm import selectinload
        statement = select(PipelinePhase).where(PipelinePhase.id == phase_id).options(
            selectinload(PipelinePhase.composal_datasets),
            selectinload(PipelinePhase.dataset_files),
            selectinload(PipelinePhase.phase_error_buckets)
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_phases_by_sequence(self, pipeline_id: str, root_phase_id: str) -> List[PipelinePhase]:
        """Get all phases in a sequence (same root phase)

        Args:
            pipeline_id: The pipeline ID
            root_phase_id: The root phase ID (first phase in the sequence)

        Returns:
            List of phases in the sequence, ordered by phase_number
        """
        # Get root phase (empty phase_path)
        root_phase = await self.get_by_id(root_phase_id)
        if not root_phase or root_phase.phase_path != "":
            return []

        # Get all phases that start with this root phase ID in their path
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id,
            (PipelinePhase.id == root_phase_id) |
            (PipelinePhase.phase_path.like(f"{root_phase_id}%"))
        ).order_by(PipelinePhase.phase_number)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_root_phases(self, pipeline_id: str) -> List[PipelinePhase]:
        """Get all root phases (phase_path = '') for a pipeline

        These are the starting phases of each sequence (typically phase_number = 0).
        """
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == pipeline_id,
            PipelinePhase.phase_path == ""
        ).order_by(PipelinePhase.created_at)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_child_phases(self, phase_id: str) -> List[PipelinePhase]:
        """Get all nested children of a phase using phase_path LIKE query

        Queries all phases where phase_path starts with the given phase_id.
        This returns all descendants (children, grandchildren, etc.).

        Args:
            phase_id: The phase ID to find children for

        Returns:
            List of all nested child phases, ordered by depth in hierarchy

        Examples:
            If phase_id="abc", finds phases with:
            - phase_path="abc" (direct child)
            - phase_path="abc/def" (grandchild)
            - phase_path="abc/def/ghi" (great-grandchild)
            etc.
        """
        phase = await self.get_by_id(phase_id)
        if not phase:
            return []

        # Find all phases where phase_path starts with this phase's ID
        # Match either exact (direct children) or starts with phase_id/ (deeper descendants)
        statement = select(PipelinePhase).where(
            PipelinePhase.pipeline_id == phase.pipeline_id,
            (PipelinePhase.phase_path == phase_id) |
            (PipelinePhase.phase_path.like(f"{phase_id}/%"))
        )

        result = await self.session.execute(statement)
        phases = list(result.scalars().all())

        # Order by depth (number of IDs in phase_path)
        phases.sort(key=lambda p: len(p.phase_path.split('/')))

        return phases

    async def get_previous_phase(self, phase_id: str) -> Optional[PipelinePhase]:
        """Get the immediate previous phase by querying the last phase_id in phase_path

        For a phase with phase_path="a/b/c", returns the phase with id="c".

        Args:
            phase_id: The phase ID to find the previous phase for

        Returns:
            The previous phase, or None if this is a base phase or phase not found
        """
        phase = await self.get_by_id(phase_id)
        if not phase or not phase.phase_path:
            # Base phase or not found - no previous phase
            return None

        previous_phase_id = phase.get_previous_phase_id()
        if not previous_phase_id:
            return None

        return await self.get_by_id(previous_phase_id)

    async def get_parent_phase(self, phase_id: str) -> Optional[PipelinePhase]:
        """Get the parent (root/base) phase by querying the first phase_id in phase_path

        For a phase with phase_path="a/b/c", returns the phase with id="a" and phase_path="".

        Args:
            phase_id: The phase ID to find the parent phase for

        Returns:
            The parent/root phase, or None if this is a base phase or phase not found
        """
        phase = await self.get_by_id(phase_id)
        if not phase or not phase.phase_path:
            # Already a base phase or not found
            return None

        parent_phase_id = phase.get_parent_phase_id()
        if not parent_phase_id:
            return None

        # Query for phase with this ID and empty phase_path (should be base phase)
        statement = select(PipelinePhase).where(
            PipelinePhase.id == parent_phase_id,
            PipelinePhase.phase_path == ""
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_all_ancestors(self, phase_id: str) -> List[PipelinePhase]:
        """Get all ancestor phases from root to immediate previous

        For a phase with phase_path="a/b/c", returns phases [a, b, c] in order.

        Args:
            phase_id: The phase ID to find ancestors for

        Returns:
            List of ancestor phases ordered from root to immediate previous
        """
        phase = await self.get_by_id(phase_id)
        if not phase or not phase.phase_path:
            return []

        ancestor_ids = phase.get_ancestor_phase_ids()
        if not ancestor_ids:
            return []

        # Query all ancestors by their IDs
        statement = select(PipelinePhase).where(
            PipelinePhase.id.in_(ancestor_ids)
        )
        result = await self.session.execute(statement)
        ancestor_map = {p.id: p for p in result.scalars().all()}

        # Return in order from root to immediate previous
        return [ancestor_map[ancestor_id] for ancestor_id in ancestor_ids if ancestor_id in ancestor_map]

    async def get_direct_children(self, phase_id: str) -> List[PipelinePhase]:
        """Get only direct children (not nested descendants)

        Queries phases where phase_path exactly equals the given phase_id.

        Args:
            phase_id: The phase ID to find direct children for

        Returns:
            List of direct child phases
        """
        phase = await self.get_by_id(phase_id)
        if not phase:
            return []

        # For base phase, children have phase_path=phase_id
        # For non-base phase, children have phase_path=parent_path/phase_id
        if phase.phase_path == "":
            # This is a base phase, children have path=phase_id
            statement = select(PipelinePhase).where(
                PipelinePhase.pipeline_id == phase.pipeline_id,
                PipelinePhase.phase_path == phase_id
            )
        else:
            # Non-base phase, children have path=current_path/phase_id
            child_path = phase.build_child_path()
            statement = select(PipelinePhase).where(
                PipelinePhase.pipeline_id == phase.pipeline_id,
                PipelinePhase.phase_path == child_path
            )

        result = await self.session.execute(statement)
        return list(result.scalars().all())
