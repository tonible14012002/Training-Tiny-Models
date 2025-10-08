from typing import Optional, List
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.models import ErrorBucket, PhaseErrorBucket, utc_now
import json


class ErrorBucketRepository:
    """Repository for ErrorBucket operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        pipeline_id: str,
        name: str,
        description: str,
        examples: str,
        data_generation_strategy: Optional[str] = None
    ) -> ErrorBucket:
        """Create a new error bucket"""
        bucket = ErrorBucket(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            examples=examples,
            data_generation_strategy=data_generation_strategy
        )
        self.session.add(bucket)
        await self.session.commit()
        await self.session.refresh(bucket)
        return bucket

    async def get_by_id(self, bucket_id: str) -> Optional[ErrorBucket]:
        """Get error bucket by ID"""
        return await self.session.get(ErrorBucket, bucket_id)

    async def get_by_name(self, pipeline_id: str, name: str) -> Optional[ErrorBucket]:
        """Get error bucket by name within a pipeline"""
        statement = select(ErrorBucket).where(
            ErrorBucket.pipeline_id == pipeline_id,
            ErrorBucket.name == name
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def list_by_pipeline(self, pipeline_id: str) -> List[ErrorBucket]:
        """Get all error buckets for a pipeline"""
        statement = select(ErrorBucket).where(
            ErrorBucket.pipeline_id == pipeline_id
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update(
        self,
        bucket_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        examples: Optional[str] = None,
        data_generation_strategy: Optional[str] = None
    ) -> Optional[ErrorBucket]:
        """Update error bucket"""
        bucket = await self.get_by_id(bucket_id)
        if not bucket:
            return None

        if name:
            bucket.name = name
        if description:
            bucket.description = description
        if examples:
            bucket.examples = examples
        if data_generation_strategy is not None:
            bucket.data_generation_strategy = data_generation_strategy

        bucket.updated_at = utc_now()

        self.session.add(bucket)
        await self.session.commit()
        await self.session.refresh(bucket)
        return bucket

    async def delete(self, bucket_id: str) -> bool:
        """Delete error bucket"""
        bucket = await self.get_by_id(bucket_id)
        if not bucket:
            return False

        self.session.delete(bucket)
        await self.session.commit()
        return True

    async def bulk_create(self, pipeline_id: str, buckets_data: List[dict]) -> List[ErrorBucket]:
        """Create multiple error buckets at once"""
        buckets = []
        for data in buckets_data:
            bucket = ErrorBucket(
                pipeline_id=pipeline_id,
                name=data["name"],
                description=data["description"],
                examples=data["examples"],
                data_generation_strategy=data.get("data_generation_strategy")
            )
            buckets.append(bucket)
            self.session.add(bucket)

        await self.session.commit()
        for bucket in buckets:
            await self.session.refresh(bucket)

        return buckets


class PhaseErrorBucketRepository:
    """Repository for PhaseErrorBucket operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def batch_create(
        self,
        phase_id: str,
        buckets: List[ErrorBucket],
        error_counts: Optional[dict[str, int]] = None,
        generation_counts: Optional[dict[str, int]] = None,
        examples_data: Optional[dict[str, str]] = None
    ) -> List[PhaseErrorBucket]:
        """
        Create multiple phase error buckets by cloning from base error buckets.
        Ignores UNIQUE constraint violations for duplicate (phase_id, bucket_id) tuples.

        Args:
            phase_id: The phase ID to associate buckets with
            buckets: List of base ErrorBucket instances to clone
            error_counts: Optional dict mapping bucket_id to error_count
            generation_counts: Optional dict mapping bucket_id to generation_count
            examples_data: Optional dict mapping bucket_id to examples JSON string
        """
        phase_buckets = []
        error_counts = error_counts or {}
        generation_counts = generation_counts or {}
        examples_data = examples_data or {}

        for bucket in buckets:
            # Check if this phase error bucket already exists
            existing = await self.get_by_phase_and_bucket(phase_id, bucket.id)
            if existing:
                # Skip duplicate, add existing to result list
                phase_buckets.append(existing)
                continue

            phase_bucket = PhaseErrorBucket(
                phase_id=phase_id,
                bucket_id=bucket.id,
                error_count=error_counts.get(bucket.id, 0),
                generation_count=generation_counts.get(bucket.id, 0),
                examples=examples_data.get(bucket.id, "[]")
            )
            phase_buckets.append(phase_bucket)
            self.session.add(phase_bucket)

        await self.session.commit()
        for phase_bucket in phase_buckets:
            await self.session.refresh(phase_bucket)

        return phase_buckets

    async def create(
        self,
        phase_id: str,
        bucket_id: str,
        error_count: int = 0,
        generation_count: int = 0,
        examples: str = "[]"
    ) -> PhaseErrorBucket:
        """Create a new phase error bucket"""
        phase_bucket = PhaseErrorBucket(
            phase_id=phase_id,
            bucket_id=bucket_id,
            error_count=error_count,
            generation_count=generation_count,
            examples=examples
        )
        self.session.add(phase_bucket)
        await self.session.commit()
        await self.session.refresh(phase_bucket)
        return phase_bucket

    async def get_by_id(self, phase_bucket_id: str) -> Optional[PhaseErrorBucket]:
        """Get phase error bucket by ID"""
        return await self.session.get(PhaseErrorBucket, phase_bucket_id)

    async def list_by_phase(self, phase_id: str):
        """Get all error buckets for a phase"""
        statement = select(PhaseErrorBucket).where(
            PhaseErrorBucket.phase_id == phase_id
        ).order_by(PhaseErrorBucket.created_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())
    
    async def list_detail_by_phase(self, base_buckets: List[ErrorBucket], phase_id: str):
        phase_buckets = await self.list_by_phase(phase_id)
        base_bucket_map = {b.id: b for b in base_buckets}

        base_bucket_map = {b.id: b for b in base_buckets}
        # Only get latest bucket for same bucket name (there might be duplicates due to retries)
        latest_buckets = {}
        for bucket in phase_buckets:
            if bucket.bucket_id in base_bucket_map and base_bucket_map[bucket.bucket_id].name not in latest_buckets:
                latest_buckets[base_bucket_map[bucket.bucket_id].name] = {
                    "error": bucket.model_dump(),
                    "base_bucket": {
                        "name": base_bucket_map[bucket.bucket_id].name,
                        "description": base_bucket_map[bucket.bucket_id].description,
                    }
                }
                latest_buckets[base_bucket_map[bucket.bucket_id].name]["error"]["examples"]\
                    = json.loads(bucket.examples) if bucket.examples else []
        
        return latest_buckets

    async def get_by_phase_and_bucket(
        self,
        phase_id: str,
        bucket_id: str
    ) -> Optional[PhaseErrorBucket]:
        """Get specific phase error bucket"""
        statement = select(PhaseErrorBucket).where(
            PhaseErrorBucket.phase_id == phase_id,
            PhaseErrorBucket.bucket_id == bucket_id
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def delete(self, phase_bucket_id: str) -> bool:
        """Delete phase error bucket"""
        phase_bucket = await self.get_by_id(phase_bucket_id)
        if not phase_bucket:
            return False

        self.session.delete(phase_bucket)
        await self.session.commit()
        return True
