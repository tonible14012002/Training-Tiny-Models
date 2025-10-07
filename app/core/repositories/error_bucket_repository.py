from typing import Optional, List
from sqlmodel import Session, select

from app.core.models.models import ErrorBucket, PhaseErrorBucket, utc_now


class ErrorBucketRepository:
    """Repository for ErrorBucket operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
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
        self.session.commit()
        self.session.refresh(bucket)
        return bucket

    def get_by_id(self, bucket_id: str) -> Optional[ErrorBucket]:
        """Get error bucket by ID"""
        return self.session.get(ErrorBucket, bucket_id)

    def get_by_name(self, pipeline_id: str, name: str) -> Optional[ErrorBucket]:
        """Get error bucket by name within a pipeline"""
        statement = select(ErrorBucket).where(
            ErrorBucket.pipeline_id == pipeline_id,
            ErrorBucket.name == name
        )
        return self.session.exec(statement).first()

    def get_by_pipeline(self, pipeline_id: str) -> List[ErrorBucket]:
        """Get all error buckets for a pipeline"""
        statement = select(ErrorBucket).where(
            ErrorBucket.pipeline_id == pipeline_id
        )
        return list(self.session.exec(statement).all())

    def update(
        self,
        bucket_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        examples: Optional[str] = None,
        data_generation_strategy: Optional[str] = None
    ) -> Optional[ErrorBucket]:
        """Update error bucket"""
        bucket = self.get_by_id(bucket_id)
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
        self.session.commit()
        self.session.refresh(bucket)
        return bucket

    def delete(self, bucket_id: str) -> bool:
        """Delete error bucket"""
        bucket = self.get_by_id(bucket_id)
        if not bucket:
            return False

        self.session.delete(bucket)
        self.session.commit()
        return True

    def bulk_create(self, pipeline_id: str, buckets_data: List[dict]) -> List[ErrorBucket]:
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

        self.session.commit()
        for bucket in buckets:
            self.session.refresh(bucket)

        return buckets


class PhaseErrorBucketRepository:
    """Repository for PhaseErrorBucket operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
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
        self.session.commit()
        self.session.refresh(phase_bucket)
        return phase_bucket

    def get_by_id(self, phase_bucket_id: str) -> Optional[PhaseErrorBucket]:
        """Get phase error bucket by ID"""
        return self.session.get(PhaseErrorBucket, phase_bucket_id)

    def get_by_phase(self, phase_id: str) -> List[PhaseErrorBucket]:
        """Get all error buckets for a phase"""
        statement = select(PhaseErrorBucket).where(
            PhaseErrorBucket.phase_id == phase_id
        )
        return list(self.session.exec(statement).all())

    def get_by_phase_and_bucket(
        self,
        phase_id: str,
        bucket_id: str
    ) -> Optional[PhaseErrorBucket]:
        """Get specific phase error bucket"""
        statement = select(PhaseErrorBucket).where(
            PhaseErrorBucket.phase_id == phase_id,
            PhaseErrorBucket.bucket_id == bucket_id
        )
        return self.session.exec(statement).first()

    def update_counts(
        self,
        phase_bucket_id: str,
        error_count: Optional[int] = None,
        generation_count: Optional[int] = None
    ) -> Optional[PhaseErrorBucket]:
        """Update error and generation counts"""
        phase_bucket = self.get_by_id(phase_bucket_id)
        if not phase_bucket:
            return None

        if error_count is not None:
            phase_bucket.error_count = error_count
        if generation_count is not None:
            phase_bucket.generation_count = generation_count

        phase_bucket.updated_at = utc_now()

        self.session.add(phase_bucket)
        self.session.commit()
        self.session.refresh(phase_bucket)
        return phase_bucket

    def increment_error_count(self, phase_bucket_id: str, count: int = 1) -> Optional[PhaseErrorBucket]:
        """Increment error count"""
        phase_bucket = self.get_by_id(phase_bucket_id)
        if not phase_bucket:
            return None

        phase_bucket.error_count += count
        phase_bucket.updated_at = utc_now()

        self.session.add(phase_bucket)
        self.session.commit()
        self.session.refresh(phase_bucket)
        return phase_bucket

    def increment_generation_count(self, phase_bucket_id: str, count: int = 1) -> Optional[PhaseErrorBucket]:
        """Increment generation count"""
        phase_bucket = self.get_by_id(phase_bucket_id)
        if not phase_bucket:
            return None

        phase_bucket.generation_count += count
        phase_bucket.updated_at = utc_now()

        self.session.add(phase_bucket)
        self.session.commit()
        self.session.refresh(phase_bucket)
        return phase_bucket

    def update_examples(self, phase_bucket_id: str, examples: str) -> Optional[PhaseErrorBucket]:
        """Update examples"""
        phase_bucket = self.get_by_id(phase_bucket_id)
        if not phase_bucket:
            return None

        phase_bucket.examples = examples
        phase_bucket.updated_at = utc_now()

        self.session.add(phase_bucket)
        self.session.commit()
        self.session.refresh(phase_bucket)
        return phase_bucket

    def get_top_error_buckets(self, phase_id: str, limit: int = 10) -> List[PhaseErrorBucket]:
        """Get top error buckets by error count for a phase"""
        statement = select(PhaseErrorBucket).where(
            PhaseErrorBucket.phase_id == phase_id
        ).order_by(PhaseErrorBucket.error_count.desc()).limit(limit)
        return list(self.session.exec(statement).all())

    def delete(self, phase_bucket_id: str) -> bool:
        """Delete phase error bucket"""
        phase_bucket = self.get_by_id(phase_bucket_id)
        if not phase_bucket:
            return False

        self.session.delete(phase_bucket)
        self.session.commit()
        return True
