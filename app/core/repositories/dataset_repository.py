from typing import Optional, List
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.models import ComposalDataset, DatasetFile


class DatasetRepository:
    """Repository for ComposalDataset operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_composal_ds(
        self,
        pipeline_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        total_samples: int = 0
    ) -> ComposalDataset:
        """Create a new composal dataset (collection/container for dataset files)"""
        dataset = ComposalDataset(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            file_path=file_path,
            total_samples=total_samples
        )
        self.session.add(dataset)
        await self.session.commit()
        await self.session.refresh(dataset)
        return dataset

    async def get_by_id(self, dataset_id: str) -> Optional[ComposalDataset]:
        """Get dataset by ID"""
        return await self.session.get(ComposalDataset, dataset_id)

    async def get_by_pipeline(self, pipeline_id: str) -> List[ComposalDataset]:
        """Get all datasets for a pipeline"""
        statement = select(ComposalDataset).where(
            ComposalDataset.pipeline_id == pipeline_id
        ).order_by(ComposalDataset.created_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_name(self, pipeline_id: str, name: str) -> Optional[ComposalDataset]:
        """Get dataset by name within a pipeline"""
        statement = select(ComposalDataset).where(
            ComposalDataset.pipeline_id == pipeline_id,
            ComposalDataset.name == name
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def update(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        total_samples: Optional[int] = None
    ) -> Optional[ComposalDataset]:
        """Update dataset"""
        dataset = await self.get_by_id(dataset_id)
        if not dataset:
            return None

        if name:
            dataset.name = name
        if description:
            dataset.description = description
        if file_path:
            dataset.file_path = file_path
        if total_samples is not None:
            dataset.total_samples = total_samples

        self.session.add(dataset)
        await self.session.commit()
        await self.session.refresh(dataset)
        return dataset

    async def increment_samples(self, dataset_id: str, count: int) -> Optional[ComposalDataset]:
        """Increment total samples count"""
        dataset = await self.get_by_id(dataset_id)
        if not dataset:
            return None

        dataset.total_samples += count

        self.session.add(dataset)
        await self.session.commit()
        await self.session.refresh(dataset)
        return dataset

    async def delete(self, dataset_id: str) -> bool:
        """Delete dataset"""
        dataset = await self.get_by_id(dataset_id)
        if not dataset:
            return False

        self.session.delete(dataset)
        await self.session.commit()
        return True

    async def get_with_files(self, dataset_id: str) -> Optional[ComposalDataset]:
        """Get dataset with files loaded"""
        dataset = await self.get_by_id(dataset_id)
        if dataset:
            _ = dataset.dataset_files
        return dataset


class DatasetFileRepository:
    """Repository for DatasetFile operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_dataset_file(
        self,
        parent_dataset_id: str,
        file_path: str,
        phase_id: str,
        file_type: Optional[str] = None,
        sample_count: int = 0
    ) -> DatasetFile:
        """Create a new dataset file entry (individual file within a composal dataset)"""
        dataset_file = DatasetFile(
            parent_dataset_id=parent_dataset_id,
            file_path=file_path,
            phase_id=phase_id,
            file_type=file_type,
            sample_count=sample_count
        )
        self.session.add(dataset_file)
        await self.session.commit()
        await self.session.refresh(dataset_file)
        return dataset_file

    async def get_by_id(self, file_id: str) -> Optional[DatasetFile]:
        """Get dataset file by ID"""
        return await self.session.get(DatasetFile, file_id)

    async def get_by_dataset(self, dataset_id: str) -> List[DatasetFile]:
        """Get all files for a dataset"""
        statement = select(DatasetFile).where(
            DatasetFile.parent_dataset_id == dataset_id
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_phase(self, phase_id: str) -> List[DatasetFile]:
        """Get all files for a phase"""
        statement = select(DatasetFile).where(
            DatasetFile.phase_id == phase_id
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_type(self, dataset_id: str, file_type: str) -> List[DatasetFile]:
        """Get files by type for a dataset"""
        statement = select(DatasetFile).where(
            DatasetFile.parent_dataset_id == dataset_id,
            DatasetFile.file_type == file_type
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_path(self, file_path: str) -> Optional[DatasetFile]:
        """Get dataset file by path"""
        statement = select(DatasetFile).where(
            DatasetFile.file_path == file_path
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def update(
        self,
        file_id: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        sample_count: Optional[int] = None
    ) -> Optional[DatasetFile]:
        """Update dataset file"""
        dataset_file = await self.get_by_id(file_id)
        if not dataset_file:
            return None

        if file_path:
            dataset_file.file_path = file_path
        if file_type:
            dataset_file.file_type = file_type
        if sample_count is not None:
            dataset_file.sample_count = sample_count

        self.session.add(dataset_file)
        await self.session.commit()
        await self.session.refresh(dataset_file)
        return dataset_file

    async def delete(self, file_id: str) -> bool:
        """Delete dataset file"""
        dataset_file = await self.get_by_id(file_id)
        if not dataset_file:
            return False

        self.session.delete(dataset_file)
        await self.session.commit()
        return True

    async def bulk_create_dataset_files(
        self,
        parent_dataset_id: str,
        phase_id: str,
        files_data: List[dict]
    ) -> List[DatasetFile]:
        """Create multiple dataset file entries at once"""
        files = []
        for data in files_data:
            dataset_file = DatasetFile(
                parent_dataset_id=parent_dataset_id,
                file_path=data["file_path"],
                phase_id=phase_id,
                file_type=data.get("file_type"),
                sample_count=data.get("sample_count", 0)
            )
            files.append(dataset_file)
            self.session.add(dataset_file)

        await self.session.commit()
        for file in files:
            await self.session.refresh(file)

        return files

    async def get_total_samples_by_phase(self, phase_id: str) -> int:
        """Get total sample count for all files in a phase"""
        statement = select(DatasetFile).where(
            DatasetFile.phase_id == phase_id
        )
        result = await self.session.execute(statement)
        files = result.scalars().all()
        return sum(f.sample_count for f in files)

    async def get_training_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all training files for a dataset"""
        return await self.get_by_type(dataset_id, "train")

    async def get_validation_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all validation files for a dataset"""
        return await self.get_by_type(dataset_id, "validation")

    async def get_test_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all test files for a dataset"""
        return await self.get_by_type(dataset_id, "test")

    async def get_generated_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all generated files for a dataset"""
        return await self.get_by_type(dataset_id, "generated")
