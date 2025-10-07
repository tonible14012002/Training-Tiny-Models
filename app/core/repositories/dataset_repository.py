from typing import Optional, List
from sqlmodel import Session, select

from app.core.models.models import ComposalDataset, DatasetFile


class DatasetRepository:
    """Repository for ComposalDataset operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        pipeline_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        total_samples: int = 0
    ) -> ComposalDataset:
        """Create a new dataset"""
        dataset = ComposalDataset(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            total_samples=total_samples
        )
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def get_by_id(self, dataset_id: str) -> Optional[ComposalDataset]:
        """Get dataset by ID"""
        return self.session.get(ComposalDataset, dataset_id)

    def get_by_pipeline(self, pipeline_id: str) -> List[ComposalDataset]:
        """Get all datasets for a pipeline"""
        statement = select(ComposalDataset).where(
            ComposalDataset.pipeline_id == pipeline_id
        )
        return list(self.session.exec(statement).all())

    def get_by_name(self, pipeline_id: str, name: str) -> Optional[ComposalDataset]:
        """Get dataset by name within a pipeline"""
        statement = select(ComposalDataset).where(
            ComposalDataset.pipeline_id == pipeline_id,
            ComposalDataset.name == name
        )
        return self.session.exec(statement).first()

    def update(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        total_samples: Optional[int] = None
    ) -> Optional[ComposalDataset]:
        """Update dataset"""
        dataset = self.get_by_id(dataset_id)
        if not dataset:
            return None

        if name:
            dataset.name = name
        if description:
            dataset.description = description
        if total_samples is not None:
            dataset.total_samples = total_samples

        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def increment_samples(self, dataset_id: str, count: int) -> Optional[ComposalDataset]:
        """Increment total samples count"""
        dataset = self.get_by_id(dataset_id)
        if not dataset:
            return None

        dataset.total_samples += count

        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def delete(self, dataset_id: str) -> bool:
        """Delete dataset"""
        dataset = self.get_by_id(dataset_id)
        if not dataset:
            return False

        self.session.delete(dataset)
        self.session.commit()
        return True

    def get_with_files(self, dataset_id: str) -> Optional[ComposalDataset]:
        """Get dataset with files loaded"""
        dataset = self.get_by_id(dataset_id)
        if dataset:
            _ = dataset.dataset_files
        return dataset


class DatasetFileRepository:
    """Repository for DatasetFile operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        parent_dataset_id: str,
        file_path: str,
        phase_id: str,
        file_type: Optional[str] = None,
        sample_count: int = 0
    ) -> DatasetFile:
        """Create a new dataset file"""
        dataset_file = DatasetFile(
            parent_dataset_id=parent_dataset_id,
            file_path=file_path,
            phase_id=phase_id,
            file_type=file_type,
            sample_count=sample_count
        )
        self.session.add(dataset_file)
        self.session.commit()
        self.session.refresh(dataset_file)
        return dataset_file

    def get_by_id(self, file_id: str) -> Optional[DatasetFile]:
        """Get dataset file by ID"""
        return self.session.get(DatasetFile, file_id)

    def get_by_dataset(self, dataset_id: str) -> List[DatasetFile]:
        """Get all files for a dataset"""
        statement = select(DatasetFile).where(
            DatasetFile.parent_dataset_id == dataset_id
        )
        return list(self.session.exec(statement).all())

    def get_by_phase(self, phase_id: str) -> List[DatasetFile]:
        """Get all files for a phase"""
        statement = select(DatasetFile).where(
            DatasetFile.phase_id == phase_id
        )
        return list(self.session.exec(statement).all())

    def get_by_type(self, dataset_id: str, file_type: str) -> List[DatasetFile]:
        """Get files by type for a dataset"""
        statement = select(DatasetFile).where(
            DatasetFile.parent_dataset_id == dataset_id,
            DatasetFile.file_type == file_type
        )
        return list(self.session.exec(statement).all())

    def get_by_path(self, file_path: str) -> Optional[DatasetFile]:
        """Get dataset file by path"""
        statement = select(DatasetFile).where(
            DatasetFile.file_path == file_path
        )
        return self.session.exec(statement).first()

    def update(
        self,
        file_id: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        sample_count: Optional[int] = None
    ) -> Optional[DatasetFile]:
        """Update dataset file"""
        dataset_file = self.get_by_id(file_id)
        if not dataset_file:
            return None

        if file_path:
            dataset_file.file_path = file_path
        if file_type:
            dataset_file.file_type = file_type
        if sample_count is not None:
            dataset_file.sample_count = sample_count

        self.session.add(dataset_file)
        self.session.commit()
        self.session.refresh(dataset_file)
        return dataset_file

    def delete(self, file_id: str) -> bool:
        """Delete dataset file"""
        dataset_file = self.get_by_id(file_id)
        if not dataset_file:
            return False

        self.session.delete(dataset_file)
        self.session.commit()
        return True

    def bulk_create(
        self,
        parent_dataset_id: str,
        phase_id: str,
        files_data: List[dict]
    ) -> List[DatasetFile]:
        """Create multiple dataset files at once"""
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

        self.session.commit()
        for file in files:
            self.session.refresh(file)

        return files

    def get_total_samples_by_phase(self, phase_id: str) -> int:
        """Get total sample count for all files in a phase"""
        statement = select(DatasetFile).where(
            DatasetFile.phase_id == phase_id
        )
        files = self.session.exec(statement).all()
        return sum(f.sample_count for f in files)

    def get_training_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all training files for a dataset"""
        return self.get_by_type(dataset_id, "train")

    def get_validation_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all validation files for a dataset"""
        return self.get_by_type(dataset_id, "validation")

    def get_test_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all test files for a dataset"""
        return self.get_by_type(dataset_id, "test")

    def get_generated_files(self, dataset_id: str) -> List[DatasetFile]:
        """Get all generated files for a dataset"""
        return self.get_by_type(dataset_id, "generated")
