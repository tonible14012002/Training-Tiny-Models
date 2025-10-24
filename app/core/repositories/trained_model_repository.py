from typing import Optional, List
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.models.models import TrainedModel, utc_now


class TrainedModelRepository:
    """Repository for TrainedModel operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        phase_id: str,
        model_name: str,
        model_save_path: str,
        train_type: str,
        dataset_file_id: Optional[str] = None,
        training_params: Optional[str] = None,
        training_argument_profile_id: Optional[str] = None,
        description: Optional[str] = None,
        status: str = "training"
    ) -> TrainedModel:
        """Create a new trained model record

        Args:
            phase_id: The phase ID this model belongs to
            model_name: Name of the model
            model_save_path: Path where the model is saved
            train_type: Type of training (e.g., FROM_SCRATCH, CONTINUE)
            dataset_file_id: Optional dataset file ID
            training_params: Optional JSON string with training parameters
            training_argument_profile_id: Optional training argument profile ID
            description: Optional textual description of how dataset was combined
            status: Training status (default: training)
        """

        trained_model = TrainedModel(
            phase_id=phase_id,
            model_name=model_name,
            model_save_path=model_save_path,
            train_type=train_type,
            dataset_file_id=dataset_file_id,
            training_params=training_params,
            training_argument_profile_id=training_argument_profile_id,
            description=description,
            status=status
        )
        self.session.add(trained_model)
        await self.session.commit()
        await self.session.refresh(trained_model)
        return trained_model

    async def get_by_id(self, model_id: str) -> Optional[TrainedModel]:
        """Get trained model by ID"""
        return await self.session.get(TrainedModel, model_id)

    async def get_by_phase(self, phase_id: str) -> List[TrainedModel]:
        """Get all trained models for a phase"""
        statement = select(TrainedModel).where(
            TrainedModel.phase_id == phase_id
        ).order_by(TrainedModel.created_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_phase_and_train_type(self, phase_id: str, train_type: str) -> Optional[TrainedModel]:
        """Get a trained model by phase_id and train_type

        Args:
            phase_id: The phase ID
            train_type: The training type (e.g., FROM_SCRATCH, CONTINUE)

        Returns:
            The TrainedModel if found, None otherwise
        """
        statement = select(TrainedModel).where(
            TrainedModel.phase_id == phase_id,
            TrainedModel.train_type == train_type
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_latest_by_phase(self, phase_id: str) -> Optional[TrainedModel]:
        """Get the latest trained model for a phase"""
        statement = select(TrainedModel).where(
            TrainedModel.phase_id == phase_id
        ).order_by(TrainedModel.created_at.desc())
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_by_status(self, status: str) -> List[TrainedModel]:
        """Get all trained models by status"""
        statement = select(TrainedModel).where(
            TrainedModel.status == status
        ).order_by(TrainedModel.created_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_dataset_file(self, dataset_file_id: str) -> List[TrainedModel]:
        """Get all trained models associated with a dataset file"""
        statement = select(TrainedModel).where(
            TrainedModel.dataset_file_id == dataset_file_id
        ).order_by(TrainedModel.created_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_completed_models(self, phase_id: Optional[str] = None) -> List[TrainedModel]:
        """Get all completed trained models, optionally filtered by phase"""
        statement = select(TrainedModel).where(
            TrainedModel.status == "completed"
        )
        if phase_id:
            statement = statement.where(TrainedModel.phase_id == phase_id)

        statement = statement.order_by(TrainedModel.completed_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update_status(
        self,
        model_id: str,
        status: str,
        completed_at: Optional[datetime] = None,
        training_time: Optional[float] = None
    ) -> Optional[TrainedModel]:
        """Update trained model status"""
        trained_model = await self.get_by_id(model_id)
        if not trained_model:
            return None

        trained_model.status = status

        if completed_at:
            trained_model.completed_at = completed_at
        elif status == "completed":
            trained_model.completed_at = utc_now()

        if training_time is not None:
            trained_model.training_time = training_time

        self.session.add(trained_model)
        await self.session.commit()
        await self.session.refresh(trained_model)
        return trained_model

    async def update(
        self,
        model_id: str,
        model_name: Optional[str] = None,
        model_save_path: Optional[str] = None,
        training_params: Optional[str] = None,
        training_time: Optional[float] = None
    ) -> Optional[TrainedModel]:
        """Update trained model information"""
        trained_model = await self.get_by_id(model_id)
        if not trained_model:
            return None

        if model_name:
            trained_model.model_name = model_name
        if model_save_path:
            trained_model.model_save_path = model_save_path
        if training_params is not None:
            trained_model.training_params = training_params
        if training_time is not None:
            trained_model.training_time = training_time

        self.session.add(trained_model)
        await self.session.commit()
        await self.session.refresh(trained_model)
        return trained_model

    async def delete(self, model_id: str) -> bool:
        """Delete trained model"""
        trained_model = await self.get_by_id(model_id)
        if not trained_model:
            return False

        self.session.delete(trained_model)
        await self.session.commit()
        return True

    async def get_with_evaluations(self, model_id: str) -> Optional[TrainedModel]:
        """Get trained model with evaluation results loaded"""
        trained_model = await self.get_by_id(model_id)
        if trained_model:
            _ = trained_model.evaluation_results
        return trained_model

    async def get_all_by_model_name(self, model_name: str) -> List[TrainedModel]:
        """Get all trained models by model name"""
        statement = select(TrainedModel).where(
            TrainedModel.model_name == model_name
        ).order_by(TrainedModel.created_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_latest_by_phase_ids(self, phase_ids: List[str]) -> dict[str, Optional[TrainedModel]]:
        """Get the latest trained model for each phase

        Args:
            phase_ids: List of phase IDs to get trained models for

        Returns:
            Dictionary mapping phase_id to the latest TrainedModel for that phase
        """
        from sqlalchemy import func

        # Subquery to get the latest created_at for each phase
        latest_subquery = (
            select(
                TrainedModel.phase_id,
                func.max(TrainedModel.created_at).label("max_created_at")
            )
            .where(TrainedModel.phase_id.in_(phase_ids))
            .group_by(TrainedModel.phase_id)
            .subquery()
        )

        # Get the trained models
        statement = (
            select(TrainedModel)
            .join(
                latest_subquery,
                (TrainedModel.phase_id == latest_subquery.c.phase_id) &
                (TrainedModel.created_at == latest_subquery.c.max_created_at)
            )
        )
        result = await self.session.execute(statement)
        trained_models = list(result.scalars().all())

        # Build result dict
        result_dict: dict[str, Optional[TrainedModel]] = {phase_id: None for phase_id in phase_ids}
        for model in trained_models:
            result_dict[model.phase_id] = model

        return result_dict
