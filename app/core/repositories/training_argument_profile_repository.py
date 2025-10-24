from typing import Optional, List
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.models import TrainingArgumentProfile, utc_now


class TrainingArgumentProfileRepository:
    """Repository for TrainingArgumentProfile operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        name: str,
        training_config: str,
        lora_config: str,
        description: Optional[str] = None
    ) -> TrainingArgumentProfile:
        """Create a new training argument profile

        Args:
            name: Unique name for the profile
            training_config: JSON string with training configuration
            lora_config: JSON string with LoRA configuration
            description: Optional description of the profile

        Raises:
            ValueError: If a profile with the same name already exists
        """
        # Check if profile with this name already exists
        existing_profile = await self.get_by_name(name)
        if existing_profile:
            raise ValueError(
                f"A training argument profile with name '{name}' already exists. "
                f"Profile names must be unique."
            )

        profile = TrainingArgumentProfile(
            name=name,
            description=description,
            training_config=training_config,
            lora_config=lora_config
        )
        self.session.add(profile)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def get_by_id(self, profile_id: str) -> Optional[TrainingArgumentProfile]:
        """Get training argument profile by ID"""
        return await self.session.get(TrainingArgumentProfile, profile_id)

    async def get_by_name(self, name: str) -> Optional[TrainingArgumentProfile]:
        """Get training argument profile by name"""
        statement = select(TrainingArgumentProfile).where(
            TrainingArgumentProfile.name == name
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_all(self) -> List[TrainingArgumentProfile]:
        """Get all training argument profiles"""
        statement = select(TrainingArgumentProfile).order_by(
            TrainingArgumentProfile.created_at.desc()
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update(
        self,
        profile_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        training_config: Optional[str] = None,
        lora_config: Optional[str] = None
    ) -> Optional[TrainingArgumentProfile]:
        """Update training argument profile

        Args:
            profile_id: ID of the profile to update
            name: New name (must be unique if provided)
            description: New description
            training_config: New training configuration JSON string
            lora_config: New LoRA configuration JSON string

        Raises:
            ValueError: If trying to set a name that already exists
        """
        profile = await self.get_by_id(profile_id)
        if not profile:
            return None

        # Check name uniqueness if changing name
        if name and name != profile.name:
            existing = await self.get_by_name(name)
            if existing:
                raise ValueError(
                    f"A training argument profile with name '{name}' already exists. "
                    f"Profile names must be unique."
                )
            profile.name = name

        if description is not None:
            profile.description = description

        if training_config is not None:
            profile.training_config = training_config

        if lora_config is not None:
            profile.lora_config = lora_config

        profile.updated_at = utc_now()

        self.session.add(profile)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def delete(self, profile_id: str) -> bool:
        """Delete training argument profile

        Args:
            profile_id: ID of the profile to delete

        Returns:
            True if deleted, False if not found
        """
        profile = await self.get_by_id(profile_id)
        if not profile:
            return False

        await self.session.delete(profile)
        await self.session.commit()
        return True
